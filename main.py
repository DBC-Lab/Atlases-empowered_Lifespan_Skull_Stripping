import os
import numpy as np
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.transforms import AsDiscrete,Activations,Compose
from networks.net import NET
from utils.data_utils import get_loader
from trainer import run_training
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from functools import partial
import argparse

parser = argparse.ArgumentParser(description='Skull stripping pipeline')
parser.add_argument('--checkpoint', default=None, help='start training from saved checkpoint')
parser.add_argument('--logdir', default='train', type=str, help='directory to save the tensorboard logs')
parser.add_argument('--pretrained_dir', default='./pretrained_models/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--data_dir', default='./Training_subjects/', type=str, help='dataset directory')
parser.add_argument('--json_list', default='./dataset/training_subjects.json', type=str, help='dataset json file')
parser.add_argument('--pretrained_model_name', default='model_best_acc.pt', type=str, help='pretrained model name')
parser.add_argument('--save_checkpoint', default=True, action='store_true', help='save checkpoint during training')
parser.add_argument('--max_epochs', default=5000, type=int, help='max number of training epochs')
parser.add_argument('--batch_size', default=1, type=int, help='number of batch size')
parser.add_argument('--sw_batch_size', default=1, type=int, help='number of sliding window batch size')
parser.add_argument('--optim_lr', default=1e-4, type=float, help='optimization learning rate')
parser.add_argument('--optim_name', default='adamw', type=str, help='optimization algorithm')
parser.add_argument('--reg_weight', default=1e-5, type=float, help='regularization weight')
parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
parser.add_argument('--val_every', default=100, type=int, help='validation frequency')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--workers', default=2, type=int, help='number of workers')
parser.add_argument('--model_name', default='net', type=str, help='model name')
parser.add_argument('--pos_embed', default='perceptron', type=str, help='type of position embedding')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization layer type in decoder')
parser.add_argument('--num_heads', default=12, type=int, help='number of attention heads in ViT encoder')
parser.add_argument('--mlp_dim', default=3072, type=int, help='mlp dimention in ViT encoder')
parser.add_argument('--hidden_size', default=768, type=int, help='hidden size dimention in ViT encoder')
parser.add_argument('--feature_size', default=16, type=int, help='feature size dimention')
parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=2, type=int, help='number of output channels')
parser.add_argument('--res_block', action='store_true', help='use residual blocks')
parser.add_argument('--conv_block', action='store_true', help='use conv blocks')
parser.add_argument('--roi_x', default=128, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=128, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=128, type=int, help='roi size in z direction')
parser.add_argument('--infer_overlap', default=0.8, type=float, help='sliding window inference overlap')
parser.add_argument('--lrschedule', default='warmup_cosine', type=str, help='type of learning rate scheduler')
parser.add_argument('--warmup_epochs', default=50, type=int, help='number of warmup epochs')
parser.add_argument('--resume_ckpt', default=False, action='store_true', help='resume training from pretrained checkpoint')
parser.add_argument('--resume_jit', action='store_true', help='resume training from pretrained torchscript checkpoint')


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = './runs/' + args.logdir

    main_worker(gpu=0, args=args)


def main_worker(gpu, args):

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    args.gpu = gpu

    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    loader = get_loader(args)
    print(args.rank, ' gpu', args.gpu)
    if args.rank == 0:
        print('Batch size is:', args.batch_size, 'epochs', args.max_epochs)
    pretrained_dir = args.pretrained_dir
    if (args.model_name is None) or args.model_name == 'net':
        model = NET(
            device = args.gpu,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            norm_name=args.norm_name,
            conv_block=True,
            res_block=False)

        if args.resume_ckpt:
            model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))
            model.load_state_dict(model_dict['state_dict'])
            print('Use pretrained weights')

        if args.resume_jit:
            if not args.noamp:
                print('Training from pre-trained checkpoint does not support AMP\nAMP is disabled.')
                args.amp = args.noamp
            model = torch.jit.load(os.path.join(pretrained_dir, args.pretrained_model_name))
    else:
        raise ValueError('Unsupported model ' + str(args.model_name))


    dice_acc = DiceMetric(include_background=True,
                          reduction=MetricReduction.MEAN,
                          get_not_nans=True)

    post_label = AsDiscrete(to_onehot=True,
                            n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True,
                           to_onehot=True,
                           n_classes=args.out_channels)
    dice_acc = DiceMetric(include_background=True,
                          reduction=MetricReduction.MEAN,
                          get_not_nans=True)
    model_inferer = partial(sliding_window_inference,
                            roi_size=inf_size,
                            sw_batch_size=args.sw_batch_size,
                            predictor=model,
                            overlap=args.infer_overlap)


    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count', pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k.replace('backbone.','')] = v
        model.load_state_dict(new_state_dict, strict=False)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)


    if args.optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.optim_lr,
                                     weight_decay=args.reg_weight)
    elif args.optim_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.optim_lr,
                                      weight_decay=args.reg_weight)
    elif args.optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.optim_lr,
                                    momentum=args.momentum,
                                    nesterov=True,
                                    weight_decay=args.reg_weight)
    elif args.optim_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=args.optim_lr)
    else:
        raise ValueError('Unsupported Optimization Procedure: ' + str(args.optim_name))

    if args.lrschedule == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=args.warmup_epochs,
                                                  max_epochs=args.max_epochs)
    elif args.lrschedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    elif args.lrschedule == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        scheduler = None
    accuracy = run_training(model=model,
                            train_loader=loader[0],
                            val_loader=loader[1],
                            optimizer=optimizer,
                            acc_func=dice_acc,
                            args=args,
                            model_inferer=model_inferer,
                            scheduler=scheduler,
                            start_epoch=start_epoch,
                            post_label=post_label,
                            post_pred=post_pred)
    return accuracy

if __name__ == '__main__':
    main()
