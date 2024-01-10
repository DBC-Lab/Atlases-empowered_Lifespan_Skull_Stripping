import os
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter
import torch.nn.parallel
from utils.utils import distributed_all_gather
import torch.utils.data.distributed
from monai.data import decollate_batch
import SimpleITK as sitk


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0,
                            self.sum / self.count,
                            self.sum)


def train_epoch(model,
                loader,
                optimizer,
                epoch,
                args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()

    #### loss function ####
    mse = nn.MSELoss()
    smoothloss = smooth_loss(penalty='l2', loss_mult=2)
    loss_func = nn.CrossEntropyLoss()
    ncc = NCC().loss
    
    for idx, batch_data in enumerate(loader):
        data, target = batch_data['image'], batch_data['label']

        data_r = data[0,0,:,:,:]
        data_r = np.rot90(data_r, k = 1, axes = (0, 2))
        data_r = np.flip(data_r, axis=0)
        data_r = np.flip(data_r, axis=1)
        data_r = np.flip(data_r, axis=2)
        data_r = data_r.copy()
        data[0,0,:,:,:] = torch.tensor(data_r)
        

        target_r = target[0,0,:,:,:]
        target_r = np.rot90(target_r, k = 1, axes = (0, 2))
        target_r = np.flip(target_r, axis=0)
        target_r = np.flip(target_r, axis=1)
        target_r = np.flip(target_r, axis=2)
        target_r = target_r.copy()
        target[0,0,:,:,:] = torch.tensor(target_r)


        img_name = batch_data['image_meta_dict']['filename_or_obj'][0].split('.')[0]
        target_name = batch_data['label_meta_dict']['filename_or_obj'][0].split('.')[0]


        age_name = batch_data['image_meta_dict']['filename_or_obj'][0].split('/')[-2]
        if (age_name=='0m'):
            atlas = sitk.ReadImage(
                './Lifespan_brain_atlases/brain-atlas-Month0-downsample.hdr')
            atlas = sitk.GetArrayFromImage(atlas)
            atlas = torch.tensor(atlas).float()
            data = torch.tensor(data).float()

            atlas_mask = sitk.ReadImage(
                './Lifespan_brain_atlases/brain-atlas-Month0-mask-downsample.hdr')
            atlas_mask = sitk.GetArrayFromImage(atlas_mask)
            atlas_mask = torch.tensor(atlas_mask).float()
            atlas_mask = atlas_mask.cuda(args.rank)

        if (age_name == '3m'):
            atlas = sitk.ReadImage(
                './Lifespan_brain_atlases/brain-atlas-Month3-downsample.hdr')
            atlas = sitk.GetArrayFromImage(atlas)
            atlas = torch.tensor(atlas).float()
            data = torch.tensor(data).float()

            atlas_mask = sitk.ReadImage(
                './Lifespan_brain_atlases/brain-atlas-Month3-mask-downsample.hdr')
            atlas_mask = sitk.GetArrayFromImage(atlas_mask)
            atlas_mask = torch.tensor(atlas_mask).float()
            atlas_mask = atlas_mask.cuda(args.rank)

        if (age_name == '6m'):
            atlas = sitk.ReadImage(
                './Lifespan_brain_atlases/brain-atlas-Month6-downsample.hdr')
            atlas = sitk.GetArrayFromImage(atlas)
            atlas = torch.tensor(atlas).float()
            data = torch.tensor(data).float()

            atlas_mask = sitk.ReadImage(
                './Lifespan_brain_atlases/brain-atlas-Month6-mask-downsample.hdr')
            atlas_mask = sitk.GetArrayFromImage(atlas_mask)
            atlas_mask = torch.tensor(atlas_mask).float()
            atlas_mask = atlas_mask.cuda(args.rank)

        if (age_name == '9m'):
            atlas = sitk.ReadImage(
                './Lifespan_brain_atlases/brain-atlas-Month9-downsample.hdr')
            atlas = sitk.GetArrayFromImage(atlas)
            atlas = torch.tensor(atlas).float()
            data = torch.tensor(data).float()

            atlas_mask = sitk.ReadImage(
                './Lifespan_brain_atlases/brain-atlas-Month9-mask-downsample.hdr')
            atlas_mask = sitk.GetArrayFromImage(atlas_mask)
            atlas_mask = torch.tensor(atlas_mask).float()
            atlas_mask = atlas_mask.cuda(args.rank)

        if (age_name == '12m'):
            atlas = sitk.ReadImage(
                './Lifespan_brain_atlases/brain-atlas-Month12-downsample.hdr')
            atlas = sitk.GetArrayFromImage(atlas)
            atlas = torch.tensor(atlas).float()
            data = torch.tensor(data).float()

            atlas_mask = sitk.ReadImage(
                './Lifespan_brain_atlases/brain-atlas-Month12-mask-downsample.hdr')
            atlas_mask = sitk.GetArrayFromImage(atlas_mask)
            atlas_mask = torch.tensor(atlas_mask).float()
            atlas_mask = atlas_mask.cuda(args.rank)

        if (age_name == '18m'):
            atlas = sitk.ReadImage(
                './Lifespan_brain_atlases/brain-atlas-Month18-downsample.hdr')
            atlas = sitk.GetArrayFromImage(atlas)
            atlas = torch.tensor(atlas).float()
            data = torch.tensor(data).float()

            atlas_mask = sitk.ReadImage(
                './Lifespan_brain_atlases/brain-atlas-Month18-mask-downsample.hdr')
            atlas_mask = sitk.GetArrayFromImage(atlas_mask)
            atlas_mask = torch.tensor(atlas_mask).float()
            atlas_mask = atlas_mask.cuda(args.rank)

        if (age_name == '24m'):
            atlas = sitk.ReadImage(
                './Lifespan_brain_atlases/brain-atlas-Month24-downsample.hdr')
            atlas = sitk.GetArrayFromImage(atlas)
            atlas = torch.tensor(atlas).float()
            data = torch.tensor(data).float()

            atlas_mask = sitk.ReadImage(
                './Lifespan_brain_atlases/brain-atlas-Month24-mask-downsample.hdr')
            atlas_mask = sitk.GetArrayFromImage(atlas_mask)
            atlas_mask = torch.tensor(atlas_mask).float()
            atlas_mask = atlas_mask.cuda(args.rank)

        if (age_name == 'Adolescent'):
            atlas = sitk.ReadImage(
                './Lifespan_brain_atlases/brain-atlas-Adolescent-downsample.hdr')
            atlas = sitk.GetArrayFromImage(atlas)
            atlas = torch.tensor(atlas).float()
            data = torch.tensor(data).float()

            atlas_mask = sitk.ReadImage(
                './Lifespan_brain_atlases/brain-atlas-Adolescent-mask-downsample.hdr')
            atlas_mask = sitk.GetArrayFromImage(atlas_mask)
            atlas_mask = torch.tensor(atlas_mask).float()
            atlas_mask = atlas_mask.cuda(args.rank)

        if (age_name == 'Adult'):
            atlas = sitk.ReadImage(
                './Lifespan_brain_atlases/brain-atlas-Adult-downsample.hdr')
            atlas = sitk.GetArrayFromImage(atlas)
            atlas = torch.tensor(atlas).float()
            data = torch.tensor(data).float()

            atlas_mask = sitk.ReadImage(
                './Lifespan_brain_atlases/brain-atlas-Adult-mask-downsample.hdr')
            atlas_mask = sitk.GetArrayFromImage(atlas_mask)
            atlas_mask = torch.tensor(atlas_mask).float()
            atlas_mask = atlas_mask.cuda(args.rank)

        if (age_name == 'Elder'):
            atlas = sitk.ReadImage(
                './Lifespan_brain_atlases/brain-atlas-Elder-downsample.hdr')
            atlas = sitk.GetArrayFromImage(atlas)
            atlas = torch.tensor(atlas).float()
            data = torch.tensor(data).float()

            atlas_mask = sitk.ReadImage(
                './Lifespan_brain_atlases/brain-atlas-Elder-mask-downsample.hdr')
            atlas_mask = sitk.GetArrayFromImage(atlas_mask)
            atlas_mask = torch.tensor(atlas_mask).float()
            atlas_mask = atlas_mask.cuda(args.rank)


        
        atlas = (atlas-torch.min(atlas))/(torch.max(atlas)-torch.min(atlas))
        data = (data-torch.min(data))/(torch.max(data)-torch.min(data))
        
        data, atlas, target = data.cuda(args.rank), atlas.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters(): param.grad = None
        with autocast(enabled=args.amp):
            logits, moving_trans, y_source, pos_flow, x_reg, atlas_mask_formable = model(data, atlas, atlas_mask)
         

            target = target.squeeze()
            target = target.unsqueeze(dim=0)
            loss1 = loss_func(logits, target.long())
            loss2 = mse(y_source, x_reg)
            loss3 = smoothloss(pos_flow)
            target = target.unsqueeze(dim=0)
            loss4 = mse(atlas_mask_formable, target)
            
            
            loss = loss1 + 0.1*loss2 + loss3 + 0.1*loss4

            print('#',loss, '#',loss1, '#', loss2, '#',loss3, '#',loss4)

            
            target = target.cpu().numpy()[0,:,:,:,:]
            atlas_mask_formable = atlas_mask_formable.squeeze()
            atlas_mask_formable = atlas_mask_formable.unsqueeze(dim=0)
            atlas_mask_formable = atlas_mask_formable.detach().cpu().numpy()
            
            data_outputs = np.zeros([1,128,128,128])
            data_outputs = np.where(atlas_mask_formable>0.5, 1.0, 0.0)
            
            
            logits = logits.detach().cpu().numpy()
            pre = np.argmax(logits,1)
            dice_list_sub = []
            for j in range(1, 2):
                organ_Dice = dice(pre[0] == j, target[0] == j)
                dice_list_sub.append(organ_Dice)

            mean_dice = np.mean(dice_list_sub)
            print("Mean Organ Dice: {}".format(mean_dice))
            
            
            
            dice_list_sub = []
            for j in range(1, 2):
                organ_Dice = dice(data_outputs[0] == j, target[0] == j)
                dice_list_sub.append(organ_Dice)

            mean_dice = np.mean(dice_list_sub)
            print("Mean Organ Dice: {}".format(mean_dice))


        loss.backward()
        optimizer.step()
        
        run_loss.update(loss.item(), n=args.batch_size)

        print('Epoch {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
              'loss: {:.4f}'.format(run_loss.avg),
              'time {:.2f}s'.format(time.time() - start_time))

        start_time = time.time()
    for param in model.parameters() : param.grad = None
    return run_loss.avg

def val_epoch(model,
              loader,
              epoch,
              acc_func,
              args,
              model_inferer=None,
              post_label=None,
              post_pred=None):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data['image'], batch_data['label']
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc = acc.cuda(args.rank)

            acc_list = acc.detach().cpu().numpy()
            avg_acc = np.mean([np.nanmean(l) for l in acc_list])

            if args.rank == 0:
                print('Val {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
                      'acc', avg_acc,
                      'time {:.2f}s'.format(time.time() - start_time))
            start_time = time.time()
    return avg_acc

def save_checkpoint(model,
                    epoch,
                    args,
                    filename='model.pt',
                    best_acc=0,
                    optimizer=None,
                    scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {
            'epoch': epoch,
            'best_acc': best_acc,
            'state_dict': state_dict
            }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    filename=os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print('Saving checkpoint', filename)

def run_training(model,
                 train_loader,
                 val_loader,
                 optimizer,
                 loss_func,
                 acc_func,
                 args,
                 model_inferer=None,
                 scheduler=None,
                 start_epoch=0,
                 post_label=None,
                 post_pred=None
                 ):
    writer = None

    writer = SummaryWriter(log_dir=args.logdir)
    print('Writing Tensorboard logs to ', args.logdir)
    scaler = None

    val_acc_max = 0.
    for epoch in range(start_epoch, args.max_epochs):
        print(time.ctime(), 'Epoch:', epoch)
        epoch_time = time.time()
        train_loss = train_epoch(model,
                                 train_loader,
                                 optimizer,
                                 scaler=scaler,
                                 epoch=epoch,
                                 loss_func=loss_func,
                                 args=args)

        print('Final training  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(train_loss),
              'time {:.2f}s'.format(time.time() - epoch_time))
        if args.rank==0 and writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
        b_new_best = False
        
        
        modelname='epoch' + str(epoch) + 'model-all.pt'
        
        if (epoch+1) % 30 == 0:
            save_checkpoint(model, (epoch), args, filename=modelname,
                            best_acc=0,
                            optimizer=optimizer,
                            scheduler=scheduler)
        
        
        if (epoch+1) % args.val_every == 0:
            epoch_time = time.time()
            val_avg_acc = val_epoch(model,
                                    val_loader,
                                    epoch=epoch,
                                    acc_func=acc_func,
                                    model_inferer=model_inferer,
                                    args=args,
                                    post_label=post_label,
                                    post_pred=post_pred)
            print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1),
                  'acc', val_avg_acc, 'time {:.2f}s'.format(time.time() - epoch_time))

            if writer is not None:
                writer.add_scalar('val_acc', val_avg_acc, epoch)
            if val_avg_acc > val_acc_max:
                print('new best ({:.6f} --> {:.6f}). '.format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                b_new_best = True
                if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                    save_checkpoint(model, epoch, args,
                                    best_acc=val_acc_max,
                                    optimizer=optimizer,
                                    scheduler=scheduler)
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model,
                                epoch,
                                args,
                                best_acc=val_acc_max,
                                filename='model_final.pt')
            if b_new_best:
                print('Copying to model.pt new best model!!!!')
                shutil.copyfile(os.path.join(args.logdir, 'model_final.pt'), os.path.join(args.logdir, 'model.pt'))

        if scheduler is not None:
            scheduler.step()

    print('Training Finished !, Best Accuracy: ', val_acc_max)

    return val_acc_max

