import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from networks.net import NET
from utils.data_utils import get_loader
from trainer import dice
import argparse
from scipy.ndimage.interpolation import shift
import SimpleITK as sitk
from skimage import measure

import subprocess

parser = argparse.ArgumentParser(description='Skull stripping pipeline')
parser.add_argument('--pretrained_dir', default='./runs/test/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--data_dir', default='./Model/', type=str, help='dataset directory')
parser.add_argument('--pretrained_model_name', default='epoch1039model-all.pt', type=str, help='pretrained model name')
parser.add_argument('--saved_checkpoint', default='ckpt', type=str, help='Supports torchscript or ckpt pretrained checkpoint type')
parser.add_argument('--mlp_dim', default=3072, type=int, help='mlp dimention in ViT encoder')
parser.add_argument('--hidden_size', default=768, type=int, help='hidden size dimention in ViT encoder')
parser.add_argument('--feature_size', default=16, type=int, help='feature size dimention')
parser.add_argument('--infer_overlap', default=0.9, type=float, help='sliding window inference overlap')
parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=2, type=int, help='number of output channels')
parser.add_argument('--num_heads', default=12, type=int, help='number of attention heads in ViT encoder')
parser.add_argument('--res_block', action='store_true', help='use residual blocks')
parser.add_argument('--conv_block', action='store_true', help='use conv blocks')
parser.add_argument('--roi_x', default=128, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=128, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=128, type=int, help='roi size in z direction')
parser.add_argument('--workers', default=1, type=int, help='number of workers')
parser.add_argument('--pos_embed', default='perceptron', type=str, help='type of position embedding')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization layer type in decoder')
parser.add_argument('--subject_name', default='', type=str, help='testing data id')
parser.add_argument('--stage', default='', type=str, help='stage of lifespan')
parser.add_argument('--age_month', default='', type=str, help='age in month')
parser.add_argument('--input_path', default='', type=str, help='path of input data')
parser.add_argument('--output_path', default='', type=str, help='path of output data')

def main():
    args = parser.parse_args()
    args.test_mode = True
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join('./Model/', model_name)
    if args.saved_checkpoint == 'torchscript':
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == 'ckpt':
        model = NET(
            device = device,
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
        model_dict = torch.load(pretrained_pth, map_location=('cpu'))
        model.load_state_dict(model_dict['state_dict'])
    model.eval()
    model.to(device)

    with torch.no_grad():
        for file in os.scandir(args.input_path):
            if file.name.endswith('.hdr'):
                # read MRI image
                T1w_img = sitk.ReadImage(file.path)
                size = T1w_img.GetSize()
                origin = T1w_img.GetOrigin()
                spacing = T1w_img.GetSpacing()
                direction = T1w_img.GetDirection()
                T1w_img = sitk.GetArrayFromImage(T1w_img)

                img_name = file.name.split('/')[-1].split('.')[0]

                # read atlas and atlas mask
                if args.stage == 'Infant':
                    if args.age_month == '0':
                        atlas = sitk.ReadImage(
                            './Lifespan_brain_atlases/brain-atlas-Month0-downsample.hdr')
                        atlas = sitk.GetArrayFromImage(atlas)
                        atlas = torch.tensor(atlas).float()

                        atlas_mask = sitk.ReadImage(
                            './Lifespan_brain_atlases/brain-atlas-Month0-mask-downsample.hdr')
                        atlas_mask = sitk.GetArrayFromImage(atlas_mask)
                        atlas_mask = torch.tensor(atlas_mask).float()

                    if args.age_month == '3':
                        atlas = sitk.ReadImage(
                            './Lifespan_brain_atlases/brain-atlas-Month3-downsample.hdr')
                        atlas = sitk.GetArrayFromImage(atlas)
                        atlas = torch.tensor(atlas).float()

                        atlas_mask = sitk.ReadImage(
                            './Lifespan_brain_atlases/brain-atlas-Month3-mask-downsample.hdr')
                        atlas_mask = sitk.GetArrayFromImage(atlas_mask)
                        atlas_mask = torch.tensor(atlas_mask).float()


                    if args.age_month == '6':
                        atlas = sitk.ReadImage(
                            './Lifespan_brain_atlases/brain-atlas-Month6-downsample.hdr')
                        atlas = sitk.GetArrayFromImage(atlas)
                        atlas = torch.tensor(atlas).float()

                        atlas_mask = sitk.ReadImage(
                            './Lifespan_brain_atlases/brain-atlas-Month6-mask-downsample.hdr')
                        atlas_mask = sitk.GetArrayFromImage(atlas_mask)
                        atlas_mask = torch.tensor(atlas_mask).float()


                    if args.age_month == '9':
                        atlas = sitk.ReadImage(
                            './Lifespan_brain_atlases/brain-atlas-Month9-downsample.hdr')
                        atlas = sitk.GetArrayFromImage(atlas)
                        atlas = torch.tensor(atlas).float()

                        atlas_mask = sitk.ReadImage(
                            './Lifespan_brain_atlases/brain-atlas-Month9-mask-downsample.hdr')
                        atlas_mask = sitk.GetArrayFromImage(atlas_mask)
                        atlas_mask = torch.tensor(atlas_mask).float()


                    if args.age_month == '12':
                        atlas = sitk.ReadImage(
                            './Lifespan_brain_atlases/brain-atlas-Month12-downsample.hdr')
                        atlas = sitk.GetArrayFromImage(atlas)
                        atlas = torch.tensor(atlas).float()

                        atlas_mask = sitk.ReadImage(
                            './Lifespan_brain_atlases/brain-atlas-Month12-mask-downsample.hdr')
                        atlas_mask = sitk.GetArrayFromImage(atlas_mask)
                        atlas_mask = torch.tensor(atlas_mask).float()


                    if args.age_month == '18':
                        atlas = sitk.ReadImage(
                            './Lifespan_brain_atlases/brain-atlas-Month18-downsample.hdr')
                        atlas = sitk.GetArrayFromImage(atlas)
                        atlas = torch.tensor(atlas).float()

                        atlas_mask = sitk.ReadImage(
                            './Lifespan_brain_atlases/brain-atlas-Month18-mask-downsample.hdr')
                        atlas_mask = sitk.GetArrayFromImage(atlas_mask)
                        atlas_mask = torch.tensor(atlas_mask).float()


                    if args.age_month == '24':
                        atlas = sitk.ReadImage(
                            './Lifespan_brain_atlases/brain-atlas-Month24-downsample.hdr')
                        atlas = sitk.GetArrayFromImage(atlas)
                        atlas = torch.tensor(atlas).float()

                        atlas_mask = sitk.ReadImage(
                            './Lifespan_brain_atlases/brain-atlas-Month24-mask-downsample.hdr')
                        atlas_mask = sitk.GetArrayFromImage(atlas_mask)
                        atlas_mask = torch.tensor(atlas_mask).float()


                if args.stage == 'Adolescent':
                    atlas = sitk.ReadImage(
                        './Lifespan_brain_atlases/brain-atlas-Adolescent-downsample.hdr')
                    atlas = sitk.GetArrayFromImage(atlas)
                    atlas = torch.tensor(atlas).float()

                    atlas_mask = sitk.ReadImage(
                        './Lifespan_brain_atlases/brain-atlas-Adolescent-mask-downsample.hdr')
                    atlas_mask = sitk.GetArrayFromImage(atlas_mask)
                    atlas_mask = torch.tensor(atlas_mask).float()


                if args.stage == 'Adult':
                    atlas = sitk.ReadImage(
                        './Lifespan_brain_atlases/brain-atlas-Adult-downsample.hdr')
                    atlas = sitk.GetArrayFromImage(atlas)
                    atlas = torch.tensor(atlas).float()

                    atlas_mask = sitk.ReadImage(
                        './Lifespan_brain_atlases/brain-atlas-Adult-mask-downsample.hdr')
                    atlas_mask = sitk.GetArrayFromImage(atlas_mask)
                    atlas_mask = torch.tensor(atlas_mask).float()


                if args.stage == 'Elder':
                    atlas = sitk.ReadImage(
                        './Lifespan_brain_atlases/brain-atlas-Elder-downsample.hdr')
                    atlas = sitk.GetArrayFromImage(atlas)
                    atlas = torch.tensor(atlas).float()

                    atlas_mask = sitk.ReadImage(
                        './Lifespan_brain_atlases/brain-atlas-Elder-mask-downsample.hdr')
                    atlas_mask = sitk.GetArrayFromImage(atlas_mask)
                    atlas_mask = torch.tensor(atlas_mask).float()

                # rescale atlas
                atlas = atlas.to(device)
                atlas_mask = atlas_mask.to(device)
                atlas = (atlas - torch.min(atlas)) / (torch.max(atlas) - torch.min(atlas))

                #######################
                #### first_testing
                template = sitk.ReadImage('./Template/template.hdr')
                #template = sitk.GetArrayFromImage(template)
                T1w_img_img = sitk.GetImageFromArray(T1w_img)

                matcher = sitk.HistogramMatchingImageFilter()
                matcher.SetNumberOfHistogramLevels(1024)
                matcher.SetNumberOfMatchPoints(7)
                matcher.ThresholdAtMeanIntensityOn()
                moving = matcher.Execute(T1w_img_img, template)

                '''
                save_dir = file.path.replace('.hdr', '-hm.hdr', 1)
                moving = sitk.GetArrayFromImage(moving)
                out = sitk.GetImageFromArray(moving)
                out.SetOrigin(origin)
                out.SetSpacing(spacing)
                out.SetDirection(direction)
                sitk.WriteImage(out, save_dir)
                '''

                xsize = float(size[0]) * float(spacing[0]) * 0.5
                ysize = float(size[1]) * float(spacing[1]) * 0.5
                zsize = float(size[2]) * float(spacing[2]) * 0.5

                resampler = sitk.ResampleImageFilter()
                resampler.SetSize([int(xsize), int(ysize), int(zsize)])
                resampler.SetOutputSpacing([2,2,2])
                resampler.SetOutputOrigin(origin)
                resampler.SetOutputDirection(direction)
                resampler.SetInterpolator(sitk.sitkLinear)

                resampled_image = resampler.Execute(moving)
                resampled_image = sitk.GetArrayFromImage(resampled_image)
                '''
                save_dir = file.path.replace('.hdr', '-hm-downsample.hdr', 1)
                resampled_image = sitk.GetArrayFromImage(resampled_image)
                out = sitk.GetImageFromArray(resampled_image)
                out.SetOrigin(origin)
                out.SetSpacing([2,2,2])
                out.SetDirection(direction)
                sitk.WriteImage(out, save_dir)
                '''

                # rescale T1w_img_hm_downsample
                T1w_img_hm_downsample = torch.tensor(resampled_image).float()
                T1w_img_hm_downsample = T1w_img_hm_downsample.to(device)
                T1w_img_hm_downsample = (T1w_img_hm_downsample - torch.min(T1w_img_hm_downsample)) / (torch.max(T1w_img_hm_downsample) - torch.min(T1w_img_hm_downsample))

                T1w_img_hm_downsample = torch.unsqueeze(T1w_img_hm_downsample, dim=0)
                T1w_img_hm_downsample = torch.unsqueeze(T1w_img_hm_downsample, dim=0)

                # testing
                logits, moving_trans, y_source, pos_flow, x_reg, atlas_mask_formable = model(T1w_img_hm_downsample, atlas, atlas_mask)


                # upsample to original space
                upsample = nn.Upsample(scale_factor=2, mode='trilinear')
                atlas_mask_formable_upsample = upsample(atlas_mask_formable)
                ini_mask = np.zeros([256, 256, 256])
                atlas_mask_formable_upsample = atlas_mask_formable_upsample.cpu().numpy()[0,0,:,:,:]
                ini_mask = np.where(atlas_mask_formable_upsample > 0.5, 1.0, 0.0)

                save_dir = os.path.join(args.input_path, file.name.replace('.hdr', '-stripped-2.hdr', 1))
                out = sitk.GetImageFromArray(ini_mask)
                sitk.WriteImage(out, save_dir)

                #######################
                #### second_testing

                #### crop
                label_img = ini_mask

                loc = np.where(label_img == 1)
                x_min = np.min(loc[0])
                x_max = np.max(loc[0])
                if (x_min-x_max)%4!=0:
                    if (x_min-x_max)%4==1:
                        x_max += 3
                    if (x_min-x_max)%4==2:
                        x_max += 2
                    if (x_min-x_max)%4==3:
                        x_max += 1

                y_min = np.min(loc[1])
                y_max = np.max(loc[1])
                if (y_min - y_max) % 4 != 0:
                    if (y_min - y_max) % 4 == 1:
                        y_max += 3
                    if (y_min - y_max) % 4 == 2:
                        y_max += 2
                    if (y_min - y_max) % 4 == 3:
                        y_max += 1
                z_min = np.min(loc[2])
                z_max = np.max(loc[2])
                if (z_min - z_max) % 4 != 0:
                    if (z_min - z_max) % 4 == 1:
                        z_max += 3
                    if (z_min - z_max) % 4 == 2:
                        z_max += 2
                    if (z_min - z_max) % 4 == 3:
                        z_max += 1

                T1w_img_crop = T1w_img[max(x_min - 25, 0):min(x_max + 25, T1w_img.shape[0]),
                            max(y_min - 25, 0):min(y_max + 25, T1w_img.shape[1]),
                            max(z_min - 25, 0):min(z_max + 25, T1w_img.shape[2])]

                save_dir = file.path.replace('.hdr', '-crop.hdr', 1)
                out = sitk.GetImageFromArray(T1w_img_crop)
                out.SetOrigin(origin)
                out.SetSpacing(spacing)
                out.SetDirection(direction)
                sitk.WriteImage(out, save_dir)


                #### hm
                T1w_img_crop = sitk.GetImageFromArray(T1w_img_crop)

                matcher = sitk.HistogramMatchingImageFilter()
                matcher.SetNumberOfHistogramLevels(1024)
                matcher.SetNumberOfMatchPoints(7)
                matcher.ThresholdAtMeanIntensityOn()
                moving = matcher.Execute(T1w_img_crop, template)
                '''
                save_dir = file.path.replace('.hdr', '-crop-hm.hdr', 1)
                moving = sitk.GetArrayFromImage(moving)
                out = sitk.GetImageFromArray(moving)
                out.SetOrigin(origin)
                out.SetSpacing(spacing)
                out.SetDirection(direction)
                sitk.WriteImage(out, save_dir)
                '''


                #### downsample to resolution 2.0
                T1w_img_crop = sitk.GetArrayFromImage(moving)
                T1w_img_crop_img = moving

                xsize = T1w_img_crop.shape[0] * float(spacing[0]) * 0.5
                ysize = T1w_img_crop.shape[1] * float(spacing[1]) * 0.5
                zsize = T1w_img_crop.shape[2] * float(spacing[2]) * 0.5

                resampler = sitk.ResampleImageFilter()
                resampler.SetSize([int(xsize), int(ysize), int(zsize)])
                resampler.SetOutputSpacing([2, 2, 2])
                resampler.SetOutputOrigin(origin)
                resampler.SetOutputDirection(direction)
                resampler.SetInterpolator(sitk.sitkLinear)

                T1w_img_resample = resampler.Execute(T1w_img_crop_img)

                '''
                save_dir = file.path.replace('.hdr', '-crop-hm-resample.hdr', 1)
                T1w_img_resample = sitk.GetArrayFromImage(T1w_img_resample)
                out = sitk.GetImageFromArray(T1w_img_resample)
                out.SetOrigin(origin)
                out.SetSpacing([2, 2, 2])
                out.SetDirection(direction)
                sitk.WriteImage(out, save_dir)
                '''

                #resize to 128
                T1w_img_resample = sitk.GetArrayFromImage(T1w_img_resample)
                x = T1w_img_resample.shape[0]
                y = T1w_img_resample.shape[1]
                z = T1w_img_resample.shape[2]

                temp = np.zeros([128, 128, 128])
                temp[max(118 - x, 0):118, int((128 - y) / 2):int((128 - y) / 2) + y,
                int((128 - z) / 2):int((128 - z) / 2) + z] = T1w_img_resample[max(x - 118, 0):, :, :]

                '''
                save_dir = file.path.replace('.hdr', '-crop-hm-resample-resize.hdr', 1)
                out = sitk.GetImageFromArray(temp)
                out.SetOrigin(origin)
                out.SetSpacing((2.0,2.0,2.0))
                out.SetDirection(direction)
                sitk.WriteImage(out, save_dir)
                '''

                # testing
                T1w_img_downsample = torch.tensor(temp).float()
                T1w_img_downsample = T1w_img_downsample.to(device)


                T1w_img_downsample = (T1w_img_downsample - torch.min(T1w_img_downsample)) / (
                        torch.max(T1w_img_downsample) - torch.min(T1w_img_downsample))

                T1w_img_downsample = torch.unsqueeze(T1w_img_downsample, dim=0)
                T1w_img_downsample = torch.unsqueeze(T1w_img_downsample, dim=0)


                logits, moving_trans, y_source, pos_flow, x_reg, atlas_mask_formable = model(T1w_img_downsample, atlas, atlas_mask)

                atlas_mask_formable = atlas_mask_formable[0, 0, :, :, :].cpu().numpy()
                atlas_mask_crop_resample = atlas_mask_formable[max(118 - x, 0):118, int((128 - y) / 2):int((128 - y) / 2) + y,
                int((128 - z) / 2):int((128 - z) / 2) + z]

                '''
                #atlas_mask_formable = np.where(atlas_mask_formable > 0.5, 1.0, 0.0)
                s_path = img_name + '-atlas_mask_formable-downsample.hdr'
                save_dir = os.path.join(args.output_path, s_path)
                out = sitk.GetImageFromArray(atlas_mask_crop_resample)
                out.SetOrigin(origin)
                out.SetSpacing((2.0,2.0,2.0))
                out.SetDirection(direction)
                sitk.WriteImage(out, save_dir)

                #print(float(xdim), float(ydim), float(zdim))
                '''



                #### resample to original resolution
                xsize = atlas_mask_crop_resample.shape[0] * 2 * 1
                ysize = atlas_mask_crop_resample.shape[1] * 2 * 1
                zsize = atlas_mask_crop_resample.shape[2] * 2 * 1
                print('######', xsize, ysize, zsize)

                resampler = sitk.ResampleImageFilter()
                resampler.SetSize([int(xsize), int(ysize), int(zsize)])
                resampler.SetOutputSpacing([1, 1, 1])
                resampler.SetOutputOrigin(origin)
                resampler.SetOutputDirection(direction)
                resampler.SetInterpolator(sitk.sitkLinear)


                atlas_mask_crop_resample = sitk.GetImageFromArray(atlas_mask_crop_resample)
                atlas_mask_crop_resample.SetSpacing((2.0, 2.0, 2.0))
                atlas_mask_crop_resample_upsample = resampler.Execute(atlas_mask_crop_resample)

                '''
                save_dir = args.input_path + img_name + '-atlas_mask_formable-downsample-upsample.hdr'
                atlas_mask_crop_resample_upsample = sitk.GetArrayFromImage(atlas_mask_crop_resample_upsample)
                out = sitk.GetImageFromArray(atlas_mask_crop_resample_upsample)
                out.SetOrigin(origin)
                out.SetSpacing([1, 1, 1])
                out.SetDirection(direction)
                sitk.WriteImage(out, save_dir)
                '''

                binary_image = atlas_mask_crop_resample_upsample > 0.5

                # Create the binary opening filter
                opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
                opening_filter.SetKernelRadius(5)
                opened_image = opening_filter.Execute(binary_image)

                '''
                save_dir = args.input_path + img_name + '-atlas_mask_formable-downsample-upsample-b-bo.hdr'
                opened_image = sitk.GetArrayFromImage(opened_image)
                out = sitk.GetImageFromArray(opened_image)
                out.SetOrigin(origin)
                out.SetSpacing([1, 1, 1])
                out.SetDirection(direction)
                sitk.WriteImage(out, save_dir)
                    `'''

                #### resize
                opened_image = sitk.GetArrayFromImage(opened_image)
                brain_mask = np.zeros([256, 256, 256])
                brain_mask[max(x_min - 25, 0):min(x_max + 25, T1w_img.shape[0]),
                max(y_min - 25, 0):min(y_max + 25, T1w_img.shape[1]),
                max(z_min - 25, 0):min(z_max + 25, T1w_img.shape[2])] = opened_image


                save_dir =args.output_path + img_name + '-brainmask.hdr'
                out = sitk.GetImageFromArray(brain_mask)
                out.SetOrigin(origin)
                out.SetSpacing(spacing)
                out.SetDirection(direction)
                sitk.WriteImage(out, save_dir)

                os.remove(os.path.join(args.input_path, file.name.replace('.hdr', '-stripped-2.hdr', 1)))


if __name__ == '__main__':
    main()
