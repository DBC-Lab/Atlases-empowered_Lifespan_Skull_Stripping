import os
import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom
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
parser.add_argument('--pretrained_dir', default='./Model/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--data_dir', default='/', type=str, help='dataset directory')
parser.add_argument('--pretrained_model_name', default='model_best_acc.pt', type=str, help='pretrained model name')
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
parser.add_argument('--N4', default='False', type=str, help='apply N4')
parser.add_argument('--age_in_month', default='', type=str, help='age in month')
parser.add_argument('--input_path', default='./Testing_subjects/', type=str, help='path of input data')
parser.add_argument('--output_path', default='./Testing_subjects/', type=str, help='path of output data')


def largest_connected_component_3d(binary_image):
    # Label all connected components in the image
    labeled_array, num_features = ndimage.label(binary_image)

    # Find the sizes of the connected components
    sizes = ndimage.sum(binary_image, labeled_array, range(num_features + 1))

    # Identify the largest connected component (excluding the background)
    mask_size = sizes < max(sizes)
    remove_pixel = mask_size[labeled_array]
    labeled_array[remove_pixel] = 0

    # Label the largest connected component
    labeled_array, _ = ndimage.label(labeled_array)

    return labeled_array




def main():
    args = parser.parse_args()
    args.test_mode = True
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(args.pretrained_dir, model_name)
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
            if '.nii' in file.name:
                # read MRI image
                T1w_img = sitk.ReadImage(file.path)
                size = T1w_img.GetSize()
                origin = T1w_img.GetOrigin()
                spacing = T1w_img.GetSpacing()
                direction = T1w_img.GetDirection()

                # reorient to RAI (Right-Anterior-Inferior)
                rai_transform = sitk.DICOMOrientImageFilter()
                rai_transform.SetDesiredCoordinateOrientation("RAI")
                rai_image = rai_transform.Execute(T1w_img)
                T1w_img = rai_image

                # allpy N4
                if args.N4 == 'True':
                    corrector = sitk.N4BiasFieldCorrectionImageFilter()
                    number_of_iterations = [50, 50, 30, 20]
                    corrector.SetMaximumNumberOfIterations(number_of_iterations)
                    output_image = corrector.Execute(T1w_img)
                    T1w_img = output_image


                T1w_img = sitk.GetArrayFromImage(T1w_img)
                img_name = file.name.split('/')[-1].split('.')[0]

                # read atlas and atlas mask
                if int(args.age_in_month)<=2:
                    month = 'Month0'
                if ((int(args.age_in_month)>2)&(int(args.age_in_month)<=4)):
                    month = 'Month3'
                if ((int(args.age_in_month)>4)&(int(args.age_in_month)<=7)):
                    month = 'Month6'
                if ((int(args.age_in_month)>7)&(int(args.age_in_month)<=10)):
                    month = 'Month9'
                if ((int(args.age_in_month)>10)&(int(args.age_in_month)<=14)):
                    month = 'Month12'
                if ((int(args.age_in_month)>14)&(int(args.age_in_month)<=20)):
                    month = 'Month18'
                if ((int(args.age_in_month)>20)&(int(args.age_in_month)<=28)):
                    month = 'Month24'
                if ((int(args.age_in_month)>28)&(int(args.age_in_month)<=540)):
                    month = 'Adolescent'
                if ((int(args.age_in_month)>540)&(int(args.age_in_month)<=1920)):
                    month = 'Adult'
                if ((int(args.age_in_month)>1920)):
                    month = 'Elder'

                atlas = sitk.ReadImage(
                    './Lifespan_brain_atlases/brain-atlas-' + month + '-downsample.hdr')
                atlas = sitk.GetArrayFromImage(atlas)
                atlas = torch.tensor(atlas).float()

                atlas_mask = sitk.ReadImage(
                    './Lifespan_brain_atlases/brain-atlas-' + month + '-mask-downsample.hdr')
                atlas_mask = sitk.GetArrayFromImage(atlas_mask)
                atlas_mask = torch.tensor(atlas_mask).float()


                # rescale atlas
                atlas = atlas.to(device)
                atlas_mask = atlas_mask.to(device)
                atlas = (atlas - torch.min(atlas)) / (torch.max(atlas) - torch.min(atlas))

                #######################
                #### First_testing ####
                #######################
                template = sitk.ReadImage('./Template/template.hdr')
                #template = sitk.GetArrayFromImage(template)
                T1w_img_img = sitk.GetImageFromArray(T1w_img)

                # histgram matching
                matcher = sitk.HistogramMatchingImageFilter()
                matcher.SetNumberOfHistogramLevels(1024)
                matcher.SetNumberOfMatchPoints(7)
                matcher.ThresholdAtMeanIntensityOn()
                T1w_img_img_hm = matcher.Execute(T1w_img_img, template)

                # downsample to resolution 2.0
                xsize = float(size[0]) * float(spacing[0]) * 0.5
                ysize = float(size[1]) * float(spacing[1]) * 0.5
                zsize = float(size[2]) * float(spacing[2]) * 0.5

                resampler = sitk.ResampleImageFilter()
                resampler.SetSize([int(xsize), int(ysize), int(zsize)])
                resampler.SetOutputSpacing([2,2,2])
                resampler.SetOutputOrigin(origin)
                resampler.SetOutputDirection(direction)
                resampler.SetInterpolator(sitk.sitkLinear)

                T1w_img_hm_downsample = resampler.Execute(T1w_img_img_hm)
                T1w_img_hm_downsample = sitk.GetArrayFromImage(T1w_img_hm_downsample)


                # rescale T1w_img_hm_downsample
                T1w_img_hm_downsample = torch.tensor(T1w_img_hm_downsample).float()
                T1w_img_hm_downsample = T1w_img_hm_downsample.to(device)
                T1w_img_hm_downsample = (T1w_img_hm_downsample - torch.min(T1w_img_hm_downsample)) / (torch.max(T1w_img_hm_downsample) - torch.min(T1w_img_hm_downsample))

                T1w_img_hm_downsample = torch.unsqueeze(T1w_img_hm_downsample, dim=0)
                T1w_img_hm_downsample = torch.unsqueeze(T1w_img_hm_downsample, dim=0)

                # testing
                logits, moving_trans, y_source, pos_flow, x_reg, atlas_mask_formable = model(T1w_img_hm_downsample, atlas, atlas_mask)

                # Extracted brain mask
                logits = logits.cpu().numpy()
                pre = np.argmax(logits, axis=1)[0,:,:,:]
                pre_upsample = zoom(pre, 2.0, order=3)
                pre_upsample_b = np.where(pre_upsample>0.5, 1.0, 0.0)

                # Binaryopening
                pre_upsample_b = pre_upsample_b.astype(int)
                pre_upsample_b = sitk.GetImageFromArray(pre_upsample_b)
                opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
                opening_filter.SetKernelRadius(8)
                opened_image = opening_filter.Execute(pre_upsample_b)

                # Remain 3D maximum connected component
                opened_image = sitk.GetArrayFromImage(opened_image)
                labeled_array, num_features = ndimage.label(opened_image)

                sizes = ndimage.sum(opened_image, labeled_array, range(num_features + 1))
                mask_size = sizes < max(sizes)
                remove_pixel = mask_size[labeled_array]
                labeled_array[remove_pixel] = 0
                labeled_array, _ = ndimage.label(labeled_array)

                save_dir = file.path.replace('.hdr', '-stripped-2.hdr', 1)
                out = sitk.GetImageFromArray(labeled_array)
                out.SetOrigin(origin)
                out.SetSpacing(spacing)
                out.SetDirection(direction)
                sitk.WriteImage(out, save_dir)

                #######################
                #### Second_testing ####
                #######################

                for i in range(2):
                    #### crop
                    ininal_label_img = labeled_array

                    loc = np.where(ininal_label_img == 1)
                    x_min = np.min(loc[0])
                    x_max = np.max(loc[0])
                    if (x_max - x_min) % 2 != 0:
                        x_max += 1
                    y_min = np.min(loc[1])
                    y_max = np.max(loc[1])
                    if (y_min - y_max) % 2 != 0:
                        y_max += 1
                    z_min = np.min(loc[2])
                    z_max = np.max(loc[2])
                    if (z_min - z_max) % 2 != 0:
                        z_max += 1

                    T1w_img_crop = T1w_img[max(x_min - 20, 0):min(x_max + 40, T1w_img.shape[0]),
                                   max(y_min - 40, 0):min(y_max + 40, T1w_img.shape[1]),
                                   max(z_min - 40, 0):min(z_max + 40, T1w_img.shape[2])]

                    #### histgram matching
                    T1w_img_crop_img = sitk.GetImageFromArray(T1w_img_crop)

                    matcher = sitk.HistogramMatchingImageFilter()
                    matcher.SetNumberOfHistogramLevels(1024)
                    matcher.SetNumberOfMatchPoints(7)
                    matcher.ThresholdAtMeanIntensityOn()
                    moving = matcher.Execute(T1w_img_crop_img, template)

                    #### downsample to resolution 2.0
                    T1w_img_crop_hm = sitk.GetArrayFromImage(moving)
                    T1w_img_crop_hm_img = moving

                    xsize = T1w_img_crop_hm.shape[0] * float(spacing[0]) * 0.5
                    ysize = T1w_img_crop_hm.shape[1] * float(spacing[1]) * 0.5
                    zsize = T1w_img_crop_hm.shape[2] * float(spacing[2]) * 0.5

                    resampler = sitk.ResampleImageFilter()
                    resampler.SetSize([round(xsize), round(ysize), round(zsize)])
                    resampler.SetOutputSpacing([2, 2, 2])
                    resampler.SetOutputOrigin(origin)
                    resampler.SetOutputDirection(direction)
                    resampler.SetInterpolator(sitk.sitkLinear)

                    T1w_img_crop_hm_downsample = resampler.Execute(T1w_img_crop_hm_img)

                    # resize to size 128
                    T1w_img_crop_hm_downsample = sitk.GetArrayFromImage(T1w_img_crop_hm_downsample)
                    x = T1w_img_crop_hm_downsample.shape[0]
                    y = T1w_img_crop_hm_downsample.shape[1]
                    z = T1w_img_crop_hm_downsample.shape[2]

                    temp = np.zeros([128, 128, 128])
                    temp[max(118 - x, 0):118, int((128 - y) / 2):int((128 - y) / 2) + y,
                    int((128 - z) / 2):int((128 - z) / 2) + z] = T1w_img_crop_hm_downsample[max(x - 118, 0):, :, :]

                    # testing
                    T1w_img_crop_hm_downsample = torch.tensor(temp).float()
                    T1w_img_crop_hm_downsample = T1w_img_crop_hm_downsample.to(device)

                    T1w_img_crop_hm_downsample = (T1w_img_crop_hm_downsample - torch.min(
                        T1w_img_crop_hm_downsample)) / (
                                                         torch.max(T1w_img_crop_hm_downsample) - torch.min(
                                                     T1w_img_crop_hm_downsample))

                    T1w_img_crop_hm_downsample = torch.unsqueeze(T1w_img_crop_hm_downsample, dim=0)
                    T1w_img_crop_hm_downsample = torch.unsqueeze(T1w_img_crop_hm_downsample, dim=0)

                    logits, moving_trans, y_source, pos_flow, x_reg, atlas_mask_formable = model(
                        T1w_img_crop_hm_downsample, atlas,
                        atlas_mask)

                    atlas_mask_formable = atlas_mask_formable[0, 0, :, :, :].cpu().numpy()
                    atlas_mask_crop = atlas_mask_formable[max(118 - x, 0):118,
                                      int((128 - y) / 2):int((128 - y) / 2) + y,
                                      int((128 - z) / 2):int((128 - z) / 2) + z]

                    # resample to original resolution
                    xsize = atlas_mask_crop.shape[0] * 2 * (1.0 / float(spacing[0]))
                    ysize = atlas_mask_crop.shape[1] * 2 * (1.0 / float(spacing[1]))
                    zsize = atlas_mask_crop.shape[2] * 2 * (1.0 / float(spacing[2]))
                    print('######', xsize, ysize, zsize)

                    resampler = sitk.ResampleImageFilter()
                    resampler.SetSize([round(xsize), round(ysize), round(zsize)])
                    resampler.SetOutputSpacing([float(spacing[0]), float(spacing[1]), float(spacing[2])])
                    resampler.SetOutputOrigin(origin)
                    resampler.SetOutputDirection(direction)
                    resampler.SetInterpolator(sitk.sitkLinear)

                    atlas_mask_crop = sitk.GetImageFromArray(atlas_mask_crop)
                    atlas_mask_crop.SetSpacing([2, 2, 2])
                    atlas_mask_crop_upsample = resampler.Execute(atlas_mask_crop)

                    save_dir = args.input_path + img_name + '-atlas_mask_formable-downsample-upsample.hdr'
                    atlas_mask_crop_upsample = sitk.GetArrayFromImage(atlas_mask_crop_upsample)
                    out = sitk.GetImageFromArray(atlas_mask_crop_upsample)
                    out.SetOrigin(origin)
                    out.SetSpacing([float(spacing[0]), float(spacing[1]), float(spacing[2])])
                    out.SetDirection(direction)
                    sitk.WriteImage(out, save_dir)

                    # resize to original size
                    binary_image = np.where(atlas_mask_crop_upsample > 0.5, 1.0, 0.0)
                    binary_image = binary_image.astype(int)
                    binary_image = sitk.GetImageFromArray(binary_image)
                    opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
                    opening_filter.SetKernelRadius(5)
                    opened_image = opening_filter.Execute(binary_image)

                    opened_image = sitk.GetArrayFromImage(opened_image)
                    brain_mask = np.zeros([size[0], size[1], size[2]])
                    brain_mask[max(x_min - 20, 0):min(x_max + 40, T1w_img.shape[0]),
                    max(y_min - 40, 0):min(y_max + 40, T1w_img.shape[1]),
                    max(z_min - 40, 0):min(z_max + 40, T1w_img.shape[2])] = opened_image

                    if i==0:
                        save_dir = file.path.replace('.hdr', '-stripped-2.nii', 1)
                        out = sitk.GetImageFromArray(brain_mask)
                        out.SetOrigin(origin)
                        out.SetSpacing(spacing)
                        out.SetDirection(direction)
                        sitk.WriteImage(out, save_dir)

                    else:
                        save_dir = args.output_path + img_name + '-brainmask.nii'
                        out = sitk.GetImageFromArray(brain_mask)
                        out.SetOrigin(origin)
                        out.SetSpacing(spacing)
                        out.SetDirection(direction)
                        sitk.WriteImage(out, save_dir)

                    # os.remove(os.path.join(args.input_path, file.name.replace('.hdr', '-stripped-2.hdr', 1)))

if __name__ == '__main__':
    main()
