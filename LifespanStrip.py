import os
import torch
import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom
from scipy.ndimage import label as ndimage_label
from scipy.ndimage import sum as ndi_sum
from monai.inferers import sliding_window_inference
import math
from networks.net import NET
import argparse
import SimpleITK as sitk

import subprocess

parser = argparse.ArgumentParser(description='Skull stripping pipeline')
parser.add_argument('--pretrained_dir', default='./Model/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--data_dir', default='/', type=str, help='dataset directory')
parser.add_argument('--pretrained_model_name', default='epoch1049model-all.pt', type=str, help='pretrained model name')
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
        if os.path.exists(args.input_path):
            print(f"Work on {args.input_path}")
        else:
            print(f"Input path {args.input_path} does not exist!")

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
            print(f"Output path {args.output_path} created.")

        for file in os.scandir(args.input_path):
            if '.nii' in file.name:
                # Read and preprocess MRI image
                img_name = os.path.splitext(file.name)[0]
                print('Processing:', img_name)

                T1w_img = sitk.ReadImage(file.path)
                original_direction = T1w_img.GetDirection()
                print("Original Direction:", original_direction)
                new_direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
                T1w_img.SetDirection(new_direction)

                # save reoriented MRI image
                save_dir = args.output_path + '/' + img_name + '-reorient.nii'
                sitk.WriteImage(T1w_img, save_dir)

                # print new direction
                size, origin, spacing, direction = T1w_img.GetSize(), T1w_img.GetOrigin(), T1w_img.GetSpacing(), T1w_img.GetDirection()
                print("New Direction:",direction)
                print('Reoriente done')

                # Apply N4 bias correction if specified
                if args.N4 == 'True':
                    corrector = sitk.N4BiasFieldCorrectionImageFilter()
                    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
                    T1w_img = corrector.Execute(T1w_img)
                    print('N4 done')
                print('Skip N4')

                # Select appropriate atlas and mask based on age
                age_month = int(args.age_in_month)
                if age_month <= 2:
                    month = 'Month0'
                elif age_month <= 4:
                    month = 'Month3'
                elif age_month <= 7:
                    month = 'Month6'
                elif age_month <= 10:
                    month = 'Month9'
                elif age_month <= 14:
                    month = 'Month12'
                elif age_month <= 20:
                    month = 'Month18'
                elif age_month <= 28:
                    month = 'Month24'
                elif age_month <= 540:
                    month = 'Adolescent'
                elif age_month <= 1920:
                    month = 'Adult'
                else:
                    month = 'Elder'

                # Load atlas and atlas mask
                atlas_path = f'./Lifespan_brain_atlases/brain-atlas-{month}-downsample.hdr'
                atlas_mask_path = f'./Lifespan_brain_atlases/brain-atlas-{month}-mask-downsample.hdr'

                atlas = sitk.ReadImage(atlas_path)
                atlas_mask = sitk.ReadImage(atlas_mask_path)

                atlas = torch.tensor(sitk.GetArrayFromImage(atlas)).float().to(device)
                atlas_mask = torch.tensor(sitk.GetArrayFromImage(atlas_mask)).float().to(device)

                # Rescale atlas intensity
                atlas = (atlas - atlas.min()) / (atlas.max() - atlas.min())
                print('Load atlas done')

                #######################
                #### First_testing ####
                #######################

                # Downsample image to 128x128x128 with 2.0 mm resolution
                resampler = sitk.ResampleImageFilter()
                resampler.SetSize([128, 128, 128])
                resampler.SetOutputSpacing([2.0, 2.0, 2.0])
                resampler.SetOutputOrigin(origin)
                resampler.SetOutputDirection(direction)
                resampler.SetInterpolator(sitk.sitkLinear)
                T1w_img_downsample = resampler.Execute(T1w_img)
                T1w_img_downsample.SetDirection(T1w_img.GetDirection())

                # Load and histogram-match the template
                template = sitk.ReadImage('./Template/template.hdr')
                matcher = sitk.HistogramMatchingImageFilter()
                matcher.SetNumberOfHistogramLevels(1024)
                matcher.SetNumberOfMatchPoints(7)
                matcher.ThresholdAtMeanIntensityOn()
                T1w_img_downsample_hm = matcher.Execute(T1w_img_downsample, template)


                # Normalize and prepare for model input
                T1w_img_downsample_hm = sitk.GetArrayFromImage(T1w_img_downsample_hm)
                T1w_img_downsample_hm = torch.tensor(T1w_img_downsample_hm, device=device).float()
                T1w_img_downsample_hm = (T1w_img_downsample_hm - T1w_img_downsample_hm.min()) / (
                        T1w_img_downsample_hm.max() - T1w_img_downsample_hm.min())
                T1w_img_downsample_hm = T1w_img_downsample_hm.unsqueeze(0).unsqueeze(0)

                # Model inference
                logits, *_ = model(T1w_img_downsample_hm, atlas, atlas_mask)

                # Extract and process brain mask
                logits = logits.cpu().numpy()
                pre = np.argmax(logits, axis=1)[0]

                # Upsample logits to original size and resolution
                upsample_factor = [osz / 128 for osz in size]
                pre_upsample = zoom(pre, upsample_factor, order=3)
                pre_upsample_b = (pre_upsample > 0.5).astype(int)

                # Apply binary opening
                pre_upsample_b = sitk.GetImageFromArray(pre_upsample_b)
                opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
                opening_filter.SetKernelRadius(3)
                opened_image = opening_filter.Execute(pre_upsample_b)

                # Extract largest connected component
                opened_image = sitk.GetArrayFromImage(opened_image)
                labeled_array, _ = ndimage_label(opened_image)
                sizes = ndi_sum(opened_image, labeled_array, range(labeled_array.max() + 1))
                mask = sizes < max(sizes)
                labeled_array[mask[labeled_array]] = 0
                labeled_array, _ = ndimage_label(labeled_array)

                '''
                # Save the processed brain mask
                output_path = file.path.replace('.nii', '-stripped-2.hdr', 1)
                out_image = sitk.GetImageFromArray(labeled_array)
                out_image.SetOrigin(origin)
                out_image.SetSpacing(spacing)
                out_image.SetDirection(direction)
                sitk.WriteImage(out_image, output_path)
                '''

                print('First testing done')


                #######################
                #### Second_testing ###
                #######################

                # Step 1: Crop the initial brain region based on the first testing result
                T1w_img = sitk.GetArrayFromImage(T1w_img)
                loc = np.where(labeled_array == 1)
                x_min = np.min(loc[0])
                x_max = np.max(loc[0])
                if ((x_max - x_min) * spacing[0]) % 2 != 0:
                    x_max -= 1
                y_min = np.min(loc[1])
                y_max = np.max(loc[1])
                if ((y_min - y_max) * spacing[1]) % 2 != 0:
                    y_max -= 1
                z_min = np.min(loc[2])
                z_max = np.max(loc[2])
                if ((z_min - z_max) * spacing[2]) % 2 != 0:
                    z_max -= 1

                T1w_img_crop = T1w_img[
                               max(x_min - 20, 0):min(x_max + 40, T1w_img.shape[0]),
                               max(y_min - 40, 0):min(y_max + 40, T1w_img.shape[1]),
                               max(z_min - 40, 0):min(z_max + 40, T1w_img.shape[2])
                               ]

                # Step 2: Downsample the cropped image to a uniform 2.0 mm resolution
                xsize = T1w_img_crop.shape[0]
                ysize = T1w_img_crop.shape[1]
                zsize = T1w_img_crop.shape[2]
                xsize_downsample = math.ceil(T1w_img_crop.shape[0] * spacing[0] * 0.5)
                ysize_downsample = math.ceil(T1w_img_crop.shape[1] * spacing[1] * 0.5)
                zsize_downsample = math.ceil(T1w_img_crop.shape[2] * spacing[2] * 0.5)

                T1w_img_crop = sitk.GetImageFromArray(T1w_img_crop)
                resampler = sitk.ResampleImageFilter()
                resampler.SetSize([zsize_downsample, ysize_downsample, xsize_downsample])
                resampler.SetOutputSpacing([2, 2, 2])
                resampler.SetOutputOrigin(T1w_img_crop.GetOrigin())
                resampler.SetOutputDirection(T1w_img_crop.GetDirection())
                resampler.SetInterpolator(sitk.sitkLinear)
                T1w_img_crop_downsample = resampler.Execute(T1w_img_crop)

                # Step 3: Center the cropped brain region within a 128x128x128 array
                T1w_img_crop_downsample = sitk.GetArrayFromImage(T1w_img_crop_downsample)
                x = T1w_img_crop_downsample.shape[0]
                y = T1w_img_crop_downsample.shape[1]
                z = T1w_img_crop_downsample.shape[2]

                temp = np.zeros([128, 128, 128])
                temp[max(118 - x, 0):118, int((128 - y) / 2):int((128 - y) / 2) + y,
                int((128 - z) / 2):int((128 - z) / 2) + z] = T1w_img_crop_downsample[max(x - 118, 0):, :, :]

                # Step 4: Perform histogram matching to align intensity distributions
                temp = sitk.GetImageFromArray(temp)
                temp = sitk.Cast(temp, sitk.sitkFloat64)
                template = sitk.Cast(template, sitk.sitkFloat64)
                matcher = sitk.HistogramMatchingImageFilter()
                matcher.SetNumberOfHistogramLevels(1024)
                matcher.SetNumberOfMatchPoints(7)
                matcher.ThresholdAtMeanIntensityOn()
                T1w_img_crop_downsample_temp_hm = matcher.Execute(temp, template)
                T1w_img_crop_downsample_temp_hm = sitk.GetArrayFromImage(T1w_img_crop_downsample_temp_hm)

                # Step 5: Normalize the centered image for model input
                T1w_img_crop_downsample_temp_hm = torch.tensor(T1w_img_crop_downsample_temp_hm, device=device).float()
                T1w_img_crop_downsample_temp_hm = (T1w_img_crop_downsample_temp_hm - T1w_img_crop_downsample_temp_hm.min()) / (
                        T1w_img_crop_downsample_temp_hm.max() - T1w_img_crop_downsample_temp_hm.min())
                T1w_img_crop_downsample_temp_hm= T1w_img_crop_downsample_temp_hm.unsqueeze(0).unsqueeze(0)

                # Step 6: Model inference to refine the brain mask
                logits, moving_trans, y_source, pos_flow, x_reg, atlas_mask_formable = model(
                    T1w_img_crop_downsample_temp_hm, atlas, atlas_mask
                )

                # Step 7: Extract and crop the atlas mask's center area
                atlas_mask_formable = atlas_mask_formable[0, 0].cpu().numpy()
                atlas_mask_crop = atlas_mask_formable[
                                  max(118 - x, 0):118,
                                  (128 - y) // 2:(128 - y) // 2 + y,
                                  (128 - z) // 2:(128 - z) // 2 + z
                                  ]

                # Step 8: Resample the cropped atlas mask to the original resolution
                resampler = sitk.ResampleImageFilter()

                resampler.SetSize([zsize, ysize, xsize])
                resampler.SetOutputSpacing(spacing)
                atlas_mask_crop = sitk.GetImageFromArray(atlas_mask_crop)
                atlas_mask_crop.SetSpacing([2.0, 2.0, 2.0])
                atlas_mask_crop_upsample = resampler.Execute(atlas_mask_crop)

                # Step 9: Threshold and refine the atlas mask to create a binary brain mask
                atlas_mask_crop_upsample = sitk.GetArrayFromImage(atlas_mask_crop_upsample)
                binary_image = np.where(atlas_mask_crop_upsample > 0.5, 1, 0).astype(np.int32)
                binary_image = sitk.GetImageFromArray(binary_image)

                # Step 10: Apply morphological opening to refine the binary brain mask
                opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
                opening_filter.SetKernelRadius(5)
                opened_image = opening_filter.Execute(binary_image)
                opened_image = sitk.GetArrayFromImage(opened_image)

                # Step 11: Restore the binary brain mask to the original position
                brain_mask = np.zeros(size, dtype=np.int32)
                brain_mask[max(x_min - 20, 0):min(x_max + 40, T1w_img.shape[0]),
                               max(y_min - 40, 0):min(y_max + 40, T1w_img.shape[1]),
                               max(z_min - 40, 0):min(z_max + 40, T1w_img.shape[2])] = opened_image

                # Step 12: Save the final brain mask
                save_dir = args.output_path + '/' + img_name + '-reorient-brainmask.nii'
                out = sitk.GetImageFromArray(brain_mask)
                out.SetOrigin(origin)
                out.SetSpacing(spacing)
                out.SetDirection(direction)
                sitk.WriteImage(out, save_dir)
                print('Skull stripping done')


if __name__ == '__main__':
    main()
