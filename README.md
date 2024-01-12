# Atlases-empowered_Lifespan_Skull_Stripping

## Model Overview

![image](https://github.com/DBC-Lab/Atlases-empowered_Lifespan_Skull_Stripping/blob/main/Fig_S2.png)

This repository contains the code for the knowledge-empowered lifespan skull stripping framework. It is designed to perform skull stripping on lifespan subjects from multiple sites by utilizing personalized prior information from atlases. The code presents the complete skull stripping process for T1-weighted MRIs under the guidance of age-specific brain atlases, including the brain extraction module and registration module. The brain extraction module utilizes a brain extraction network to extract the brain parenchyma and generate an initial estimation. In the registration module, an age-specific atlas is registered to the estimated brain, incorporating personalized prior knowledge. The deformation field generated during the registration process is then applied to the corresponding atlas, resulting in the final brain mask.

Note: The current model is suitable for lifespan subjects from birth to old age, with minor tissue deformities. We are now working on incorporating fetal subjects and pathological cases into our training dataset. Please stay tuned.

## Data and Data preprocessing
### Data
We selected fifteen representative lifespan subjects' MRIs as demo data in the ***'./Testing_subjects/'***, including 3 neonate subjects' scans, 3 infant subjects' scans, 3 adolescent subjects' scans, 3 adult subjects' scans, and 3 elder subjects' scans obtained from different scanners/protocols.
    

### Data preprocessing
For each MRI, the preprocessing steps include:  (1) adjusting the orientation of the images to a standard reference frame; (2) performing inhomogeneity correction (10.1109/TMI.2010.2046908); (3) resampling the image resolution into 2×2×2 mm3; (4) normalizing the intensity range across subjects to the same scale; and (5) rigidly moving the center of the brain part (based on ground-truth masks for training subjects or the estimated brain for testing subjects) to ensure a consistent position across subjects and facilitate the registration.


## File descriptions
> Lifespan_brain_atlases:
> 
> We employ dense atlases at 0, 3, 6, 9, 12, 18, 24 months of age, which were built from UNC/UMN baby connectome project [1]. For the later age, we employ three sparse atlases, including an adolescent brain atlas covering 3-18 years old from [2], an adult brain atlas covering 19-64 years old from [3], an elderly brain atlas covering 65+ years old from [4].
>> [1]. Chen, L., Wu, Z., Hu, D., Wang, Y., Zhao, F., Zhong, T., Lin, W., Wang, L., Li, G.: A 4d infant brain volumetric atlas based on the unc/umn baby connectome project (bcp) cohort. Neuroimage 253, 119097 (2022)
>> [2]. Fonov, V., Evans, A.C., Botteron, K., Almli, C.R., McKinstry, R.C., Collins, D.L., Group, B.D.C.: Unbiased average age-appropriate atlases for pediatric studies. Neuroimage 54(1), 313–327 (2011)
>> [3]. Rohlfing, T., Zahr, N.M., Sullivan, E.V., Pfefferbaum, A.: The sri24 multichannel atlas of normal adult human brain structure. Human brain mapping 31(5), 798–819 (2010)
>> [4]. Wu, Y., Ridwan, A.R., Niaz, M.R., Qi, X., Zhang, S., Bennett, D.A., Arfanakis, K., Initiative, A.D.N.: Development of high quality t1-weighted and diffusion tensor templates of the older adult brain in a common space. NeuroImage 260, 119417 (2022)

>> 10 T1-weighted brain atlases and the corresponding brain masks covering 0, 3, 6, 9, 12, 18, 24 months, 3-18 years, 18-64 years, and 65+ years old in downsample space.

>> ***brain-atlas-x-downsample.hdr***: the brain atlas image at x age in downsample space.

>> ***brain-atlas-x-mask-downsample.hdr***: the brain mask of atlas at x age in downsample space.


> Testing_subjects

>> The folder ***Testing_subjects*** contains 5 subfolders, each of which includes 3 T1-weighted MRIs, covering neonates, infants, adolescents, adults, and elders.

>> ***subject-x-T1w.hdr***: the T1w MRI.

> Histogram matching template

>> The subjects in the folder ***Template*** are the template for histogram matching.

>> ***template.hdr***: the template T1w MRI.


## Training and Testing
### System Requirements
#### Hardware Requirements
This model requires only a standard computer with enough RAM to support the operations defined by a user. For optimal performance, we recommend a computer with a 16GB or higher memory GPU.

#### Software Requirements
##### OS Requirements
This model is supported by Linux, which has been tested on ***Red Hat Enterprise Linux Server release 8***.
##### Python Dependencies
This model mainly depends on the Python scientific stack.

    torch==1.9.1
    numpy==1.24.3
    monai==0.7.0
    nibabel==3.1.1
    tqdm==4.59.0
    einops==0.3.0
    tensorboardX==2.1
    SimpleITK==2.2.1 


### Training

1. Setting the hyper-parameters for the network

In the ***networks*** folder, our network with standard hyper-parameters for the task of knowledge-empowered lifespan skull stripping can be defined as follows:

   ```
   model = Net(
        in_channels=1,
        out_channels=2,
        img_size=(128, 128, 128),
        conv_block=True,
        dropout_rate=0.0)
   ```
   
The above model is used for brain T1w MR image (1-channel input) and for 2-class outputs, and the network expects resampled input images with the size of (128, 128, 128) and the resolution of (2, 2, 2). 

2. Initiating training

In the ***main.py***, our initial training setting is as follows:

   ```
   python main.py
   --batch_size=1
   --data_dir=
   --json_list=
   --optim_lr=1e-4
   --lrschedule=warmup_cosine
   --infer_overlap=0.5
   --save_checkpoint
   --data_dir=/dataset/dataset0/
   --pretrained_dir='./pretrained_models/'
   --pretrained_model_name='model_best_acc.pth'
   --resume_ckpt
   ```

If you would like to train on your own data, please note that you need to provide the location of your dataset directory by using ***--data_dir*** and specify the training data by using ***--json_list***.

3. Running training

You can initiate the training process by executing the following command:

```
python3 main.py
```

### Testing
1. Initiating testing

In the ***testing.py***, our initial testing setting is as follows:

```
--data_dir=
--pretrained_dir='./Model/'
--pretrained_model_name
--saved_checkpoint=ckpt
```
We have provided our pre-trained model (Lifespan_Skull_Stripping.pt), which can be downloaded from this link: https://www.dropbox.com/scl/fo/3f9o9sgls4e88jved8ooo/h?rlkey=h46zb5ulwbacrbh8vtsjwygn0&dl=0

After downloading the model, it should be placed in ***/Model/*** folder for convenient subsequent testing.

2. Running testing

You can run inference using the provided checkpoint by executing the following command:

```
python testing.py --input_path  --output_path --stage --age_month
```
In this command, ***--input_path*** specifies the path to the testing data, ***--output_path*** indicates where the skull stripping results will be saved, ***--stage*** identifies the life stage of the test data (options include Neonate, Infant, Adolescent, Adult, Elder), and ***--age_month*** is used to define the precise age in months for cases when the ***--stage*** is Neonate or Infant (options are 0m, 3m, 6m, 9m, 12m, 18m, 24m).

The demo testing subjects are located in the ***/Testing_subjects/*** folder, along with their corresponding --stage information. For subjects in the Neonate folder, the ***--age_month*** is set to 0m, while for those in the Infant folder, the ***--age_month*** is set to 24m.


