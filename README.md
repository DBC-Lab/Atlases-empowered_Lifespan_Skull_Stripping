# Atlases-empowered_Lifespan_Skull_Stripping

## Model Overview

![](https://github.com/limeiwang5050/Knowledge-empowered_Lifespan_Skull_Stripping/blob/main/Picture3-20.png)

This repository contains the code for the knowledge-empowered lifespan skull stripping framework. It is designed to perform skull stripping on lifespan subjects from multiple sites by utilizing personalized prior information from atlases. The code presents the complete skull stripping process for T1-weighted MRIs under the guidance of age-specific brain atlases, including the brain extraction module and registration module. The brain extraction module utilizes a brain extraction network to extract the brain parenchyma and generate an initial estimation. In the registration module, an age-specific atlas is registered to the estimated brain, incorporating personalized prior knowledge. The deformation field generated during the registration process is then applied to the corresponding atlas, resulting in the final brain mask.

## Data and Data preprocessing
### Data
We selected five representative lifespan subjects' MRIs as demo data, including a neonate subject from the Developing Human Connectome Project (dHCP), an infant subject from National Database for Autism Research (NDAR), an adolescent subject from Autism Brain Imaging Data Exchange (ABIDE), an adult subject from 3R-BRAIN, and an elder subject from Alzheimer’s Disease Neuroimaging Initiative (ADNI).

    dHCP: Philips scanner (<http://www.developingconnectome.org/>)
    
    NDAR: Siemens scanner (<https://nda.nih.gov/edit collection.html?id=19>)
    
    ABIDE: Philips scanner (<https://fcon 1000.projects.nitrc.org/indi/abide/>)
    
    3R-BRAIN: GE scanner (<http://deepneuro.bnu.edu.cn/?p=163>)
    
    ADNI: Siemens scanner (<https://ida.loni.usc.edu/login.jsp>)
    

### Data preprocessing
For each MRI, the preprocessing steps include: (1) adjusting the orientation of the images to a standard reference frame; (2) performing inhomogeneity correction; (3) resampling the image resolution into 2×2×2 mm3; and (4) histogram matching with the template.

## File descriptions
> Brain atlases

>> 10 T1w brain atlases covering 0, 3, 6, 9, 12, 18, 24 months, 3-18 years, 18-64 years, and 65+ years old.

>> ***atlas-x.hdr***: the brain atlas image at x age.

> Training_subjects

>> The subjects in the folder ***Training_subjects*** are 300 T1w MRIs with corresponding manual labels.

>> ***subject-x-T1w.hdr***: the T1w MRI.

>> ***subejct-x-label.hdr***: the manual label.

> Testing_subjects

>> The subjects in the folder ***Testing_subjects*** are 5 representative T1w MRIs at neonate, infant, adolescent, adult, and elder age groups.

>> ***subject-x-T1.hdr***: the T1w MRI.

> Histogram matching template

>> The subjects in the folder ***Histogram matching template*** are the template for histogram matching.

>> ***HM-template.hdr***: the template T1w MRI.


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
        norm_name='instance',
        conv_block=True,
        dropout_rate=0.0)
   ```
   
The above model is used for brain T1w MR image (1-channel input) and for 2-class outputs, and the network expects resampled input images with the size of (128, 128, 128). 

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

Using the following command to perform training:

```
python3 main.py
```

### Testing
1. Initiating testing

In the ***test.py***, our initial testing setting is as follows:

```
--infer_overlap=0.5
--data_dir=
--json_list=
--pretrained_dir='./pretrained_models/'
--pretrained_model_name
--saved_checkpoint=ckpt
```

We provide our pre-trained checkpoint for the knowledge-empowered lifespan skull stripping task in the ***/runs/test/*** folder. ***--infer_overlap*** determines the overlap between the sliding window patches. A higher value typically results in more accurate segmentation outputs but with the cost of longer inference time.

If you would like to test on your own data, please note that you also need to provide the location of your dataset directory by using ***--data_dir*** and specify the testing data by using ***--json_list*** and indicate the model by using ***----pretrained_model_name***.

2. Running testing

The following command runs inference using the provided checkpoint:

```
python test.py
```

