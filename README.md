# Atlases-empowered_Lifespan_Skull_Stripping
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14047793.svg)](https://doi.org/10.5281/zenodo.14047793)

The LifespanStrip framework leverages atlas-based prior information to enable skull stripping across lifespan MRIs from multiple sites. We have further enhanced the model to support skull stripping on both T1-weighted and T2-weighted MRIs.

### Getting Started: Containerized Execution (Docker)

We have provided the Docker container for LifespanStrip. The container is pre-configured with all necessary dependencies and follows the BIDS-Apps protocol principles.
#### Prerequisites

1. Install Docker
   
    Ensure Docker is installed on your system.

2. Pull the Docker Image
   
    Use the following command to pull the pre-built Docker image:

    ```
    docker pull limeiw/lifespanstrip:v2.0.1
    ```
3. Prepare Your Data

    Ensure your input data adheres to the BIDS standard (if applicable). Alternatively, you can test it on the Testing_BIDS we provide.

#### How to Run the Container

1. Basic Command
   
    Run the Docker container using the following command:

    ```
    docker run --gpus all -v /path/to/input:/app/data limeiw/lifespanstrip:v2.0.1 --bids_root filename_of_BIDS_dataset --subject_id id_of_subject --session_id id_of_session
    ```
    ***'-v /path/to/input'*** mounts the input data directory to the container's ***'/app/data'*** directory.
   
    ***'--bids_root'*** specifies the BIDS dataset to be processed.
   
    ***'--subject_id'*** specifies a subject within the BIDS dataset to be processed (optional).
   
    ***'--session_id'*** specifies a session within the BIDS dataset to be processed (optional).
   

3. Example Usage
   
    For example, using the Testing_BIDS we provided. The following command will process all the data that meets the criteria within the Testing_BIDS.

    ```
    docker run --gpus all -v /home/user/data:/app/data limeiw/lifespanstrip:v2.0.1 --bids_root Testing_BIDS
    ```

    The following command will process a specific subject when the ***'--subject_id'*** is provided (e.g. 001).
    ```
    docker run --gpus all -v /home/user/data:/app/data limeiw/lifespanstrip:v2.0.1 --bids_root Testing_BIDS --subject_id A00032170
    ```

    The following command will process a specific session when the ***'--session_id'*** (e.g. V02) is provided.
    ```
    docker run --gpus all -v /home/user/data:/app/data limeiw/lifespanstrip:v2.0.1 --bids_root Testing_BIDS --session_id V02
    ```


### Getting Started: A Guide to Using Code Directly from GitHub
![image](https://github.com/DBC-Lab/Atlases-empowered_Lifespan_Skull_Stripping/blob/main/Tutorial.jpg)

### Model overview

![image](https://github.com/DBC-Lab/Atlases-empowered_Lifespan_Skull_Stripping/blob/main/Picture2.png)

This repository includes the code for the atlases-empowered lifespan skull stripping framework, which is designed to perform skull stripping on T1-weighted MRIs subjects across the lifespan from multiple sites, using personalized prior information derived from atlases. The code presents the complete skull stripping process for T1-weighted MRIs under the guidance of age-specific brain atlases, including the brain extraction module and registration module. The brain extraction module utilizes a brain extraction network to extract the brain parenchyma and generate an initial estimation. In the registration module, an age-specific atlas is registered to the estimated brain, incorporating personalized prior knowledge. The deformation field generated during the registration process is then applied to the corresponding atlas, resulting in the final brain mask.

> [!NOTE]
> The current model is suitable for lifespan T1-weighted MRIs from birth to old age, with minor tissue deformities. We have also updated the extended model to handle multi-modality images and pathological MRIs with high distortion (HD), as shown in the skull stripping results below. 

### Update: Our single model can handle T1w/T2w/FLAIR MRIs. Here are demos of skull stripping results from lifespan T2w/FLAIR MRIs (left to right: raw MRI, estimated brain probability, and brain mask):
<div align=center><img src="https://github.com/DBC-Lab/Atlases-empowered_Lifespan_Skull_Stripping/assets/110405481/a06754bc-d525-4873-bae1-56792aae75cd" width="400px"></div>

### Update: By augmenting training subjects with simulated tissue loss, our model can handle pathological MRIs with High Distortion (HD). Here are demos of skull stripping results from pathological T1w MRIs (left to right: raw MRI, estimated brain probability, and brain mask):
<div align=center><img src="https://github.com/DBC-Lab/Atlases-empowered_Lifespan_Skull_Stripping/blob/main/Performance_on_pathological_MRIs.png" width="400px"></div>

### Update: We have expanded our model to perform skull stripping across various imaging sequences and modalities, such as FLAIR (Fluid-Attenuated Inversion Recovery), PD (Proton Density)-Weighted Imaging, MRA (Magnetic Resonance Angiography), EPI (Echo Planar Imaging), and DWI (Diffusion-Weighted Imaging). Here are demos of skull stripping results from different images (left to right: raw MRI, estimated brain probability, and brain mask):
<div align=center><img src="https://github.com/DBC-Lab/Atlases-empowered_Lifespan_Skull_Stripping/blob/main/Performance_on_different_modalities.png" width="400px"></div>


---
## File Descriptions

### Training_subjects: 
This folder contains 10 sub-folders (*0month, 3months, 6months, 9months, 12months, 18months, 24months, Adolescent, Adult, Elder*), with each sub-folder containing 30 training data. The data sources have been summarized in the table below.

| Age group | Data source | Number | Format |
| --- | --- | --- | --- |
| 0 month | [Developing Human Connectome Project (dHCP)](http://www.developingconnectome.org/) | 6 | .nii.gz |
| 0 month | [Baby Connectome Project (BCP)](https://nda.nih.gov/edit_collection.html?id=2848/) | 24| .nii.gz |
| 3 months | [Baby Connectome Project (BCP)](https://nda.nih.gov/edit_collection.html?id=2848/) | 30 | .nii.gz |
| 6 months | [Baby Connectome Project (BCP)](https://nda.nih.gov/edit_collection.html?id=2848/) | 30 | .nii.gz |
| 9 months | [Baby Connectome Project (BCP)](https://nda.nih.gov/edit_collection.html?id=2848/) | 30 | .nii.gz |
| 12 months | [Baby Connectome Project (BCP)](https://nda.nih.gov/edit_collection.html?id=2848/) | 30 | .nii.gz |
| 18 months | [Baby Connectome Project (BCP)](https://nda.nih.gov/edit_collection.html?id=2848/) | 30 | .nii.gz |
| 24 months | [Baby Connectome Project (BCP)](https://nda.nih.gov/edit_collection.html?id=2848/) | 30 | .nii.gz |
| Adolescent | [Baby Connectome Project (BCP)](https://nda.nih.gov/edit_collection.html?id=2848/) | 12 | .nii.gz |
| Adolescent | [Autism Brain Imaging Data Exchange (ABIDE)](https://fcon_1000.projects.nitrc.org/indi/abide/) | 18 | .nii.gz |
| Adult | [International Consortium for Brain Mapping (ICBM)](https://ida.loni.usc.edu/login.jsp) | 30 | .nii.gz |
| Elder | [Alzheimer’s Disease Neuroimaging Initiative (ADNI)](https://ida.loni.usc.edu) | 30 | .nii.gz |

> [!NOTE]
> Since we do not have the rights to share the raw data, users may have to download the raw data from the above links in the table and perform the following steps to match the provided brain masks:
> (1) Converting to the required format;
> (2) Reorienting MRI data to the RAI (Right-Anterior-Inferior) orientation;
> (3) Resampling the reorientated MRI data to the size of 256\*256\*256 with the resolution of 1\*1\*1.


### Testing_subjects: 

We will gradually release our annotations for over 20,000 scans to serve as a resource for the community. So far, we have released annotations for dHCP, MAP, CCNP, ABVIB and SLIM datasets, with more to come.

| Age group | Data source | Number | Format |
| --- | --- | --- | --- |
| 24-45 days | [Developing Human Connectome Project (dHCP)](http://www.developingconnectome.org/) | 375 | .nii.gz |
| 0-72 months | [Multi-visit Advanced Pediatric (MAP)](https://circlelab.unc.edu/studies/completed-data-collection/multi-visit-advanced-pediatric-brain-imaging-map/) | 265| .nii.gz |
| 18-30 years | [Chinese Color Nest Project (CCNP)](https://ccnp.scidb.cn/en) | 211 | .nii.gz |
| 62-90 years | [Aging Brain: Vasculature, Ischemia, and Behavior (ABVIB)](https://nda.nih.gov/edit_collection.html?id=2848/) | 228 | .nii.gz |
| 17-27 years | [Southwest University Longitudinal Imaging Multimodal (SLIM) Brain Data Repository](http://fcon_1000.projects.nitrc.org/indi/retro/southwestuni_qiu_index.html) | 572 | .nii.gz |
|  | More will be coming | | |

> [!NOTE]
> Since we do not have the rights to share the raw data, users may have to download the raw data from the above links in the table and perform the following steps to match the provided brain masks:
> (1) Converting to the required format;
> (2) Reorienting MRI data to the RAI (Right-Anterior-Inferior) orientation;
> (3) Resampling the reorientated MRI data to the size of 256\*256\*256 with the resolution of 1\*1\*1.
> 

### Lifespan_brain_atlases:
> 
> We employ dense atlases at 0, 3, 6, 9, 12, 18, 24 months of age, which were built from UNC/UMN baby connectome project [1]. For the later age, we employ three sparse atlases, including an adolescent brain atlas covering 3-18 years old from [2], an adult brain atlas covering 19-64 years old from [3], an elderly brain atlas covering 65+ years old from [4].
>> [1]. Chen, L., Wu, Z., Hu, D., Wang, Y., Zhao, F., Zhong, T., Lin, W., Wang, L., Li, G.: A 4d infant brain volumetric atlas based on the unc/umn baby connectome project (bcp) cohort. Neuroimage 253, 119097 (2022)
>> [2]. Fonov, V., Evans, A.C., Botteron, K., Almli, C.R., McKinstry, R.C., Collins, D.L., Group, B.D.C.: Unbiased average age-appropriate atlases for pediatric studies. Neuroimage 54(1), 313–327 (2011)
>> [3]. Rohlfing, T., Zahr, N.M., Sullivan, E.V., Pfefferbaum, A.: The sri24 multichannel atlas of normal adult human brain structure. Human brain mapping 31(5), 798–819 (2010)
>> [4]. Wu, Y., Ridwan, A.R., Niaz, M.R., Qi, X., Zhang, S., Bennett, D.A., Arfanakis, K., Initiative, A.D.N.: Development of high quality t1-weighted and diffusion tensor templates of the older adult brain in a common space. NeuroImage 260, 119417 (2022)

>> 10 T1-weighted brain atlases and the corresponding brain masks covering 0, 3, 6, 9, 12, 18, 24 months, 3-18 years, 18-64 years, and 65+ years old in downsample space.

>> ***brain-atlas-x-downsample.hdr***: the brain atlas image at x age in downsample space.

>> ***brain-atlas-x-mask-downsample.hdr***: the brain mask of atlas at x age in downsample space.



### Histogram Matching Template:  

>> The subjects in the folder ***Template*** are the template for histogram matching.

>> ***template.hdr***: the template T1-weighted MRI.



---
## Testing Instruction

### 1. System Requirements
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

You can use the following command to install all dependencies.

```
pip3 install -r requirements.txt
```

### 2. Data and Model

#### Data Preparation

We have provided 3 example adult T1-weighted MR images in the ***'./Testing_subjects/'*** folder. These images are sourced from the [Chinese Color Nest Project (CCNP)](https://www.scidb.cn/en/detail?dataSetId=c81f0e90a51b4cfca348ce4da6ca734e). The age of the subjects in months at the time of image acquisition is integrated into the filenames of the images. If you want to test your own data located in different folders, you can use the ***--input_path*** argument to modify the input path. Note all imaging data was stored in the Analyze 7.5 file format, comprising of ***'.hdr'*** and ***'.img'*** files. 

#### Model Preparation

We have provided our pre-trained model (Lifespan_Skull_Stripping.pt), which can be downloaded from this link: https://www.dropbox.com/scl/fo/3f9o9sgls4e88jved8ooo/h?rlkey=h46zb5ulwbacrbh8vtsjwygn0&dl=0

After downloading the model, it should be placed in ***./Model/*** folder for convenient subsequent testing.


### 3. Testing Tutorial

You can achieve skull stripping on brain MRIs using the provided checkpoint by executing the following command:

```
python LifespanStrip.py --input_path  --output_path  --age_in_month
```

In this command, ***--input_path*** specifies the folder to the testing data (the default is ***'Testing_subjects'***). The ***--output_path*** indicates where the skull stripping results will be saved (with ***'Testing_subjects'*** as the default location). Additionally, the ***--age_in_month*** option is utilized to specify the exact age in months of the test subjects. Based on the provided age information, our testing procedure will select the most appropriate brain atlas for testing. 

For each provided T1-weighted MR image, our testing procedure initially rotates the image to the RAI (Right-Anterior-Inferior) orientation. Subsequently, it conducts inhomogeneity correction (10.1109/TMI.2010.2046908). If your testing dataset already includes images with corrected inhomogeneity, you can utilize the option ***--N4=Flase*** to bypass this step, which can help save processing time.

After the preprocessing step, the images are resized to a uniform size of 128 × 128 × 128, with a resolution of 2×2×2 mm³, and then rescaled to the same range. Following this, the testing procedure utilizes the trained model to perform skull stripping and then saves the final brain mask, resampled to the input image size, in the output path ***--output_path***.


---
## Training Instruction

### 1. Setting Hyper-parameters

In the ***networks*** folder, our network with standard hyper-parameters for the task of knowledge-empowered lifespan skull stripping can be defined as follows:

   ```
   model = Net(
        in_channels=1,
        out_channels=2,
        img_size=(128, 128, 128),
        conv_block=True)
   ```
   
The above model is used for brain T1w MR image (1-channel input) and for 2-class outputs, and the network expects resampled input images with the size of (128, 128, 128) and the resolution of (2, 2, 2). 

### 2. Initiating Training

In the ***main.py***, our initial training setting is as follows:

   ```
   python main.py
   --batch_size=1
   --data_dir='./Training_data/' 
   --json_list='./datasets/Training_data.json'
   --optim_lr=1e-4
   --lrschedule=warmup_cosine
   --infer_overlap=0.8
   ```

If you would like to train on your own data, please note that you need to provide the location of your dataset directory by using ***--data_dir*** and specify the training data by using ***--json_list***.

### 3. Running Training

You can initiate the training process by executing the following command:

```
python3 main.py
```
---
## Citation

## License information
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements 
This work was supported by the National Institute of Mental Health under Award Numbers MH133845, MH117943, MH123202, and MH116225.

## Support

- **Issue**: If you encounter any issues or have concerns, please submit them here [(https://github.com/DBC-Lab/Atlases-empowered_Lifespan_Skull_Stripping/issues)](https://github.com/DBC-Lab/Atlases-empowered_Lifespan_Skull_Stripping/issues) 


