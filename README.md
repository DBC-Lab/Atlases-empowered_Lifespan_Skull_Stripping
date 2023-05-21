# Knowledge-empowered_Lifespan_Skull_Stripping

## Model Overview

![](https://github.com/limeiwang5050/Knowledge-empowered_Lifespan_Skull_Stripping/blob/main/Picture3-20.png)

This repository contains the code for an end-to-end infant brain parcellation pipeline.

The code is composed of two network folders: (1) Global_ROIs_localization_network, and (2) Local_ROIs_Refinement_network, and their respective pretrained models are placed in ```./model/``` files in their respective folders.

The Global_ROIs_localization_network first employs raw MRIs (T1w or/and T2w MRIs) to produce 146 regions probability maps. The Local_ROIs_Refinement_network further uses the 146 regions probability maps, together with raw MRIs, to refine the 146 regions probability maps.

For convenience, we provide a ```pipeline.csh``` by merging two networks for an end-to-end parcellation.



## Installing Dependencies

```
pip install -r requirements.txt
```

## Testing 
You can use the pre-trained model or test it on demo images.

If you would like to test on the demo images in ```./Demo_Images/```, you can directly run the following command:

```
./ pipeline.csh
```

If you would like to test on your own data, please follow below steps:

#### 1. Data preprocessing

The testing data should contain T1w or/and aligned T2w MRIs, after inhomogeneity correction. The image resolution is uniformly resampled to (1, 1, 1) mm3 with a size of (192, 192, 192).

#### 2. Update testing data name and file address

You need to change the *.json file in ```./Global_ROIs_localization_network/dataset/``` and ```./Local_ROIs_Refinement_network/dataset/``` to indicate your own testing data, and then provide the location of your dataset directory by using --data_dir.

Then run the ```./ pipeline.csh``` command for testing.
