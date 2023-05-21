# Knowledge-empowered_Lifespan_Skull_Stripping

## Model Overview

![](https://github.com/limeiwang5050/Knowledge-empowered_Lifespan_Skull_Stripping/blob/main/Picture3-20.png)

This repository contains the code for the knowledge-empowered lifespan skull stripping framework. It is designed to perform skull stripping on lifespan subjects from multiple sites by utilizing personalized prior information from atlases. The code presents the complete skull stripping process for T1-weighted MRIs under the guidence of age-specific brain atlases, including brain extraction module and registration module. The brain extraction module utilizes a brain extraction network to extract the brain parenchyma and generate an initial estimation. In the registration module, an age-specific atlas is registered to the estimated brain, incorporating personalized prior knowledge. The deformation field generated during the registration process is then applied to the corresponding atlas, resulting in the final brain mask.

## Installing Dependencies

```
pip install -r requirements.txt
```

## Training

## Testing 


#### 1. Data preprocessing



#### 2. Update testing data name and file address

