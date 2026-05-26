# Bias Adjustment of Seasonal Arctic Sea Ice Predictions from the Canadian Earth System Model Version 5

This repository contains scripts for designing, running, and tuning hyperparameters of **post-processing scripts** for deep learning-based bias adjustment of Arctic Sea Ice Concentrations from the CanESM5 model.  

The **deterministic bias adjustment** can be performed using:
- **MLP**  
- **Simple CNN**  
- **Partial Convolution UNet** (based on ConvNeXt)  
- **ConvLSTM**  

The **probabilistic bias adjustment** is based on the **4-way conditional generative model** from Sohn et al. (2015), implemented with **partial convolution ConvNeXt blocks**.

Latest version is cVAE_1031 which includes the cVAE-CRPS model where the Gaussian decoder is replaced with a probabilstic generator. The training script is run_training_CVAE_1031_NOAA.py and 
predict_cVAE.py can be used for inference. The deterministic UNet-based initial guess can be trained using run_training_CNNs_NOAA.py and inference can be made using predict.py. 

**Note:** This repository is intended for internal script sharing and does not include final, cleaned versions of the codes and scripts.

## Contributors
This work was developed by **Parsa Gooya** in collaboration with **Reinel Sospedra-Alfonso** at the **Canadian Centre for Climate Modeling and Analysis (CCCma)**. 

## Copyright
© Environment and Climate Change Canada and the contributors, 2025. All rights reserved.  
For inquiries, contact **parsa.gooya@ec.gc.ca**.  
Do not copy or reproduce without proper citation.
