## Problem Overview
The DynUnet architecture, enhanced with the Duck3D Block , is deployed for segmenting three different types of brain tumors from MRI scans: whole tumor (WT), tumor core (TC), and enhancing tumor (ET). Each MRI scan is represented as a tensor in  $R^{h \times w \times \times d \times 4}$, encompassing four distinct imaging modalities: T1-weighted (T1w), post-contrast T1-weighted with Gadolinium (T1Gd), T2-weighted (T2w), and Fluid Attenuated Inversion Recovery (FLAIR).

The objective is to classify each voxel into one of four categories: WT, TC, ET, or non-tumor background. The segmentation process employs a binary mask $F$  in the space $R^{h \times w \times \times \times 4}$, where each dimension in the fourth axis represents the probability that a voxel belongs to one of the tumor categories.

## Model and Duck3D Block
