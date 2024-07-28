# Enhanced DynUNet for Brain Tumor Segmentation

## Introduction
This repository contains the implementation of the Enhanced DynUNet architecture, which integrates the Duck3D Block for improved brain tumor segmentation. Our method has been validated on the BraTS 2018 dataset, demonstrating superior performance in segmenting Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET).

## Highlights
- **State-of-the-art Performance**: Achieves higher Dice scores compared to existing models.
- **Innovative Architecture**: Incorporation of the Duck3D Block enhances feature extraction capabilities.
- **Clinical Relevance**: Provides a reliable tool for aiding in the diagnosis and treatment planning of brain tumors.

## Model Overview
The Enhanced DynUNet model incorporates multiple novel components:
- **Duck3D Block**: Enhances the model's ability to focus on relevant features within MRI scans.
- **Residual Blocks**: Implemented at various points to prevent gradient vanishing and ensure deep feature learning.
- **Multi-Scale Processing**: Facilitates accurate segmentation across different tumor regions.

## Installation
To set up this project, clone the repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/enhanced-dynunet-brain-tumor-segmentation.git
cd enhanced-dynunet-brain-tumor-segmentation
pip install -r requirements.txt
