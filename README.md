# 🧠 Brain Stroke Detection and Segmentation with MONAI

This repository contains a collection of Jupyter Notebooks for detecting and segmenting brain stroke regions from medical imaging data, using MONAI — a deep learning framework specialized for medical image analysis.

## 📌 Project Overview
This project focuses on:
- **Binary classification**: Detecting whether a brain stroke is present or not  
- **Segmentation**: Identifying and localizing stroke lesions in brain scans  
- Exploring different training strategies:
  - Patch-based training  
  - Full image-based training

The workflow is implemented step-by-step in notebook format, making it easy to follow and modify.

## 🛠️ Tools & Libraries
- [MONAI](https://monai.io/)
- PyTorch
- NumPy, OpenCV
- Jupyter Notebook

## 📁 Project Structure 
```
brain-stroke-detection/
├── data/
│   ├── data.txt
│
├── notebooks/
│   ├── classification/
│   │   ├── 01_raw_patchless.ipynb       # Raw image, patchless classification
│   │   └── 02_raw_patch.ipynb           # Raw image, patched classification
│   │   
│   │   
│   │
│   └── segmentation/
│       ├── 01_synthetic_image_segmentation.ipynb
│       └── 02_raw_patchless.ipynb       # Raw image, patchless segmentation
│       
│       
```

## 🚧 Status
This project is currently in progress. Notebooks will be added and updated as the work evolves.  

## 📌 Notes
- Medical image data (e.g., CT or MRI) is required for training.
- Model performance will be evaluated using standard metrics like accuracy and Dice score.
- All training and testing will be conducted using MONAI pipelines.

---

> This repository is created as part of a biomedical engineering course project.
