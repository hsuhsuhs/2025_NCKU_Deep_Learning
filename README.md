# 2025 NCKU Deep Learning

This repository contains coursework implementations for the **Deep Learning** course at National Cheng Kung University (NCKU).  
It includes two main assignments:

- [**DL_hw1**](https://github.com/hsuhsuhs/2025_NCKU_Deep_Learning/tree/main/DL_hw1): Image Classification using MLP and CNN 
- [**DL_hw2**](https://github.com/hsuhsuhs/2025_NCKU_Deep_Learning/tree/main/DL_hw2): Industrial Surface Defect Detection and Segmentation

---

---

# 🔹 DL_hw1 — Image Classification

##  Task Overview

The objective of HW1 is to perform **multi-class image classification** on an industrial dataset with **10 defect categories**, including:

- hole
- blackscratch_long
- whitescratch
- fold
- dent
- bulge
- dirt
- insect
- edge
- weldpoint :contentReference[oaicite:0]{index=0}

Two models are implemented:

- **MLP (Multi-Layer Perceptron)**
- **VGG16 (CNN-based model)**

---

##  Model Design

### 1. MLP
- Fully connected neural network
- Flexible number of hidden layers
- Used as a baseline model

### 2. VGG16
- Deep convolutional neural network
- Pretrained or trained from scratch
- Strong performance for image classification

---

##  Pipeline

The overall workflow follows:
```bach
Data Preparation → Model Design → Loss Function & Optimizer → Training → Evaluation
```

This pipeline is explicitly described in the assignment instructions :contentReference[oaicite:1]{index=1}.

---

##  Training Details

Typical components include:

- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam / SGD
- **Metrics**:
  - Accuracy
  - Precision
  - Recall

---

##  Results

- Training and testing accuracy curves are generated
- Performance comparison between MLP and VGG16
- VGG16 generally outperforms MLP due to spatial feature extraction



---

# 🔹 DL_hw2 — Defect Detection & Segmentation

##  Task Overview

This assignment focuses on **industrial surface defect analysis**, including:

1. **Object Detection** 
2. **Semantic Segmentation** 

Dataset:
- Kolektor Surface-Defect Dataset 
- Real-world industrial images
- Includes both defective and non-defective samples

---

##  Dataset Description

- 356 images with defects
- 2979 images without defects
- Image size ≈ 230 × 630
- Various defect types (scratches, spots, imperfections) 

---

##  Part 1 — Object Detection

### Goal
Detect and localize defects using bounding boxes.

### Supported Models

- YOLO (v5, v8, v11)
- SSD
- RetinaNet
- Faster R-CNN

### Evaluation Metric

- **mAP (Mean Average Precision)**

---

##  Part 2 — Segmentation

### Goal
Classify each pixel as:

- Defect
- Background


### Evaluation Metric

- **Dice Coefficient**

---

##  Implementation Details

### Data Processing
- Image resizing
- Normalization
- Data augmentation

### Training Setup
- Batch size
- Learning rate
- Optimizer
- Epochs

---

##  Experimental Results

The report includes:

- Detection performance (mAP@0.5)
- Segmentation performance (Dice score)
- Precision-Recall curves
- Failure case analysis

---

##  Failure Analysis

Typical failure cases include:

- False positives (noise detected as defect)
- False negatives (small defects missed)

Possible reasons:

- Imbalanced dataset
- Small defect size
- Complex background

---

##  GUI System

A GUI is implemented for demonstration:

- Input image
- Detection results (bounding boxes)
- Segmentation mask overlay :contentReference[oaicite:6]{index=6}

---

# ⚠️ Notes

- Large model files (`.pth`, `.pt`) are excluded from this repository
- Output folders are ignored to keep the repo lightweight

---

