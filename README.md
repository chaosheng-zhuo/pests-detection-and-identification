# **Pests Detection and Identification**

This repository contains our COMP9517 group project for pest detection and identification based on the **AgroPest-12** dataset.
The project includes traditional machine learning baselines, YOLO-based detection, a two-stage YOLO+ResNet pipeline, exploratory data analysis (EDA), and explainability (XAI) using Captum.

------

## **📁 Repository Structure**

```
/
├── Traditional machine learning/
│     └── machine_learning.ipynb           # HOG/LBP/SIFT + ML baselines
│
├── eda_outputs/                           # Dataset EDA figures + summary
│     ├── class_dist.png
│     ├── class_samples.png
│     ├── objects_hist.png
│     ├── quality.png                      # Brightness & blur distribution
│     ├── samples.png
│     ├── sizes.png
│     └── report.md                        # EDA explanation
│
├── yolo_resnet/                           # Detection & Two-Stage Pipeline
│     ├── crop_from_yolo_labels.py         # Crop training data from YOLO boxes
│     ├── train_detect_yolo.py             # Train YOLO detector
│     ├── train_resnet_classifier.py       # Train ResNet classifier
│     ├── infer_two_stage.py               # Two-stage YOLO+ResNet inference
│     ├── yolo+resnet.ipynb                # End-to-end pipeline notebook
│     └── plot.ipynb                       # Training curves & metrics plotting
│
├── xAi/                                   # Explainability (XAI)
│     ├── xAi.ipynb                           # Saliency + Integrated Gradients
│     └── README.md                        # Module-specific instructions
│
├── 数据分析.ipynb
│
│
└── (root-level README — this file)
```

------

## **📌 Project Overview**

### **Goal**

Perform **pest detection** and **fine-grained classification** using images collected in natural farmland environments.

### **Methods**

This project implements three main approaches:

1. **Traditional ML baseline**
   - Features: HOG / LBP / SIFT-BoW
   - Classifiers: SVM / RandomForest / KNN
   - Sliding-window detection
2. **One-stage deep learning**
   - YOLO-based detection + classification
   - Evaluate mAP @ 0.5 and per-class AP
3. **Two-stage YOLO + ResNet**
   - Stage 1: YOLO detector for bounding boxes
   - Stage 2: ResNet-50 classifier on cropped insects
   - Evaluate overall accuracy & confusion matrix
4. **Explainability (XAI)**
   - Captum Saliency
   - Integrated Gradients
   - Visualization of correct & misclassified cases

------

## **📊 Exploratory Data Analysis (EDA)**

The `eda_outputs/` directory contains:

- Class distribution
- Object-per-image histogram
- Brightness & blur distribution
- Sample images with bounding boxes
- Size / aspect ratio analysis
- Short EDA report (`report.md`)

These help identify dataset imbalance, image quality variations, and detection difficulty.

------

## **🚀 How to Run**

### **1. Install dependencies**

Please use the dependencies listed at the top of each script.

------

### **2. Train YOLO Detector**

```
python yolo_resnet/train_detect_yolo.py --data data.yaml --epochs <num>
```

------

### **3. Crop Classification Dataset**

```
python yolo_resnet/crop_from_yolo_labels.py
```

This creates:

```
cls_data/train/<class>/*.jpg
cls_data/valid/<class>/*.jpg
```

------

### **4. Train ResNet50 Classifier**

```
python yolo_resnet/train_resnet_classifier.py
```

------

### **5. Two-Stage Inference**

```
python yolo_resnet/infer_two_stage.py
```

------

### **6. Run Explainability**

```
python xAi/xAi.ipynb
```

Outputs will be saved to:

```
xai_resnet/correct/
xai_resnet/wrong/
```


