# 🧠 Brain Tumor Detection Using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) to classify brain MRI scans into four categories:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The model achieves high accuracy and can assist in early detection of brain tumors.

---

## Dataset
The dataset used in this project is from Kaggle:  
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  

The dataset contains MRI images for training and testing, organized into the following classes:
- Training/
  - glioma
  - meningioma
  - pituitary
  - notumor
- Testing/
  - glioma
  - meningioma
  - pituitary
  - notumor

---

## Model Architecture
The CNN model consists of:
- 3 Convolutional layers with ReLU activation and MaxPooling
- Flatten layer
- Dense layer with 128 units
- Dropout layer for regularization
- Output Dense layer with 4 units (softmax)

Optimizer: `AdamW`  
Loss: `Categorical Crossentropy`  
Metrics: `Accuracy`

---

## Installation
1. Clone the repository:
```bash
git clone https://github.com/MahmoudAttia111/Brain_Tumor_Detection_CNN.git
cd Brain_Tumor_Detection_CNN
```
## Install dependencies:
pip install -r requirements.txt

## Running the Streamlit App
streamlit run task2_brain_tumor_detection_(cnn).py

## Live Demo
Try the app online here:
🧠 Brain Tumor Detection App[ https://braintumordetectioncnn- ](https://braintumordetectioncnn-8xq3y5nr94wzvyooeenzyh.streamlit.app/)
