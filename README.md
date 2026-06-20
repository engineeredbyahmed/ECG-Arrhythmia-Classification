# ECG Arrhythmia Classification Using Deep Learning

## Overview

This project develops a deep learning model for automated ECG arrhythmia classification using the MIT-BIH Arrhythmia Dataset.

The objective is to classify heartbeats into five arrhythmia categories and investigate the challenges of class imbalance in medical datasets. The project includes data preprocessing, imbalance handling, neural network training, model evaluation, and performance analysis.

This work was completed as part of my MSc Data Science studies and combines my background in Biomedical Engineering with Machine Learning.

---

## Dataset

Dataset: MIT-BIH Arrhythmia Database

The dataset contains ECG heartbeat recordings represented by 187 numerical features and one target label.

### Classes

| Class | Description |
|---------|-------------|
| 0 | Normal Beat |
| 1 | Supraventricular Premature Beat |
| 2 | Premature Ventricular Contraction |
| 3 | Fusion Beat |
| 4 | Unclassifiable Beat |

Due to licensing and file size considerations, the dataset is not included in this repository.

---

## Project Pipeline

### 1. Data Preprocessing

- Data cleaning
- Feature standardization using StandardScaler
- Stratified train-validation splitting
- Exploratory analysis of class distributions

### 2. Handling Class Imbalance

Medical datasets are often highly imbalanced.

Techniques explored:

- SMOTE (Synthetic Minority Oversampling Technique)
- Random Under-Sampling
- Class Weights

### 3. Model Development

A Multi-Layer Perceptron (MLP) neural network was developed using TensorFlow/Keras.

#### Best Architecture

```text
Input Layer (187 features)
↓
Dense(256) + ReLU
↓
Dense(128) + ReLU
↓
Dense(64) + ReLU
↓
Dense(32) + ReLU
↓
Output Layer (5 Classes)
```

Additional techniques:

- Dropout Regularization
- Early Stopping
- Adam Optimizer
- Learning Rate = 0.001

---

## Results

### Best Performance

| Metric | Score |
|----------|---------|
| Accuracy | 95% |
| Macro F1 Score | 80% |

### Key Findings

- High overall classification accuracy.
- Strong performance on majority classes.
- Significant improvement in minority-class recall through SMOTE.
- Fusion Beat (Class 3) remained challenging due to severe class imbalance and feature overlap.
- Model achieved high recall for rare arrhythmias but generated additional false positives.

---

## Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

Example outputs:

- Training and Validation Loss Curves
- Confusion Matrix
- Class-wise Performance Analysis

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- TensorFlow / Keras
- Imbalanced-Learn

---

## Repository Structure

```text
ECG-Arrhythmia-Classification/
│
├── ECG_classification.ipynb
├── README.md
├── requirements.txt
├── figures/
│   ├── confusion_matrix.png
│   ├── loss_curves.png
│   ├── PCA.png
│   ├── System Pipeline.png
│   └── class_distribution.png
│
└── results/
    ├── ANN_model.h5
    └── classification_report.txt
```

---

## Future Improvements

Potential future work includes:

- Convolutional Neural Networks (CNNs)
- Transformer-based architectures
- Cost-sensitive learning
- Explainable AI (SHAP/LIME)
- Deployment as a clinical decision-support system

---

## Author

Ahmed Al Dulaim

MSc Data Science, University of Exeter

BSc Biomedical Engineering

Interested in Machine Learning, Healthcare AI, Computer Vision, and Medical Data Analytics.
