# Adversarial Training with Circle Loss and CIFAR-10

## Overview
This project explores adversarial attacks and defenses using the CIFAR-10 dataset and ResNet-18. It utilizes the Circle Loss function, which improves optimization flexibility by balancing intra-class and inter-class similarities, leading to enhanced performance over traditional methods.

## Objectives
1. Understand the concept of adversarial attacks.
2. Enhance CIFAR-10 dataset robustness using adversarial training with Circle Loss.
3. Utilize a pre-trained ResNet-18 model for representation.
4. Evaluate the model performance under various training and testing scenarios.

## Dataset
The CIFAR-10 dataset consists of 60,000 color images (32×32) in 10 classes, with each class containing 6,000 images. The dataset is split as follows:
- 20% for training
- 80% for validation


## Implementation Steps
### 1. Load Dataset and Model
- Load CIFAR-10 dataset.
- Load the pre-trained ResNet-18 model.

### 2. Generate Adversarial Examples
- Provide a brief explanation of PGD (Projected Gradient Descent) attacks.
- Generate adversarial examples using FGSM (Fast Gradient Sign Method) with epsilon = 0.1.
- Modify a subset of image pixels randomly and visualize some adversarial samples.

### 3. Model Training and Evaluation
#### **Scenario A: Training with Original Data (Cross-Entropy Loss)**
- Train the model using the original dataset and cross-entropy loss.
- Evaluate model performance on:
  - Original test dataset
  - Adversarial test dataset

#### **Scenario B: Training with Augmented Data (Cross-Entropy Loss)**
- Train the model with an augmented dataset where each sample has a 50% chance of being perturbed.
- Evaluate model performance on:
  - Original test dataset
  - Adversarial test dataset

### 4. Circle Loss Implementation
- Provide a brief explanation of Circle Loss and its advantages over traditional loss functions.
- Train the model using Circle Loss.
- Evaluate model performance on:
  - Original test dataset
  - Adversarial test dataset

### 5. Results and Analysis
- Compare and analyze the results obtained from Scenarios A, B, and Circle Loss training.
- Report accuracy on test datasets and plot the loss curves.
- Visualize 512-dimensional representations in 2D using UMAP.

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- UMAP-learn

## Author
Zeinab Yazdani