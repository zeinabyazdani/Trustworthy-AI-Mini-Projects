## Implementation of ResNet-18 and Generalization Analysis Across Datasets

### Steps:
1. **ResNet-18 Implementation**: The ResNet-18 architecture will be implemented manually using PyTorch without relying on prebuilt models.
2. **Dataset Preparation**:
   - The SVHN dataset (colored street view house numbers) will be loaded and split into training and test sets.
   - The MNIST dataset (black-and-white handwritten digits) will be loaded for additional testing, ensuring compatibility with the trained model.
3. **Training the Model**: The model will be trained on the SVHN training set.
4. **Evaluation**:
   - The trained model will be evaluated on the SVHN test set.
   - The model will also be tested on the MNIST test set, requiring conversion of MNIST images to 3-channel format.
5. **Results and Discussion**: The classification accuracy on both datasets will be reported, and the challenges in cross-dataset generalization will be analyzed.

### Handling Single-Channel MNIST Data
Since the MNIST dataset contains grayscale images (1 channel) and the ResNet-18 model expects 3-channel inputs, we will convert MNIST images to 3 channels by duplicating the single-channel data across three channels.


## Generalization Strategies
1. **Dropout and Batch Normalization**: Investigate the impact of removing Batch Normalization (BN) from ResNet-18.
   - Dropout randomly deactivates neurons during training to prevent over-reliance on specific features, reducing overfitting and improving generalization.
   - Batch Normalization normalizes the mean and variance of each layer’s inputs, reducing internal covariate shift, leading to faster convergence, more stable training, and better generalization.
2. **Loss function**
   - Utilize Label Smoothing Cross Entropy (adding Label Smoothing Regularization to standard Cross Entropy).
     Label smoothing replaces one-hot encoded label vector y_hot with a mixture of y_hot and the uniform distribution: y_ls = (1 - α) * y_hot + α / K

     where K is the number of label classes, and α is a hyperparameter that determines the amount of smoothing. If α = 0, we obtain the original one-hot encoded y_hot. If α = 1, we get the uniform distribution.
3. **Data augmentation**
   - Apply techniques that increase data diversity and expose the model to a larger variety of samples, thereby improving generalization.

      The main challenge is that SVHN images are colorful and complex, while MNIST is grayscale and much simpler. So, choosing the right Data Augmentation can help the model generalize better and become less sensitive to color differences and background details.
      - **Recommended Data Augmentation for this Task:**
         - Use ColorJitter to Reduce Color Dependence: Brightness and Contrast help the model become robust to different lighting conditions.
      Saturation is applied slightly to reduce reliance on color. Hue is set to a low value to avoid drastic color shifts.
         - Adding Noise & Geometric Transformations: Since MNIST consists of handwritten digits that vary in shape and size, the following augmentations will help.
         - Random Rotation (±10°) to account for slight tilts in digits.
         - RandomResizedCrop to introduce scale and position variations.
         - GaussianBlur to reduce dependency on sharp details.
         - Convert images to grayscale with a 50% probability.
4. Optimizer
   - Use another optimizer like Adam.
5. **Feature extraction**
   - Use pre-trained models, especially those trained on large datasets like ImageNet, which have learned rich feature representations from diverse data.
   - Utilize a **pre-trained ResNet-18 model** trained on ImageNet (`torchvision.models` can be used with `pretrained=True`, or pre-trained weights can be manually loaded into the custom model).


## Final Results (Generalization Performance)
- **Model:** The best combination from the above settings to evaluate cross-dataset generalization.
1. Train the model on MNIST and evaluate its generalization by testing on both MNIST and SVHN.
2. Train the model on MNIST, fine-tune the classifier layer (final classification head) on SVHN, and assess generalization by testing on both MNIST and SVHN.
