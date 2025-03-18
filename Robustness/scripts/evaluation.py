
import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variable to store extracted features
features_list = []


# Hook function to extract output from the avgpool layer
def hook_fn(module, input, output):
    global features_list
    features_list.append(output.flatten(1).cpu().numpy())


# Register a forward hook on the avgpool layer
def register_hook(model):
    layer = model.avgpool  # Get the avgpool layer
    handle = layer.register_forward_hook(hook_fn)
    return handle  # Return handle for later removal


# Extract features from the model
def extract_features(model, dataloader):
    global features_list
    features_list = []  # Clear previous feature list
    hook_handle = register_hook(model)  # Register the hook
    
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _ = model(images)  # Forward pass, hook captures avgpool output
    
    hook_handle.remove()  # Remove the hook after extraction
    return np.vstack(features_list)  # Convert list to NumPy array


# Evaluate model using KNN
def knn_accuracy(model, test_loader):
    features = extract_features(model, test_loader)
    labels = np.array([label for _, label in test_loader.dataset])  # Get labels
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features, labels)
    accuracy = knn.score(features, labels)
    
    return accuracy


# Visualize feature distribution using UMAP
def plot_umap(model, test_loader):
    features = extract_features(model, test_loader)
    labels = np.array([label for _, label in test_loader.dataset])
    
    umap_proj = umap.UMAP(n_components=2).fit_transform(features)
    plt.scatter(umap_proj[:, 0], umap_proj[:, 1], c=labels, cmap='Spectral', alpha=0.5)
    plt.title("UMAP Projection of Test Data")
    plt.show()
