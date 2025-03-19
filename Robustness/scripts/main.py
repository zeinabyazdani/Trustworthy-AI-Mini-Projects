import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import losses
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import yaml


# Find Project directory
try:
    # If running in a `.py` script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir    = os.path.abspath(os.path.join(current_dir, os.pardir))
except NameError:
    # If running in a `.ipynb` file
    base_dir    = os.getcwd()
print("Base Directory:", base_dir)


# Append necessary directories to system path
sys.path.append(os.path.join(base_dir, "data"))
sys.path.append(os.path.join(base_dir, "scripts"))
sys.path.append(os.path.join(base_dir, "config"))

# Import custom modules
from data_loader import load_data, generate_adversary_samples
from train import training
from evaluation import knn_accuracy, plot_umap


# Load configuration settings from yaml file
with open(os.path.join(base_dir, "config", "config.yaml"), 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

# Set hyperparameters
learning_rate = config["learning_rate"]
epochs     = config["epochs"]
batch_size = config["batch_size"] 
val_split  = config["val_split"]
num_classes= config["num_classes"]     # Number of classes (for CrossEntropyLoss)
embedding_size = config["embedding_size"] # Size of the embedding (for Circle Loss)
model_name = config["model_name"]
# Pathes to save data and trained model
model_path    = os.path.join(base_dir, 'saved_models')
path_save_data= os.path.join(base_dir, 'data', 'dataset')
# Seting for check robustness
epsilon = config["epsilon"]
adv_tr  = config["adversary_training"]
use_circle_loss = config["use_circle_loss"]

# Check and set device
print("cuda available?", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, val_loader, test_loader = load_data(batch_size, val_split, path_save_data)


### Define model and loss function ###
# Load pretrained ResNet18 model
model = models.resnet18(pretrained=True)
if use_circle_loss:
    # Replace the final fc layer with a new one for embeddings
    model.fc = nn.Linear(512, embedding_size)
    # Define Circle Loss
    loss_fn = losses.CircleLoss()
else:
    # Replace the final fc layer with a new one for classification
    model.fc = nn.Linear(512, num_classes)
    # Define CrossEntropyLoss
    loss_fn = nn.CrossEntropyLoss()


### Training ###
# Define optimiizer
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

# Training
training(model, device, train_loader, val_loader, 
         loss_fn, optimizer, scheduler, epochs, 
         generate_adversary_samples if adv_tr else None, 
         model_path, model_name=model_name, plot_loss=True)


### Generate adversary test loader ###
# List to store noisy images and labels
adv_images = []
adv_labels = []
# Generate noisy images
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    # Generate adversarial samples
    adversary_images = generate_adversary_samples(model, images, labels, loss_fn, epsilon=0.1, manual=False)
    adv_images.append(adversary_images)
    adv_labels.append(labels)
# Concatenate all tensors along the batch dimension
adv_images = torch.cat(adv_images, dim=0)
adv_labels = torch.cat(adv_labels, dim=0)
# Create a new DataLoader with noisy images
test_loader_adv = DataLoader(TensorDataset(adv_images, adv_labels), batch_size=test_loader.batch_size, shuffle=False)


### Evaluate the model on test data ###
print('Model Performance on test set')
knn_accuracy(model, test_loader)
plot_umap(model, test_loader)
print('Model Performance on adversary test set')
knn_accuracy(model, test_loader)
plot_umap(model, test_loader_adv)
