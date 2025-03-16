import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import losses
import os
import sys
import yaml


# Find Project directory
try:
    # If running in a `.py` script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir    = os.path.join(current_dir, os.pardir)
except NameError:
    # If running in a `.ipynb` file
    base_dir    = os.getcwd()
print("Base Directory:", base_dir)

# Append necessary directories to system path
sys.path.append(os.path.join(base_dir, "data"))
sys.path.append(os.path.join(base_dir, "config"))

# Import custom modules
from data_loader import load_data

# Load configuration settings from yaml file
with open(os.path.join(base_dir, "config", "config.yaml"), 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

# Set hyperparameters
batch_size = config["batch_size"] 
val_split  = config["val_split"]
data_path  = config["data_path"]
learning_rate = config["learning_rate"]
epsilon       = config["epsilon"]
loss_func     = config["loss_func"]

# Check and set device
print("cuda available?", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, val_loader, test_loader = load_data(batch_size, val_split, data_path)

# Define model
model = models.resnet18(pretrained=True)

# Define loss function
if loss_func == "circle":
    loss_fn = losses.CircleLoss()
else:
    loss_fn = nn.CrossEntropyLoss()

# Define optimiizer
optimizer = optim.Adam(model.parameters, lr=learning_rate)

# Training

# # Evaluate the model on test data
# print('Model Performance on test set')
# print(eval(model, device, test_loader).item())
