import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
import torch
import yaml
import os
import sys


try:
    # If running in a `.py` script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
except NameError:
    # If running in Google Colab
    base_dir = os.getcwd()

print("Base Directory:", base_dir)


# Append necessary directories to system path
sys.path.append(os.path.join(base_dir, 'data'))  # Append data directory to system path
sys.path.append(os.path.join(base_dir, 'model'))  # Append models directory to system path
sys.path.append(os.path.join(base_dir, 'scripts'))  # Append scripts directory to system path
sys.path.append(os.path.join(base_dir, 'config'))  # Append config directory to system path
sys.path.append(os.path.join(base_dir, 'utils'))  # Append utils directory to system path


# Import custom modules
from data_loader import load_data  # Custom data loader
from resnet18 import ResNet18 # Custom modele
from train import training  # Custom training function
from evaluation import eval  # Custom evaluation function
from losses import LabelSmoothingCrossEntropy  # Custom layer visualization function
from visualization import data_visualization
# Set path to save data and trained models


# Load configuration settings from yaml file
with open(os.path.join(base_dir, 'config', 'config.yaml'), 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

# Set hyperparameters
learning_rate = config['learning_rate']
momentum      = config['momentum']
epochs        = config['epochs']
batch_size    = config['batch_size']
val_split     = config['val_split']
step_size     = config['step_size']
gamma         = config['gamma']
loss_function = config['loss_function']
optimizer_    = config['optimizer_']
pre_train     = config['pre_train']
fine_tune     = config['fine_tune']
augmentation  = config['augmentation']
use_batch_norm= config['use_batch_norm']
train_dataset = config['train_dataset']
model_name    = config['model_name']
# Pathes to save data and trained model
model_path    = os.path.join(base_dir, 'model', 'saved_models')
path_save_data= os.path.join(base_dir, 'data', 'dataset')
# Random seed
SEED = config['SEED']
np.random.seed(SEED)
torch.manual_seed(SEED)


# Check and set device
print("cuda available?", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load data
if train_dataset.lower() == 'mnist':
    _, train_loader, _, val_loader, svhn_test_loader, mnist_test_loader = load_data(batch_size, 0.1, path_save_data, augmentation=augmentation)
else:
    train_loader, _, val_loader, _, svhn_test_loader, mnist_test_loader = load_data(batch_size, 0.1, path_save_data, augmentation=augmentation)


# Plot sample image for each dataset
print("Plot sample image from SVHN dataset")
sample = next(iter(svhn_test_loader))
data_visualization(batch_data=sample, figsize = (3,3))
print("Plot sample image from MNIST dataset")
sample = next(iter(mnist_test_loader))
data_visualization(batch_data=sample, figsize = (3,3))


# Define model
if pre_train:
    # Load the ResNet-18 model with pre-trained weights on ImageNet
    model = models.resnet18(pretrained=True)
    # Modify the fully connected layer (classifier) for fine-tuning
    model.fc = nn.Linear(model.fc.in_features, 10)  # Change output to 10 classes
else:
    # Load custom ResNet-18 model
    model = ResNet18(in_channels=3, n_classes=10, use_bn=use_batch_norm)
    if fine_tune:
        # Load the full model (since it was saved using torch.save(model, PATH))
        pretrained_model = torch.load(os.path.join(model_path, "best_model.pth"))
        # Extract the state dictionary
        pretrained_weights = pretrained_model.state_dict()
        # Load into the new model
        model.load_state_dict(pretrained_weights)

        # Freeze all convolutional layers (keep their weights unchanged)
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False



# Define loss function
if loss_function == 'LSCE':
    loss_func = LabelSmoothingCrossEntropy(label_smoothing=0.1)
else:
    loss_func = nn.CrossEntropyLoss()

# Define optimizer and learning rate scheduler
if optimizer_.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
else:
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma)


# Training
training(model, device, train_loader, val_loader, loss_func, optimizer, scheduler, epochs, model_path, model_name=model_name, plot_loss=True)

# Evaluate the model on test data
print('Model Performance on SVHN test set')
print(eval(model, device, svhn_test_loader).item())
print('Model Performance on MNIST test set')
print(eval(model, device, mnist_test_loader).item())
