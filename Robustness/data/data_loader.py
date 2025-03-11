import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import math


def load_data(batch_size=64, val_split=0.2, data_path='.'):

    # Define transforms
    transform = transforms.compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.229, 0.224, 0.225]),
    ])

    # Load CIFAR-10 dataset
    train_ds = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=transform, download=True)
    test_ds  = torchvision.datasets.CIFAR10(root=data_path, train=False, transform=transform,download=True)

    # Split train and validation datasets
    indices = np.arange(len(train_ds))
    np.random.shuffle(indices)
    val_size = math.floor(len(train_ds) * val_split)
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    train_sampler = data.SubsetRandomSampler(train_idx)
    val_sampler   = data.SubsetRandomSampler(val_idx)

    # Create data loader
    train_loader = DataLoader(train_ds, batch_size, sampler=train_sampler)
    val_loader   = DataLoader(train_ds, batch_size, sampler=val_sampler)
    test_loader  = DataLoader(test_ds, batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

