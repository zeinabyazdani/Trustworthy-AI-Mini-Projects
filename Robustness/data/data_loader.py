import torchvision
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import math


def load_data(batch_size=64, val_split=0.8, data_path='.', augmentation=False):

    # Define transforms
    if augmentation:
        train_transform = transforms.compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.229, 0.224, 0.225]),
        ])
    test_transform = transforms.compose([
            transforms.ToTensor(),
    ])

    train_ds = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=train_transform, download=True)
    test_ds  =torchvision.datasets.CIFAR10(root=data_path, train=False, transform=test_transform,download=True)
    val_size = math.floor(len(train_ds) * val_split)
    tr_size  = len(train_ds) * val_size
    train_ds, val_ds = random_split(train_ds, [tr_size, val_size])
    print(f"train_ds: {len(train_ds)}, \nsvhn_test_ds: {len(test_ds)}, \nvalidation_ds: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size, shuffle=True)

    return train_loader, val_loader, test_loader
