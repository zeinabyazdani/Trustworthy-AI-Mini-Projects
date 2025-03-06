
import torchvision
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import math
    
def load_data(batch_size, val_split, path_save_data = "Generalization\data\dataset", augmentation=False):

    # Define transforms
    if augmentation:
        #The MNIST dataset is single-channel (grayscale), unlike the SVHN dataset, which is three-channel (RGB).
        # To ensure compatibility when testing, we can use torchvision.transforms.Grayscale(num_output_channels=3) to replicate the grayscale values across three channels.

        svhn_train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # Scale variations
            transforms.RandomRotation(10),  # Random rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color variations
            transforms.RandomGrayscale(p=0.5),  # Convert to grayscale with 50% probability
            transforms.GaussianBlur(kernel_size=3),  # Slight blurring
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (ImageNet)
        ])
        mnist_train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
            transforms.Resize((32, 32)),  # Resize MNIST to match SVHN
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
    else:
        svhn_train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        mnist_train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((32, 32)),  # Resize MNIST to match SVHN
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    svhn_test_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    mnist_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32)),  # Resize MNIST to match SVHN
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    svhn_train_ds = torchvision.datasets.SVHN(root= path_save_data, split= "train", transform= svhn_train_transform, download = True)
    val_size = math.floor(len(svhn_train_ds) * val_split)
    train_size = len(svhn_train_ds) - val_size
    svhn_train_ds, svhn_val_ds = random_split(svhn_train_ds, [train_size, val_size])

    mnist_train_ds = torchvision.datasets.MNIST(root= path_save_data, train=True, transform= mnist_train_transform, download = True)
    val_size = math.floor(len(mnist_train_ds) * val_split)
    train_size = len(mnist_train_ds) - val_size
    mnist_train_ds, mnist_val_ds = random_split(mnist_train_ds, [train_size, val_size])


    svhn_test_ds  = torchvision.datasets.SVHN(root= path_save_data, split= "test",  transform= svhn_test_transform, download = True)
    mnist_test_ds = torchvision.datasets.MNIST(root= path_save_data, train=False,   transform= mnist_test_transform, download = True)
    print(f"svhn_train_ds: {len(svhn_train_ds)}, \nmnist_train_ds: {len(mnist_train_ds)}, \nsvhn_test_ds: {len(svhn_test_ds)}, \nmnist_test_ds: {len(mnist_test_ds)}")

    svhn_train_loader = DataLoader(svhn_train_ds, batch_size=batch_size, shuffle=True)
    mnist_train_loader   = DataLoader(mnist_train_ds, batch_size=batch_size, shuffle=True)
    svhn_val_loader = DataLoader(svhn_val_ds, batch_size=batch_size, shuffle=True)
    mnist_val_loader   = DataLoader(mnist_val_ds, batch_size=batch_size, shuffle=True)
    svhn_test_loader  = DataLoader(svhn_test_ds, batch_size=batch_size, shuffle=True)
    mnist_test_loader = DataLoader(mnist_test_ds, batch_size=batch_size, shuffle=True)
    

    return svhn_train_loader, mnist_train_loader, svhn_val_loader, mnist_val_loader, svhn_test_loader, mnist_test_loader
