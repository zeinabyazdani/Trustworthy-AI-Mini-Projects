import torchvision
import torchvision.transforms as transforms


def load_data(batch_size, val_split, data_path, augmentation):

    # Define transforms
    if augmentation:
        train_transform = transforms.compose([
            transforms.ToTensor()
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.229, 0.224, 0.225]),
        ])

    