# src/dataset.py
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32):
    """
    Creates PyTorch DataLoaders for training and validation.
    Expects data_dir to have 'train' and 'val' subfolders.
    """
    # ResNet standard transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(), # Data augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], # Standard ResNet normalization
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=2)
        for x in ['train', 'val']
    }

    class_names = image_datasets['train'].classes
    return dataloaders, class_names

if __name__ == "__main__":
    print("Dataset module ready. Waiting for actual data to test!")