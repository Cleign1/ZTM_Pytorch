"""
Contains functionality for setting up the data for Pytorch DataLoader
"""
import os

import torchvision.transforms.v2 as trans
from torchvision import datasets
from torch.utils.data import DataLoader

num_workers = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: trans.Compose,
    batch_size: int,
    num_workers: int=num_workers, ):
    """
    Creates training and validation DataLoader objects.
    Args:
        train_dir (str): Directory path to the training dataset.
        test_dir (str): Directory path to the validation dataset.
        transform (trans.Compose): Transformations to apply to the dataset.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
    Returns:
        tuple: A tuple containing:
            - DataLoader: DataLoader object for the training dataset.
            - DataLoader: DataLoader object for the validation dataset.
    """
    # pake ImageFolder to create datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    
    # Get class name
    class_names = train_data.classes
    
    # Turn images into DataLoader
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_dataloader, test_dataloader, class_names
