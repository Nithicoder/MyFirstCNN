"""PyTorch Dataset & DataLoader utilities for Fashion-MNIST."""

import torch
import torchvision
from torchvision.transforms import transforms

FASHION_CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def get_data_loaders(data_dir: str = "./data", batch_size: int = 64):
    """Downloads (if necessary) and returns train and test DataLoaders for Fashion-MNIST.

    Args:
        data_dir: Directory where the dataset will be stored.
        batch_size: Mini-batch size for training and testing.

    Returns:
        tuple of (train_loader, test_loader, train_dataset, test_dataset)
    """
    transform = transforms.ToTensor()

    train_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader, train_dataset, test_dataset
