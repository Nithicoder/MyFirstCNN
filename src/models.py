"""Convolutional Neural Network architectures for Fashion-MNIST classification."""

import torch
import torch.nn as nn


class FashionCNN(nn.Module):
    """Baseline Single-Layer CNN Architecture.

    Structure:
        - Conv2d: 1 -> 16 channels, 3x3 kernel, pad=1
        - ReLU activation
        - MaxPool2d: 2x2 kernel, stride=2 (28x28 -> 14x14)
        - Flatten
        - Linear: 3136 -> 10 classes
    Total Parameters: 31,530
    """

    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(14 * 14 * 16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


class DeepFashionCNN(nn.Module):
    """Improved Two-Layer Deep CNN Architecture.

    Structure:
        - Conv Block 1: 1 -> 32 channels, 3x3, pad=1 + ReLU + MaxPool2d (28x28 -> 14x14)
        - Conv Block 2: 32 -> 64 channels, 3x3, pad=1 + ReLU + MaxPool2d (14x14 -> 7x7)
        - Flatten (64 * 7 * 7 = 3136)
        - Linear Hidden: 3136 -> 128 + ReLU
        - Linear Output: 128 -> 10 classes
    Total Parameters: 421,642
    """

    def __init__(self):
        super(DeepFashionCNN, self).__init__()

        # Conv Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        # Classifier Head
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
