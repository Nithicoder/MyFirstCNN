# -*- coding: utf-8 -*-
"""01_baseline_1layer_cnn.py

Baseline: Single-Layer Convolutional Neural Network (CNN) trained on Fashion-MNIST.
Achieves ~88-89% accuracy with 31,530 parameters.
"""

import torch
import torchvision
from torchvision.transforms import transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 1. Dataset & DataLoader Setup
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

test_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    shuffle=False,
    batch_size=64
)

print(f"Pipeline active. Total training images loaded: {len(train_dataset)}")


# 2. Baseline Architecture Definition (1 Conv Layer)
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        # Conv layer: 1 input channel (grayscale) -> 16 feature maps (3x3 kernel, pad=1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # MaxPool layer: 2x2 kernel, stride=2 -> 28x28 down to 14x14
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # Fully connected layer: 16 channels * 14 * 14 = 3136 features -> 10 output classes
        self.fc1 = nn.Linear(14 * 14 * 16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


model = FashionCNN()
print(model)

# 3. Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 300 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# 5. Evaluation on Unseen Test Dataset
model.eval()
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

accuracy = 100 * correct_predictions / total_predictions
print(f'Final Model Accuracy on 10,000 unseen test images: {accuracy:.2f}%')

# 6. Sample Prediction Visualization
dataiter = iter(test_loader)
images, labels = next(dataiter)

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.ravel()

classes = (
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
)

for i in range(16):
    img = images[i].numpy()
    img = np.transpose(img, (1, 2, 0))

    true_label = classes[labels[i].item()]
    guess_label = classes[predicted[i].item()]

    text_color = 'green' if true_label == guess_label else 'red'

    axes[i].imshow(img.squeeze(), cmap='gray')
    axes[i].set_title(f"True: {true_label}\nGuess: {guess_label}", color=text_color, fontsize=10)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.5)
plt.show()
