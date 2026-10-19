# -*- coding: utf-8 -*-
"""02_improved_2layer_cnn.py

Two-Layer Deep Convolutional Neural Network (CNN) trained on Fashion-MNIST.
Incorporates dual convolution-pooling blocks and an intermediate dense hidden layer.
Achieves ~91.5-92.5% accuracy with 421,642 parameters.
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
print(f"Total test images loaded: {len(test_dataset)}")


# 2. Two-Layer Deep CNN Architecture Definition
class DeepFashionCNN(nn.Module):
    def __init__(self):
        super(DeepFashionCNN, self).__init__()

        # Conv Block 1: 1 -> 32 channels (3x3 kernel, pad=1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14

        # Conv Block 2: 32 -> 64 channels (3x3 kernel, pad=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7

        self.flatten = nn.Flatten()

        # Fully Connected Classifier: 7 * 7 * 64 = 3136 -> 128 -> 10
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


deep_model = DeepFashionCNN()
print(deep_model)
total_params = sum(p.numel() for p in deep_model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params:,}")


# 3. Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(deep_model.parameters(), lr=0.001)


# 4. Training Loop
epochs = 5
for epoch in range(epochs):
    deep_model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        outputs = deep_model(images)
        loss = criterion(outputs, labels)

        deep_model.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 300 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


# 5. Evaluation on Unseen Test Dataset
deep_model.eval()
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = deep_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

accuracy = 100 * correct_predictions / total_predictions
print(f'Deep CNN Accuracy on 10,000 unseen test images: {accuracy:.2f}%')


# 6. Checkpoint Export
file_path = 'deep_fashion_cnn.pth'
torch.save(deep_model.state_dict(), file_path)
print(f"Success: {total_params:,} parameters saved to {file_path}")


# 7. Sample Prediction Visualization
dataiter = iter(test_loader)
images, labels = next(dataiter)

with torch.no_grad():
    outputs = deep_model(images)
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
