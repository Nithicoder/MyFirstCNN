"""CLI Training and Evaluation Script for Fashion-MNIST CNNs.

Usage:
    python train.py --model baseline --epochs 5 --batch-size 64
    python train.py --model deep --epochs 5 --batch-size 64 --save-path deep_fashion_cnn.pth
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_data_loaders, FASHION_CLASSES
from src.models import FashionCNN, DeepFashionCNN


def train_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 300 == 0:
            print(
                f"Epoch [{epoch + 1}/{total_epochs}], "
                f"Batch [{batch_idx + 1}/{len(loader)}], "
                f"Loss: {loss.item():.4f}"
            )
    return running_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Train CNN on Fashion-MNIST")
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "deep"],
        help="Model architecture to train (baseline or deep)",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory for dataset")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save weights (.pth)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, train_ds, test_ds = get_data_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size
    )
    print(f"Loaded {len(train_ds):,} training samples and {len(test_ds):,} test samples.")

    if args.model == "baseline":
        model = FashionCNN().to(device)
    else:
        model = DeepFashionCNN().to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Instantiated '{args.model}' architecture with {total_params:,} trainable parameters.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        test_acc = evaluate(model, test_loader, device)
        print(f"--> Epoch {epoch + 1} Complete. Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%")

    final_acc = evaluate(model, test_loader, device)
    print(f"\nFinal Test Accuracy on 10,000 unseen samples: {final_acc:.2f}%")

    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"Model checkpoint successfully saved to {args.save_path}")


if __name__ == "__main__":
    main()
