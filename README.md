# MyFirstCNN: Fashion-MNIST Classifier with PyTorch

A lightweight, production-grade Convolutional Neural Network (CNN) built with **PyTorch** and **Torchvision** to classify clothing items from the **Fashion-MNIST** dataset.

---

## 📌 Project Overview & Objective

The objective of this project is to implement an end-to-end computer vision pipeline using PyTorch:
- Load and normalize the 10-class Fashion-MNIST dataset.
- Design a compact CNN feature extractor with spatial convolution, non-linear activation (ReLU), and max pooling.
- Train the model using the Adam optimizer with cross-entropy loss over multiple epochs.
- Evaluate the model on 10,000 unseen test samples and visualize batch predictions with truth vs. prediction status.

---

## 🧠 Complete CNN Architecture Breakdown

The model (`FashionCNN`) takes single-channel grayscale images of dimension $28 \times 28$ and maps them to class logits across 10 categories.

### Layer-by-Layer Specifications

| # | Layer / Operation | Specification / Config | Input Shape | Output Shape | Parameters | Purpose |
|---|---|---|---|---|---|---|
| **1** | `Conv2d` | `in=1, out=16, kernel=3x3, stride=1, padding=1` | `(B, 1, 28, 28)` | `(B, 16, 28, 28)` | 160 | Extracts 16 local spatial feature maps while maintaining spatial dimensions via padding |
| **2** | `ReLU` | Element-wise activation function | `(B, 16, 28, 28)` | `(B, 16, 28, 28)` | 0 | Introduces non-linearity |
| **3** | `MaxPool2d` | `kernel=2x2, stride=2` | `(B, 16, 28, 28)` | `(B, 16, 14, 14)` | 0 | Halves spatial resolution, providing translation invariance |
| **4** | `Flatten` | Flatten feature map into 1D feature vector | `(B, 16, 14, 14)` | `(B, 3136)` | 0 | Converts multi-channel 2D features into vector for classification |
| **5** | `Linear (fc1)` | `in_features=3136 (14*14*16), out_features=10` | `(B, 3136)` | `(B, 10)` | 31,370 | Computes raw unnormalized class logits for 10 classes |

**Total Trainable Parameters:** 31,530

```
Input (1x28x28) 
   │
   ▼
[Conv2d (1 -> 16, 3x3, pad=1)]  ──> (16x28x28)
   │
   ▼
[ReLU]                          ──> (16x28x28)
   │
   ▼
[MaxPool2d (2x2, stride=2)]     ──> (16x14x14)
   │
   ▼
[Flatten]                       ──> (3136)
   │
   ▼
[Linear (3136 -> 10)]           ──> (10 Logits)
```

---

## ⚙️ Hyperparameters & Training Setup

- **Framework:** PyTorch (`torch`, `torchvision`)
- **Dataset:** Fashion-MNIST (60,000 training images, 10,000 test images)
- **Input Dimensions:** Grayscale $1 \times 28 \times 28$ pixels
- **Batch Size:** 64
- **Epochs:** 5
- **Optimizer:** Adam (`torch.optim.Adam`)
- **Learning Rate:** `0.001`
- **Loss Criterion:** `nn.CrossEntropyLoss()` (combines LogSoftmax and NLLLoss)
- **Data Transforms:** `transforms.ToTensor()` (scales pixel values to $[0.0, 1.0]$)

---

## 📊 Performance & Evaluation Results

- **Dataset Size:**
  - Training: 60,000 samples (938 batches per epoch at batch size 64)
  - Testing: 10,000 unseen samples
- **Training Progression:** The cross-entropy loss steadily converges over the 5 epochs (dropping from initial values $\approx 0.60$ down to $\approx 0.20-0.30$).
- **Test Set Accuracy:** Achieves **~88% – 90%** top-1 classification accuracy on the 10,000 unseen test images with only a single convolutional layer.
- **Visual Validation:** Includes a $4 \times 4$ Matplotlib grid rendering true labels vs. model guesses, with color-coded agreement (green for correct, red for misclassification).

### Target Classes (10 classes):
`T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, `Ankle boot`

---

## 🚀 Setup & Reproduction Instructions

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Clone the Repository
```bash
git clone https://github.com/Nithicoder/MyFirstCNN.git
cd MyFirstCNN
```

### 3. Install Dependencies
```bash
pip install torch torchvision matplotlib numpy
```

*(Or with CUDA support if GPU is available: visit [pytorch.org](https://pytorch.org) for your CUDA version).*

### 4. Run Training and Evaluation
```bash
python MyFirstCNN.py
```

The script will automatically:
1. Download Fashion-MNIST into `./data/` if not already present.
2. Train the CNN for 5 epochs and stream batch loss logs.
3. Compute the overall accuracy on the 10,000 test images.
4. Display a $4 \times 4$ visualization grid comparing ground truth against model predictions.

---

## 📂 Repository Structure

```
MyFirstCNN/
├── .gitignore          # Ignores downloaded datasets, cache, and virtualenvs
├── MyFirstCNN.py       # End-to-end PyTorch training, evaluation & visualization script
└── README.md           # Project documentation and architectural overview
```
