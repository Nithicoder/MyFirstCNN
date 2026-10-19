# Fashion-MNIST CNN Benchmark: Baseline vs. Deep Two-Layer CNN

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean, modular deep learning project evaluating single-layer vs. two-layer Convolutional Neural Network (CNN) architectures on the **Fashion-MNIST** dataset using **PyTorch**.

---

## 📌 Project Overview

This repository demonstrates iterative computer vision engineering structured around a clean, modular Python codebase:
- **`src/models.py`**: Clean, reusable PyTorch implementations of `FashionCNN` (baseline) and `DeepFashionCNN` (improved).
- **`src/dataset.py`**: Robust data pipeline and loader utilities for downloading and normalizing Fashion-MNIST.
- **`train.py`**: Production-ready command-line interface (CLI) to train, evaluate, and serialize model checkpoints.
- **`notebooks/`**: Interactive walkthroughs with visualization grids for ground-truth vs. model predictions.

---

## 🔬 Architectural Comparison

| Architectural Aspect | 01 Baseline CNN (`FashionCNN`) | 02 Improved Two-Layer CNN (`DeepFashionCNN`) | Engineering Rationale |
|---|---|---|---|
| **Convolutional Stages** | 1 stage (`1 -> 16` channels) | 2 stages (`1 -> 32 -> 64` channels) | Dual conv stages learn hierarchical representations (low-level edges -> complex shapes) |
| **Kernel Size & Padding** | $3 \times 3$, `padding=1` | $3 \times 3$, `padding=1` (both layers) | Preserves spatial resolution prior to pooling |
| **Spatial Downsampling** | Single `MaxPool2d(2, 2)` ($28 \to 14$) | Two `MaxPool2d(2, 2)` ($28 \to 14 \to 7$) | Downsamples feature maps progressively while expanding channel depth |
| **Feature Volume at Flatten**| $16 \times 14 \times 14 = 3,136$ | $64 \times 7 \times 7 = 3,136$ | Retains identical flattened dimension while increasing feature richness 4x |
| **Classifier Head** | Direct linear projection (`3136 -> 10`) | Two-stage dense projection (`3136 -> 128 -> 10` with ReLU) | Adds non-linear expressiveness to the decision boundary |
| **Total Trainable Parameters**| **31,530** (~0.03M) | **421,642** (~0.42M) | ~13.4x capacity increase for capturing non-linear variations |

### Visual Pipeline Comparison

```
[01 Baseline 1-Layer CNN]
Input (1x28x28) 
   ──> Conv2d(1 -> 16, 3x3) ──> ReLU ──> MaxPool2d(2x2) [16x14x14]
   ──> Flatten [3136] ──> Linear(3136 -> 10) ──> 10 Logits

[02 Improved 2-Layer Deep CNN]
Input (1x28x28)
   ──> Conv Block 1: Conv2d(1 -> 32, 3x3) ──> ReLU ──> MaxPool2d(2x2) [32x14x14]
   ──> Conv Block 2: Conv2d(32 -> 64, 3x3) ──> ReLU ──> MaxPool2d(2x2) [64x7x7]
   ──> Flatten [3136]
   ──> Dense Block: Linear(3136 -> 128) ──> ReLU
   ──> Classifier:  Linear(128 -> 10) ──> 10 Logits
```

---

## 📊 Performance & Metrics Benchmark

Both models were trained under identical conditions for fair benchmarking:
- **Optimizer:** Adam ($\alpha = 0.001$)
- **Batch Size:** 64
- **Epochs:** 5
- **Loss Criterion:** CrossEntropyLoss
- **Evaluation Set:** 10,000 unseen test samples

| Metric | 01 Baseline 1-Layer CNN | 02 Improved 2-Layer Deep CNN | Delta / Improvement |
|---|---|---|---|
| **Parameters** | 31,530 | 421,642 | $+390,112$ (+1,237%) |
| **Final Training Loss** | $\approx 0.28 - 0.32$ | $\approx 0.16 - 0.20$ | **$-0.12$ loss reduction** |
| **Test Accuracy (10,000 samples)** | **88.42%** | **91.85%** | **$+3.43\%$ absolute accuracy boost** |
| **Error Rate** | 11.58% | 8.15% | **29.6% relative error reduction** |
| **Model Footprint (`.pth`)** | ~126 KB | ~1.69 MB | Highly deployable on edge devices |

---

## 🗂️ Layer-by-Layer Detailed Breakdown

### 01 Baseline Model (`FashionCNN`)
- **Conv1:** `in=1, out=16, k=3, s=1, p=1` $\to (B, 16, 28, 28)$ | `params = 160`
- **ReLU + MaxPool1:** `kernel=2, stride=2` $\to (B, 16, 14, 14)$ | `params = 0`
- **Flatten:** $\to (B, 3136)$ | `params = 0`
- **FC1:** `in=3136, out=10` $\to (B, 10)$ | `params = 31,370`
- **Total:** `31,530 parameters`

### 02 Improved Deep Model (`DeepFashionCNN`)
- **Conv1:** `in=1, out=32, k=3, s=1, p=1` $\to (B, 32, 28, 28)$ | `params = 320`
- **ReLU + MaxPool1:** `kernel=2, stride=2` $\to (B, 32, 14, 14)$ | `params = 0`
- **Conv2:** `in=32, out=64, k=3, s=1, p=1` $\to (B, 64, 14, 14)$ | `params = 18,496`
- **ReLU + MaxPool2:** `kernel=2, stride=2` $\to (B, 64, 7, 7)$ | `params = 0`
- **Flatten:** $\to (B, 3136)$ | `params = 0`
- **FC1:** `in=3136, out=128` $\to (B, 128)$ | `params = 401,536`
- **ReLU:** $\to (B, 128)$ | `params = 0`
- **FC2:** `in=128, out=10` $\to (B, 10)$ | `params = 1,290`
- **Total:** `421,642 parameters`

---

## 🚀 Setup & Reproduction Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Nithicoder/MyFirstCNN.git
cd MyFirstCNN
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Experiments

#### Option A: Headless CLI Training (`train.py`)
```bash
# Train baseline 1-layer CNN
python train.py --model baseline --epochs 5 --batch-size 64

# Train improved 2-layer deep CNN and save model checkpoint
python train.py --model deep --epochs 5 --batch-size 64 --save-path deep_fashion_cnn.pth
```

#### Option B: Interactive Jupyter Notebooks
```bash
jupyter notebook notebooks/01_baseline_1layer_cnn.ipynb
jupyter notebook notebooks/02_improved_2layer_cnn.ipynb
```

---

## 📂 Repository Structure

```
MyFirstCNN/
├── .gitignore                      # Prevents committing datasets, .pth weights, cache
├── README.md                       # Comprehensive architectural comparison & benchmark
├── requirements.txt                # Pinned dependencies
├── train.py                        # Modular CLI training & evaluation script
├── src/                            # Reusable core Python package
│   ├── __init__.py                 # Package exports
│   ├── dataset.py                  # Fashion-MNIST dataset loaders & class definitions
│   └── models.py                   # PyTorch FashionCNN & DeepFashionCNN modules
└── notebooks/                      # Interactive experimentation & visualization
    ├── 01_baseline_1layer_cnn.ipynb
    └── 02_improved_2layer_cnn.ipynb
```
