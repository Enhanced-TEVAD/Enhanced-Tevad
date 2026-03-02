# Enhanced TEVAD: Improved Video Anomaly Detection with Captions

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **🚀 Enhanced Implementation** of the paper _TEVAD: Improved video anomaly detection with captions_, featuring significant architectural improvements and performance gains.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Enhancements](#key-enhancements)
- [Performance Results](#performance-results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Usage Examples](#usage-examples)
- [Contributors](#contributors)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

Enhanced TEVAD is an improved implementation of the original TEVAD (Text-Enhanced Video Anomaly Detection) method, featuring advanced Multi-Task Network (MTN) architecture with significant performance improvements. This implementation achieves superior results on benchmark datasets with reduced training time.

### Key Features

- **Enhanced MTN Architecture**: Improved Multi-Task Network with Pyramid Dilated Convolutions (PDC)
- **Advanced Attention Mechanisms**: Squeeze-and-Excitation (SE) modules and Transformer encoders
- **Multi-Modal Fusion**: Sophisticated visual and textual feature fusion strategies
- **Stability Improvements**: Better training stability with BatchNorm and dropout
- **Performance Gains**: 99.22% AUC on UCSD-Ped2 (vs 98.6% original) with 11× fewer epochs

## 🚀 Key Enhancements

### 1. Enhanced Model Architecture (`e_model.py`)

- **Pyramid Dilated Convolution (PDC) Block**: Multi-scale temporal feature extraction
- **Squeeze-and-Excitation (SE) Module**: Channel-wise attention for feature recalibration
- **Transformer Encoder Blocks**: Long-range temporal modeling with self-attention
- **Improved Multi-Modal Fusion**: Better integration of visual and textual features

### 2. Advanced Training Pipeline (`e_main.py`)

- **Enhanced Loss Functions**: Improved convergence with refined loss computation
- **Better Optimization**: Advanced training procedures with stability improvements
- **Improved Checkpointing**: Better model saving and loading mechanisms
- **Enhanced Logging**: Comprehensive experiment tracking and monitoring

### 3. Stability Improvements (`e_mtn.py`)

- **Batch Normalization**: Added throughout the network for training stability
- **Proper Weight Initialization**: Xavier uniform initialization for better convergence
- **Dropout Regularization**: Strategic dropout placement to prevent overfitting
- **Gradient Scaling**: Residual connections with proper scaling to prevent gradient explosion

## 📊 Performance Results

| Dataset      | Original TEVAD | Enhanced TEVAD | Epochs Used | Improvement |
| ------------ | -------------- | -------------- | ----------- | ----------- |
| UCSD-Ped2    | 98.6%          | **99.22% AUC** | Reduced     | +0.62 AUC   |
| UCF-Crime    | 84.90%         | **85.34%**     | Reduced     | +0.44       |
| ShanghaiTech | Baseline       | Baseline       | Reduced     | Baseline    |

## 🛠 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- 8GB+ RAM recommended

### Step 1: Clone the Repository

```bash
git clone https://github.com/Enhanced-TEVAD/Enhanced_Tevad
cd Enhanced_Tevad
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Setup Visdom (Required)

```bash
# Install visdom
pip install visdom

# Start visdom server (in a separate terminal)
python -m visdom.server
```

## 🚀 Quick Start

### Training

```bash
# UCSD-Ped2 dataset
python e_main.py --dataset ped2 --feature-group both --fusion add --aggregate_text --max-epoch 5000 --extra_loss --batch-size 2

# UCF-Crime dataset
python e_main.py --dataset ucf --feature-group both --fusion concat --aggregate_text --extra_loss

# ShanghaiTech dataset
python e_main.py --dataset shanghai_v2 --feature-group both --fusion add --aggregate_text --extra_loss

# XD-Violence dataset
python e_main.py --dataset violence --feature-group both --fusion add --aggregate_text --extra_loss --feature-size 1024
```

### Testing

```bash
# Test with pre-trained models
python main_test.py --dataset ped2 --pretrained-ckpt ./ckpt/my_best/ped2-both-text_agg-add-1-1-extra_loss-755-4869-i3d.pkl --feature-group both --fusion add --aggregate_text --save_test_results
```

## 🏗 Architecture

### Enhanced Multi-Task Network (MTN)

The enhanced MTN architecture consists of several key components:

1. **Pyramid Dilated Convolution (PDC) Block**

   - Multi-scale temporal feature extraction
   - Parallel dilated convolutions with different dilation rates
   - Batch normalization for stability

2. **Squeeze-and-Excitation (SE) Module**

   - Channel-wise attention mechanism
   - Adaptive feature recalibration
   - Improved feature representation

3. **Transformer Encoder Block**
   - Long-range temporal modeling
   - Self-attention mechanisms
   - Pre-normalization for stability

### Multi-Modal Fusion Strategies

- **Concatenation**: Direct feature concatenation
- **Addition**: Element-wise feature addition
- **Product**: Element-wise feature multiplication
- **Up-sampling**: Dimensional alignment with learned projections

## 📁 Project Structure

```
Enhanced_Tevad/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── config.py                # Configuration settings
├── dataset.py               # Dataset loading utilities
├── e_main.py                # Enhanced main training script
├── e_model.py               # Enhanced model architecture
├── e_mtn.py                 # Enhanced MTN components
├── train.py                 # Training utilities
├── test_10crop.py           # Testing utilities
├── utils.py                 # Utility functions
├── option.py                # Command-line arguments
├── list/                    # Dataset lists and ground truth
│   ├── gt-ped2.npy
│   ├── gt-sh2.npy
│   ├── gt-ucf.npy
│   ├── gt-violence.npy
│   └── *.list files
├── ckpt/                    # Model checkpoints
│   └── my_best/
└── save/                    # Extracted features
    ├── Crime/
    ├── Shanghai/
    ├── UCSDped2/
    └── Violence/
```

## 🗂 Datasets

### Supported Datasets

1. **UCSD-Ped2**: Pedestrian anomaly detection
2. **UCF-Crime**: Crime detection in surveillance videos
3. **ShanghaiTech**: Campus surveillance anomaly detection
4. **XD-Violence**: Violence detection in videos

### Feature Requirements

#### Visual Features

- **I3D Features**: Extracted using I3D with ResNet-50 backbone
- **Multi-crop Augmentation**: 10-crop augmentation for robustness
- **Download**: Available from [feature repository](https://1drv.ms/u/s!AlbDzA9D8VkhoO8dcvJNaAMkk5bbgA?e=Eh2LCB)

#### Text Features

- **Sentence Embeddings**: Generated using SimCSE
- **Caption Generation**: Using SwinBERT for video captioning
- **Download**: Available from [text features repository](https://1drv.ms/u/s!AlbDzA9D8VkhoO8dcvJNaAMkk5bbgA?e=Eh2LCB)

## 💡 Usage Examples

### Basic Training

```python
# Train on UCSD-Ped2 with enhanced settings
python e_main.py \
    --dataset ped2 \
    --feature-group both \
    --fusion add \
    --aggregate_text \
    --max-epoch 5000 \
    --extra_loss \
    --batch-size 2
```

### Advanced Configuration

```python
# Custom learning rate and feature size
python e_main.py \
    --dataset ucf \
    --feature-group both \
    --fusion concat \
    --aggregate_text \
    --extra_loss \
    --lr 0.0001 \
    --feature-size 2048
```

### Testing with Pre-trained Models

```python
# Test with best model
python main_test.py \
    --dataset ped2 \
    --pretrained-ckpt ./ckpt/my_best/ped2-both-text_agg-add-1-1-extra_loss-755-4869-i3d.pkl \
    --feature-group both \
    --fusion add \
    --aggregate_text \
    --save_test_results
```

## 🔧 Configuration Options

### Key Parameters

- `--dataset`: Dataset name (ped2, ucf, shanghai_v2, violence)
- `--feature-group`: Feature type (both, visual, text)
- `--fusion`: Fusion method (concat, add, product, add_up)
- `--aggregate_text`: Enable text aggregation
- `--extra_loss`: Enable additional loss terms
- `--max-epoch`: Maximum training epochs
- `--batch-size`: Training batch size
- `--lr`: Learning rate

### Advanced Options

- `--feature-size`: Feature dimension (default: 2048)
- `--emb-dim`: Embedding dimension (default: 768)
- `--alpha`: Loss balancing parameter
- `--normal-weight`: Normal sample weight
- `--abnormal-weight`: Abnormal sample weight

## 📈 Training Process

### Loss Functions

The enhanced training uses multiple loss components:

1. **Feature Magnitude Loss (L_fm)**:

   ```
   L_fm = |c - f_FM(v_j; k)| if y_j = 1
          |f_FM(v_j; k)|     if y_j = 0
   ```

2. **Binary Cross-Entropy Loss (L_bce)**:

   ```
   L_bce = -1/|V| Σ[y_j log f_s(v_j; k) + (1-y_j) log(1-f_s(v_j; k))]
   ```

3. **Total Loss**:
   ```
   L = α * L_fm + L_bce
   ```

### Training Strategy

- **Weakly Supervised Learning**: Only video-level labels required
- **Top-K Selection**: Focus on most anomalous snippets
- **Multi-Modal Fusion**: Combine visual and textual features
- **Progressive Training**: Gradual complexity increase

## 🧪 Evaluation

### Metrics

- **AUC (Area Under Curve)**: Primary metric for most datasets
- **AP (Average Precision)**: Used for XD-Violence dataset
- **Frame-level Accuracy**: Snippet-to-frame propagation

### Evaluation Process

1. **Feature Extraction**: Extract visual and textual features
2. **Anomaly Scoring**: Compute snippet-level anomaly scores
3. **Threshold Selection**: Optimize detection threshold
4. **Performance Calculation**: Compute final metrics

## 🔬 Research Applications

### Anomaly Detection Tasks

- **Surveillance Monitoring**: Real-time anomaly detection
- **Security Applications**: Crime and violence detection
- **Industrial Monitoring**: Manufacturing anomaly detection
- **Healthcare**: Medical video analysis

### Academic Research

- **Weakly Supervised Learning**: Video-level label utilization
- **Multi-Modal Learning**: Visual-textual feature fusion
- **Temporal Modeling**: Long-range dependency capture
- **Attention Mechanisms**: Focus on relevant features

### Contributions

- **Architecture Design**: Enhanced MTN with PDC, SE, and Transformer blocks
- **Training Pipeline**: Improved loss functions and optimization procedures
- **Stability Improvements**: BatchNorm, dropout, and weight initialization
- **Performance Optimization**: Reduced training time with better convergence

## 🔗 Related Work

- **Original TEVAD**: [GitHub Repository](https://github.com/coranholmes/TEVAD)
- **RTFM**: [GitHub Repository](https://github.com/tianyu0207/RTFM/)
- **I3D Feature Extraction**: [GitHub Repository](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet)
- **SwinBERT**: [GitHub Repository](https://github.com/coranholmes/SwinBERT)

