# EII-DAN: EEG-based Individual Identification via Domain Adversarial Network

> A Deep Learning Framework for Cross-Domain EEG Individual Recognition

## Model Overview

EII-DAN is a deep learning model designed for **cross-domain EEG individual identification**, integrating **multi-scale spatiotemporal encoding**, **hybrid attention mechanisms**, and **contrast-enhanced domain adversarial training** to achieve robust feature extraction and cross-domain generalization from EEG signals.


## Model Architecture

### Input Data
- **Dimensions**: `[64, 128, 5, 5, 3]` (batch_size, time_steps, feature_matrix, frequency_bands)
- **Features**: EEG microstate feature matrix (24-dimensional features reshaped to 5×5)

### Core Modules

1. **Multi-scale Spatiotemporal Encoder**
   - Temporal dynamic convolution (3×1, 5×1)
   - Spatial depthwise separable convolution
   - Multi-scale pyramid (dilated convolution + pooling)

2. **Hybrid Attention Module**
   - Local window attention + global sparse sampling
   - Grouped multi-head mechanism (75% parameter reduction)
   - Causal masking for temporal causality

3. **Contrast-enhanced Domain Adversarial Training**
   - Dynamic adversarial loss weighting
   - Contrastive loss for same-subject feature alignment
   - Gradient reversal layer for domain-invariant feature learning

## Quick Start

### Environment Setup
```bash
git clone https://gitee.com/LEELILI/EII-DAN.git
cd EII-DAN
pip install -r requirements.txt