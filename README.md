# GC-AIS
Official Python implementation of the framework proposed in the paper: **"GC-AIS: A Structure-Aware Graph Contrastive Autoencoder for Robust Instance Selection in High-Dimensional Manifolds"**.

## Overview
This repository contains the source code required to reproduce the core methodology of the GC-AIS framework. The algorithm shifts the paradigm from geometric distance-based filtering to topological representation learning by integrating Graph Attention Networks (GAT) with a dual-branch objective (Reconstruction + Structural Contrastive Loss).

## Core Modules Included:
1. **Topological Graph Construction**: Cosine-based k-NN and Gaussian Heat Kernel.
2. **Structure-Aware Encoding**: Multi-head Graph Attention Networks.
3. **Dual-Branch Optimization**: Generative (MSE) and Discriminative (InfoNCE) branches.
4. **Dual Importance Scoring**: Pruning instances based on Reconstruction Confidence (RC) and Structural Hardness (SH).

## Dependencies
- `torch` (PyTorch)
- `scikit-learn`
- `pandas`
- `numpy`
