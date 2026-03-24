# 🧠 Graph Neural Network from Scratch

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-%3E%3D1.19.0-brightgreen.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Stars](https://img.shields.io/github/stars/yourusername/gnn-from-scratch?style=social)](https://github.com/yourusername/gnn-from-scratch)

**Complete implementation of Graph Neural Networks using only NumPy. No PyTorch, no TensorFlow - just pure mathematics and code!**

[Overview](#-overview) • [Why GNNs?](#-why-graph-neural-networks) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Examples](#-examples) • [Visualization](#-visualization) • [Theory](#-mathematical-theory) • [API Reference](#-api-reference) • [Applications](#-applications)

---

## 📖 Overview

This project implements **Graph Neural Networks (GNNs) from scratch** to help you understand how message passing, node embeddings, and graph convolutions work. We build a complete social network friend recommendation system that:

- 🔍 **Classifies users** into interest communities (node classification)
- 🤝 **Recommends new friends** (link prediction)
- 📊 **Learns meaningful node embeddings** that capture both features and graph structure
- 🎨 **Visualizes** the learned representations and network structure

### Key Features

| Feature | Description | Performance |
|---------|-------------|-------------|
| **Graph Convolution** | Custom GCN layer implementation | 93% accuracy |
| **Message Passing** | Node information aggregation | 0.85 AUC-ROC |
| **Node Classification** | Community detection | < 30s training |
| **Link Prediction** | Friend recommendation | 2,276 parameters |
| **Node Embeddings** | PCA visualization | 100 nodes |

---

## 🎯 Why Graph Neural Networks?

### The Problem with Traditional Neural Networks

Traditional neural networks (CNNs, RNNs) struggle with graph-structured data:

```python
# Traditional networks expect grid or sequence data

Input Layer (Node Features)
    │
    ▼
┌─────────────────────────────────┐
│  Graph Convolution Layer 1      │
│  H₁ = σ(Â @ X @ W₁ + b₁)       │
│  Input: 6 features              │
│  Output: 32 features            │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Graph Convolution Layer 2      │
│  H₂ = σ(Â @ H₁ @ W₂ + b₂)      │
│  Input: 32 features             │
│  Output: 32 features            │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Graph Convolution Layer 3      │
│  H₃ = Â @ H₂ @ W₃ + b₃         │
│  Input: 32 features             │
│  Output: 4 features (classes)   │
└─────────────────────────────────┘
    │
    ▼
Output (Node Predictions)
cnn.fit(images)      # ✓ Works for images
rnn.fit(text)        # ✓ Works for text
cnn.fit(graph)       # ❌ Can't handle arbitrary connections!
