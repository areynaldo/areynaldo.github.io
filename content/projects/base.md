---
title: "Base"
description: "A minimal C base library for personal projects."
date: 2026-02-08
---

[GitHub Repository](https://github.com/areynaldo/base)

A minimal C base library for personal projects. Base provides essential building blocks for C projects: memory management, common types, and utilities. Designed to be simple, dependency-free, and easy to drop into any project.

## Modules

### Core (`base.h`)

- Arena-based memory allocation
- Common type definitions (float32_t, etc.)
- String utilities
- Error handling
- Byte manipulation (endian swap, etc.)

### Tensor (`tensor.h`)

- N-dimensional tensors (float32)
- Element-wise operations
- Activations (ReLU, Sigmoid, Tanh, Softmax)
- Loss functions (MSE, Cross-Entropy)
- Linear layers with backprop
- SGD optimizer

## Examples

### MNIST

A 2-layer MLP trained on MNIST demonstrating the tensor module:

```
784 (input) -> 128 (ReLU) -> 10 (Softmax)
```

Results: ~98% accuracy after 5 epochs.

## Structure

```
base.h            - Core library
tensor.h          - Tensor/NN module
examples/
  mnist.c         - MNIST training example
```