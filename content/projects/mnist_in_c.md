---
title: "MNIST in C"
description: "A minimal MNIST neural network implementation in C."
date: 2026-02-08
---

[GitHub Repository](https://github.com/areynaldo/mnist_in_c)

# MNIST in C

A minimal MNIST neural network implementation in C with minimal dependencies.

## Overview

This project demonstrates a simple 2-layer MLP (Multi-Layer Perceptron) trained on the MNIST dataset, written in C.

## Model Architecture

```
784 (input) -> 128 (ReLU) -> 10 (Softmax)
```

Results: ~98% accuracy after 5 epochs.

## Data

Place the MNIST data files in the `data/` directory:

- train-images.idx3-ubyte
- train-labels.idx1-ubyte
- t10k-images.idx3-ubyte
- t10k-labels.idx1-ubyte

Download MNIST from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and extract to `data/`.

## Structure

```
base.h            - Core library
tensor.h          - Tensor/NN module
examples/
  mnist.c         - MNIST training example
```
