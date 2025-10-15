"""Neural network modules for Coeus.

This module provides PyTorch-compatible neural network components including
layers, activations, loss functions, and utilities.
"""

from .._coeus import (
    Linear, Sequential,
    ReLU, Sigmoid, Tanh,
    Conv2d,
    BatchNorm2d,
    LayerNorm, GroupNorm, InstanceNorm2d,
    MultiheadAttention,
    Dropout,
    Embedding,
    relu, sigmoid, tanh, mse_loss, cross_entropy_loss, bce_with_logits_loss, l1_loss, smooth_l1_loss, nll_loss, kldiv_loss, triplet_margin_loss
)

# Expose PyTorch-compatible API
__all__ = [
    # Modules
    "Module",
    "Linear",
    "Sequential",

    # Convolutional layers
    "Conv2d",

    # Normalization layers
    "BatchNorm2d",
    "LayerNorm",
    "GroupNorm",
    "InstanceNorm2d",
    "MultiheadAttention",

    # Regularization layers
    "Dropout",

    # Embedding layers
    "Embedding",

    # Activation modules
    "ReLU", "Sigmoid", "Tanh",

    # Functional API
    "functional",
    "relu",
    "sigmoid",
    "tanh",
    "mse_loss",
    "cross_entropy_loss",
    "bce_with_logits_loss",
    "l1_loss",
    "smooth_l1_loss",
    "nll_loss",
    "kldiv_loss",
    "triplet_margin_loss",
]

# Create functional submodule
from . import functional
