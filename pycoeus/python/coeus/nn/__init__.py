"""Neural network modules for Coeus.

This module provides PyTorch-compatible neural network components including
layers, activations, loss functions, and utilities.
"""

from .._coeus import (
    Sequential,
    Linear,
    Conv2D,
    BatchNorm2d,
    Dropout,
    Embedding,
    LayerNorm,
    RNN,
    LSTM,
    GRU,
    ReLU,
    GELU,
    SiLU,
    PReLU,
    Sigmoid,
    Tanh,
    MaxPool1d,
    MaxPool2d,
    AvgPool1d,
    AvgPool2d,
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
)

# Expose PyTorch-compatible API
__all__ = [
    "Sequential",
    "Linear",
    "Conv2D",
    "BatchNorm2d",
    "Dropout",
    "Embedding",
    "LayerNorm",
    "RNN",
    "LSTM",
    "GRU",
    "ReLU",
    "GELU",
    "SiLU",
    "PReLU",
    "Sigmoid",
    "Tanh",
    "MaxPool1d",
    "MaxPool2d",
    "AvgPool1d",
    "AvgPool2d",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
]

# Create functional submodule
from . import functional
