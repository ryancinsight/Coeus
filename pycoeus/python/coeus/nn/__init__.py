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
)

# Expose PyTorch-compatible API
__all__ = [
    "Sequential",
    "Linear",
    "Conv2D",
    "BatchNorm2d",
    "Dropout",
]

# Create functional submodule - commented out until functional operations are implemented
# from . import functional
