"""Functional API for neural network operations.

This module provides functional versions of neural network operations
that can be used without creating module instances.
"""

from .. import relu, sigmoid, tanh, gelu, silu, leaky_relu, elu
from .. import mse_loss, cross_entropy as cross_entropy_loss
from .. import max_pool2d, avg_pool2d, dropout, layer_norm
from .. import Linear, Conv2D, BatchNorm2d, Dropout, Embedding

# Expose functional operations
__all__ = [
    "relu",
    "sigmoid",
    "tanh",
    "gelu",
    "silu",
    "leaky_relu",
    "elu",
    "mse_loss",
    "cross_entropy_loss",
    "conv2d",
    "batch_norm",
    "dropout",
    "max_pool2d",
    "avg_pool2d",
    "layer_norm",
]
