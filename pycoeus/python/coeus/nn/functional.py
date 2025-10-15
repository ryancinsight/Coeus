"""Functional API for neural network operations.

This module provides functional versions of neural network operations
that can be used without creating module instances.
"""

from .. import relu, sigmoid, tanh, mse_loss, cross_entropy_loss
from .. import conv2d_py as conv2d, batch_norm_py as batch_norm, dropout_py as dropout
from .. import gelu_py as gelu, silu_py as silu, leaky_relu_py as leaky_relu, elu_py as elu
from .. import max_pool2d_py as max_pool2d, avg_pool2d_py as avg_pool2d
from .. import layer_norm_py as layer_norm

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
