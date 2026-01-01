"""Optimization algorithms for Coeus.

This module provides PyTorch-compatible optimizers for training neural networks.
"""

from .._coeus import SGD, Adam, AdamW, RMSprop, Adagrad
# Create lr_scheduler submodule
from . import lr_scheduler

# Expose PyTorch-compatible API
__all__ = [
    "SGD",
    "Adam",
    "AdamW",
    "RMSprop",
    "Adagrad",
    "lr_scheduler",
]
