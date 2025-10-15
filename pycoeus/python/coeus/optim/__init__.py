"""Optimization algorithms for Coeus.

This module provides PyTorch-compatible optimizers for training neural networks.
"""

from .. import SGD, Adam, AdamW, RMSprop, Adagrad, StepLR, CosineAnnealingLR, ExponentialLR

# Create lr_scheduler submodule
from . import lr_scheduler

# Expose PyTorch-compatible API
__all__ = [
    "Optimizer",
    "SGD",
    "Adam",
    "AdamW",
    "RMSprop",
    "Adagrad",
    "StepLR",
    "CosineAnnealingLR",
    "ExponentialLR",
    "lr_scheduler",
]
