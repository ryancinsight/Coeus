"""Optimization algorithms for Coeus.

This module provides PyTorch-compatible optimizers for training neural networks.
"""

from .._coeus import SGD, Adam, AdamW, RMSprop, Adagrad
# Schedulers might need to be imported differently if they are NOT in _coeus
# For now, let's just get the optimizers working.
# from .. import StepLR, CosineAnnealingLR, ExponentialLR 

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
