"""
Optimization algorithms for PyCoeus

All optimizers delegate to Rust implementations.
"""

# Import only what's actually implemented in Rust
from pycoeus._core import (
    Sgd,
    Adam,
    AdamW,
    RMSprop,
    Adagrad,
    Lbfgs,
)

# PyTorch compatibility aliases
SGD = Sgd
LBFGS = Lbfgs

# Import learning rate schedulers (when implemented)
from . import lr_scheduler

# Export all optimizers
__all__ = [
    "Sgd", "SGD",  # Both Rust name and PyTorch alias
    "Adam", 
    "AdamW",
    "RMSprop",
    "Adagrad",
    "Lbfgs", "LBFGS",  # Both Rust name and PyTorch alias
    "lr_scheduler",
]