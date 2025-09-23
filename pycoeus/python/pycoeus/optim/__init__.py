"""
Optimization algorithms for PyCoeus

All optimizers delegate to Rust implementations.
"""

# Import only what's actually implemented in Rust
from pycoeus._core import (
    Sgd,
    Adam,
    AdamW,
)

# PyTorch compatibility aliases
SGD = Sgd

# Import learning rate schedulers (when implemented)
from . import lr_scheduler

# Export all optimizers
__all__ = [
    "Sgd", "SGD",  # Both Rust name and PyTorch alias
    "Adam",
    "AdamW",
    "lr_scheduler",
]