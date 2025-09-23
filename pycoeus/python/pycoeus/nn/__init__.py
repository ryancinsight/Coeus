"""
Neural network modules for PyCoeus

This module provides PyTorch-compatible neural network layers, activation functions,
loss functions, and utilities. All implementations are in Rust - this just provides
a clean Python API that delegates to the Rust implementations.
"""

# Import only what's actually implemented in Rust
from pycoeus._core import (
    # Actually implemented layers
    Linear,
    Conv1d,
    Conv2d,
    BatchNorm1d,

    # Actually implemented activation functions
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,

    # Actually implemented loss functions
    MseLoss,
    CrossEntropyLoss,
)

# PyTorch compatibility aliases
MSELoss = MseLoss

# Import submodules (these also delegate to Rust)
from . import functional
from . import init
from . import utils

# PyTorch compatibility aliases
MSE = MSELoss
CrossEntropy = CrossEntropyLoss

# Parameter class - delegates to Rust tensor with requires_grad
class Parameter:
    """Parameter wrapper - delegates to Rust tensor implementation."""
    
    def __init__(self, data, requires_grad=True):
        if hasattr(data, 'set_requires_grad'):
            data.set_requires_grad(requires_grad)
        self.data = data
        self.requires_grad = requires_grad
    
    def __repr__(self):
        return f"Parameter containing:\n{self.data}"

def parameter(data, requires_grad=True):
    """Create a parameter - delegates to Rust tensor."""
    return Parameter(data, requires_grad)

# Export only what's actually implemented
__all__ = [
    # Classes
    "Parameter",
    "parameter",

    # Actually implemented layers
    "Linear",
    "Conv1d",
    "Conv2d",
    "BatchNorm1d",

    # Actually implemented activation functions
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",

    # Actually implemented loss functions
    "MseLoss", "MSELoss",  # Both Rust name and PyTorch alias
    "MSE",
    "CrossEntropyLoss",
    "CrossEntropy",

    # Submodules
    "functional",
    "init",
    "utils",
]