"""
Neural network modules for PyCoeus

This module provides PyTorch-compatible neural network layers, activation functions,
loss functions, and utilities. All implementations are in Rust - this just provides
a clean Python API that delegates to the Rust implementations.
"""

# Import only what's actually implemented in Rust
from pycoeus._core import (
    # Base module
    NNModule,
    
    # Actually implemented layers
    Linear,
    Conv2d,
    Rnn,
    Lstm,
    Gru,
    
    # Actually implemented activation functions
    ReLU,
    
    # Actually implemented loss functions
    MseLoss,
    CrossEntropyLoss,
    
    # Actually implemented models
    GPT2,
)

# PyTorch compatibility aliases
RNN = Rnn
LSTM = Lstm
GRU = Gru
MSELoss = MseLoss

# Import submodules (these also delegate to Rust)
from . import functional
from . import init
from . import utils

# PyTorch compatibility aliases
Module = NNModule
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
    # Base classes
    "Module",
    "NNModule",
    "Parameter",
    "parameter",
    
    # Actually implemented layers
    "Linear",
    "Conv2d",
    "Rnn", "RNN",  # Both Rust name and PyTorch alias
    "Lstm", "LSTM",  # Both Rust name and PyTorch alias
    "Gru", "GRU",  # Both Rust name and PyTorch alias
    
    # Actually implemented activation functions
    "ReLU",
    
    # Actually implemented loss functions
    "MseLoss", "MSELoss",  # Both Rust name and PyTorch alias
    "MSE",
    "CrossEntropyLoss",
    "CrossEntropy",
    
    # Actually implemented models
    "GPT2",
    
    # Submodules
    "functional",
    "init",
    "utils",
]