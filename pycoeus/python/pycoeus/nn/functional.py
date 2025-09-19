"""
Functional interface for neural network operations

All functions delegate to Rust implementations. No Python implementations here.
"""

from typing import Optional, Union
from pycoeus import PyTensor

# All functional operations delegate to Rust implementations
def relu(input: PyTensor, inplace: bool = False) -> PyTensor:
    """ReLU activation - delegates to Rust."""
    return input.relu()  # PyTensor has relu method

def sigmoid(input: PyTensor) -> PyTensor:
    """Sigmoid activation - delegates to Rust."""
    return input.sigmoid()  # PyTensor has sigmoid method

def tanh(input: PyTensor) -> PyTensor:
    """Tanh activation - delegates to Rust."""
    return input.tanh()  # PyTensor has tanh method

def mse_loss(input: PyTensor, target: PyTensor, reduction: str = 'mean') -> PyTensor:
    """MSE loss - delegates to Rust."""
    from pycoeus._core import MseLoss
    loss_fn = MseLoss()
    return loss_fn.forward(input, target)

def cross_entropy(input: PyTensor, target: PyTensor, 
                 weight: Optional[PyTensor] = None,
                 reduction: str = 'mean') -> PyTensor:
    """Cross entropy loss - delegates to Rust."""
    from pycoeus._core import CrossEntropyLoss
    loss_fn = CrossEntropyLoss()
    return loss_fn.forward(input, target)

# Note: These functions would need proper tensor operations implemented
# For now, they're placeholders that would use the actual tensor methods

def linear(input: PyTensor, weight: PyTensor, bias: Optional[PyTensor] = None) -> PyTensor:
    """Linear transformation - would use tensor matmul when implemented."""
    raise NotImplementedError("Functional linear requires tensor matmul - use nn.Linear instead")

def conv2d(input: PyTensor, weight: PyTensor, bias: Optional[PyTensor] = None,
           stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0,
           dilation: Union[int, tuple] = 1, groups: int = 1) -> PyTensor:
    """2D convolution - would use tensor conv2d when implemented."""
    raise NotImplementedError("Functional conv2d requires tensor operations - use nn.Conv2d instead")

# Export only what's actually implemented
__all__ = [
    "relu", "sigmoid", "tanh",
    "mse_loss", "cross_entropy", 
    "linear", "conv2d",
]