"""
Functional interface for neural network operations

All functions delegate to Rust implementations. No Python implementations here.
"""

from typing import Optional, Union
import pycoeus as pc
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

def softmax(input: PyTensor, dim: Optional[int] = None) -> PyTensor:
    """Softmax activation - delegates to Rust."""
    if dim is None:
        dim = -1
    return input.softmax(dim)  # PyTensor has softmax method

def mse_loss(input: PyTensor, target: PyTensor, reduction: str = 'mean') -> PyTensor:
    """MSE loss - delegates to Rust."""
    from pycoeus._core import MseLoss

    # Map string reduction to Rust enum values
    if reduction == 'mean':
        loss_fn = MseLoss.with_reduction_mean()
    elif reduction == 'sum':
        loss_fn = MseLoss.with_reduction_sum()
    elif reduction == 'none':
        loss_fn = MseLoss.with_reduction_none()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Must be 'mean', 'sum', or 'none'")

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
    """Linear transformation using tensor matmul."""
    # Handle 1D input by unsqueezing to 2D (batch_size=1, features)
    if len(input.shape()) == 1:
        input = input.unsqueeze(0)  # Add batch dimension
        needs_squeeze = True
    else:
        needs_squeeze = False

    # PyTorch functional.linear: output = input @ weight.t() + bias
    # For compatibility with PyTorch's functional.linear, we need to transpose weight
    weight_t = weight.t()
    output = input @ weight_t

    if bias is not None:
        # Use proper broadcasting: bias [out_features] -> [batch_size, out_features]
        output = output + bias

    # Squeeze back to original shape if input was 1D
    if needs_squeeze:
        output = output.squeeze(0)

    return output

def conv2d(input: PyTensor, weight: PyTensor, bias: Optional[PyTensor] = None,
           stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0,
           dilation: Union[int, tuple] = 1, groups: int = 1) -> PyTensor:
    """2D convolution - use nn.Conv2d layer for now.

    Note: Functional conv2d is not yet implemented. Use the nn.Conv2d layer instead:

    ```python
    conv = pc.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    output = conv(input)
    ```
    """
    raise NotImplementedError(
        "Functional conv2d is not yet implemented. Use nn.Conv2d layer instead:\n"
        "conv = pc.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)\n"
        "output = conv(input)"
    )

# Export only what's actually implemented
__all__ = [
    "relu", "sigmoid", "tanh", "softmax",
    "mse_loss", "cross_entropy",
    "linear",  # ✅ Implemented
    # "conv2d",  # ❌ Not implemented - use nn.Conv2d layer
]