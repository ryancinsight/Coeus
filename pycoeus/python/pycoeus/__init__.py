"""
PyCoeus: PyTorch-compatible neural network library built in Rust

PyCoeus provides a PyTorch-compatible API with high performance Rust backend,
offering automatic differentiation, neural network modules, and GPU acceleration.

All implementations are in Rust - this Python layer just provides a clean API.
"""

from typing import List, Optional, Union, Any
import numpy as np

# Import the core Rust module
try:
    from pycoeus._core import *
except ImportError as e:
    raise ImportError(
        "Failed to import PyCoeus core module. "
        "Make sure PyCoeus is properly installed with: pip install pycoeus"
    ) from e

# Import neural network classes from submodules
from .nn import Linear, Conv1d, Conv2d, BatchNorm1d, ReLU, Sigmoid, Tanh, Softmax, MseLoss, CrossEntropyLoss

# Import submodules
from . import nn
from . import optim
from . import utils

# Version information
__version__ = "0.1.0"
__author__ = "Coeus Team"
__email__ = "team@coeus.ai"

# Tensor creation functions - delegate to PyTensor static methods
def tensor(data: Union[List, np.ndarray, float, int],
          dtype: Optional[str] = None,
          device: Optional[str] = None,
          requires_grad: bool = False) -> PyTensor:
    """Create a tensor from data - delegates to PyTensor static methods."""
    if isinstance(data, np.ndarray):
        tensor = PyTensor.from_numpy(data.astype(np.float32))
    elif isinstance(data, (list, tuple)):
        array = np.array(data, dtype=np.float32)
        tensor = PyTensor.from_numpy(array)
    elif isinstance(data, (int, float)):
        array = np.array([data], dtype=np.float32)
        tensor = PyTensor.from_numpy(array)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    if requires_grad:
        tensor.requires_grad_(True)

    return tensor

# All tensor creation functions delegate to PyTensor static methods
def zeros(shape: List[int], **kwargs) -> PyTensor:
    """Create zero tensor - delegates to PyTensor.zeros."""
    return PyTensor.zeros(shape)

def ones(shape: List[int], **kwargs) -> PyTensor:
    """Create ones tensor - delegates to PyTensor.ones."""
    return PyTensor.ones(shape)

def randn(*shape: int, **kwargs) -> PyTensor:
    """Create random normal tensor - delegates to PyTensor.randn."""
    return PyTensor.randn(list(shape))

def rand(*shape: int, **kwargs) -> PyTensor:
    """Create random uniform tensor - delegates to PyTensor.rand."""
    return PyTensor.rand(list(shape))

def arange(start: float, end: Optional[float] = None, step: float = 1.0, **kwargs) -> PyTensor:
    """Create range tensor - delegates to PyTensor.arange."""
    if end is None:
        return PyTensor.arange(0.0, start, step)
    return PyTensor.arange(start, end, step)

def eye(n: int, m: Optional[int] = None, **kwargs) -> PyTensor:
    """Create identity matrix - delegates to PyTensor.eye."""
    return PyTensor.eye(n, m or n)

# Utility functions - delegate to PyTensor static methods
def manual_seed(seed: int) -> None:
    """Set random seed - delegates to PyTensor.manual_seed."""
    PyTensor.manual_seed(seed)

def cuda_is_available() -> bool:
    """Check CUDA availability - delegates to PyTensor.cuda_is_available."""
    return PyTensor.cuda_is_available()

def set_num_threads(num_threads: int) -> None:
    """Set thread count - delegates to PyTensor.set_num_threads."""
    PyTensor.set_num_threads(num_threads)

def get_num_threads() -> int:
    """Get thread count - delegates to PyTensor.get_num_threads."""
    return PyTensor.get_num_threads()

# Export main classes and functions
__all__ = [
    # Core tensor operations
    "PyTensor",
    "Device",
    "tensor",
    "zeros",
    "ones",
    "randn",
    "rand",
    "arange",
    "eye",

    # Neural network components
    "Linear", "Conv1d", "Conv2d", "BatchNorm1d", "ReLU", "Sigmoid", "Tanh", "Softmax",
    "MseLoss", "CrossEntropyLoss",

    # Submodules
    "nn",
    "optim",
    "utils",

    # Utility functions
    "manual_seed",
    "cuda_is_available",
    "set_num_threads",
    "get_num_threads",
]
