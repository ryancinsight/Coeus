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

# Import submodules
from . import nn
from . import optim
from . import utils

# Version information
__version__ = "0.1.0"
__author__ = "Coeus Team"
__email__ = "team@coeus.ai"

# Tensor creation functions - delegate to Rust implementations
def tensor(data: Union[List, np.ndarray, float, int], 
          dtype: Optional[str] = None,
          device: Optional[str] = None,
          requires_grad: bool = False) -> PyTensor:
    """Create a tensor from data - delegates to Rust implementation."""
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

# All tensor creation functions delegate to Rust
def zeros(shape: List[int], **kwargs) -> PyTensor:
    """Create zero tensor - delegates to Rust."""
    from pycoeus._core import zeros as rust_zeros
    return rust_zeros(shape)

def ones(shape: List[int], **kwargs) -> PyTensor:
    """Create ones tensor - delegates to Rust."""
    from pycoeus._core import ones as rust_ones
    return rust_ones(shape)

def randn(*shape: int, **kwargs) -> PyTensor:
    """Create random normal tensor - delegates to Rust."""
    from pycoeus._core import randn as rust_randn
    return rust_randn(list(shape))

def rand(*shape: int, **kwargs) -> PyTensor:
    """Create random uniform tensor - delegates to Rust."""
    from pycoeus._core import rand_tensor as rust_rand
    return rust_rand(list(shape))

def arange(start: float, end: Optional[float] = None, step: float = 1.0, **kwargs) -> PyTensor:
    """Create range tensor - delegates to Rust."""
    from pycoeus._core import arange as rust_arange
    if end is None:
        return rust_arange(0.0, start, step)
    return rust_arange(start, end, step)

def eye(n: int, m: Optional[int] = None, **kwargs) -> PyTensor:
    """Create identity matrix - delegates to Rust."""
    from pycoeus._core import eye as rust_eye
    return rust_eye(n, m or n)

# Utility functions - delegate to Rust
def manual_seed(seed: int) -> None:
    """Set random seed - delegates to Rust."""
    from pycoeus._core import manual_seed as rust_manual_seed
    rust_manual_seed(seed)

def cuda_is_available() -> bool:
    """Check CUDA availability - delegates to Rust."""
    from pycoeus._core import cuda_is_available as rust_cuda_is_available
    return rust_cuda_is_available()

def set_num_threads(num_threads: int) -> None:
    """Set thread count - delegates to Rust."""
    from pycoeus._core import set_num_threads as rust_set_num_threads
    rust_set_num_threads(num_threads)

def get_num_threads() -> int:
    """Get thread count - delegates to Rust."""
    from pycoeus._core import get_num_threads as rust_get_num_threads
    return rust_get_num_threads()

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
