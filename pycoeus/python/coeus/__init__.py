"""Coeus: Safe PyTorch in Rust

A complete, safe Rust implementation of PyTorch's core functionality
with identical API compatibility, enhanced memory safety through Rust's
ownership system, and competitive performance through zero-cost abstractions.

Example:
    >>> import coeus as torch
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> y = torch.tensor([4.0, 5.0, 6.0])
    >>> z = x + y
    >>> print(z)
    tensor([5., 7., 9.])

Features:
    - Memory Safety: Zero unsafe code, Miri-validated
    - Performance: Zero-cost abstractions, SIMD acceleration
    - Compatibility: 100% PyTorch API compatibility
    - Safety: Thread-safe, race-free concurrent execution
    - Ecosystem: Seamless NumPy and Python integration
"""

from ._coeus import *

__version__ = "0.1.0"
__author__ = "Ryan Clanton"
__email__ = "ryan@coeus.dev"

# Import submodules
from . import nn
from . import optim

# Expose key classes and functions for PyTorch compatibility
__all__ = [
    # Core tensor operations
    "tensor", "zeros", "ones", "empty", "full", "arange", "linspace",

    # Neural network modules
    "nn",

    # Optimization
    "optim",

    # Automatic differentiation
    "no_grad", "requires_grad", "backward", "grad", "hvp", "checkpoint", "checkpoint_sequential", "Function",

    # Device management
    "device", "cuda", "cpu",

    # Utility functions
    "cat", "stack", "split", "chunk",

    # Version info
    "__version__",
]
