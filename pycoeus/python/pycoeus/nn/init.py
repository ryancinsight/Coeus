"""
Parameter initialization functions

All initialization functions delegate to Rust implementations.
No Python implementations here - just a clean API.
"""

import math
from typing import Optional
from pycoeus import PyTensor

def calculate_gain(nonlinearity: str, param: Optional[float] = None) -> float:
    """Calculate gain value - delegates to Rust implementation."""
    # This should delegate to Rust implementation
    from pycoeus._core import calculate_gain as rust_calculate_gain
    return rust_calculate_gain(nonlinearity, param)

def uniform_(tensor: PyTensor, a: float = 0.0, b: float = 1.0) -> PyTensor:
    """Uniform initialization - delegates to Rust."""
    return tensor.uniform_(a, b)

def normal_(tensor: PyTensor, mean: float = 0.0, std: float = 1.0) -> PyTensor:
    """Normal initialization - delegates to Rust."""
    return tensor.normal_(mean, std)

def constant_(tensor: PyTensor, val: float) -> PyTensor:
    """Constant initialization - delegates to Rust."""
    return tensor.fill_(val)

def zeros_(tensor: PyTensor) -> PyTensor:
    """Zero initialization - delegates to Rust."""
    return tensor.zero_()

def ones_(tensor: PyTensor) -> PyTensor:
    """Ones initialization - delegates to Rust."""
    return tensor.fill_(1.0)

def eye_(tensor: PyTensor) -> PyTensor:
    """Identity initialization - delegates to Rust."""
    return tensor.eye_()

def xavier_uniform_(tensor: PyTensor, gain: float = 1.0) -> PyTensor:
    """Xavier uniform initialization - delegates to Rust."""
    return tensor.xavier_uniform_(gain)

def xavier_normal_(tensor: PyTensor, gain: float = 1.0) -> PyTensor:
    """Xavier normal initialization - delegates to Rust."""
    return tensor.xavier_normal_(gain)

def kaiming_uniform_(tensor: PyTensor, a: float = 0, mode: str = 'fan_in',
                    nonlinearity: str = 'leaky_relu') -> PyTensor:
    """Kaiming uniform initialization - delegates to Rust."""
    return tensor.kaiming_uniform_(a, mode, nonlinearity)

def kaiming_normal_(tensor: PyTensor, a: float = 0, mode: str = 'fan_in',
                   nonlinearity: str = 'leaky_relu') -> PyTensor:
    """Kaiming normal initialization - delegates to Rust."""
    return tensor.kaiming_normal_(a, mode, nonlinearity)

# Aliases for compatibility
glorot_uniform_ = xavier_uniform_
glorot_normal_ = xavier_normal_
he_uniform_ = kaiming_uniform_
he_normal_ = kaiming_normal_

# Export all functions
__all__ = [
    "calculate_gain", "uniform_", "normal_", "constant_", "zeros_", "ones_", "eye_",
    "xavier_uniform_", "xavier_normal_", "kaiming_uniform_", "kaiming_normal_",
    "glorot_uniform_", "glorot_normal_", "he_uniform_", "he_normal_",
]