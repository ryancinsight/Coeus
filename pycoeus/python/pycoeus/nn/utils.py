"""
Neural network utility functions

All utilities delegate to Rust implementations.
"""

from typing import Union, Iterable
from pycoeus import PyTensor

def clip_grad_norm_(parameters: Iterable[PyTensor], max_norm: float, 
                   norm_type: float = 2.0) -> float:
    """Clip gradient norm - delegates to Rust."""
    from pycoeus._core import clip_grad_norm as rust_clip_grad_norm
    return rust_clip_grad_norm(list(parameters), max_norm, norm_type)

def clip_grad_value_(parameters: Iterable[PyTensor], clip_value: float) -> None:
    """Clip gradient values - delegates to Rust."""
    from pycoeus._core import clip_grad_value as rust_clip_grad_value
    rust_clip_grad_value(list(parameters), clip_value)

def parameters_to_vector(parameters: Iterable[PyTensor]) -> PyTensor:
    """Convert parameters to vector - delegates to Rust."""
    from pycoeus._core import parameters_to_vector as rust_parameters_to_vector
    return rust_parameters_to_vector(list(parameters))

def vector_to_parameters(vec: PyTensor, parameters: Iterable[PyTensor]) -> None:
    """Convert vector to parameters - delegates to Rust."""
    from pycoeus._core import vector_to_parameters as rust_vector_to_parameters
    rust_vector_to_parameters(vec, list(parameters))

def weight_norm(module, name: str = 'weight', dim: int = 0):
    """Apply weight normalization - delegates to Rust."""
    from pycoeus._core import apply_weight_norm
    return apply_weight_norm(module, name, dim)

def remove_weight_norm(module, name: str = 'weight'):
    """Remove weight normalization - delegates to Rust."""
    from pycoeus._core import remove_weight_norm as rust_remove_weight_norm
    return rust_remove_weight_norm(module, name)

def spectral_norm(module, name: str = 'weight', n_power_iterations: int = 1,
                 dim: int = 0, eps: float = 1e-12):
    """Apply spectral normalization - delegates to Rust."""
    from pycoeus._core import apply_spectral_norm
    return apply_spectral_norm(module, name, n_power_iterations, dim, eps)

def remove_spectral_norm(module, name: str = 'weight'):
    """Remove spectral normalization - delegates to Rust."""
    from pycoeus._core import remove_spectral_norm as rust_remove_spectral_norm
    return rust_remove_spectral_norm(module, name)

# Export all functions
__all__ = [
    "clip_grad_norm_", "clip_grad_value_", "parameters_to_vector", "vector_to_parameters",
    "weight_norm", "remove_weight_norm", "spectral_norm", "remove_spectral_norm",
]