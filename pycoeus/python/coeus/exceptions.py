"""
Coeus Exception Hierarchy

This module defines the complete exception hierarchy for PyCoeus,
providing specific exception types for different error categories
to enable better error handling in user code.

The hierarchy follows PyTorch's exception patterns while adding
Rust-specific error context.
"""


class CoeusError(Exception):
    """Base exception for all Coeus errors.
    
    All Coeus-specific exceptions inherit from this base class,
    allowing users to catch all Coeus errors with a single except clause.
    
    Example:
        >>> try:
        ...     result = some_coeus_operation()
        ... except CoeusError as e:
        ...     print(f"Coeus error occurred: {e}")
    """
    pass


class TensorError(CoeusError):
    """Raised when tensor operations fail.
    
    This exception is raised for general tensor operation failures,
    including invalid operations, incompatible tensors, or internal
    tensor computation errors.
    
    Example:
        >>> try:
        ...     result = tensor1.matmul(tensor2)
        ... except TensorError as e:
        ...     print(f"Tensor operation failed: {e}")
    """
    pass


class BackendError(CoeusError):
    """Raised when backend operations fail.
    
    This exception is raised when the underlying compute backend
    (CPU, GPU, TPU, NPU) encounters an error during execution.
    
    Example:
        >>> try:
        ...     tensor = tensor.to('cuda')
        ... except BackendError as e:
        ...     print(f"Backend error: {e}")
    """
    pass


class OptimizerError(CoeusError):
    """Raised when optimizer operations fail.
    
    This exception is raised for optimizer-specific errors such as
    invalid hyperparameters, state loading failures, or step errors.
    
    Example:
        >>> try:
        ...     optimizer.step()
        ... except OptimizerError as e:
        ...     print(f"Optimizer error: {e}")
    """
    pass


class NNError(CoeusError):
    """Raised when neural network operations fail.
    
    This exception is raised for neural network layer errors,
    including invalid layer configurations, forward pass failures,
    or parameter management errors.
    
    Example:
        >>> try:
        ...     output = model(input)
        ... except NNError as e:
        ...     print(f"Neural network error: {e}")
    """
    pass


class StorageError(CoeusError):
    """Raised when storage operations fail.
    
    This exception is raised for storage-related errors such as
    memory allocation failures, storage format conversion errors,
    or invalid storage operations.
    
    Example:
        >>> try:
        ...     sparse_tensor = tensor.to_sparse()
        ... except StorageError as e:
        ...     print(f"Storage error: {e}")
    """
    pass


class ShapeError(TensorError):
    """Raised when tensor shapes are incompatible.
    
    This exception is raised when operations receive tensors with
    incompatible shapes, such as matrix multiplication with mismatched
    dimensions or broadcasting failures.
    
    Example:
        >>> try:
        ...     result = tensor1 + tensor2  # Different shapes
        ... except ShapeError as e:
        ...     print(f"Shape mismatch: {e}")
    """
    pass


class DeviceError(BackendError):
    """Raised when device operations fail.
    
    This exception is raised for device-specific errors such as
    device not available, device memory exhausted, or device
    transfer failures.
    
    Example:
        >>> try:
        ...     tensor = tensor.to('cuda:0')
        ... except DeviceError as e:
        ...     print(f"Device error: {e}")
    """
    pass


# Export all exception classes
__all__ = [
    'CoeusError',
    'TensorError',
    'BackendError',
    'OptimizerError',
    'NNError',
    'StorageError',
    'ShapeError',
    'DeviceError',
]
