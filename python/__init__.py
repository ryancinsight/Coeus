# PyCoeus - Python bindings for Coeus tensor library
"""
PyCoeus provides PyTorch-compatible tensor operations with automatic differentiation.

Example:
    >>> import pycoeus as pc
    >>> import numpy as np
    >>>
    >>> # Create tensor from numpy array
    >>> data = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> tensor = pc.PyTensor.from_numpy(data)
    >>>
    >>> # Perform operations
    >>> result = tensor + tensor
    >>> print(result.data())  # [[2.0, 4.0], [6.0, 8.0]]
"""

# Import the main module
import pycoeus

# Re-export main classes for convenience
PyTensor = pycoeus.PyTensor
Device = pycoeus.Device

# Neural network modules
try:
    NNModule = pycoeus.NNModule
except AttributeError:
    NNModule = None

try:
    Linear = pycoeus.Linear
except AttributeError:
    Linear = None

try:
    Conv2d = pycoeus.Conv2d
except AttributeError:
    Conv2d = None

try:
    ReLU = pycoeus.ReLU
except AttributeError:
    ReLU = None

# Loss functions
MSELoss = pycoeus.MSELoss
CrossEntropyLoss = pycoeus.CrossEntropyLoss

# Optimizers
SGD = pycoeus.SGD
Adam = pycoeus.Adam

__version__ = "0.1.0"
__all__ = [
    "PyTensor",
    "Device",
    "NNModule",
    "Linear",
    "Conv2d",
    "ReLU",
    "MSELoss",
    "CrossEntropyLoss",
    "SGD",
    "Adam",
]
