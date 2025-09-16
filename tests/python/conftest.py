"""Pytest configuration and fixtures for PyCoeus testing"""

import pytest
import numpy as np
import sys
import os

# Try to import PyCoeus - it should be installed via maturin/pip in CI
try:
    import pycoeus as pc
    PYCOEUS_AVAILABLE = True
except ImportError:
    PYCOEUS_AVAILABLE = False
    pc = None

# Try to import PyTorch for compatibility testing
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None


@pytest.fixture(scope="session")
def pycoeus_available():
    """Check if PyCoeus is available for testing"""
    if not PYCOEUS_AVAILABLE:
        pytest.skip("PyCoeus not available - run 'cargo build --release' first")
    return True


@pytest.fixture(scope="session")
def pytorch_available():
    """Check if PyTorch is available for compatibility testing"""
    if not PYTORCH_AVAILABLE:
        pytest.skip("PyTorch not available - install with 'pip install torch'")
    return True


@pytest.fixture
def sample_data():
    """Provide sample data for testing"""
    return {
        'vector': [1.0, 2.0, 3.0, 4.0],
        'matrix': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'tensor_3d': list(range(24)),  # 2x3x4 tensor
        'shapes': {
            'vector': [4],
            'matrix': [2, 3],
            'tensor_3d': [2, 3, 4]
        }
    }


@pytest.fixture
def pycoeus_tensors(sample_data):
    """Create PyCoeus tensors for testing"""
    if not PYCOEUS_AVAILABLE:
        pytest.skip("PyCoeus not available")

    return {
        'vector': pc.PyTensor(sample_data['vector'], sample_data['shapes']['vector']),
        'matrix': pc.PyTensor(sample_data['matrix'], sample_data['shapes']['matrix']),
        'tensor_3d': pc.PyTensor(sample_data['tensor_3d'], sample_data['shapes']['tensor_3d'])
    }


@pytest.fixture
def torch_tensors(sample_data):
    """Create PyTorch tensors for comparison"""
    if not PYTORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    return {
        'vector': torch.tensor(sample_data['vector']),
        'matrix': torch.tensor(sample_data['matrix']).reshape(2, 3),
        'tensor_3d': torch.tensor(sample_data['tensor_3d']).reshape(2, 3, 4)
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup for each test"""
    # Reset any global state if needed
    pass


def assert_tensors_close(pycoeus_tensor, torch_tensor, rtol=1e-5, atol=1e-6):
    """Assert that PyCoeus and PyTorch tensors are numerically close"""
    if not PYTORCH_AVAILABLE:
        pytest.skip("PyTorch not available for comparison")

    pc_data = pycoeus_tensor.data()
    torch_data = torch_tensor.flatten().tolist()

    assert len(pc_data) == len(torch_data), f"Shape mismatch: {len(pc_data)} vs {len(torch_data)}"

    for pc_val, torch_val in zip(pc_data, torch_data):
        assert abs(pc_val - torch_val) <= atol + rtol * abs(torch_val), \
            f"Values differ: {pc_val} vs {torch_val}"


def assert_shapes_match(pycoeus_tensor, torch_tensor):
    """Assert that tensor shapes match"""
    pc_shape = pycoeus_tensor.shape()
    torch_shape = list(torch_tensor.shape)

    assert pc_shape == torch_shape, f"Shape mismatch: {pc_shape} vs {torch_shape}"
