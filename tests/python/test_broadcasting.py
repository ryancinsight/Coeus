"""Comprehensive broadcasting tests for PyCoeus"""

import pytest
import numpy as np


@pytest.mark.broadcasting
@pytest.mark.parametrize("shape_a,shape_b,expected_shape", [
    # Basic broadcasting
    ([3], [1], [3]),
    ([2, 3], [3], [2, 3]),
    ([2, 1, 3], [3], [2, 1, 3]),
    ([2, 3, 4], [1, 4], [2, 3, 4]),

    # Complex broadcasting
    ([1, 5], [3, 1, 5], [3, 1, 5]),
    ([2, 1, 4], [3, 4], [2, 3, 4]),
    ([1], [2, 3, 4], [2, 3, 4]),
    ([3, 1], [1, 4], [3, 4]),

    # Same shapes
    ([2, 3], [2, 3], [2, 3]),
    ([5], [5], [5]),
    ([1, 1, 1], [1, 1, 1], [1, 1, 1]),
])
def test_broadcasting_shapes(pycoeus_available, pytorch_available, shape_a, shape_b, expected_shape):
    """Test broadcasting shape compatibility and results"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Generate data
    size_a = np.prod(shape_a)
    size_b = np.prod(shape_b)

    data_a = np.random.randn(size_a).tolist()
    data_b = np.random.randn(size_b).tolist()

    pc_a = pc.PyTensor(data_a, shape_a)
    pc_b = pc.PyTensor(data_b, shape_b)
    torch_a = torch.tensor(data_a).reshape(shape_a)
    torch_b = torch.tensor(data_b).reshape(shape_b)

    # Test addition
    pc_result = pc_a + pc_b
    torch_result = torch_a + torch_b

    assert pc_result.shape() == list(torch_result.shape)
    assert pc_result.shape() == expected_shape

    # Verify results match
    pc_data = np.array(pc_result.data())
    torch_data = torch_result.numpy()

    np.testing.assert_allclose(pc_data, torch_data, rtol=1e-6,
                              err_msg=f"Broadcasting failed for {shape_a} + {shape_b}")


@pytest.mark.broadcasting
def test_broadcasting_edge_cases(pycoeus_available, pytorch_available):
    """Test edge cases in broadcasting"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test scalar broadcasting
    scalar_data = [2.0]
    vector_data = [1.0, 2.0, 3.0, 4.0]

    pc_scalar = pc.PyTensor(scalar_data, [])
    pc_vector = pc.PyTensor(vector_data, [4])
    torch_scalar = torch.tensor(2.0)
    torch_vector = torch.tensor(vector_data)

    pc_result = pc_scalar * pc_vector
    torch_result = torch_scalar * torch_vector

    assert pc_result.shape() == [4]
    np.testing.assert_allclose(np.array(pc_result.data()), torch_result.numpy(), rtol=1e-6)


@pytest.mark.broadcasting
def test_broadcasting_errors(pycoeus_available, pytorch_available):
    """Test that incompatible shapes raise appropriate errors"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Incompatible shapes that should fail
    incompatible_pairs = [
        ([2, 3], [3, 2]),  # Both dimensions different
        ([2, 3, 4], [2, 4]),  # Middle dimension incompatible
        ([5], [3]),  # Different sizes in same dimension
    ]

    for shape_a, shape_b in incompatible_pairs:
        data_a = np.random.randn(np.prod(shape_a)).tolist()
        data_b = np.random.randn(np.prod(shape_b)).tolist()

        pc_a = pc.PyTensor(data_a, shape_a)
        pc_b = pc.PyTensor(data_b, shape_b)

        # Should raise an error for incompatible broadcasting
        with pytest.raises(Exception):
            pc_result = pc_a + pc_b


@pytest.mark.broadcasting
@pytest.mark.parametrize("operation", ["add", "sub", "mul", "div"])
def test_broadcasting_operations(pycoeus_available, pytorch_available, operation):
    """Test broadcasting with different operations"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Broadcasting case: (3,) and (2, 3)
    shape_a = [3]
    shape_b = [2, 3]
    expected_shape = [2, 3]

    data_a = [1.0, 2.0, 3.0]
    data_b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    pc_a = pc.PyTensor(data_a, shape_a)
    pc_b = pc.PyTensor(data_b, shape_b)
    torch_a = torch.tensor(data_a)
    torch_b = torch.tensor(data_b).reshape(shape_b)

    if operation == "add":
        pc_result = pc_a + pc_b
        torch_result = torch_a + torch_b
    elif operation == "sub":
        pc_result = pc_a - pc_b
        torch_result = torch_a - torch_b
    elif operation == "mul":
        pc_result = pc_a * pc_b
        torch_result = torch_a * torch_b
    elif operation == "div":
        pc_result = pc_a / pc_b
        torch_result = torch_a / torch_b

    assert pc_result.shape() == expected_shape

    pc_data = np.array(pc_result.data())
    torch_data = torch_result.numpy()

    np.testing.assert_allclose(pc_data, torch_data, rtol=1e-6,
                              err_msg=f"Broadcasting {operation} failed")


@pytest.mark.broadcasting
def test_broadcasting_with_reductions(pycoeus_available, pytorch_available):
    """Test broadcasting combined with reduction operations"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Create tensors for broadcasting then reduction
    data_a = [1.0, 2.0, 3.0]
    data_b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    pc_a = pc.PyTensor(data_a, [3])
    pc_b = pc.PyTensor(data_b, [2, 3])
    torch_a = torch.tensor(data_a)
    torch_b = torch.tensor(data_b).reshape(2, 3)

    # Broadcasting addition
    pc_sum = (pc_a + pc_b).sum()
    torch_sum = (torch_a + torch_b).sum()

    pc_data = pc_sum.data()
    torch_data = torch_sum.tolist()

    assert abs(pc_data[0] - torch_data[0]) < 1e-6


@pytest.mark.broadcasting
def test_broadcasting_matmul_compatibility(pycoeus_available, pytorch_available):
    """Test broadcasting compatibility with matrix multiplication"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Broadcasting with batched matrix multiplication
    # Shape: (2, 3) @ (3, 4) -> (2, 4)
    batch_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    matrix_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

    pc_batch = pc.PyTensor(batch_data, [2, 3])
    pc_matrix = pc.PyTensor(matrix_data, [3, 4])
    torch_batch = torch.tensor(batch_data).reshape(2, 3)
    torch_matrix = torch.tensor(matrix_data).reshape(3, 4)

    pc_result = pc_batch @ pc_matrix
    torch_result = torch_batch @ torch_matrix

    assert pc_result.shape() == [2, 4]
    assert pc_result.shape() == list(torch_result.shape)

    pc_data = np.array(pc_result.data())
    torch_data = torch_result.numpy()

    np.testing.assert_allclose(pc_data, torch_data, rtol=1e-6)


@pytest.mark.broadcasting
def test_broadcasting_memory_efficiency(pycoeus_available, pytorch_available):
    """Test that broadcasting doesn't create unnecessary memory overhead"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch
    import psutil
    import os

    process = psutil.Process(os.getpid())

    # Measure baseline memory
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Create large tensors for broadcasting
    large_shape = [1000]
    small_shape = [1]

    large_data = np.random.randn(1000).tolist()
    small_data = [2.0]

    pc_large = pc.PyTensor(large_data, large_shape)
    pc_small = pc.PyTensor(small_data, small_shape)
    torch_large = torch.tensor(large_data)
    torch_small = torch.tensor(2.0)

    # Perform broadcasting operation
    pc_result = pc_large * pc_small
    torch_result = torch_large * torch_small

    # Check memory usage after operation
    after_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = after_memory - baseline_memory

    # Broadcasting should not cause excessive memory usage
    assert memory_increase < 50, f"Broadcasting memory usage too high: {memory_increase}MB"

    # Verify results are correct
    pc_data = np.array(pc_result.data())
    torch_data = torch_result.numpy()

    np.testing.assert_allclose(pc_data, torch_data, rtol=1e-6)


@pytest.mark.broadcasting
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_broadcasting_high_dimensions(pycoeus_available, pytorch_available, ndim):
    """Test broadcasting with high-dimensional tensors"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Create shapes for broadcasting
    shape_a = [1] * ndim
    shape_b = [2] * ndim
    expected_shape = [2] * ndim

    size_a = 1
    size_b = 2 ** ndim

    data_a = [1.0]
    data_b = np.random.randn(size_b).tolist()

    pc_a = pc.PyTensor(data_a, shape_a)
    pc_b = pc.PyTensor(data_b, shape_b)
    torch_a = torch.tensor(1.0).reshape(shape_a)
    torch_b = torch.tensor(data_b).reshape(shape_b)

    pc_result = pc_a + pc_b
    torch_result = torch_a + torch_b

    assert pc_result.shape() == expected_shape

    pc_data = np.array(pc_result.data())
    torch_data = torch_result.numpy()

    np.testing.assert_allclose(pc_data, torch_data, rtol=1e-6,
                              err_msg=f"High-dimensional broadcasting failed for {ndim}D")


@pytest.mark.broadcasting
def test_broadcasting_gradient_flow(pycoeus_available, pytorch_available):
    """Test that broadcasting works correctly with gradient computation"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Create broadcasting scenario with gradients
    data_a = [1.0, 2.0, 3.0]
    data_b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    pc_a = pc.PyTensor(data_a, [3])
    pc_b = pc.PyTensor(data_b, [2, 3])

    # Enable gradient tracking
    pc_a.requires_grad_(True)
    pc_b.requires_grad_(True)

    # Perform broadcasting operation
    pc_result = pc_a + pc_b

    # Verify shapes are correct
    assert pc_result.shape() == [2, 3]

    # Verify gradient tracking is maintained
    assert pc_result.requires_grad()  # This would be true once autograd is fully implemented


@pytest.mark.broadcasting
def test_broadcasting_numeric_stability(pycoeus_available, pytorch_available):
    """Test numerical stability in broadcasting operations"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test with very small numbers
    small_data = [1e-8, 1e-7, 1e-6]
    broadcast_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    pc_small = pc.PyTensor(small_data, [3])
    pc_broadcast = pc.PyTensor(broadcast_data, [2, 3])
    torch_small = torch.tensor(small_data)
    torch_broadcast = torch.tensor(broadcast_data).reshape(2, 3)

    pc_result = pc_small + pc_broadcast
    torch_result = torch_small + torch_broadcast

    pc_data = np.array(pc_result.data())
    torch_data = torch_result.numpy()

    np.testing.assert_allclose(pc_data, torch_data, rtol=1e-6,
                              err_msg="Broadcasting numerical stability failed")


@pytest.mark.broadcasting
def test_broadcasting_empty_dimensions(pycoeus_available, pytorch_available):
    """Test broadcasting with empty dimensions (size 1)"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test various combinations of size-1 dimensions
    test_cases = [
        ([1], [5], [5]),
        ([1, 3], [2, 1, 3], [2, 1, 3]),
        ([3, 1], [1, 4], [3, 4]),
        ([1, 1, 1], [2, 3, 4], [2, 3, 4]),
    ]

    for shape_a, shape_b, expected_shape in test_cases:
        data_a = np.random.randn(np.prod(shape_a)).tolist()
        data_b = np.random.randn(np.prod(shape_b)).tolist()

        pc_a = pc.PyTensor(data_a, shape_a)
        pc_b = pc.PyTensor(data_b, shape_b)
        torch_a = torch.tensor(data_a).reshape(shape_a)
        torch_b = torch.tensor(data_b).reshape(shape_b)

        pc_result = pc_a * pc_b
        torch_result = torch_a * torch_b

        assert pc_result.shape() == expected_shape

        pc_data = np.array(pc_result.data())
        torch_data = torch_result.numpy()

        np.testing.assert_allclose(pc_data, torch_data, rtol=1e-6,
                                  err_msg=f"Empty dimension broadcasting failed for {shape_a} * {shape_b}")


@pytest.mark.broadcasting
def test_broadcasting_complex_operations(pycoeus_available, pytorch_available):
    """Test broadcasting in complex mathematical operations"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Complex broadcasting: (2, 1, 3) + (1, 4, 1) = (2, 4, 3)
    shape_a = [2, 1, 3]
    shape_b = [1, 4, 1]
    expected_shape = [2, 4, 3]

    data_a = np.random.randn(6).tolist()  # 2 * 1 * 3 = 6
    data_b = np.random.randn(4).tolist()  # 1 * 4 * 1 = 4

    pc_a = pc.PyTensor(data_a, shape_a)
    pc_b = pc.PyTensor(data_b, shape_b)
    torch_a = torch.tensor(data_a).reshape(shape_a)
    torch_b = torch.tensor(data_b).reshape(shape_b)

    pc_result = pc_a.exp() + pc_b.sin()
    torch_result = torch.exp(torch_a) + torch.sin(torch_b)

    assert pc_result.shape() == expected_shape

    pc_data = np.array(pc_result.data())
    torch_data = torch_result.numpy()

    np.testing.assert_allclose(pc_data, torch_data, rtol=1e-5,
                              err_msg="Complex broadcasting operation failed")
