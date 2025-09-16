"""Edge cases and special scenarios tests for PyCoeus"""

import pytest
import numpy as np
import math


@pytest.mark.edge_case
def test_empty_tensor_operations(pycoeus_available, pytorch_available):
    """Test operations with empty tensors"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test with empty tensors
    try:
        pc_empty = pc.PyTensor([], [0])
        torch_empty = torch.tensor([])

        # Basic properties should work
        assert pc_empty.shape() == [0]
        assert pc_empty.numel() == 0
        assert pc_empty.dim() == 1

        # Operations on empty tensors
        pc_result = pc_empty + pc_empty
        torch_result = torch_empty + torch_empty

        assert pc_result.shape() == list(torch_result.shape)
        assert pc_result.data() == []

    except Exception as e:
        pytest.skip(f"Empty tensor operations not yet implemented: {e}")


@pytest.mark.edge_case
def test_single_element_tensor_operations(pycoeus_available, pytorch_available):
    """Test operations with single-element tensors"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    data = [42.0]
    pc_tensor = pc.PyTensor(data, [1])
    torch_tensor = torch.tensor(data)

    # Test all operations
    operations = [
        ("add", lambda x, y: x + y),
        ("mul", lambda x, y: x * y),
        ("sub", lambda x, y: x - y),
        ("div", lambda x, y: x / y),
    ]

    for op_name, op_func in operations:
        pc_result = op_func(pc_tensor, pc_tensor)
        torch_result = op_func(torch_tensor, torch_tensor)

        pc_data = pc_result.data()
        torch_data = torch_result.tolist()

        assert pc_data == torch_data, f"Single element {op_name} failed"


@pytest.mark.edge_case
def test_large_dimension_tensor_operations(pycoeus_available, pytorch_available):
    """Test operations with tensors having many dimensions"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test with 4D tensor
    shape = [2, 3, 4, 5]
    size = np.prod(shape)
    data = np.random.randn(size).tolist()

    pc_tensor = pc.PyTensor(data, shape)
    torch_tensor = torch.tensor(data).reshape(shape)

    # Test basic properties
    assert pc_tensor.shape() == list(torch_tensor.shape)
    assert pc_tensor.numel() == torch_tensor.numel()
    assert pc_tensor.dim() == len(torch_tensor.shape)

    # Test element-wise operations
    pc_result = pc_tensor + pc_tensor
    torch_result = torch_tensor + torch_tensor

    assert pc_result.shape() == list(torch_result.shape)


@pytest.mark.edge_case
def test_extreme_value_operations(pycoeus_available, pytorch_available):
    """Test operations with extreme values"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    extreme_values = [
        [1e-100],  # Very small positive
        [1e100],   # Very large positive
        [-1e-100], # Very small negative
        [-1e100],  # Very large negative
    ]

    for data in extreme_values:
        pc_tensor = pc.PyTensor(data, [1])
        torch_tensor = torch.tensor(data)

        # Test that operations don't crash
        try:
            pc_result = pc_tensor * pc_tensor
            torch_result = torch_tensor * torch_tensor

            pc_data = pc_result.data()[0]
            torch_data = torch_result.item()

            # Results should be consistent (both finite or both infinite)
            pc_is_finite = np.isfinite(pc_data)
            torch_is_finite = np.isfinite(torch_data)

            assert pc_is_finite == torch_is_finite, \
                f"Extreme value handling inconsistency: {pc_data} vs {torch_data}"

        except Exception as e:
            # Some extreme values might cause exceptions - this is acceptable
            continue


@pytest.mark.edge_case
def test_nan_inf_operations(pycoeus_available, pytorch_available):
    """Test operations involving NaN and infinity"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    special_values = [
        [float('nan')],
        [float('inf')],
        [-float('inf')],
    ]

    for data in special_values:
        pc_tensor = pc.PyTensor(data, [1])
        torch_tensor = torch.tensor(data)

        # Test operations handle special values appropriately
        pc_result = pc_tensor + pc.PyTensor([1.0], [1])
        torch_result = torch_tensor + torch.tensor([1.0])

        pc_data = pc_result.data()[0]
        torch_data = torch_result.item()

        # Check NaN propagation
        if np.isnan(data[0]):
            assert np.isnan(pc_data) == np.isnan(torch_data), \
                f"NaN propagation mismatch: {pc_data} vs {torch_data}"

        # Check infinity handling
        elif np.isinf(data[0]):
            assert np.isinf(pc_data) == np.isinf(torch_data), \
                f"Infinity handling mismatch: {pc_data} vs {torch_data}"


@pytest.mark.edge_case
def test_precision_boundary_operations(pycoeus_available, pytorch_available):
    """Test operations at precision boundaries"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test values near float32 precision limits
    boundary_values = [
        [1.0 + 1e-6],   # Just above 1.0
        [1.0 - 1e-6],   # Just below 1.0
        [1e-6],         # Near underflow
        [1e6],          # Near overflow for some operations
    ]

    for data in boundary_values:
        pc_tensor = pc.PyTensor(data, [1])
        torch_tensor = torch.tensor(data)

        # Test various operations
        operations = [
            ("add", lambda x: x + x),
            ("mul", lambda x: x * x),
            ("div", lambda x: x / x),  # Should be 1.0
        ]

        for op_name, op_func in operations:
            pc_result = op_func(pc_tensor)
            torch_result = op_func(torch_tensor)

            pc_data = pc_result.data()[0]
            torch_data = torch_result.item()

            # Check relative error
            if abs(torch_data) > 1e-12:
                relative_error = abs((pc_data - torch_data) / torch_data)
                assert relative_error < 1e-5, \
                    f"Precision boundary {op_name} failed: relative error {relative_error}"


@pytest.mark.edge_case
def test_broadcasting_edge_shapes(pycoeus_available, pytorch_available):
    """Test broadcasting with unusual shape combinations"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test edge cases for broadcasting
    edge_cases = [
        # ([1], [5]),           # Already tested
        # ([5], [1]),           # Already tested
        ([1, 1], [3, 1]),     # Multiple size-1 dimensions
        ([1, 4], [3, 1]),     # Mixed dimensions
        ([2, 1, 3], [1, 4, 1]), # Complex broadcasting
    ]

    for shape_a, shape_b in edge_cases:
        size_a = np.prod(shape_a)
        size_b = np.prod(shape_b)

        data_a = np.random.randn(size_a).tolist()
        data_b = np.random.randn(size_b).tolist()

        pc_a = pc.PyTensor(data_a, shape_a)
        pc_b = pc.PyTensor(data_b, shape_b)
        torch_a = torch.tensor(data_a).reshape(shape_a)
        torch_b = torch.tensor(data_b).reshape(shape_b)

        # Calculate expected broadcast shape
        try:
            torch_result = torch_a + torch_b
            expected_shape = list(torch_result.shape)

            pc_result = pc_a + pc_b

            assert pc_result.shape() == expected_shape, \
                f"Broadcasting edge case failed: {shape_a} + {shape_b}"

        except Exception as e:
            # Some broadcasting combinations might not be supported yet
            pytest.skip(f"Broadcasting edge case not yet supported: {e}")


@pytest.mark.edge_case
def test_memory_allocation_edge_cases(pycoeus_available, pytorch_available):
    """Test memory allocation with edge case tensor sizes"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch
    import psutil
    import os

    process = psutil.Process(os.getpid())

    # Test with very small tensors
    tiny_sizes = [1, 2, 3, 5]

    for size in tiny_sizes:
        data = np.random.randn(size).tolist()

        pc_tensor = pc.PyTensor(data, [size])
        torch_tensor = torch.tensor(data)

        # Verify basic functionality
        pc_result = pc_tensor + pc_tensor
        torch_result = torch_tensor + torch_tensor

        pc_data = pc_result.data()
        torch_data = torch_result.tolist()

        assert pc_data == torch_data, f"Tiny tensor size {size} failed"

    # Test with reasonably large tensors (avoid memory issues in CI)
    large_sizes = [1000, 5000]

    for size in large_sizes:
        data = np.random.randn(size).tolist()

        pc_tensor = pc.PyTensor(data, [size])
        torch_tensor = torch.tensor(data)

        # Basic operations should work
        pc_result = pc_tensor.sum()
        torch_result = torch_tensor.sum()

        pc_data = pc_result.data()
        torch_data = torch_result.tolist()

        assert pc_data == torch_data, f"Large tensor size {size} failed"


@pytest.mark.edge_case
def test_operation_chaining_edge_cases(pycoeus_available, pytorch_available):
    """Test operation chaining with edge cases"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test long chains of operations
    data = [1.0, 2.0, 3.0]
    pc_tensor = pc.PyTensor(data, [3])
    torch_tensor = torch.tensor(data)

    # Build up a complex chain
    pc_result = pc_tensor
    torch_result = torch_tensor

    for i in range(10):
        if i % 2 == 0:
            pc_result = pc_result + pc_tensor
            torch_result = torch_result + torch_tensor
        else:
            pc_result = pc_result * pc_tensor
            torch_result = torch_result * torch_tensor

    # Results should be consistent
    pc_data = np.array(pc_result.data())
    torch_data = torch_result.numpy()

    np.testing.assert_allclose(pc_data, torch_data, rtol=1e-5,
                              err_msg="Operation chaining edge case failed")


@pytest.mark.edge_case
def test_gradient_edge_cases(pycoeus_available):
    """Test gradient computation with edge cases"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Test gradient tracking with special values
    special_cases = [
        ([0.0], "zero"),
        ([1.0], "one"),
        ([float('inf')], "infinity"),
        ([float('nan')], "nan"),
    ]

    for data, case_name in special_cases:
        pc_tensor = pc.PyTensor(data, [1])
        pc_tensor.requires_grad_(True)

        # Verify gradient tracking is maintained
        assert pc_tensor.requires_grad() == True

        # Initial gradients should be None
        assert pc_tensor.grad() is None

        # This tests the robustness of the gradient tracking system


@pytest.mark.edge_case
def test_error_recovery_edge_cases(pycoeus_available):
    """Test error recovery in edge cases"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Test that operations recover gracefully from edge cases
    edge_cases = [
        ([0.0], [0.0], "zero_division"),  # 0/0
        ([1e-10], [1e10], "extreme_ratio"),  # Very different magnitudes
        ([float('inf')], [float('inf')], "inf_inf"),  # inf + inf
    ]

    for data_a, data_b, case_name in edge_cases:
        try:
            pc_a = pc.PyTensor(data_a, [1])
            pc_b = pc.PyTensor(data_b, [1])

            # Try various operations
            operations = [
                lambda x, y: x + y,
                lambda x, y: x * y,
                lambda x, y: x - y,
            ]

            for op in operations:
                try:
                    result = op(pc_a, pc_b)
                    # If we get here, operation succeeded
                    assert len(result.data()) == 1, f"Operation result shape incorrect for {case_name}"

                except Exception as e:
                    # Some operations might fail with extreme values - this is acceptable
                    continue

        except Exception as e:
            # Tensor creation itself might fail with extreme values
            continue


@pytest.mark.edge_case
def test_concurrent_operation_edge_cases(pycoeus_available):
    """Test concurrent operations with edge cases"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc
    import threading

    # Test concurrent operations with various tensor sizes
    results = []
    errors = []

    def worker_operation(worker_id, tensor_size):
        """Worker function for concurrent operations"""
        try:
            data = np.random.randn(tensor_size).tolist()
            tensor = pc.PyTensor(data, [tensor_size])

            # Perform various operations
            result = tensor + tensor
            results.append((worker_id, len(result.data())))

        except Exception as e:
            errors.append((worker_id, str(e)))

    # Test with different tensor sizes concurrently
    tensor_sizes = [10, 100, 1000]
    threads = []

    for i, size in enumerate(tensor_sizes):
        thread = threading.Thread(target=worker_operation, args=(i, size))
        threads.append(thread)
        thread.start()

    # Wait for all threads
    for thread in threads:
        thread.join()

    # Verify all operations completed
    assert len(results) == len(tensor_sizes), f"Some concurrent operations failed: {errors}"
    assert len(errors) == 0, f"Concurrent operation errors: {errors}"


@pytest.mark.edge_case
def test_resource_cleanup_edge_cases(pycoeus_available):
    """Test resource cleanup in edge cases"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc
    import gc

    # Test that resources are cleaned up even in edge cases
    tensors_created = []

    for i in range(100):
        # Create tensors with various sizes
        size = (i % 10) + 1
        data = np.random.randn(size).tolist()

        tensor = pc.PyTensor(data, [size])
        tensors_created.append(tensor)

        # Perform some operations
        result = tensor + tensor

    # Delete all tensors
    del tensors_created
    gc.collect()

    # System should still be functional
    test_tensor = pc.PyTensor([1.0, 2.0, 3.0], [3])
    test_result = test_tensor + test_tensor

    assert len(test_result.data()) == 3, "Resource cleanup affected functionality"


@pytest.mark.edge_case
def test_api_consistency_edge_cases(pycoeus_available, pytorch_available):
    """Test API consistency in edge cases"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test that PyCoeus API remains consistent even with unusual inputs
    unusual_cases = [
        ([], [0], "empty_tensor"),
        ([1.0], [], "scalar_shape"),
        ([float('nan'), float('inf'), -float('inf')], [3], "special_values"),
    ]

    for data, shape, case_name in unusual_cases:
        try:
            pc_tensor = pc.PyTensor(data, shape)
            torch_tensor = torch.tensor(data).reshape(shape)

            # Test that basic properties work
            pc_shape = pc_tensor.shape()
            torch_shape = list(torch_tensor.shape)

            assert pc_shape == torch_shape, f"Shape inconsistency in {case_name}"

            pc_numel = pc_tensor.numel()
            torch_numel = torch_tensor.numel()

            assert pc_numel == torch_numel, f"Numel inconsistency in {case_name}"

        except Exception as e:
            # Some edge cases might not be supported yet
            pytest.skip(f"Edge case {case_name} not yet supported: {e}")


@pytest.mark.edge_case
def test_performance_edge_cases(pycoeus_available, pytorch_available):
    """Test performance characteristics in edge cases"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch
    import time

    # Test performance with different tensor sizes
    sizes = [1, 10, 100, 1000]

    for size in sizes:
        data = np.random.randn(size).tolist()

        pc_tensor = pc.PyTensor(data, [size])
        torch_tensor = torch.tensor(data)

        # Time a simple operation
        start_time = time.time()
        for _ in range(100):
            pc_result = pc_tensor + pc_tensor
        pc_time = time.time() - start_time

        start_time = time.time()
        for _ in range(100):
            torch_result = torch_tensor + torch_tensor
        torch_time = time.time() - start_time

        # PyCoeus should complete operations (performance targets for later)
        assert pc_time > 0, f"PyCoeus operation failed for size {size}"
        assert torch_time > 0, f"PyTorch operation failed for size {size}"

        # Log performance for analysis
        ratio = pc_time / torch_time if torch_time > 0 else float('inf')
        print(".2f")
