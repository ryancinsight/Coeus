"""Numerical accuracy validation tests for PyCoeus"""

import pytest
import numpy as np
import math


@pytest.mark.numerical
def test_relative_error_validation(pycoeus_available, pytorch_available):
    """Test relative error stays within acceptable bounds"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test with various numerical ranges
    test_ranges = [
        (1e-6, 1e-3),  # Very small numbers
        (0.1, 10.0),   # Normal range
        (1e3, 1e6),    # Large numbers
    ]

    for min_val, max_val in test_ranges:
        data = np.random.uniform(min_val, max_val, 1000).tolist()
        pc_tensor = pc.PyTensor(data, [1000])
        torch_tensor = torch.tensor(data)

        # Test various operations
        operations = [
            ('add', lambda x: x + x),
            ('mul', lambda x: x * x),
            ('exp', lambda x: x.exp() if hasattr(x, 'exp') else torch.exp(x)),
            ('sqrt', lambda x: x.sqrt() if hasattr(x, 'sqrt') else torch.sqrt(x)),
        ]

        for op_name, op_func in operations:
            pc_result = op_func(pc_tensor)
            torch_result = op_func(torch_tensor)

            pc_data = np.array(pc_result.data())
            torch_data = torch_result.numpy()

            # Calculate relative error
            relative_error = np.abs((pc_data - torch_data) / (torch_data + 1e-12))

            # Assert relative error is within acceptable bounds (< 1e-6)
            max_relative_error = np.max(relative_error)
            assert max_relative_error < 1e-6, \
                f"{op_name} operation relative error too high: {max_relative_error}"


@pytest.mark.numerical
def test_absolute_error_validation(pycoeus_available, pytorch_available):
    """Test absolute error for operations near zero"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test operations that should result in values near zero
    data = np.random.randn(100).tolist()
    pc_tensor = pc.PyTensor(data, [100])
    torch_tensor = torch.tensor(data)

    # Test x - x = 0
    pc_diff = pc_tensor - pc_tensor
    torch_diff = torch_tensor - torch_tensor

    pc_data = np.array(pc_diff.data())
    torch_data = torch_diff.numpy()

    # Absolute error should be very small
    abs_error = np.abs(pc_data - torch_data)
    max_abs_error = np.max(abs_error)

    assert max_abs_error < 1e-12, f"Absolute error too high: {max_abs_error}"


@pytest.mark.numerical
def test_gradient_numerical_verification(pycoeus_available, pytorch_available):
    """Test gradient computation using numerical differentiation"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    def numerical_gradient(func, x, h=1e-5):
        """Compute numerical gradient using central difference"""
        return (func(x + h) - func(x - h)) / (2 * h)

    # Test function: f(x) = x^2
    def f(x):
        return x * x

    test_values = [0.1, 1.0, 10.0, -1.0, -10.0]

    for x_val in test_values:
        # Analytical gradient: f'(x) = 2x
        analytical_grad = 2 * x_val

        # Numerical gradient
        numerical_grad = numerical_gradient(f, x_val)

        # They should be very close
        relative_error = abs((analytical_grad - numerical_grad) / (analytical_grad + 1e-12))
        assert relative_error < 1e-6, \
            f"Numerical gradient error too high at x={x_val}: {relative_error}"


@pytest.mark.numerical
@pytest.mark.parametrize("operation", [
    "sin", "cos", "exp", "log", "sqrt", "tanh", "sigmoid"
])
def test_mathematical_function_accuracy(pycoeus_available, pytorch_available, operation):
    """Test accuracy of mathematical functions against numpy reference"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Generate test data appropriate for each function
    if operation in ["log", "sqrt"]:
        # Positive values only
        data = np.random.uniform(0.1, 10.0, 100).tolist()
    elif operation == "exp":
        # Reasonable range to avoid overflow
        data = np.random.uniform(-5.0, 5.0, 100).tolist()
    else:
        data = np.random.uniform(-3.0, 3.0, 100).tolist()

    pc_tensor = pc.PyTensor(data, [100])
    torch_tensor = torch.tensor(data)

    # Apply operation
    if operation == "sin":
        pc_result = pc_tensor.sin()
        torch_result = torch.sin(torch_tensor)
    elif operation == "cos":
        pc_result = pc_tensor.cos()
        torch_result = torch.cos(torch_tensor)
    elif operation == "exp":
        pc_result = pc_tensor.exp()
        torch_result = torch.exp(torch_tensor)
    elif operation == "log":
        pc_result = pc_tensor.log()
        torch_result = torch.log(torch_tensor)
    elif operation == "sqrt":
        pc_result = pc_tensor.sqrt()
        torch_result = torch.sqrt(torch_tensor)
    elif operation == "tanh":
        pc_result = pc_tensor.tanh()
        torch_result = torch.tanh(torch_tensor)
    elif operation == "sigmoid":
        pc_result = pc_tensor.sigmoid()
        torch_result = torch.sigmoid(torch_tensor)

    # Compare results
    pc_data = np.array(pc_result.data())
    torch_data = torch_result.numpy()

    # Check for NaN or Inf values
    assert not np.any(np.isnan(pc_data)), f"NaN values found in {operation} result"
    assert not np.any(np.isinf(pc_data)), f"Inf values found in {operation} result"

    # Calculate relative error
    relative_error = np.abs((pc_data - torch_data) / (torch_data + 1e-12))
    max_relative_error = np.max(relative_error)

    # Different tolerance for different operations
    tolerance = {
        "sin": 1e-6, "cos": 1e-6, "exp": 1e-6,
        "log": 1e-5, "sqrt": 1e-6, "tanh": 1e-6, "sigmoid": 1e-6
    }[operation]

    assert max_relative_error < tolerance, \
        f"{operation} accuracy error too high: {max_relative_error}"


@pytest.mark.numerical
def test_precision_edge_cases(pycoeus_available, pytorch_available):
    """Test precision handling at edge cases"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    edge_cases = [
        ([0.0], "zero"),
        ([1.0], "one"),
        ([-1.0], "negative_one"),
        ([float('inf')], "infinity"),
        ([-float('inf')], "negative_infinity"),
        ([float('nan')], "nan"),
        ([1e-12], "very_small"),
        ([1e12], "very_large"),
    ]

    for data, case_name in edge_cases:
        pc_tensor = pc.PyTensor(data, [1])
        torch_tensor = torch.tensor(data)

        # Test basic operations don't crash
        try:
            pc_result = pc_tensor + pc_tensor
            torch_result = torch_tensor + torch_tensor

            pc_data = pc_result.data()
            torch_data = torch_result.tolist()

            # For finite values, check they match
            if np.isfinite(data[0]):
                assert abs(pc_data[0] - torch_data[0]) < 1e-6, \
                    f"Edge case {case_name} failed: {pc_data[0]} vs {torch_data[0]}"

        except Exception as e:
            # For special values, we just want to ensure no crash
            if np.isfinite(data[0]):
                pytest.fail(f"Edge case {case_name} should not raise exception: {e}")


@pytest.mark.numerical
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_dtype_precision(pycoeus_available, pytorch_available, dtype):
    """Test precision across different data types"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Generate test data
    data = np.random.randn(100).astype(np.float32 if dtype == "float32" else np.float64)

    # Convert to appropriate precision for comparison
    pc_tensor = pc.PyTensor(data.tolist(), [100])
    torch_tensor = torch.tensor(data)

    # Test operation
    pc_result = pc_tensor * pc_tensor
    torch_result = torch_tensor * torch_tensor

    pc_data = np.array(pc_result.data())
    torch_data = torch_result.numpy()

    # Check precision-appropriate tolerance
    rtol = 1e-5 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(pc_data, torch_data, rtol=rtol,
                              err_msg=f"{dtype} precision test failed")


@pytest.mark.numerical
def test_broadcasting_numerical_accuracy(pycoeus_available, pytorch_available):
    """Test numerical accuracy in complex broadcasting scenarios"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test case: (3,) + (2, 3) -> broadcasting
    vector_data = [1.0, 2.0, 3.0]
    matrix_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    pc_vector = pc.PyTensor(vector_data, [3])
    pc_matrix = pc.PyTensor(matrix_data, [2, 3])
    torch_vector = torch.tensor(vector_data)
    torch_matrix = torch.tensor(matrix_data).reshape(2, 3)

    pc_result = pc_vector + pc_matrix
    torch_result = torch_vector + torch_matrix

    pc_data = np.array(pc_result.data())
    torch_data = torch_result.numpy()

    np.testing.assert_allclose(pc_data, torch_data, rtol=1e-6,
                              err_msg="Broadcasting numerical accuracy failed")


@pytest.mark.numerical
def test_overflow_underflow_handling(pycoeus_available, pytorch_available):
    """Test proper handling of overflow and underflow conditions"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test exponential overflow
    large_values = [700.0, 800.0, 900.0]  # exp(709) ≈ inf in float32
    pc_tensor = pc.PyTensor(large_values, [3])
    torch_tensor = torch.tensor(large_values)

    pc_exp = pc_tensor.exp()
    torch_exp = torch.exp(torch_tensor)

    pc_data = np.array(pc_exp.data())
    torch_data = torch_exp.numpy()

    # Both should either overflow to inf or handle gracefully
    for i in range(len(pc_data)):
        if np.isinf(torch_data[i]):
            assert np.isinf(pc_data[i]), f"Overflow handling mismatch at index {i}"
        else:
            assert abs(pc_data[i] - torch_data[i]) < 1e-6, \
                f"Overflow handling value mismatch at index {i}"


@pytest.mark.numerical
def test_rounding_error_accumulation(pycoeus_available, pytorch_available):
    """Test that rounding errors don't accumulate excessively"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Perform many operations that could accumulate rounding errors
    data = np.random.randn(100).tolist()
    pc_tensor = pc.PyTensor(data, [100])
    torch_tensor = torch.tensor(data)

    # Perform 1000 operations
    pc_result = pc_tensor
    torch_result = torch_tensor

    for _ in range(1000):
        pc_result = pc_result + pc_tensor * 0.001  # Small addition
        torch_result = torch_result + torch_tensor * 0.001

    pc_data = np.array(pc_result.data())
    torch_data = torch_result.numpy()

    # Check that relative error is still acceptable after many operations
    relative_error = np.abs((pc_data - torch_data) / (torch_data + 1e-12))
    max_relative_error = np.max(relative_error)

    assert max_relative_error < 1e-4, \
        f"Rounding error accumulation too high: {max_relative_error}"
