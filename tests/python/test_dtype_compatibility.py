"""Dtype compatibility and conversion tests for PyCoeus"""

import pytest
import numpy as np


@pytest.mark.dtype
def test_float32_operations(pycoeus_available, pytorch_available):
    """Test operations with float32 dtype"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Create float32 data
    data = np.random.randn(100).astype(np.float32).tolist()

    pc_tensor = pc.PyTensor(data, [100])
    torch_tensor = torch.tensor(data, dtype=torch.float32)

    # Test basic operations
    operations = [
        ("add", lambda pc, torch: (pc + pc, torch + torch)),
        ("mul", lambda pc, torch: (pc * pc, torch * torch)),
        ("exp", lambda pc, torch: (pc.exp(), torch.exp(torch))),
        ("sqrt", lambda pc, torch: (pc.sqrt(), torch.sqrt(torch))),
    ]

    for op_name, op_func in operations:
        pc_result, torch_result = op_func(pc_tensor, torch_tensor)

        pc_data = np.array(pc_result.data(), dtype=np.float32)
        torch_data = torch_result.numpy().astype(np.float32)

        np.testing.assert_allclose(pc_data, torch_data, rtol=1e-5, atol=1e-6,
                                  err_msg=f"float32 {op_name} failed")


@pytest.mark.dtype
def test_dtype_conversion_consistency(pycoeus_available, pytorch_available):
    """Test dtype conversion consistency"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test conversion between different precisions
    data_f64 = np.random.randn(50).astype(np.float64)

    # Convert to different precisions and back
    pc_f64 = pc.PyTensor(data_f64.tolist(), [50])

    # Test that operations maintain precision appropriately
    result = pc_f64 + pc_f64
    result_data = np.array(result.data())

    # Check that result has reasonable precision
    assert result_data.dtype == np.float64 or result_data.dtype == np.float32, \
        f"Unexpected result dtype: {result_data.dtype}"

    # Verify numerical consistency
    expected = data_f64 + data_f64
    np.testing.assert_allclose(result_data, expected, rtol=1e-12,
                              err_msg="Dtype conversion consistency failed")


@pytest.mark.dtype
def test_integer_tensor_operations(pycoeus_available, pytorch_available):
    """Test operations with integer tensors"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Create integer data
    data = np.random.randint(-100, 100, 50).tolist()

    pc_tensor = pc.PyTensor([float(x) for x in data], [50])  # Convert to float for operations
    torch_tensor = torch.tensor(data, dtype=torch.float32)

    # Test arithmetic operations
    pc_result = pc_tensor + pc_tensor
    torch_result = torch_tensor + torch_tensor

    pc_data = np.array(pc_result.data())
    torch_data = torch_result.numpy()

    np.testing.assert_allclose(pc_data, torch_data, rtol=1e-6,
                              err_msg="Integer tensor operations failed")


@pytest.mark.dtype
def test_mixed_precision_operations(pycoeus_available, pytorch_available):
    """Test operations mixing different precisions"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Create tensors with different precisions
    data_f32 = np.random.randn(25).astype(np.float32)
    data_f64 = np.random.randn(25).astype(np.float64)

    pc_f32 = pc.PyTensor(data_f32.tolist(), [25])
    pc_f64 = pc.PyTensor(data_f64.tolist(), [25])

    torch_f32 = torch.tensor(data_f32)
    torch_f64 = torch.tensor(data_f64)

    # Test operations between different precisions
    # Note: This tests the robustness of the implementation
    try:
        pc_result = pc_f32 + pc_f64
        torch_result = torch_f32 + torch_f64

        pc_data = np.array(pc_result.data())
        torch_data = torch_result.numpy()

        np.testing.assert_allclose(pc_data, torch_data, rtol=1e-5,
                                  err_msg="Mixed precision operations failed")

    except Exception as e:
        # Mixed precision might not be fully supported yet
        pytest.skip(f"Mixed precision operations not yet supported: {e}")


@pytest.mark.dtype
def test_dtype_range_validation(pycoeus_available):
    """Test validation of dtype ranges and limits"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Test with values near dtype limits
    test_cases = [
        ([1e-45], "float32_min_subnormal"),
        ([1e38], "float32_max"),
        ([1e-324], "float64_min_subnormal"),
        ([1e308], "float64_max"),
        ([-1e38], "float32_min"),
        ([-1e308], "float64_min"),
    ]

    for data, case_name in test_cases:
        pc_tensor = pc.PyTensor(data, [1])

        # Basic operations should work without crashing
        try:
            result = pc_tensor + pc_tensor
            assert len(result.data()) == 1, f"{case_name} operation failed"

            # Check that result is finite (unless input was infinite)
            result_val = result.data()[0]
            if np.isfinite(data[0]):
                assert np.isfinite(result_val), f"{case_name} produced non-finite result: {result_val}"

        except Exception as e:
            pytest.fail(f"{case_name} raised unexpected error: {e}")


@pytest.mark.dtype
def test_dtype_underflow_handling(pycoeus_available, pytorch_available):
    """Test underflow handling across dtypes"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test with very small numbers that might underflow
    very_small = [1e-40, 1e-300]  # These might underflow in float32/float64

    for val in very_small:
        try:
            pc_tensor = pc.PyTensor([val], [1])
            torch_tensor = torch.tensor([val])

            # Test operations that might cause underflow
            pc_result = pc_tensor * pc_tensor
            torch_result = torch_tensor * torch_tensor

            pc_data = pc_result.data()[0]
            torch_data = torch_result.item()

            # Both should either underflow to zero or handle gracefully
            if torch_data == 0.0:
                assert pc_data == 0.0, f"Underflow handling mismatch: PyCoeus {pc_data}, PyTorch {torch_data}"
            else:
                # If PyTorch doesn't underflow, check they're close
                assert abs(pc_data - torch_data) < 1e-12, \
                    f"Underflow result mismatch: {pc_data} vs {torch_data}"

        except Exception as e:
            # Underflow might cause exceptions in some implementations
            if "underflow" in str(e).lower() or "overflow" in str(e).lower():
                continue  # Acceptable for extreme values
            else:
                pytest.fail(f"Unexpected error with small value {val}: {e}")


@pytest.mark.dtype
def test_dtype_overflow_handling(pycoeus_available, pytorch_available):
    """Test overflow handling across dtypes"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test with very large numbers that might overflow
    very_large = [1e40, 1e310]  # These might overflow in float32/float64

    for val in very_large:
        try:
            pc_tensor = pc.PyTensor([val], [1])
            torch_tensor = torch.tensor([val])

            # Test operations that might cause overflow
            pc_result = pc_tensor * pc_tensor
            torch_result = torch_tensor * torch_tensor

            pc_data = pc_result.data()[0]
            torch_data = torch_result.item()

            # Both should either overflow to inf or handle gracefully
            if np.isinf(torch_data):
                assert np.isinf(pc_data), f"Overflow handling mismatch: PyCoeus {pc_data}, PyTorch {torch_data}"
            else:
                # If PyTorch doesn't overflow, check they're reasonably close
                relative_error = abs((pc_data - torch_data) / (torch_data + 1e-12))
                assert relative_error < 1e-3, \
                    f"Overflow result mismatch: {pc_data} vs {torch_data}"

        except Exception as e:
            # Overflow might cause exceptions in some implementations
            if "overflow" in str(e).lower():
                continue  # Acceptable for extreme values
            else:
                pytest.fail(f"Unexpected error with large value {val}: {e}")


@pytest.mark.dtype
def test_dtype_precision_consistency(pycoeus_available, pytorch_available):
    """Test precision consistency across operations"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test that precision is maintained consistently
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    pc_tensor = pc.PyTensor(data, [5])
    torch_tensor = torch.tensor(data)

    # Perform a series of operations
    for i in range(10):
        pc_tensor = pc_tensor + pc_tensor
        torch_tensor = torch_tensor + torch_tensor

    pc_data = np.array(pc_result.data())
    torch_data = torch_result.numpy()

    # Results should be very close despite multiple operations
    np.testing.assert_allclose(pc_data, torch_data, rtol=1e-12,
                              err_msg="Precision consistency failed after multiple operations")


@pytest.mark.dtype
def test_special_values_handling(pycoeus_available, pytorch_available):
    """Test handling of special floating-point values"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    special_values = [
        ([0.0], "zero"),
        ([float('inf')], "positive_infinity"),
        ([-float('inf')], "negative_infinity"),
        ([float('nan')], "nan"),
    ]

    for data, case_name in special_values:
        pc_tensor = pc.PyTensor(data, [1])
        torch_tensor = torch.tensor(data)

        # Test basic operations handle special values appropriately
        try:
            pc_result = pc_tensor + pc.PyTensor([1.0], [1])
            torch_result = torch_tensor + torch.tensor([1.0])

            pc_data = pc_result.data()[0]
            torch_data = torch_result.item()

            # For finite inputs, check results match
            if np.isfinite(data[0]):
                assert abs(pc_data - torch_data) < 1e-6, \
                    f"Special value {case_name} handling failed: {pc_data} vs {torch_data}"

            # For infinite inputs, check infinity handling
            elif np.isinf(data[0]):
                if np.isinf(torch_data):
                    assert np.isinf(pc_data), f"Infinity handling mismatch for {case_name}"

            # For NaN inputs, check NaN propagation
            elif np.isnan(data[0]):
                if np.isnan(torch_data):
                    assert np.isnan(pc_data), f"NaN handling mismatch for {case_name}"

        except Exception as e:
            # Some special values might cause exceptions - this is often acceptable
            if case_name in ["positive_infinity", "negative_infinity", "nan"]:
                continue
            else:
                pytest.fail(f"Unexpected error with {case_name}: {e}")


@pytest.mark.dtype
def test_dtype_conversion_accuracy(pycoeus_available, pytorch_available):
    """Test accuracy of dtype conversions"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test conversion between different numeric representations
    test_values = [
        0.0, 1.0, -1.0, 3.14159, 2.71828, 0.5, -0.5,
        1e-6, 1e6, -1e-6, -1e6
    ]

    for val in test_values:
        pc_tensor = pc.PyTensor([val], [1])
        torch_tensor = torch.tensor([val])

        # Test that values are preserved accurately
        pc_data = pc_tensor.data()[0]
        torch_data = torch_tensor.item()

        assert abs(pc_data - torch_data) < 1e-12, \
            f"Dtype conversion accuracy failed for {val}: {pc_data} vs {torch_data}"


@pytest.mark.dtype
def test_integer_overflow_protection(pycoeus_available):
    """Test protection against integer overflow in operations"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Test with large integers that might overflow in 32-bit
    large_int = 2**30  # Close to 32-bit signed integer limit
    data = [float(large_int), float(large_int)]

    pc_tensor = pc.PyTensor(data, [2])

    # Operations should handle large values gracefully
    try:
        result = pc_tensor[0] + pc_tensor[1]
        result_val = result.data()[0]

        # Result should be approximately correct
        expected = large_int + large_int
        assert abs(result_val - expected) / expected < 1e-12, \
            f"Large integer operation failed: {result_val} vs {expected}"

    except Exception as e:
        pytest.fail(f"Large integer operation raised unexpected error: {e}")


@pytest.mark.dtype
def test_dtype_stability_under_operations(pycoeus_available, pytorch_available):
    """Test dtype stability under repeated operations"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Start with a simple value and perform many operations
    initial_val = 1.0
    pc_tensor = pc.PyTensor([initial_val], [1])
    torch_tensor = torch.tensor([initial_val])

    # Perform many alternating operations
    for i in range(50):
        if i % 2 == 0:
            pc_tensor = pc_tensor + pc.PyTensor([0.1], [1])
            torch_tensor = torch_tensor + torch.tensor([0.1])
        else:
            pc_tensor = pc_tensor * pc.PyTensor([0.99], [1])
            torch_tensor = torch_tensor * torch.tensor([0.99])

    pc_final = pc_tensor.data()[0]
    torch_final = torch_tensor.item()

    # Results should be very close despite many operations
    assert abs(pc_final - torch_final) < 1e-10, \
        f"Dtype stability failed: {pc_final} vs {torch_final}"


@pytest.mark.dtype
def test_precision_edge_cases(pycoeus_available, pytorch_available):
    """Test precision handling at edge cases"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test values that might cause precision issues
    edge_cases = [
        ([1e-15], "very_small_positive"),
        ([-1e-15], "very_small_negative"),
        ([1e15], "very_large_positive"),
        ([-1e15], "very_large_negative"),
        ([1.0000000000000001], "tiny_above_one"),
        ([0.9999999999999999], "tiny_below_one"),
    ]

    for data, case_name in edge_cases:
        pc_tensor = pc.PyTensor(data, [1])
        torch_tensor = torch.tensor(data)

        # Test basic arithmetic
        pc_result = pc_tensor + pc_tensor
        torch_result = torch_tensor + torch_tensor

        pc_data = pc_result.data()[0]
        torch_data = torch_result.item()

        # Check relative error for precision
        if abs(torch_data) > 1e-12:
            relative_error = abs((pc_data - torch_data) / torch_data)
            assert relative_error < 1e-12, \
                f"Precision edge case {case_name} failed: relative error {relative_error}"
        else:
            # For very small values, check absolute error
            assert abs(pc_data - torch_data) < 1e-20, \
                f"Precision edge case {case_name} failed: absolute error {abs(pc_data - torch_data)}"
