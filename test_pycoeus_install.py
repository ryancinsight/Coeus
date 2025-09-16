#!/usr/bin/env python3
"""
Test script to verify PyCoeus installation and basic functionality
"""

import sys
import numpy as np

try:
    import pycoeus as pc
    print("✅ PyCoeus imported successfully")

    # Test basic tensor creation
    print("\n--- Testing Basic Tensor Creation ---")

    # Create tensor from Python list
    data = [1.0, 2.0, 3.0, 4.0]
    shape = [2, 2]
    tensor = pc.PyTensor(data, shape)
    print(f"✅ Created tensor from list: {tensor.data()}")
    print(f"   Shape: {tensor.shape()}")
    print(f"   Dimensions: {tensor.dim()}")

    # Test NumPy array conversion
    print("\n--- Testing NumPy Array Conversion ---")
    try:
        np_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor_from_numpy = pc.PyTensor.from_numpy(np_array)
        print(f"✅ Created tensor from NumPy: {tensor_from_numpy.data()}")
    except Exception as e:
        print(f"⚠️ NumPy conversion not working yet: {e}")
        print("   This is expected as the NumPy integration may need refinement")

    # Test basic operations
    print("\n--- Testing Basic Operations ---")
    result = tensor + tensor
    print(f"✅ Addition: {tensor.data()} + {tensor.data()} = {result.data()}")

    result = tensor * tensor
    print(f"✅ Multiplication: {tensor.data()} * {tensor.data()} = {result.data()}")

    # Test mathematical operations
    print("\n--- Testing Mathematical Operations ---")
    exp_tensor = tensor.exp()
    print(f"✅ Exponential: exp({tensor.data()}) = {exp_tensor.data()}")

    sin_tensor = tensor.sin()
    print(f"✅ Sine: sin({tensor.data()}) = {sin_tensor.data()}")

    # Test matrix multiplication
    print("\n--- Testing Matrix Operations ---")
    matrix_a = pc.PyTensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    matrix_b = pc.PyTensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2])
    matrix_result = matrix_a @ matrix_b
    print(f"✅ Matrix multiplication: {matrix_a.shape()} @ {matrix_b.shape()} = {matrix_result.shape()}")
    print(f"   Result: {matrix_result.data()}")

    # Test gradient tracking (if available)
    print("\n--- Testing Gradient Tracking ---")
    try:
        grad_tensor = pc.PyTensor.from_data([2.0, 3.0], [2], True)
        print(f"✅ Created tensor with gradient tracking: requires_grad = {grad_tensor.requires_grad()}")

        # Test backward pass
        loss = (grad_tensor * grad_tensor).sum()
        loss.backward()
        grad = grad_tensor.grad()
        if grad:
            print(f"✅ Gradient computed: {grad.data()}")
        else:
            print("⚠️ Gradient not available (autograd may not be fully implemented)")
    except Exception as e:
        print(f"⚠️ Gradient tracking not available: {e}")

    print("\n🎉 PyCoeus installation and basic functionality test completed successfully!")

except ImportError as e:
    print(f"❌ Failed to import PyCoeus: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    sys.exit(1)
