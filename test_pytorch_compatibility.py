#!/usr/bin/env python3
"""
Test PyTorch compatibility of Coeus implementation.

This script tests the PyTorch-compatible API exposed by pycoeus.
"""

import sys
import os

# Add the built wheel to Python path
wheel_dir = os.path.join(os.path.dirname(__file__), 'target', 'wheels')
if wheel_dir not in sys.path:
    sys.path.insert(0, wheel_dir)

try:
    import coeus as torch
    import coeus.nn as nn
    print("[OK] Successfully imported coeus as torch")
except ImportError as e:
    print(f"[ERROR] Failed to import coeus: {e}")
    sys.exit(1)

def test_basic_tensor_operations():
    """Test basic tensor creation and operations."""
    print("\n[TEST] Testing basic tensor operations...")

    try:
        # Test tensor creation
        x = torch.tensor([1.0, 2.0, 3.0])
        print(f"[OK] Created tensor: {x}")

        # Test tensor operations
        y = torch.tensor([4.0, 5.0, 6.0])
        z = x + y
        print(f"[OK] Addition: {z}")

        # Test zeros, ones
        zeros = torch.zeros([3])
        ones = torch.ones([3])
        print(f"[OK] Zeros: {zeros}, Ones: {ones}")

        return True
    except Exception as e:
        print(f"[ERROR] Basic tensor operations failed: {e}")
        return False

def test_device_management():
    """Test device management."""
    print("\n[TEST] Testing device management...")

    try:
        # Test device creation
        cpu_dev = torch.device("cpu")
        print(f"[OK] CPU device: {cpu_dev}")

        cuda_dev = torch.cuda()
        print(f"[OK] CUDA device: {cuda_dev}")

        cpu_dev2 = torch.cpu()
        print(f"[OK] CPU device (alternative): {cpu_dev2}")

        return True
    except Exception as e:
        print(f"[ERROR] Device management failed: {e}")
        return False

def test_tensor_manipulation():
    """Test tensor manipulation operations."""
    print("\n[TEST] Testing tensor manipulation...")

    try:
        # Create test tensors
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([3.0, 4.0])

        # Test concatenation
        cat_result = torch.cat([t1, t2])
        print(f"[OK] Concatenation: {cat_result}")

        # Test stacking
        stack_result = torch.stack([t1, t2])
        print(f"[OK] Stacking: {stack_result}")

        # Test splitting
        split_result = torch.split(torch.tensor([1.0, 2.0, 3.0, 4.0]), 2)
        print(f"[OK] Splitting: {len(split_result)} chunks")

        # Test chunking
        chunk_result = torch.chunk(torch.tensor([1.0, 2.0, 3.0, 4.0]), 2)
        print(f"[OK] Chunking: {len(chunk_result)} chunks")

        return True
    except Exception as e:
        print(f"[ERROR] Tensor manipulation failed: {e}")
        return False

def test_neural_networks():
    """Test neural network components."""
    print("\n[TEST] Testing neural networks...")

    try:
        # Test Linear layer
        linear = nn.Linear(10, 5)
        input_tensor = torch.ones([3, 10])  # batch_size=3, input_size=10
        output = linear(input_tensor)
        print(f"[OK] Linear layer: input {input_tensor.shape} -> output {output.shape}")

        # Test activation functions
        relu = nn.ReLU()
        activated = relu(output)
        print(f"[OK] ReLU activation: {activated.shape}")

        sigmoid = nn.Sigmoid()
        sigmoid_out = sigmoid(output)
        print(f"[OK] Sigmoid activation: {sigmoid_out.shape}")

        tanh = nn.Tanh()
        tanh_out = tanh(output)
        print(f"[OK] Tanh activation: {tanh_out.shape}")

        return True
    except Exception as e:
        print(f"[ERROR] Neural networks failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_functional_api():
    """Test functional API."""
    print("\n[TEST] Testing functional API...")

    try:
        x = torch.tensor([-1.0, 0.5, 2.0])

        # Test functional activations
        relu_result = torch.nn.functional.relu(x)
        print(f"[OK] Functional ReLU: {relu_result}")

        sigmoid_result = torch.nn.functional.sigmoid(x)
        print(f"[OK] Functional Sigmoid: {sigmoid_result}")

        tanh_result = torch.nn.functional.tanh(x)
        print(f"[OK] Functional Tanh: {tanh_result}")

        return True
    except Exception as e:
        print(f"[ERROR] Functional API failed: {e}")
        return False

def main():
    """Run all compatibility tests."""
    print("Testing Coeus PyTorch Compatibility")
    print("=" * 50)

    tests = [
        test_basic_tensor_operations,
        test_device_management,
        test_tensor_manipulation,
        test_neural_networks,
        test_functional_api,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("SUCCESS: All PyTorch compatibility tests PASSED!")
        return 0
    else:
        print("FAILURE: Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
