#!/usr/bin/env python3

import coeus as torch
import numpy as np

def test_basic_tensor_ops():
    print("Testing basic tensor operations...")

    # Create tensors
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    print(f"a: {a}")
    print(f"b: {b}")

    # Test addition
    c = a + b
    print(f"a + b: {c}")

    # Test shape
    print(f"a.shape: {a.shape}")
    print(f"b.shape: {b.shape}")
    print(f"c.shape: {c.shape}")

def test_tensor_creation():
    print("\nTesting tensor creation...")

    # Test different creation methods
    zeros = torch.zeros((3, 3))
    ones = torch.ones((2, 4))

    print(f"zeros: {zeros}")
    print(f"ones: {ones}")

def test_nn_modules():
    print("\nTesting neural network modules...")

    # Create a linear layer
    linear = torch.Linear(10, 5)
    print(f"Linear layer: {linear}")

    # Create input
    x = torch.ones((3, 10))
    print(f"Input shape: {x.shape}")

    # Forward pass
    output = linear.forward(x)
    print(f"Output shape: {output.shape}")

def test_optimizers():
    print("\nTesting optimizers...")

    # Create optimizer
    optimizer = torch.SGD(lr=0.01)
    print(f"Optimizer: {optimizer}")

if __name__ == "__main__":
    try:
        test_basic_tensor_ops()
        test_tensor_creation()
        test_nn_modules()
        test_optimizers()
        print("\n[SUCCESS] All tests passed!")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
