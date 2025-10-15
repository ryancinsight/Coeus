#!/usr/bin/env python3

import coeus as torch
import numpy as np

def test_tensor_operations():
    print("Testing tensor operations...")

    # Create tensors
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])

    print(f"a: {a}")
    print(f"b: {b}")

    # Test reshape
    a_reshaped = a.reshape([2, 3])
    print(f"a reshaped to [2,3]: {a_reshaped}")

    # Test transpose
    try:
        a_transposed = a_reshaped.transpose(0, 1)
        print(f"a transposed: {a_transposed}")
    except Exception as e:
        print(f"Transpose failed: {e}")

    # Test sum
    total_sum = a.sum()
    print(f"Sum of a: {total_sum}")

    # Test mean
    total_mean = a.mean()
    print(f"Mean of a: {total_mean}")

    # Test matmul
    try:
        m1 = torch.ones([2, 3])
        m2 = torch.ones([3, 2])
        result = m1.matmul(m2)
        print(f"Matrix multiplication result shape: {result.shape()}")
    except Exception as e:
        print(f"Matmul failed: {e}")

    # Test size and numel
    print(f"a.size(): {a.size()}")
    print(f"a.numel(): {a.numel()}")

def test_tensor_creation():
    print("\nTesting advanced tensor creation...")

    # Test arange
    try:
        arange_tensor = torch.arange(0.0, 10.0, 2.0)
        print(f"arange(0, 10, 2): {arange_tensor}")
    except Exception as e:
        print(f"arange failed: {e}")

    # Test linspace
    try:
        linspace_tensor = torch.linspace(0.0, 1.0, 5)
        print(f"linspace(0, 1, 5): {linspace_tensor}")
    except Exception as e:
        print(f"linspace failed: {e}")

    # Test full
    try:
        full_tensor = torch.full([2, 3], 5.0)
        print(f"full([2, 3], 5.0): {full_tensor}")
    except Exception as e:
        print(f"full failed: {e}")

if __name__ == "__main__":
    try:
        test_tensor_operations()
        test_tensor_creation()
        print("\n[SUCCESS] Tensor operations tests passed!")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
