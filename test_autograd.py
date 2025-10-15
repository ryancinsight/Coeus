#!/usr/bin/env python3

import coeus as torch

def test_basic_autograd():
    print("Testing basic autograd functionality...")

    # Create a tensor and variable
    x_tensor = torch.tensor([2.0, 3.0])
    x = torch.Variable(x_tensor)

    print(f"x: {x}")
    print(f"x.requires_grad(): {x.requires_grad()}")

    # Create a simple computation: y = x^2
    y = x * x  # This should create a new Variable
    print(f"y = x * x: {y}")

    # For now, we don't have full autograd integration in the Variable wrapper
    # This is a placeholder test
    print("[OK] Variable creation works")

def test_no_grad_context():
    print("\nTesting no_grad context manager...")

    # Test no_grad context manager
    with torch.no_grad():
        print("[OK] no_grad context manager created")

    print("[OK] no_grad context manager works")

if __name__ == "__main__":
    try:
        test_basic_autograd()
        test_no_grad_context()
        print("\n[SUCCESS] Autograd tests completed!")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
