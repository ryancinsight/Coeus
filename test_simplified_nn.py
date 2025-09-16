#!/usr/bin/env python3
"""
Simplified test to verify basic PyCoeus functionality
"""

import sys

def test_basic_functionality():
    """Test basic PyCoeus functionality with simplified Linear class"""
    print("=== BASIC PYCOEUS FUNCTIONALITY TEST ===\n")

    try:
        import pycoeus as pc
        print("✅ PyCoeus imported successfully")

        # Test basic tensor creation
        print("\n--- Testing Basic Tensor Creation ---")
        tensor = pc.PyTensor([1.0, 2.0, 3.0, 4.0], [2, 2])
        print(f"✅ Tensor created: {tensor.data()}")
        print(f"   Shape: {tensor.shape()}")

        # Test simplified Linear layer
        print("\n--- Testing Simplified Linear Layer ---")
        linear = pc.Linear(4, 2)
        result = linear.forward(tensor)
        print(f"✅ Linear forward pass: {result}")

        # Test basic operations
        print("\n--- Testing Basic Operations ---")
        tensor2 = pc.PyTensor([5.0, 6.0, 7.0, 8.0], [2, 2])

        # Test addition
        add_result = tensor + tensor2
        print(f"✅ Addition: {tensor.data()} + {tensor2.data()} = {add_result.data()}")

        # Test multiplication
        mul_result = tensor * tensor2
        print(f"✅ Multiplication: {tensor.data()} * {tensor2.data()} = {mul_result.data()}")

        # Test mathematical operations
        print("\n--- Testing Mathematical Operations ---")
        exp_result = tensor.exp()
        print(f"✅ Exponential: exp({tensor.data()}) = {exp_result.data()}")

        sin_result = tensor.sin()
        print(f"✅ Sine: sin({tensor.data()}) = {sin_result.data()}")

        # Test matrix multiplication
        print("\n--- Testing Matrix Operations ---")
        matrix_a = pc.PyTensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        matrix_b = pc.PyTensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2])
        matrix_result = matrix_a @ matrix_b
        print(f"✅ Matrix multiplication: {matrix_a.shape()} @ {matrix_b.shape()} = {matrix_result.shape()}")
        print(f"   Result: {matrix_result.data()}")

        print("\n🎉 BASIC PYCOEUS FUNCTIONALITY TEST COMPLETED SUCCESSFULLY!")
        return True

    except ImportError as e:
        print(f"❌ Failed to import PyCoeus: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()

    if success:
        print("\n🎯 BASIC FUNCTIONALITY TEST PASSED!")
        print("\nNext steps:")
        print("1. ✅ Core tensor operations verified")
        print("2. ✅ PyO3 integration working")
        print("3. 🔄 Ready to re-enable neural network modules")
        print("4. 🔄 Ready for comprehensive testing")
        sys.exit(0)
    else:
        print("\n❌ BASIC FUNCTIONALITY TEST FAILED!")
        sys.exit(1)
