#!/usr/bin/env python3
"""
Quick test script for PyCoeus development

This script tests the basic functionality that's actually implemented.
"""

import sys
import traceback

def test_basic_functionality():
    """Test basic PyCoeus functionality."""
    print("🧪 Testing basic PyCoeus functionality...")
    
    try:
        # Add python directory to path for testing
        import sys
        sys.path.insert(0, 'python')
        
        # Test import
        import pycoeus as pc
        print("✅ PyCoeus imported successfully")
        
        # Test tensor creation from data
        import numpy as np
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor = pc.tensor(data)
        print(f"✅ Tensor created: shape {tensor.shape()}")
        
        # Test tensor creation functions
        zeros_tensor = pc.zeros([2, 3])
        print(f"✅ Zeros tensor: shape {zeros_tensor.shape()}")
        
        ones_tensor = pc.ones([2, 3])
        print(f"✅ Ones tensor: shape {ones_tensor.shape()}")
        
        # Test tensor operations
        result = tensor + tensor
        print(f"✅ Tensor addition successful")
        
        # Test tensor methods
        sum_result = tensor.sum()
        print(f"✅ Tensor sum: {sum_result.data()}")
        
        # Test ReLU
        relu_result = tensor.relu()
        print(f"✅ ReLU operation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_neural_networks():
    """Test neural network components."""
    print("🧪 Testing neural network components...")
    
    try:
        # Add python directory to path for testing
        import sys
        sys.path.insert(0, 'python')
        
        import pycoeus as pc
        import numpy as np
        
        # Test Linear layer
        linear = pc.nn.Linear(3, 2)
        print("✅ Linear layer created")
        
        # Test forward pass
        input_data = np.random.randn(2, 3).astype(np.float32)
        input_tensor = pc.tensor(input_data)
        output = linear.forward(input_tensor)
        print(f"✅ Linear forward: {input_tensor.shape()} -> {output.shape()}")
        
        # Test ReLU activation
        relu = pc.nn.ReLU()
        activated = relu.forward(output)
        print("✅ ReLU activation successful")
        
        # Test MSE loss
        target_data = np.random.randn(2, 2).astype(np.float32)
        target_tensor = pc.tensor(target_data)
        mse_loss = pc.nn.MseLoss()
        loss = mse_loss.forward(activated, target_tensor)
        print(f"✅ MSE loss: {loss.data()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Neural network test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run quick tests."""
    print("🚀 PyCoeus Quick Test Suite")
    print("=" * 40)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Neural Networks", test_neural_networks),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())