#!/usr/bin/env python3
"""
Installation test for PyCoeus

This script tests that PyCoeus is properly installed and working.
"""

import sys
import traceback

def test_basic_import():
    """Test basic import functionality."""
    print("🧪 Testing basic import...")
    try:
        import pycoeus as pc
        print("✅ PyCoeus imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import PyCoeus: {e}")
        return False

def test_tensor_operations():
    """Test basic tensor operations."""
    print("🧪 Testing tensor operations...")
    try:
        import pycoeus as pc
        import numpy as np
        
        # Test tensor creation
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor = pc.tensor(data, requires_grad=True)
        print(f"✅ Tensor created with shape: {tensor.shape()}")
        
        # Test tensor operations
        result = tensor + tensor
        print(f"✅ Tensor addition successful")
        
        # Test tensor methods
        sum_result = tensor.sum()
        print(f"✅ Tensor sum: {sum_result.data()}")
        
        return True
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        traceback.print_exc()
        return False

def test_neural_networks():
    """Test neural network modules."""
    print("🧪 Testing neural network modules...")
    try:
        import pycoeus as pc
        import numpy as np
        
        # Test Linear layer
        linear = pc.nn.Linear(4, 2)
        input_data = np.random.randn(3, 4).astype(np.float32)
        input_tensor = pc.tensor(input_data)
        output = linear.forward(input_tensor)
        print(f"✅ Linear layer: {input_tensor.shape()} -> {output.shape()}")
        
        # Test activation
        relu = pc.nn.ReLU()
        activated = relu.forward(output)
        print(f"✅ ReLU activation successful")
        
        # Test loss function
        target_data = np.random.randn(3, 2).astype(np.float32)
        target_tensor = pc.tensor(target_data)
        loss_fn = pc.nn.MseLoss()  # Note: using MseLoss not MSELoss
        loss = loss_fn.forward(activated, target_tensor)
        print(f"✅ MSE loss computed: {loss.data()}")
        
        return True
    except Exception as e:
        print(f"❌ Neural network test failed: {e}")
        traceback.print_exc()
        return False

def test_optimizers():
    """Test optimizer functionality."""
    print("🧪 Testing optimizers...")
    try:
        import pycoeus as pc
        import numpy as np
        
        # Create a simple parameter
        param_data = np.random.randn(2, 3).astype(np.float32)
        param = pc.tensor(param_data, requires_grad=True)
        
        # Test SGD optimizer (using actual Rust implementation)
        sgd = pc.optim.SGD([param], lr=0.01)
        print("✅ SGD optimizer created")
        
        # Test Adam optimizer (using actual Rust implementation)
        adam = pc.optim.Adam([param], lr=0.001)
        print("✅ Adam optimizer created")
        
        return True
    except Exception as e:
        print(f"❌ Optimizer test failed: {e}")
        traceback.print_exc()
        return False

def test_utilities():
    """Test utility functions."""
    print("🧪 Testing utilities...")
    try:
        import pycoeus as pc
        
        # Test utility functions
        num_threads = pc.get_num_threads()
        print(f"✅ Number of threads: {num_threads}")
        
        cuda_available = pc.cuda_is_available()
        print(f"✅ CUDA available: {cuda_available}")
        
        # Test random seed
        pc.manual_seed(42)
        print("✅ Random seed set")
        
        return True
    except Exception as e:
        print(f"❌ Utility test failed: {e}")
        traceback.print_exc()
        return False

def test_comprehensive_workflow():
    """Test a comprehensive ML workflow."""
    print("🧪 Testing comprehensive workflow...")
    try:
        import pycoeus as pc
        import numpy as np
        
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100, 1).astype(np.float32)
        
        # Create tensors
        X_tensor = pc.tensor(X, requires_grad=False)
        y_tensor = pc.tensor(y, requires_grad=False)
        
        # Create model
        model_layers = [
            pc.nn.Linear(5, 10),
            pc.nn.ReLU(),
            pc.nn.Linear(10, 1)
        ]
        
        # Forward pass through layers
        x = X_tensor
        for layer in model_layers:
            x = layer.forward(x)
        
        # Compute loss
        loss_fn = pc.nn.MSELoss()
        loss = loss_fn.forward(x, y_tensor)
        
        print(f"✅ Comprehensive workflow completed")
        print(f"   Input shape: {X_tensor.shape()}")
        print(f"   Output shape: {x.shape()}")
        print(f"   Loss: {loss.data()}")
        
        return True
    except Exception as e:
        print(f"❌ Comprehensive workflow failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 PyCoeus Installation Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Import", test_basic_import),
        ("Tensor Operations", test_tensor_operations),
        ("Neural Networks", test_neural_networks),
        ("Optimizers", test_optimizers),
        ("Utilities", test_utilities),
        ("Comprehensive Workflow", test_comprehensive_workflow),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PyCoeus is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())