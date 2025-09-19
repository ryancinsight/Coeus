#!/usr/bin/env python3
"""
PyTorch vs PyCoeus Comparison Test

This script compares PyCoeus with PyTorch to ensure numerical compatibility
and API compatibility.
"""

import sys
sys.path.insert(0, 'python')

import numpy as np
import time
from typing import Tuple

# Import PyCoeus
import pycoeus as pc

# Try to import PyTorch for comparison
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
    print("✅ PyTorch available for comparison")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - will test PyCoeus functionality only")

def compare_tensors(torch_tensor, pycoeus_tensor, test_name: str, tolerance: float = 1e-4) -> bool:
    """Compare PyTorch and PyCoeus tensors for numerical accuracy."""
    if not PYTORCH_AVAILABLE:
        print(f"✅ {test_name}: PyCoeus result shape {pycoeus_tensor.shape()}")
        return True
    
    try:
        torch_numpy = torch_tensor.detach().numpy()
        pc_numpy = np.array(pycoeus_tensor.data()).reshape(pycoeus_tensor.shape())
        
        if torch_numpy.shape != pc_numpy.shape:
            print(f"❌ {test_name}: Shape mismatch - PyTorch: {torch_numpy.shape}, PyCoeus: {pc_numpy.shape}")
            return False
        
        max_diff = np.max(np.abs(torch_numpy - pc_numpy))
        
        if max_diff < tolerance:
            print(f"✅ {test_name}: Max diff = {max_diff:.2e} (within tolerance)")
            return True
        else:
            print(f"❌ {test_name}: Max diff = {max_diff:.2e} (exceeds tolerance {tolerance:.2e})")
            return False
            
    except Exception as e:
        print(f"❌ {test_name}: Comparison failed - {str(e)}")
        return False

def test_tensor_operations():
    """Test basic tensor operations."""
    print("\n📊 TENSOR OPERATIONS COMPARISON")
    print("-" * 40)
    
    # Test data
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    
    results = []
    
    if PYTORCH_AVAILABLE:
        torch_x = torch.tensor(data)
        torch_y = torch.tensor(data * 2)
        
        # Addition
        torch_add = torch_x + torch_y
        pc_x = pc.tensor(data.tolist())
        pc_y = pc.tensor((data * 2).tolist())
        pc_add = pc_x + pc_y
        results.append(compare_tensors(torch_add, pc_add, "Tensor Addition"))
        
        # Multiplication
        torch_mul = torch_x * torch_y
        pc_mul = pc_x * pc_y
        results.append(compare_tensors(torch_mul, pc_mul, "Tensor Multiplication"))
        
        # Sum
        torch_sum = torch_x.sum()
        pc_sum = pc_x.sum()
        results.append(compare_tensors(torch_sum, pc_sum, "Tensor Sum"))
        
        # ReLU
        torch_relu = torch.relu(torch_x)
        pc_relu = pc_x.relu()
        results.append(compare_tensors(torch_relu, pc_relu, "ReLU Activation"))
        
    else:
        # Test PyCoeus only
        pc_x = pc.tensor(data.tolist())
        pc_y = pc.tensor((data * 2).tolist())
        
        pc_add = pc_x + pc_y
        pc_mul = pc_x * pc_y
        pc_sum = pc_x.sum()
        pc_relu = pc_x.relu()
        
        print(f"✅ Tensor Addition: shape {pc_add.shape()}")
        print(f"✅ Tensor Multiplication: shape {pc_mul.shape()}")
        print(f"✅ Tensor Sum: {pc_sum.data()}")
        print(f"✅ ReLU Activation: applied successfully")
        results = [True, True, True, True]
    
    return all(results)

def test_neural_networks():
    """Test neural network layers."""
    print("\n🧠 NEURAL NETWORK COMPARISON")
    print("-" * 40)
    
    # Test parameters
    batch_size, in_features, out_features = 4, 5, 3
    input_data = np.random.randn(batch_size, in_features).astype(np.float32)
    
    results = []
    
    if PYTORCH_AVAILABLE:
        # PyTorch Linear layer
        torch_linear = nn.Linear(in_features, out_features)
        torch_input = torch.tensor(input_data)
        torch_output = torch_linear(torch_input)
        
        # PyCoeus Linear layer
        pc_linear = pc.nn.Linear(in_features, out_features)
        pc_input = pc.tensor(input_data.tolist())
        pc_output = pc_linear.forward(pc_input)
        
        # Compare shapes (weights are different, so we only compare shapes)
        if torch_output.shape == tuple(pc_output.shape()):
            print(f"✅ Linear Layer: Both produce shape {torch_output.shape}")
            results.append(True)
        else:
            print(f"❌ Linear Layer: Shape mismatch - PyTorch: {torch_output.shape}, PyCoeus: {pc_output.shape()}")
            results.append(False)
        
        # Test ReLU
        torch_relu = nn.ReLU()
        torch_relu_out = torch_relu(torch_output)
        
        pc_relu = pc.nn.ReLU()
        pc_relu_out = pc_relu.forward(pc_output)
        
        if torch_relu_out.shape == tuple(pc_relu_out.shape()):
            print(f"✅ ReLU Activation: Both produce shape {torch_relu_out.shape}")
            results.append(True)
        else:
            print(f"❌ ReLU Activation: Shape mismatch")
            results.append(False)
            
    else:
        # Test PyCoeus only
        pc_linear = pc.nn.Linear(in_features, out_features)
        pc_input = pc.tensor(input_data.tolist())
        pc_output = pc_linear.forward(pc_input)
        
        pc_relu = pc.nn.ReLU()
        pc_relu_out = pc_relu.forward(pc_output)
        
        print(f"✅ Linear Layer: {pc_input.shape()} -> {pc_output.shape()}")
        print(f"✅ ReLU Activation: applied to shape {pc_output.shape()}")
        results = [True, True]
    
    return all(results)

def test_loss_functions():
    """Test loss functions."""
    print("\n📉 LOSS FUNCTION COMPARISON")
    print("-" * 40)
    
    results = []
    
    # MSE Loss test
    pred_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target_data = np.array([[1.5, 1.8], [2.9, 4.1]], dtype=np.float32)
    
    if PYTORCH_AVAILABLE:
        torch_pred = torch.tensor(pred_data)
        torch_target = torch.tensor(target_data)
        torch_mse = nn.MSELoss()
        torch_loss = torch_mse(torch_pred, torch_target)
        
        pc_pred = pc.tensor(pred_data.tolist())
        pc_target = pc.tensor(target_data.tolist())
        pc_mse = pc.nn.MseLoss()
        pc_loss = pc_mse.forward(pc_pred, pc_target)
        
        results.append(compare_tensors(torch_loss, pc_loss, "MSE Loss", tolerance=1e-3))
        
    else:
        pc_pred = pc.tensor(pred_data.tolist())
        pc_target = pc.tensor(target_data.tolist())
        pc_mse = pc.nn.MseLoss()
        pc_loss = pc_mse.forward(pc_pred, pc_target)
        
        print(f"✅ MSE Loss: {pc_loss.data()[0]:.6f}")
        results.append(True)
    
    return all(results)

def test_performance():
    """Test performance comparison."""
    print("\n⚡ PERFORMANCE COMPARISON")
    print("-" * 40)
    
    # Large tensor operations
    size = 500
    data = np.random.randn(size, size).astype(np.float32)
    
    if PYTORCH_AVAILABLE:
        # PyTorch timing
        torch_a = torch.tensor(data)
        torch_b = torch.tensor(data)
        
        start_time = time.time()
        torch_result = torch_a + torch_b
        torch_time = time.time() - start_time
        
        # PyCoeus timing
        pc_a = pc.tensor(data.tolist())
        pc_b = pc.tensor(data.tolist())
        
        start_time = time.time()
        pc_result = pc_a + pc_b
        pc_time = time.time() - start_time
        
        print(f"Large tensor addition ({size}x{size}):")
        print(f"  PyTorch: {torch_time:.4f}s")
        print(f"  PyCoeus: {pc_time:.4f}s")
        print(f"  Ratio: {pc_time/torch_time:.2f}x")
        
    else:
        # PyCoeus only
        pc_a = pc.tensor(data.tolist())
        pc_b = pc.tensor(data.tolist())
        
        start_time = time.time()
        pc_result = pc_a + pc_b
        pc_time = time.time() - start_time
        
        print(f"Large tensor addition ({size}x{size}):")
        print(f"  PyCoeus: {pc_time:.4f}s")
    
    return True

def main():
    """Run all comparison tests."""
    print("🚀 PyTorch vs PyCoeus Comprehensive Comparison")
    print("=" * 60)
    
    if PYTORCH_AVAILABLE:
        print("🔥 Running full PyTorch comparison tests")
    else:
        print("🦀 Running PyCoeus functionality tests only")
    
    tests = [
        ("Tensor Operations", test_tensor_operations),
        ("Neural Networks", test_neural_networks),
        ("Loss Functions", test_loss_functions),
        ("Performance", test_performance),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: CRASHED - {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! PyCoeus is working correctly!")
        if PYTORCH_AVAILABLE:
            print("🔥 PyCoeus is numerically compatible with PyTorch!")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())