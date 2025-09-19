#!/usr/bin/env python3
"""
Test script to fix the optimizer step issue.
"""

import sys
sys.path.insert(0, 'python')

import pycoeus as pc
import numpy as np

def test_parameter_requires_grad():
    """Test if model parameters have requires_grad enabled."""
    print("=" * 50)
    print("TESTING PARAMETER REQUIRES_GRAD")
    print("=" * 50)
    
    # Create a linear layer
    linear = pc.nn.Linear(2, 1)
    weight = linear.weight()
    
    print(f"Weight requires_grad: {weight.requires_grad()}")
    
    # Try to enable requires_grad
    try:
        weight.requires_grad_(True)
        print(f"After enabling: {weight.requires_grad()}")
        return weight
    except Exception as e:
        print(f"Failed to enable requires_grad: {e}")
        return weight

def test_direct_optimizer():
    """Test optimizer with a simple tensor that has requires_grad=True."""
    print("\n" + "=" * 50)
    print("TESTING DIRECT OPTIMIZER")
    print("=" * 50)
    
    # Create a tensor with requires_grad=True
    param = pc.tensor([[1.0, 2.0]], requires_grad=True)
    print(f"Created param: {param.data()}, requires_grad: {param.requires_grad()}")
    
    # Create optimizer
    optimizer = pc.optim.SGD([param], lr=0.1)
    
    # Set gradient manually
    grad = pc.tensor([[0.1, 0.2]])
    param.set_grad(grad)
    print(f"Set gradient: {grad.data()}")
    
    # Get initial value
    initial_data = param.data().copy()
    print(f"Initial param: {initial_data}")
    
    # Take optimizer step
    optimizer.step()
    
    # Check if parameter changed
    final_data = param.data()
    print(f"Final param: {final_data}")
    
    # Calculate expected change
    expected = [initial_data[0] - 0.1 * 0.1, initial_data[1] - 0.1 * 0.2]
    print(f"Expected param: {expected}")
    
    changed = not np.allclose(initial_data, final_data)
    print(f"Parameter changed: {changed}")
    
    return changed

def test_model_parameter_training():
    """Test training with model parameters that have requires_grad=True."""
    print("\n" + "=" * 50)
    print("TESTING MODEL PARAMETER TRAINING")
    print("=" * 50)
    
    # Create model
    linear = pc.nn.Linear(2, 1)
    weight = linear.weight()
    
    # Enable requires_grad for the weight
    weight.requires_grad_(True)
    print(f"Weight requires_grad: {weight.requires_grad()}")
    
    # Create optimizer with the weight that has requires_grad=True
    optimizer = pc.optim.SGD([weight], lr=0.01)
    
    # Create data
    X = pc.tensor([[1.0, 2.0]])
    y = pc.tensor([[3.0]])
    
    # Get initial state
    initial_weight = weight.data().copy()
    print(f"Initial weight: {initial_weight}")
    
    # Forward pass
    output = linear.forward(X)
    print(f"Output: {output.data()}")
    
    # Compute loss
    loss_fn = pc.nn.MseLoss()
    loss = loss_fn.forward(output, y)
    print(f"Loss: {loss.data()}")
    
    # Manual gradient computation (since backward might not work)
    # For MSE loss: d_loss/d_output = 2 * (output - target)
    # For linear layer: d_output/d_weight = input
    # So: d_loss/d_weight = d_loss/d_output * d_output/d_weight = 2 * (output - target) * input
    
    output_val = output.data()[0]
    target_val = y.data()[0]
    input_vals = X.data()
    
    # Gradient calculation
    d_loss_d_output = 2 * (output_val - target_val)
    d_loss_d_weight = [d_loss_d_output * input_vals[0], d_loss_d_output * input_vals[1]]
    
    print(f"Manual gradient calculation: {d_loss_d_weight}")
    
    # Set the gradient manually
    grad_tensor = pc.tensor([d_loss_d_weight])
    weight.set_grad(grad_tensor)
    print(f"Set gradient: {grad_tensor.data()}")
    
    # Take optimizer step
    optimizer.step()
    
    # Check results
    final_weight = weight.data()
    print(f"Final weight: {final_weight}")
    
    # Calculate expected weight
    expected_weight = [
        initial_weight[0] - 0.01 * d_loss_d_weight[0],
        initial_weight[1] - 0.01 * d_loss_d_weight[1]
    ]
    print(f"Expected weight: {expected_weight}")
    
    # Check if weight changed
    weight_changed = not np.allclose(initial_weight, final_weight)
    print(f"Weight changed: {weight_changed}")
    
    # Forward pass again to see if loss changed
    output2 = linear.forward(X)
    loss2 = loss_fn.forward(output2, y)
    print(f"New loss: {loss2.data()}")
    
    loss_changed = abs(loss.data()[0] - loss2.data()[0]) > 1e-6
    print(f"Loss changed: {loss_changed}")
    
    return weight_changed and loss_changed

def main():
    """Run all tests."""
    print("🔧 PyCoeus Optimizer Fix Testing")
    print("=" * 60)
    
    # Test 1: Parameter requires_grad
    weight = test_parameter_requires_grad()
    
    # Test 2: Direct optimizer test
    direct_success = test_direct_optimizer()
    
    # Test 3: Model parameter training
    model_success = test_model_parameter_training()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"Direct optimizer test: {'✅ PASS' if direct_success else '❌ FAIL'}")
    print(f"Model parameter test:  {'✅ PASS' if model_success else '❌ FAIL'}")
    
    if direct_success and model_success:
        print("\n🎉 SUCCESS: Optimizer is working correctly!")
    elif direct_success:
        print("\n⚠️ PARTIAL: Direct optimizer works, but model parameters need requires_grad=True")
    else:
        print("\n❌ FAILURE: Optimizer step() method is not updating parameters")
        print("   The Rust optimizer implementation needs to be fixed.")

if __name__ == "__main__":
    main()