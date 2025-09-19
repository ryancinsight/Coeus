#!/usr/bin/env python3
"""
Debug script to investigate why PyCoeus training is not working.

This script will test each component individually to identify the root cause.
"""

import sys
sys.path.insert(0, 'python')

import pycoeus as pc
import numpy as np

def test_gradient_support():
    """Test if gradients are properly supported."""
    print("=" * 50)
    print("TESTING GRADIENT SUPPORT")
    print("=" * 50)
    
    # Create a tensor with gradient tracking
    x = pc.tensor([[2.0, 3.0]], requires_grad=True)
    print(f"Created tensor x: {x.data()}, requires_grad: {x.requires_grad()}")
    
    # Check if gradient is initially None
    grad = x.grad()
    print(f"Initial gradient: {grad}")
    
    # Try to set a gradient manually
    try:
        grad_tensor = pc.tensor([[1.0, 1.0]])
        x.set_grad(grad_tensor)
        print("✅ Successfully set gradient manually")
        print(f"Gradient after setting: {x.grad().data() if x.grad() else None}")
    except Exception as e:
        print(f"❌ Failed to set gradient: {e}")
    
    return x

def test_parameter_access():
    """Test if we can access and modify model parameters."""
    print("\n" + "=" * 50)
    print("TESTING PARAMETER ACCESS")
    print("=" * 50)
    
    # Create a linear layer
    linear = pc.nn.Linear(2, 1)
    print(f"Created linear layer: {linear.in_features()} -> {linear.out_features()}")
    
    # Try to access weights
    try:
        weight = linear.weight()
        print(f"✅ Weight access successful: shape {weight.shape()}")
        print(f"Initial weights: {weight.data()}")
        
        # Check if weight requires grad
        print(f"Weight requires_grad: {weight.requires_grad()}")
        
        return linear, weight
    except Exception as e:
        print(f"❌ Failed to access weights: {e}")
        return linear, None

def test_optimizer_step():
    """Test if optimizer step actually updates parameters."""
    print("\n" + "=" * 50)
    print("TESTING OPTIMIZER STEP")
    print("=" * 50)
    
    # Create model and data
    linear = pc.nn.Linear(2, 1)
    weight = linear.weight()
    
    print(f"Initial weight: {weight.data()}")
    
    # Create optimizer
    optimizer = pc.optim.SGD([weight], lr=0.1)
    print("Created SGD optimizer")
    
    # Manually set a gradient
    try:
        grad_data = [0.1, 0.2]  # Simple gradient
        grad_tensor = pc.tensor([grad_data])
        weight.set_grad(grad_tensor)
        print(f"Set gradient: {grad_tensor.data()}")
        
        # Take optimizer step
        print("Taking optimizer step...")
        optimizer.step()
        
        # Check if weight changed
        new_weight = linear.weight()
        print(f"Weight after step: {new_weight.data()}")
        
        # Calculate expected change: weight = weight - lr * grad
        expected = [weight.data()[0] - 0.1 * 0.1, weight.data()[1] - 0.1 * 0.2]
        print(f"Expected weight: {expected}")
        
        # Check if weights actually changed
        old_data = weight.data()
        new_data = new_weight.data()
        
        if old_data == new_data:
            print("❌ PROBLEM: Weights did not change after optimizer step!")
            return False
        else:
            print("✅ Weights changed after optimizer step")
            return True
            
    except Exception as e:
        print(f"❌ Error during optimizer step: {e}")
        return False

def test_loss_backward():
    """Test if loss.backward() computes gradients."""
    print("\n" + "=" * 50)
    print("TESTING LOSS BACKWARD")
    print("=" * 50)
    
    # Create simple computation
    x = pc.tensor([[1.0, 2.0]], requires_grad=True)
    target = pc.tensor([[3.0]])
    
    # Create model
    linear = pc.nn.Linear(2, 1)
    
    # Forward pass
    output = linear.forward(x)
    print(f"Forward pass output: {output.data()}")
    
    # Compute loss
    loss_fn = pc.nn.MseLoss()
    loss = loss_fn.forward(output, target)
    print(f"Loss: {loss.data()}")
    
    # Try backward pass
    try:
        loss.backward()
        print("✅ Backward pass completed")
        
        # Check if gradients were computed
        weight = linear.weight()
        weight_grad = weight.grad()
        
        if weight_grad is not None:
            print(f"✅ Weight gradient computed: {weight_grad.data()}")
            return True
        else:
            print("❌ No gradient computed for weights")
            return False
            
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return False

def test_full_training_step():
    """Test a complete training step."""
    print("\n" + "=" * 50)
    print("TESTING FULL TRAINING STEP")
    print("=" * 50)
    
    # Create data
    X = pc.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
    y = pc.tensor([[5.0], [7.0]], requires_grad=False)
    
    # Create model
    model = pc.nn.Linear(2, 1)
    loss_fn = pc.nn.MseLoss()
    optimizer = pc.optim.SGD([model.weight()], lr=0.01)
    
    print("Created model, loss function, and optimizer")
    
    # Get initial state
    initial_weight = model.weight().data().copy()
    print(f"Initial weights: {initial_weight}")
    
    # Forward pass
    output = model.forward(X)
    loss = loss_fn.forward(output, y)
    initial_loss = loss.data()[0]
    print(f"Initial loss: {initial_loss}")
    
    # Backward pass
    try:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check results
        final_weight = model.weight().data()
        print(f"Final weights: {final_weight}")
        
        # Forward pass again
        output = model.forward(X)
        loss = loss_fn.forward(output, y)
        final_loss = loss.data()[0]
        print(f"Final loss: {final_loss}")
        
        # Check if anything changed
        weight_changed = not np.allclose(initial_weight, final_weight)
        loss_changed = abs(initial_loss - final_loss) > 1e-6
        
        print(f"Weight changed: {weight_changed}")
        print(f"Loss changed: {loss_changed}")
        
        if weight_changed and loss_changed:
            print("✅ Training step successful!")
            return True
        else:
            print("❌ Training step failed - no learning occurred")
            return False
            
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all debug tests."""
    print("🔍 PyCoeus Training Debug Analysis")
    print("=" * 60)
    
    results = {}
    
    # Test each component
    test_gradient_support()
    results['parameter_access'] = test_parameter_access()[1] is not None
    results['optimizer_step'] = test_optimizer_step()
    results['loss_backward'] = test_loss_backward()
    results['full_training'] = test_full_training_step()
    
    # Summary
    print("\n" + "=" * 60)
    print("DEBUG RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
    
    # Identify the root cause
    print("\n" + "=" * 60)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 60)
    
    if not results['optimizer_step']:
        print("🎯 PRIMARY ISSUE: Optimizer step() method is not updating parameters")
        print("   - The optimizer.step() call completes without error")
        print("   - But parameters remain unchanged")
        print("   - This suggests the Rust optimizer implementation is incomplete")
    
    if not results['loss_backward']:
        print("🎯 SECONDARY ISSUE: Backward pass is not computing gradients")
        print("   - loss.backward() may not be implemented")
        print("   - Gradients are not being computed for parameters")
        print("   - This prevents the optimizer from having gradients to work with")
    
    print("\n📋 REQUIRED FIXES:")
    print("1. Implement actual parameter updates in Rust optimizer step() methods")
    print("2. Implement backward pass and gradient computation")
    print("3. Ensure gradients flow from loss to parameters")
    print("4. Test the complete training loop")

if __name__ == "__main__":
    main()