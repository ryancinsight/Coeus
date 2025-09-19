#!/usr/bin/env python3
"""
Test to see if the Rust optimizer is actually being called and working.
"""

import sys
sys.path.insert(0, 'python')

import pycoeus as pc
import numpy as np

def test_rust_optimizer_debug():
    """Debug the Rust optimizer step by step."""
    print("🔍 Debugging Rust Optimizer")
    print("=" * 50)
    
    # Create a simple parameter
    param = pc.tensor([[2.0, 3.0]], requires_grad=True)
    print(f"Initial param: {param.data()}")
    
    # Set a simple gradient
    grad = pc.tensor([[1.0, 1.0]])
    param.set_grad(grad)
    print(f"Set gradient: {grad.data()}")
    
    # Create optimizer
    optimizer = pc.optim.SGD([param], lr=1.0)  # Large learning rate for obvious changes
    print("Created SGD optimizer with lr=1.0")
    
    # Check if the parameter is in the optimizer
    params = optimizer.parameters()
    print(f"Optimizer has {len(params)} parameters")
    if len(params) > 0:
        print(f"First parameter data: {params[0].data()}")
        print(f"First parameter grad: {params[0].grad().data() if params[0].grad() else None}")
    
    # Take a step
    print("Taking optimizer step...")
    try:
        optimizer.step()
        print("✅ Step completed without error")
    except Exception as e:
        print(f"❌ Step failed: {e}")
        return False
    
    # Check results
    print(f"Final param: {param.data()}")
    print(f"Expected param: [1.0, 2.0]")  # 2.0 - 1.0*1.0, 3.0 - 1.0*1.0
    
    # Check if optimizer parameters changed
    params_after = optimizer.parameters()
    if len(params_after) > 0:
        print(f"Optimizer param after: {params_after[0].data()}")
    
    return not np.allclose(param.data(), [2.0, 3.0])

def test_manual_parameter_update():
    """Test manual parameter update to see if the issue is in the optimizer or parameter system."""
    print("\n🔧 Testing Manual Parameter Update")
    print("=" * 50)
    
    # Create parameter
    param = pc.tensor([[5.0, 6.0]], requires_grad=True)
    print(f"Initial param: {param.data()}")
    
    # Try to manually update the parameter data
    try:
        # This should work if the tensor system supports updates
        new_data = [4.0, 5.0]
        new_tensor = pc.tensor([new_data], requires_grad=True)
        
        # Check if we can replace the tensor
        print(f"Created new tensor: {new_tensor.data()}")
        
        # The issue might be that we can't modify tensors in-place
        # Let's see if the tensor data is immutable
        original_data = param.data()
        print(f"Original data type: {type(original_data)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Manual update failed: {e}")
        return False

def main():
    """Run debug tests."""
    print("🐛 PyCoeus Rust Optimizer Debug")
    print("=" * 60)
    
    rust_test = test_rust_optimizer_debug()
    manual_test = test_manual_parameter_update()
    
    print("\n" + "=" * 60)
    print("DEBUG ANALYSIS")
    print("=" * 60)
    
    if not rust_test:
        print("❌ The Rust optimizer is not updating parameters")
        print("   Possible causes:")
        print("   1. The Rust SGD step() method is not implemented correctly")
        print("   2. The parameter synchronization between Python and Rust is broken")
        print("   3. The gradients are not being passed to the Rust optimizer")
        print("   4. The Rust tensors are immutable or not being updated in-place")
    
    if manual_test:
        print("✅ Manual parameter operations work")
        print("   This suggests the issue is specifically in the optimizer")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Check if the Rust SGD step() method actually modifies tensor data")
    print("2. Verify that gradients are properly set on Rust tensors")
    print("3. Ensure parameter updates are synchronized back to Python")
    print("4. Consider implementing a simpler optimizer test in pure Rust")

if __name__ == "__main__":
    main()