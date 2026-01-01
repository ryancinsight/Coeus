#!/usr/bin/env python3
"""
Test Parity for Optimizers and Schedulers in Coeus.
"""

import sys
import os
import time

# Add the built wheel to Python path
wheel_dir = os.path.join(os.path.dirname(__file__), 'pycoeus', 'target', 'wheels')
# Also try direct import if installed in venv
try:
    import coeus
    print(f"[DEBUG] coeus file: {coeus.__file__}")
    import coeus.optim
    print(f"[DEBUG] imported coeus.optim: {coeus.optim}")
    try:
        import coeus.optim.lr_scheduler
        print(f"[DEBUG] imported coeus.optim.lr_scheduler: {coeus.optim.lr_scheduler}")
    except ImportError as e:
        print(f"[DEBUG] Failed to import coeus.optim.lr_scheduler directly: {e}")

    from coeus.optim import lr_scheduler
    print("[OK] Imported coeus.optim and lr_scheduler")
except ImportError as e:
    print(f"[FATAL] Could not import coeus: {e}")
    try:
        import coeus.optim
        print(f"[DEBUG] dir(coeus.optim): {dir(coeus.optim)}")
    except:
        print("[DEBUG] Could not import coeus.optim to inspect dir")
    print(f"[DEBUG] sys.path: {sys.path}")
    sys.exit(1)

def test_optimizer_state_dict():
    print("\n[TEST] Testing Optimizer state_dict...")
    try:
        # Create dummy tensors
        params = [coeus.tensor([1.0, 2.0], name="p1"), coeus.tensor([3.0], name="p2")]
        
        # Create optimizer
        opt = optim.Adam(params, lr=0.1)
        
        # Step once to generate state (moments)
        opt.zero_grad()
        # Mock gradients (if accessible via rust? or just step does nothing if no grad?)
        # Current Rust implementation needs gradients on params to do meaningful update
        # But step() should run regardless and initialize state (m, v) as zeros if not present
        opt.step()
        
        # Get state dict
        state = opt.state_dict()
        print(f"[OK] Got state_dict with keys: {list(state.keys())}")
        
        # Create new optimizer
        opt2 = optim.Adam(params, lr=0.01) # Different LR
        
        # Load state dict
        opt2.load_state_dict(state)
        print("[OK] Loaded state_dict")
        
        # Verify state transferred? 
        # State dict usually keys by param names or ids.
        # Rust impl keys by param name.
        
        return True
    except Exception as e:
        print(f"[ERROR] Optimizer state_dict failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_schedulers():
    print("\n[TEST] Testing Learning Rate Schedulers...")
    try:
        # StepLR
        print("Testing StepLR...")
        # Scheduler in Coeus bindings currently ignores optimizer arg for simple logic testing
        scheduler = lr_scheduler.StepLR(None, step_size=1, gamma=0.1)
        initial_lr = scheduler.get_lr()
        print(f"Initial LR: {initial_lr}")
        
        scheduler.step()
        lr_epoch1 = scheduler.get_lr()
        print(f"Epoch 1 LR: {lr_epoch1}")
        
        if abs(lr_epoch1 - (initial_lr * 0.1)) > 1e-6:
             print(f"[FAIL] StepLR decay incorrect: {lr_epoch1} != {initial_lr * 0.1}")
             return False
             
        # ExponentialLR
        print("Testing ExponentialLR...")
        scheduler = lr_scheduler.ExponentialLR(None, gamma=0.9)
        scheduler.step()
        print(f"Epoch 1 LR: {scheduler.get_lr()}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Scheduler tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Testing Coeus Optimizers & Schedulers")
    print("=" * 50)
    
    tests = [
        test_optimizer_state_dict,
        test_schedulers,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
            
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("SUCCESS")
        sys.exit(0)
    else:
        print("FAILURE")
        sys.exit(1)

if __name__ == "__main__":
    main()
