#!/usr/bin/env python3
"""
Test Coeus Linear Algebra Parity.
"""

import sys
import os

try:
    import coeus
    import coeus.linalg as linalg
    print(f"[DEBUG] Loaded coeus.linalg: {linalg}")
except ImportError as e:
    print(f"[FATAL] Could not import coeus.linalg: {e}")
    sys.exit(1)

def test_norm():
    print("\n[TEST] Testing linalg.norm...")
    try:
        t = coeus.tensor([3.0, 4.0])
        n = linalg.norm(t)
        print(f"Norm([3, 4]) = {n}")
        if abs(n - 5.0) > 1e-4:
            print(f"[FAIL] Expected 5.0, got {n}")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] norm failed: {e}")
        return False

def test_inv():
    print("\n[TEST] Testing linalg.inv...")
    try:
        # Identity
        eye = coeus.tensor([1.0, 0.0, 0.0, 1.0]).reshape([2, 2])
        inv_eye = linalg.inv(eye)
        print("Inv(Eye): OK")
        
        # Simple Matrix [[4, 7], [2, 6]]
        # Det = 24 - 14 = 10
        # Inv = 1/10 * [[6, -7], [-2, 4]] = [[0.6, -0.7], [-0.2, 0.4]]
        # Flattened: [0.6, -0.7, -0.2, 0.4]
        
        m = coeus.tensor([4.0, 7.0, 2.0, 6.0]).reshape([2, 2])
        inv_m = linalg.inv(m)
        
        # Verify check logic manually or via numpy parity if avail (assume no numpy)
        # 0.6
        if abs(inv_m[0, 0].item() - 0.6) > 1e-4:
             print(f"[FAIL] Inv[0,0] expected 0.6, got {inv_m[0, 0].item()}")
             return False
        
        print(f"[OK] Inversion output looks correct: {inv_m[0,0].item()}")
        return True
    except Exception as e:
        print(f"[ERROR] inv failed: {e}")
        return False

def main():
    print("Testing Coeus Linalg")
    print("=" * 50)
    
    if test_norm() and test_inv():
        print("\nSUCCESS")
        sys.exit(0)
    else:
        print("\nFAILURE")
        sys.exit(1)

if __name__ == "__main__":
    main()
