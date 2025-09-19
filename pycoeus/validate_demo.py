#!/usr/bin/env python3
"""
PyCoeus Demo Validation Script

This script validates that the demo runs successfully and all features work.
"""

import sys
import subprocess
import time

def run_demo():
    """Run the demo and check if it completes successfully."""
    print("🧪 Validating PyCoeus Demo...")
    print("=" * 40)
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, "demo.py"], 
                              capture_output=True, text=True, cwd=".")
        end_time = time.time()
        
        print(f"Demo execution time: {end_time - start_time:.2f}s")
        print(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ Demo completed successfully!")
            
            # Check for key success indicators in output
            output = result.stdout
            success_indicators = [
                "Demo Results: 9/9 sections completed successfully",
                "PyCoeus Demo Completed Successfully!",
                "All PyTorch-compatible features working correctly"
            ]
            
            all_found = True
            for indicator in success_indicators:
                if indicator in output:
                    print(f"✅ Found: {indicator}")
                else:
                    print(f"❌ Missing: {indicator}")
                    all_found = False
            
            if all_found:
                print("\n🎉 All validation checks passed!")
                return True
            else:
                print("\n⚠️ Some validation checks failed")
                return False
                
        else:
            print("❌ Demo failed with errors:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Failed to run demo: {e}")
        return False

def main():
    """Main validation function."""
    print("🚀 PyCoeus Demo Validation")
    print("=" * 40)
    
    if run_demo():
        print("\n✅ PyCoeus demo validation PASSED")
        return 0
    else:
        print("\n❌ PyCoeus demo validation FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())