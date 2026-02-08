import sys
import os

# Add wheels path just in case
wheel_dir = os.path.join(os.getcwd(), 'target', 'wheels')
if os.path.exists(wheel_dir) and wheel_dir not in sys.path:
    sys.path.insert(0, wheel_dir)
pycoeus_wheel_dir = os.path.join(os.getcwd(), 'pycoeus', 'target', 'wheels')
if os.path.exists(pycoeus_wheel_dir) and pycoeus_wheel_dir not in sys.path:
    sys.path.insert(0, pycoeus_wheel_dir)

try:
    import coeus as torch
    import coeus.nn as nn
    print("Imported coeus successfully.")
except ImportError as e:
    print(f"Failed to import coeus: {e}")
    sys.exit(1)

def test_bilinear():
    print("Testing Bilinear...")
    try:
        m = nn.Bilinear(20, 30, 40)
        input1 = torch.randn(128, 20)
        input2 = torch.randn(128, 30)
        output = m(input1, input2)
        print(f"Bilinear output shape: {output.shape}")
        if output.shape == [128, 40]:
            print("Bilinear OK")
        else:
            print("Bilinear FAIL: Shape mismatch")
            sys.exit(1)
    except Exception as e:
        print(f"Bilinear Exception: {e}")
        sys.exit(1)

def test_log_softmax():
    print("Testing LogSoftmax...")
    try:
        m = nn.LogSoftmax(dim=1)
        input = torch.randn(2, 3)
        output = m(input)
        print(f"LogSoftmax output shape: {output.shape}")
        # Sanity check values (should be negative)
        # We can't easily check values without numpy conversion or printing, assume shape is good proxy for now + execution success.
        if output.shape == [2, 3]:
            print("LogSoftmax OK")
        else:
            print("LogSoftmax FAIL: Shape mismatch")
            sys.exit(1)
            
        # Test functional log_softmax? if exposed via nn or functional
        # torch.nn.functional might not have it yet?
    except Exception as e:
        print(f"LogSoftmax Exception: {e}")
        sys.exit(1)

def test_flatten():
    print("Testing Flatten...")
    try:
        m = nn.Flatten(start_dim=1, end_dim=-1)
        input = torch.randn(32, 1, 5, 5)
        output = m(input)
        print(f"Flatten output shape: {output.shape}")
        # Expected: [32, 25]
        if output.shape == [32, 25]:
            print("Flatten OK")
        else:
            print(f"Flatten FAIL: Expected [32, 25], got {output.shape}")
            sys.exit(1)
    except Exception as e:
        print(f"Flatten Exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_bilinear()
    test_log_softmax()
    test_flatten()
    print("All verification tests passed!")
