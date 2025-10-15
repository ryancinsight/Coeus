#!/usr/bin/env python3

def test_pytorch_imports():
    print("Testing PyTorch-compatible imports...")

    # Test basic imports
    import coeus as torch
    print("[OK] import coeus as torch")

    # Test tensor operations
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.zeros([3])
    z = torch.ones([2, 3])
    print("[OK] Basic tensor creation functions")

    # Test nn submodule
    linear = torch.nn.Linear(10, 5)
    print("[OK] torch.nn.Linear import")

    # Test functional API
    relu_result = torch.nn.functional.relu(x)
    print("[OK] torch.nn.functional.relu import")

    # Test optim submodule
    sgd = torch.optim.SGD(lr=0.01)
    adam = torch.optim.Adam(lr=0.001)
    print("[OK] torch.optim.SGD and torch.optim.Adam imports")

    # Test loss functions
    mse = torch.nn.functional.mse_loss
    ce = torch.nn.functional.cross_entropy_loss
    print("[OK] Loss function imports")

    print("\n[SUCCESS] All PyTorch-compatible imports work!")

if __name__ == "__main__":
    try:
        test_pytorch_imports()
    except Exception as e:
        print(f"[ERROR] Import test failed: {e}")
        import traceback
        traceback.print_exc()
