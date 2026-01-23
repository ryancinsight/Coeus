
import torch
import coeus
import numpy as np

def test_math_ops():
    print("Testing math operations...")
    ops = [
        "acos", "acosh", "asin", "asinh", "atan", "atanh",
        "ceil", "floor", "trunc", "frac",
        "erfc", "erfinv", "exp", "log", "sqrt",
        "sin", "cos", "tan", "sinh", "cosh", "abs"
    ]
    
    x_np = np.random.rand(5).astype(np.float32)
    # Ensure domain validity for acos/asin etc
    x_np = x_np * 0.9 
    
    t_torch = torch.from_numpy(x_np)
    t_coeus = coeus.Tensor(x_np)
    
    for op in ops:
        if not hasattr(coeus, op):
            print(f"MISSING: {op}")
            continue
            
        print(f"Testing {op}...", end="")
        try:
            res_torch = getattr(torch, op)(t_torch)
            res_coeus = getattr(coeus, op)(t_coeus)
            
            # Simple check
            assert np.allclose(res_torch.numpy(), res_coeus.numpy(), atol=1e-4)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")

if __name__ == "__main__":
    test_math_ops()
