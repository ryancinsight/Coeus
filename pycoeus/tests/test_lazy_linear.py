import torch
import coeus
import numpy as np

def test_lazy_linear():
    print("Testing LazyLinear...")
    
    # Create LazyLinear with 10 output features
    lazy = coeus.nn.LazyLinear(10, bias=True)
    print(f"Created LazyLinear: {lazy}")
    
    # Verify parameters are empty initially
    params = lazy.parameters()
    print(f"Initial params len: {len(params)}")
    assert len(params) == 0, "Parameters should be empty before first forward"
    
    # Create input tensor (Batch=2, In=5)
    input_data = np.random.randn(2, 5).astype(np.float32)
    # coeus tensor from numpy (assuming from_numpy exists or we create manually)
    # If from_numpy is not yet bound for coeus, we might need a workaround.
    # But usually torch-like frameworks have it.
    # Let's check if coeus has tensor factory or from_numpy.
    # For now, let's assume we can create it via coeus.tensor or similar.
    # If not, we'll see errors.
    
    try:
        input_tensor = coeus.from_numpy(input_data)
    except AttributeError:
        # Fallback if from_numpy not top level
        input_tensor = coeus.tensor(input_data)

    print("Running forward pass...")
    output = lazy(input_tensor)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 10)
    
    # Verify parameters are now initialized
    params = lazy.parameters()
    print(f"Post-forward params len: {len(params)}")
    assert len(params) == 2, "Should have weight and bias after forward"
    
    weight = params[0]
    # Weight should be [10, 5] (out, in) or [5, 10] depending on implementation. 
    # PyTorch Linear uses [out, in]. Coeus Linear dense implementation uses [in, out] in struct but [out, in] in logic?
    # Coeus Linear logic: `weight: [input_features, output_features]`.
    # Wait, `dense.rs` says: `pub weight: Parameter<B, S, T>, // [input_features, output_features]`
    # So it is transposed compared to PyTorch (which is [out, in]).
    # Let's verify shape.
    print(f"Weight shape: {weight.shape}")
    
    print("LazyLinear test passed!")

if __name__ == "__main__":
    test_lazy_linear()
