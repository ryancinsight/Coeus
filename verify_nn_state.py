import coeus as torch
import pytest

def test_sequential_state_dict():
    print("Testing Sequential state_dict...")
    model = torch.nn.Sequential()
    model.add_linear("fc1", 10, 5)
    model.add_batch_norm2d("bn1", 5)
    
    # Check parameters
    params = dict(model.named_parameters())
    assert "fc1.weight" in params
    assert "fc1.bias" in params
    assert "bn1.weight" in params # BatchNorm weight (gamma)
    
    # Check buffers
    # Coeus doesn't have explicit buffers() method exposed?
    # But state_dict should contain them
    state = model.state_dict()
    print("State dict keys:", state.keys())
    try:
        print("Named buffers:", list(model.named_buffers()))
    except Exception as e:
        print("Error calling named_buffers:", e)

    assert "fc1.weight" in state
    assert "bn1.running_mean" in state

    assert "bn1.running_var" in state
    
    # Test saving/loading
    print("Testing load_state_dict roundtrip...")
    
    # Modify state
    new_weights = torch.tensor([[1.0]*10]*5) # 5x10
    # Assign via direct parameter modification if exposed?
    # Or just modify the state dict and load
    state["fc1.weight"] = new_weights
    
    model.load_state_dict(state)
    
    # Verify loaded
    loaded_params = dict(model.named_parameters())
    # Can't compare tensors directly easily without numpy
    # Assuming .numpy() exists (it does)
    assert abs(loaded_params["fc1.weight"].numpy().mean() - 1.0) < 1e-5
    
    print("Sequential state_dict tests passed!")

def test_batchnorm_buffers():
    print("Testing BatchNorm buffers...")
    bn = torch.nn.BatchNorm2d(3)
    
    # Verify initial state
    state = bn.state_dict()
    assert "running_mean" in state
    assert "running_var" in state
    
    # Check values (zeros and ones)
    assert abs(state["running_mean"].numpy().sum()) < 1e-5
    assert abs(state["running_var"].numpy().sum() - 3.0) < 1e-5
    
    print("BatchNorm buffers test passed!")

if __name__ == "__main__":
    try:
        test_sequential_state_dict()
        test_batchnorm_buffers()
        print("All state_dict tests passed successfully!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
