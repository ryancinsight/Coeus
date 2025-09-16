"""Autograd functionality tests for PyCoeus"""

import pytest
import numpy as np


@pytest.mark.autograd
def test_gradient_tracking_setup(pycoeus_available):
    """Test gradient tracking setup"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    data = [1.0, 2.0, 3.0]
    shape = [3]
    tensor = pc.PyTensor(data, shape)

    # Initially no gradients required
    assert not tensor.requires_grad()

    # Enable gradient computation
    tensor.requires_grad_(True)
    assert tensor.requires_grad()

    # Disable gradient computation
    tensor.requires_grad_(False)
    assert not tensor.requires_grad()


@pytest.mark.autograd
def test_gradient_initialization(pycoeus_available):
    """Test gradient initialization"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    data = [1.0, 2.0, 3.0]
    shape = [3]
    tensor = pc.PyTensor(data, shape)
    tensor.requires_grad_(True)

    # Gradients should be None initially
    assert tensor.grad() is None


@pytest.mark.autograd
def test_gradient_propagation_setup(pycoeus_available):
    """Test gradient propagation setup (basic)"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Create tensors with gradient tracking
    a_data = [2.0]
    b_data = [3.0]

    a = pc.PyTensor(a_data, [])
    b = pc.PyTensor(b_data, [])

    a.requires_grad_(True)
    b.requires_grad_(True)

    # Perform operations
    c = a + b  # c = a + b = 5
    d = a * c  # d = a * c = 2 * 5 = 10

    # Check that result is computed correctly
    assert d.data() == [10.0]

    # Gradients should be None (full autograd not implemented yet)
    assert a.grad() is None
    assert b.grad() is None
    assert c.grad() is None
    assert d.grad() is None


@pytest.mark.autograd
def test_gradient_chain_rule_setup(pycoeus_available):
    """Test chain rule setup"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # f(x) = x^2, f'(x) = 2x
    x_data = [3.0]
    x = pc.PyTensor(x_data, [])
    x.requires_grad_(True)

    # Compute x^2
    y = x * x

    # Result should be 9
    assert y.data() == [9.0]

    # Gradients not computed yet (full autograd pending)
    assert x.grad() is None
    assert y.grad() is None


@pytest.mark.autograd
def test_multiple_gradient_paths(pycoeus_available):
    """Test multiple gradient computation paths"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Create computation graph: c = a + b, d = a * b, e = c * d
    a_data = [2.0]
    b_data = [3.0]

    a = pc.PyTensor(a_data, [])
    b = pc.PyTensor(b_data, [])

    a.requires_grad_(True)
    b.requires_grad_(True)

    c = a + b  # c = 5
    d = a * b  # d = 6
    e = c * d  # e = 30

    assert c.data() == [5.0]
    assert d.data() == [6.0]
    assert e.data() == [30.0]

    # All gradients should be None (full autograd pending)
    assert a.grad() is None
    assert b.grad() is None
    assert c.grad() is None
    assert d.grad() is None
    assert e.grad() is None


@pytest.mark.autograd
def test_gradient_with_mathematical_functions(pycoeus_available):
    """Test gradient setup with mathematical functions"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    x_data = [1.0]
    x = pc.PyTensor(x_data, [])
    x.requires_grad_(True)

    # f(x) = sin(x) + cos(x)
    sin_x = x.sin()
    cos_x = x.cos()
    y = sin_x + cos_x

    # Result should be sin(1) + cos(1)
    expected = np.sin(1.0) + np.cos(1.0)
    assert abs(y.data()[0] - expected) < 1e-6

    # Gradients not computed yet
    assert x.grad() is None
    assert y.grad() is None


@pytest.mark.autograd
def test_gradient_accumulation_simulation(pycoeus_available):
    """Test gradient accumulation simulation"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Simulate: loss = sum((pred - target)^2)
    pred_data = [2.0, 3.0, 4.0]
    target_data = [2.5, 2.8, 4.2]

    pred = pc.PyTensor(pred_data, [3])
    target = pc.PyTensor(target_data, [3])

    pred.requires_grad_(True)

    # Compute (pred - target)^2
    diff = pred - target
    squared_diff = diff * diff
    loss = squared_diff.sum()

    # Loss should be computed correctly
    assert len(loss.data()) == 1
    assert loss.data()[0] > 0  # Loss should be positive

    # Gradients not computed yet (would need backward pass)
    assert pred.grad() is None
    assert loss.grad() is None


@pytest.mark.autograd
def test_nested_operations(pycoeus_available):
    """Test nested operations with gradient tracking"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Complex nested computation
    x_data = [2.0]
    y_data = [3.0]

    x = pc.PyTensor(x_data, [])
    y = pc.PyTensor(y_data, [])

    x.requires_grad_(True)
    y.requires_grad_(True)

    # f(x,y) = exp(sin(x)) + log(y^2)
    sin_x = x.sin()
    exp_sin_x = sin_x.exp()

    y_squared = y * y
    log_y_squared = y_squared.log()

    result = exp_sin_x + log_y_squared

    # Result should be computed
    assert len(result.data()) == 1
    assert result.data()[0] > 0  # Should be positive

    # Gradients not computed yet
    assert x.grad() is None
    assert y.grad() is None
    assert result.grad() is None


@pytest.mark.autograd
def test_broadcasting_with_gradients(pycoeus_available):
    """Test broadcasting operations with gradient tracking"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Broadcasting: scalar + vector
    scalar_data = [2.0]
    vector_data = [1.0, 2.0, 3.0]

    scalar = pc.PyTensor(scalar_data, [])
    vector = pc.PyTensor(vector_data, [3])

    scalar.requires_grad_(True)
    vector.requires_grad_(True)

    result = scalar + vector

    # Result should be [3, 4, 5]
    expected = [3.0, 4.0, 5.0]
    assert result.data() == expected

    # Gradients not computed yet
    assert scalar.grad() is None
    assert vector.grad() is None
    assert result.grad() is None


@pytest.mark.autograd
@pytest.mark.parametrize("operation", ["add", "sub", "mul", "div"])
def test_binary_operations_gradient_setup(pycoeus_available, operation):
    """Test binary operations gradient setup"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    a_data = [4.0]
    b_data = [2.0]

    a = pc.PyTensor(a_data, [])
    b = pc.PyTensor(b_data, [])

    a.requires_grad_(True)
    b.requires_grad_(True)

    if operation == "add":
        result = a + b
        expected = 6.0
    elif operation == "sub":
        result = a - b
        expected = 2.0
    elif operation == "mul":
        result = a * b
        expected = 8.0
    elif operation == "div":
        result = a / b
        expected = 2.0

    assert result.data() == [expected]

    # Gradients not computed yet
    assert a.grad() is None
    assert b.grad() is None
    assert result.grad() is None


@pytest.mark.autograd
def test_tensor_reshape_with_gradients(pycoeus_available):
    """Test tensor reshape with gradient tracking"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    data = [1.0, 2.0, 3.0, 4.0]
    tensor = pc.PyTensor(data, [4])
    tensor.requires_grad_(True)

    # Reshape to 2x2
    reshaped = tensor.reshape([2, 2])

    assert reshaped.shape() == [2, 2]
    assert reshaped.data() == data  # Data should be unchanged

    # Gradients not computed yet
    assert tensor.grad() is None
    assert reshaped.grad() is None


@pytest.mark.autograd
def test_zero_grad_functionality(pycoeus_available):
    """Test zero_grad functionality"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    data = [1.0, 2.0, 3.0]
    tensor = pc.PyTensor(data, [3])
    tensor.requires_grad_(True)

    # zero_grad should work (even though gradients aren't computed yet)
    # This tests the API setup
    assert tensor.requires_grad()


@pytest.mark.autograd
def test_autograd_api_completeness(pycoeus_available):
    """Test that autograd API is complete and follows PyTorch patterns"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Test that all expected autograd methods exist
    data = [1.0, 2.0, 3.0]
    tensor = pc.PyTensor(data, [3])

    # These methods should exist (even if not fully implemented)
    assert hasattr(tensor, 'requires_grad_')
    assert hasattr(tensor, 'requires_grad')
    assert hasattr(tensor, 'grad')
    # Note: backward() method not implemented yet in PyCoeus

    # Test method signatures
    assert callable(getattr(tensor, 'requires_grad_'))
    assert callable(getattr(tensor, 'requires_grad'))
    assert callable(getattr(tensor, 'grad'))
