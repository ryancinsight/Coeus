"""PyTorch compatibility tests for PyCoeus"""

import pytest
import numpy as np


@pytest.mark.pytorch_compat
def test_tensor_creation(pycoeus_tensors, torch_tensors, pytorch_available):
    """Test basic tensor creation matches PyTorch"""
    for key in ['vector', 'matrix', 'tensor_3d']:
        pc_tensor = pycoeus_tensors[key]
        torch_tensor = torch_tensors[key]

        # Check shapes match
        assert pc_tensor.shape() == list(torch_tensor.shape)

        # Check data matches
        pc_data = pc_tensor.data()
        torch_data = torch_tensor.flatten().tolist()
        assert pc_data == torch_data


@pytest.mark.pytorch_compat
def test_arithmetic_operations(pycoeus_tensors, torch_tensors, pytorch_available):
    """Test arithmetic operations match PyTorch"""
    pc_vec = pycoeus_tensors['vector']
    torch_vec = torch_tensors['vector']

    # Addition
    pc_result = pc_vec + pc_vec
    torch_result = torch_vec + torch_vec

    pc_data = pc_result.data()
    torch_data = torch_result.tolist()
    assert pc_data == torch_data

    # Subtraction
    pc_result = pc_vec - pc_vec
    torch_result = torch_vec - torch_vec

    pc_data = pc_result.data()
    torch_data = torch_result.tolist()
    assert pc_data == torch_data

    # Element-wise multiplication
    pc_result = pc_vec * pc_vec
    torch_result = torch_vec * torch_vec

    pc_data = pc_result.data()
    torch_data = torch_result.tolist()
    assert pc_data == torch_data


@pytest.mark.pytorch_compat
def test_matrix_operations(pycoeus_available, pytorch_available):
    """Test matrix operations"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Create 2x3 matrix
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    shape = [2, 3]

    pc_matrix = pc.PyTensor(data, shape)
    torch_matrix = torch.tensor(data).reshape(2, 3)

    # Transpose
    pc_transposed = pc_matrix.transpose(0, 1)
    torch_transposed = torch_matrix.t()

    assert pc_transposed.shape() == list(torch_transposed.shape)
    assert pc_transposed.data() == torch_transposed.flatten().tolist()


@pytest.mark.pytorch_compat
def test_matmul_operation(pycoeus_available, pytorch_available):
    """Test matrix multiplication"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # 2x3 matrix @ 3x2 matrix = 2x2 result
    a_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    a_shape = [2, 3]
    b_data = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    b_shape = [3, 2]

    pc_a = pc.PyTensor(a_data, a_shape)
    pc_b = pc.PyTensor(b_data, b_shape)
    torch_a = torch.tensor(a_data).reshape(2, 3)
    torch_b = torch.tensor(b_data).reshape(3, 2)

    # Matrix multiplication
    pc_result = pc_a @ pc_b
    torch_result = torch_a @ torch_b

    assert pc_result.shape() == list(torch_result.shape)
    assert pc_result.data() == torch_result.flatten().tolist()


@pytest.mark.pytorch_compat
def test_broadcasting(pycoeus_available, pytorch_available):
    """Test broadcasting behavior"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Scalar + vector broadcasting
    scalar_data = [2.0]
    scalar_shape = []
    vector_data = [1.0, 2.0, 3.0]
    vector_shape = [3]

    pc_scalar = pc.PyTensor(scalar_data, scalar_shape)
    pc_vector = pc.PyTensor(vector_data, vector_shape)
    torch_scalar = torch.tensor(2.0)
    torch_vector = torch.tensor([1.0, 2.0, 3.0])

    # Broadcasting addition
    pc_result = pc_scalar + pc_vector
    torch_result = torch_scalar + torch_vector

    assert pc_result.data() == torch_result.tolist()


@pytest.mark.pytorch_compat
def test_mathematical_functions(pycoeus_available, pytorch_available):
    """Test mathematical functions"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    data = [0.0, 1.0, 2.0]
    shape = [3]

    pc_tensor = pc.PyTensor(data, shape)
    torch_tensor = torch.tensor(data)

    # Exponential
    pc_exp = pc_tensor.exp()
    torch_exp = torch.exp(torch_tensor)
    assert pc_exp.data() == torch_exp.tolist()

    # Sine
    pc_sin = pc_tensor.sin()
    torch_sin = torch.sin(torch_tensor)
    np.testing.assert_allclose(pc_sin.data(), torch_sin.tolist(), rtol=1e-6)


@pytest.mark.pytorch_compat
def test_reduction_operations(pycoeus_available, pytorch_available):
    """Test reduction operations"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    shape = [2, 3]

    pc_tensor = pc.PyTensor(data, shape)
    torch_tensor = torch.tensor(data).reshape(2, 3)

    # Sum
    pc_sum = pc_tensor.sum()
    torch_sum = torch_tensor.sum()
    assert pc_sum.data() == [torch_sum.item()]

    # Mean
    pc_mean = pc_tensor.mean()
    torch_mean = torch_tensor.mean()
    assert pc_mean.data() == [torch_mean.item()]


@pytest.mark.pytorch_compat
def test_gradient_tracking(pycoeus_available):
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

    # Gradients should be None initially (full autograd not implemented yet)
    assert tensor.grad() is None


@pytest.mark.pytorch_compat
def test_device_management(pycoeus_available):
    """Test device management"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    data = [1.0, 2.0, 3.0]
    shape = [3]

    tensor = pc.PyTensor(data, shape)

    # Should be on CPU by default
    assert tensor.device().type() == "cpu"

    # CPU operation should work
    cpu_tensor = tensor.cpu()
    assert cpu_tensor.device().type() == "cpu"

    # CUDA should fail gracefully (not implemented yet)
    with pytest.raises(Exception):  # Should raise NotImplementedError
        tensor.cuda()


@pytest.mark.pytorch_compat
@pytest.mark.parametrize("shape", [
    [3],           # 1D
    [2, 3],        # 2D
    [2, 2, 3],     # 3D
])
def test_shape_operations(pycoeus_available, pytorch_available, shape):
    """Test shape manipulation operations"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    size = shape[0] * shape[1] if len(shape) >= 2 else shape[0]
    if len(shape) == 3:
        size *= shape[2]

    data = list(range(size))

    pc_tensor = pc.PyTensor(data, shape)
    torch_tensor = torch.tensor(data).reshape(shape)

    # Test reshape (only if total elements match)
    if len(shape) == 2:
        new_shape = [shape[1], shape[0]]
        pc_reshaped = pc_tensor.reshape(new_shape)
        torch_reshaped = torch_tensor.reshape(new_shape)

        assert pc_reshaped.shape() == list(torch_reshaped.shape)


@pytest.mark.pytorch_compat
def test_tensor_properties(pycoeus_tensors, torch_tensors, pytorch_available):
    """Test tensor properties match PyTorch"""
    for key in ['vector', 'matrix']:
        pc_tensor = pycoeus_tensors[key]
        torch_tensor = torch_tensors[key]

        # Test number of elements
        assert pc_tensor.numel() == torch_tensor.numel()

        # Test dimensionality
        assert pc_tensor.dim() == len(torch_tensor.shape)


@pytest.mark.pytorch_compat
@pytest.mark.slow
def test_numerical_stability(pycoeus_available, pytorch_available):
    """Test numerical stability with edge cases"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Test with very small numbers
    small_data = [1e-8, 1e-7, 1e-6]
    pc_tensor = pc.PyTensor(small_data, [3])
    torch_tensor = torch.tensor(small_data)

    pc_exp = pc_tensor.exp()
    torch_exp = torch.exp(torch_tensor)

    # Should handle small numbers without overflow/underflow
    for pc_val, torch_val in zip(pc_exp.data(), torch_exp.tolist()):
        assert abs(pc_val - torch_val) < 1e-6
