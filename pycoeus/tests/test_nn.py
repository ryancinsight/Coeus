import torch
import pycoeus as pc
import pytest
import numpy as np

def test_linear_compat():
    torch.manual_seed(42)
    pc.manual_seed(42)  # assume seed impl

    input_size, output_size = 10, 5
    torch_linear = torch.nn.Linear(input_size, output_size)
    pc_linear = pc.nn.Linear(input_size, output_size)

    x_torch = torch.randn(3, input_size)
    x_pc = pc.PyTensor(x_torch.numpy())

    y_torch = torch_linear(x_torch)
    y_pc = pc_linear(x_pc)

    assert np.allclose(y_torch.detach().numpy(), np.array(y_pc.data()), rtol=1e-6)

def test_conv1d_compat():
    torch_conv = torch.nn.Conv1d(3, 6, kernel_size=3)
    pc_conv = pc.nn.Conv1d(3, 6, 3)

    x_torch = torch.randn(1, 3, 10)
    x_pc = pc.PyTensor(x_torch.numpy())

    y_torch = torch_conv(x_torch)
    y_pc = pc_conv(x_pc)

    assert np.allclose(y_torch.detach().numpy(), np.array(y_pc.data()), rtol=1e-6)

def test_batchnorm1d_compat():
    torch_bn = torch.nn.BatchNorm1d(10)
    pc_bn = pc.nn.BatchNorm1d(10)

    x_torch = torch.randn(5, 10)
    x_pc = pc.PyTensor(x_torch.numpy())

    y_torch = torch_bn(x_torch)
    y_pc = pc_bn(x_pc)

    assert np.allclose(y_torch.detach().numpy(), np.array(y_pc.data()), rtol=1e-6)

# Add pooling, attention tests similar
# Edges: empty batch (shape [0, in]), dtype f32/i32, grad flow torch.autograd.gradcheck vs pc

@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_dtype_compat(dtype):
    # test linear/conv with dtype, assert allclose

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
