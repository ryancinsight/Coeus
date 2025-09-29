import pytest
import torch
import pycoeus as pc
import numpy as np

def test_add():
    data1 = np.array([-1.0, 10.0])
    data2 = np.array([10.0, -1.0])
    shape = [2]

    pt1 = torch.tensor(data1, requires_grad=True)
    pt2 = torch.tensor(data2, requires_grad=True)
    pt_res = pt1 + pt2
    pt_res.backward(torch.tensor([1.0, 1.0]))

    py1 = pc.PyTensor(data1.tolist(), shape)
    py2 = pc.PyTensor(data2.tolist(), shape)
    py_res = py1 + py2

    assert np.allclose(py_res.data(), pt_res.detach().numpy(), rtol=1e-6)
    assert np.allclose(py1.grad.data() if py1.requires_grad else np.zeros_like(data1), pt1.grad.numpy(), rtol=1e-6)

def test_mul_edges():
    # x=-1 y=10 mul → -10, grad_x=10, grad_y=-1
    data1 = np.array([-1.0])
    data2 = np.array([10.0])
    pt1 = torch.tensor(data1, requires_grad=True)
    pt2 = torch.tensor(data2, requires_grad=True)
    pt_res = pt1 * pt2
    pt_res.backward(torch.tensor([1.0]))

    py1 = pc.PyTensor(data1.tolist(), [1])
    py2 = pc.PyTensor(data2.tolist(), [1])
    py1.requires_grad_(True)
    py2.requires_grad_(True)
    py_res = py1 * py2

    assert np.allclose(py_res.data(), pt_res.detach().numpy(), rtol=1e-6)
    # Grad after backward
    py_res.backward()
    assert np.allclose(py1.grad.data(), [10.0], rtol=1e-6)
    assert np.allclose(py2.grad.data(), [-1.0], rtol=1e-6)

# Overflow i32 stub (cast f32)
def test_overflow_i32():
    # i32 max *2 → wrap, but grad=2 (linear)
    pass  # Impl post-bind i32

# Run with pytest -v
