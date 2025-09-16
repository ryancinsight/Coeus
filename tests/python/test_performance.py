"""Performance benchmarks for PyCoeus vs PyTorch"""

import pytest
import numpy as np
import time

pytestmark = pytest.mark.performance


@pytest.mark.parametrize("size", [100, 1000, 10000])
def test_tensor_creation_performance(pycoeus_available, pytorch_available, size):
    """Measure tensor creation performance (without pytest-benchmark for comparison)"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch
    import time

    data = np.random.randn(size).astype(np.float32)

    # Measure PyCoeus OLD method (list conversion) - for comparison
    start_time = time.perf_counter()
    for _ in range(100):  # Run multiple times for better measurement
        pc_tensor_old = pc.PyTensor(data.tolist(), [size])
    pc_old_time = (time.perf_counter() - start_time) / 100

    # Measure PyCoeus NEW method (direct NumPy)
    start_time = time.perf_counter()
    for _ in range(100):
        pc_tensor_new = pc.PyTensor.from_numpy(data)
    pc_new_time = (time.perf_counter() - start_time) / 100

    # Measure PyTorch performance
    start_time = time.perf_counter()
    for _ in range(100):
        torch_tensor = torch.tensor(data)
    torch_time = (time.perf_counter() - start_time) / 100

    # Basic validation
    assert pc_tensor_old.shape() == [size]
    assert pc_tensor_new.shape() == [size]
    assert torch_tensor.shape == (size,)

    # Log performance comparison
    old_ratio = pc_old_time / torch_time
    new_ratio = pc_new_time / torch_time
    improvement = pc_old_time / pc_new_time

    print(".4f")
    print(".4f")
    print(".4f")
    print(".2f")
    print(".2f")
    print(".2f")

    # Performance targets based on data size
    if size >= 1000:
        # For medium/large arrays, new method should be significantly better
        assert improvement > 1.2, ".2f"
    else:
        # For small arrays, new method should be competitive (within 20% of old method)
        assert improvement > 0.8, ".2f"

    # New method should be competitive with PyTorch
    assert new_ratio < 2.0, ".2f"


@pytest.mark.benchmark
@pytest.mark.parametrize("size", [100, 1000])
def test_matrix_multiplication_benchmark(pycoeus_available, pytorch_available, benchmark, size):
    """Benchmark matrix multiplication performance"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Create matrices
    a_data = np.random.randn(size, size).astype(np.float32)
    b_data = np.random.randn(size, size).astype(np.float32)

    pc_a = pc.PyTensor(a_data.flatten().tolist(), [size, size])
    pc_b = pc.PyTensor(b_data.flatten().tolist(), [size, size])

    torch_a = torch.tensor(a_data)
    torch_b = torch.tensor(b_data)

    def pc_matmul():
        return pc_a @ pc_b

    def torch_matmul():
        return torch_a @ torch_b

    # Benchmark operations
    pc_result = benchmark(pc_matmul)
    torch_result = benchmark(torch_matmul)

    # Basic validation
    assert pc_result.shape() == [size, size]
    assert torch_result.shape == (size, size)


@pytest.mark.benchmark
def test_elementwise_operations_benchmark(pycoeus_available, pytorch_available, benchmark):
    """Benchmark element-wise operations performance"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    size = 10000
    a_data = np.random.randn(size).astype(np.float32)
    b_data = np.random.randn(size).astype(np.float32)

    pc_a = pc.PyTensor(a_data.tolist(), [size])
    pc_b = pc.PyTensor(b_data.tolist(), [size])

    torch_a = torch.tensor(a_data)
    torch_b = torch.tensor(b_data)

    def pc_add():
        return pc_a + pc_b

    def pc_mul():
        return pc_a * pc_b

    def pc_exp():
        return pc_a.exp()

    def torch_add():
        return torch_a + torch_b

    def torch_mul():
        return torch_a * torch_b

    def torch_exp():
        return torch_a.exp()

    # Benchmark operations
    pc_add_result = benchmark(pc_add)
    pc_mul_result = benchmark(pc_mul)
    pc_exp_result = benchmark(pc_exp)

    torch_add_result = benchmark(torch_add)
    torch_mul_result = benchmark(torch_mul)
    torch_exp_result = benchmark(torch_exp)

    # Basic validation
    assert pc_add_result.shape() == [size]
    assert pc_mul_result.shape() == [size]
    assert pc_exp_result.shape() == [size]


def test_memory_usage_comparison(pycoeus_available, pytorch_available):
    """Compare memory usage between PyCoeus and PyTorch"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch
    import psutil
    import os

    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB

    initial_memory = get_memory_usage()

    # Create large tensors
    size = 5000
    data = np.random.randn(size, size).astype(np.float32)

    # PyCoeus tensor
    pc_tensor = pc.PyTensor(data.flatten().tolist(), [size, size])
    pc_memory = get_memory_usage() - initial_memory

    del pc_tensor

    # PyTorch tensor
    torch_tensor = torch.tensor(data)
    torch_memory = get_memory_usage() - initial_memory

    # Log memory usage for comparison
    print(f"PyCoeus memory usage: {pc_memory:.2f} MB")
    print(f"PyTorch memory usage: {torch_memory:.2f} MB")

    # Basic assertion that memory usage is reasonable (within 5x of each other)
    assert pc_memory > 0
    assert torch_memory > 0
    assert pc_memory / torch_memory < 5.0, f"PyCoeus uses {pc_memory/torch_memory:.2f}x more memory than PyTorch"