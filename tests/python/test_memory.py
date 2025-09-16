"""Memory profiling tests for PyCoeus"""

import pytest
import gc
import psutil
import os

pytestmark = pytest.mark.memory


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests"""
    initial_memory = get_memory_usage()

    class MemoryMonitor:
        def __init__(self, initial):
            self.initial = initial
            self.peak = initial

        def check(self):
            current = get_memory_usage()
            self.peak = max(self.peak, current)
            return current - self.initial

        def peak_usage(self):
            return self.peak - self.initial

    return MemoryMonitor(initial_memory)


def test_tensor_memory_allocation(pycoeus_available, memory_monitor):
    """Test memory allocation for tensor creation"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Test different sizes
    sizes = [100, 1000, 10000]

    for size in sizes:
        # Force garbage collection
        gc.collect()

        initial_mem = memory_monitor.check()

        # Create tensor
        data = list(range(size))
        tensor = pc.PyTensor(data, [size])

        # Check memory usage
        mem_usage = memory_monitor.check() - initial_mem

        # Basic validation
        assert tensor.shape() == [size]
        assert len(tensor.data()) == size

        # Memory should be reasonable (less than 1MB for these sizes)
        assert mem_usage < 1.0, f"Memory usage too high: {mem_usage} MB for size {size}"

        # Clean up
        del tensor


def test_tensor_operations_memory(pycoeus_available, memory_monitor):
    """Test memory usage during tensor operations"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Create test tensors
    size = 1000
    a_data = [i * 1.0 for i in range(size)]
    b_data = [i * 2.0 for i in range(size)]

    a = pc.PyTensor(a_data, [size])
    b = pc.PyTensor(b_data, [size])

    # Test operations don't leak memory
    initial_mem = memory_monitor.check()

    for _ in range(10):
        # Perform operations
        c = a + b
        d = a * b
        e = c.exp()

        # Clean up intermediate results
        del c, d, e
        gc.collect()

    final_mem = memory_monitor.check()
    mem_delta = final_mem - initial_mem

    # Memory should not grow significantly (less than 10MB growth)
    assert mem_delta < 10.0, f"Memory leak detected: {mem_delta} MB growth"


def test_large_tensor_memory(pycoeus_available, memory_monitor):
    """Test memory handling for large tensors"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    # Create a moderately large tensor
    size = 100000  # 100K elements
    data = [i * 0.001 for i in range(size)]

    initial_mem = memory_monitor.check()

    try:
        tensor = pc.PyTensor(data, [size])

        # Check memory usage is reasonable
        mem_usage = memory_monitor.check() - initial_mem

        # Should be less than 10MB for 100K floats (400KB of data)
        # Allow some overhead for Python wrapper
        assert mem_usage < 10.0, f"Memory usage too high: {mem_usage} MB for {size} elements"

        # Basic operations should work
        squared = tensor.pow(2.0)
        assert squared.shape() == [size]

    finally:
        # Clean up
        del tensor
        gc.collect()


def test_memory_cleanup(pycoeus_available, memory_monitor):
    """Test that memory is properly cleaned up"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc

    def create_tensors(count=100):
        tensors = []
        for i in range(count):
            data = [float(i) for _ in range(100)]
            tensor = pc.PyTensor(data, [100])
            tensors.append(tensor)
        return tensors

    # Create many tensors
    initial_mem = memory_monitor.check()
    tensors = create_tensors(50)

    # Check memory usage
    with_tensors_mem = memory_monitor.check()
    mem_with_tensors = with_tensors_mem - initial_mem

    # Delete tensors
    del tensors
    gc.collect()

    # Check memory cleanup
    final_mem = memory_monitor.check()
    mem_after_cleanup = final_mem - initial_mem

    # Memory should be mostly recovered (less than 20% remaining)
    cleanup_ratio = mem_after_cleanup / mem_with_tensors if mem_with_tensors > 0 else 0
    assert cleanup_ratio < 0.2, f"Poor memory cleanup: {cleanup_ratio:.1%} of memory retained"