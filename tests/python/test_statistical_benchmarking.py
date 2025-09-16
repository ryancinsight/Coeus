"""Statistical performance benchmarking for PyCoeus vs PyTorch"""

import pytest
import numpy as np
import time
import statistics
import math
from collections import defaultdict


@pytest.mark.performance
@pytest.mark.statistical
def test_statistical_performance_analysis(pycoeus_available, pytorch_available):
    """Comprehensive statistical performance analysis"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    def run_statistical_benchmark(operation_name, pc_func, torch_func, iterations=50):
        """Run statistical benchmarking for an operation"""
        pc_times = []
        torch_times = []

        for _ in range(iterations):
            # Time PyCoeus
            start_time = time.perf_counter()
            pc_result = pc_func()
            pc_time = time.perf_counter() - start_time
            pc_times.append(pc_time)

            # Time PyTorch
            start_time = time.perf_counter()
            torch_result = torch_func()
            torch_time = time.perf_counter() - start_time
            torch_times.append(torch_time)

        # Calculate statistics
        pc_stats = calculate_statistics(pc_times)
        torch_stats = calculate_statistics(torch_times)

        return {
            'operation': operation_name,
            'pc_stats': pc_stats,
            'torch_stats': torch_stats,
            'performance_ratio': pc_stats['mean'] / torch_stats['mean'] if torch_stats['mean'] > 0 else float('inf'),
            'iterations': iterations
        }

    def calculate_statistics(times):
        """Calculate comprehensive statistics for timing data"""
        if not times:
            return {}

        mean_time = statistics.mean(times)
        median_time = statistics.median(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)

        # Calculate confidence interval (95%)
        if len(times) > 1:
            confidence_level = 1.96  # 95% confidence
            margin_of_error = confidence_level * (std_dev / math.sqrt(len(times)))
            confidence_interval = (mean_time - margin_of_error, mean_time + margin_of_error)
        else:
            confidence_interval = (mean_time, mean_time)

        # Calculate percentiles
        sorted_times = sorted(times)
        p95 = np.percentile(sorted_times, 95)
        p99 = np.percentile(sorted_times, 99)

        return {
            'mean': mean_time,
            'median': median_time,
            'std_dev': std_dev,
            'min': min_time,
            'max': max_time,
            'confidence_interval': confidence_interval,
            'p95': p95,
            'p99': p99,
            'cv': std_dev / mean_time if mean_time > 0 else 0  # Coefficient of variation
        }

    # Test different tensor sizes and operations
    sizes = [100, 1000, 10000]
    operations = ['add', 'mul', 'matmul']

    results = []

    for size in sizes:
        data = np.random.randn(size).tolist()

        for operation in operations:
            if operation == 'matmul':
                # For matrix multiplication, create 2D tensors
                matrix_size = int(math.sqrt(size))
                if matrix_size * matrix_size != size:
                    matrix_size = int(math.sqrt(size)) + 1
                    actual_size = matrix_size * matrix_size
                    data = np.random.randn(actual_size).tolist()
                else:
                    actual_size = size

                shape = [matrix_size, matrix_size]

                pc_a = pc.PyTensor(data[:actual_size//2], shape)
                pc_b = pc.PyTensor(data[actual_size//2:], shape)
                torch_a = torch.tensor(data[:actual_size//2]).reshape(shape)
                torch_b = torch.tensor(data[actual_size//2:]).reshape(shape)

                pc_func = lambda: pc_a @ pc_b
                torch_func = lambda: torch_a @ torch_b
            else:
                pc_tensor = pc.PyTensor(data, [size])
                torch_tensor = torch.tensor(data)

                if operation == 'add':
                    pc_func = lambda: pc_tensor + pc_tensor
                    torch_func = lambda: torch_tensor + torch_tensor
                elif operation == 'mul':
                    pc_func = lambda: pc_tensor * pc_tensor
                    torch_func = lambda: torch_tensor * torch_tensor

            result = run_statistical_benchmark(
                f"{operation}_size_{size}",
                pc_func,
                torch_func
            )
            results.append(result)

    # Analyze results
    for result in results:
        operation = result['operation']
        pc_stats = result['pc_stats']
        torch_stats = result['torch_stats']
        ratio = result['performance_ratio']

        print(f"\n=== {operation.upper()} PERFORMANCE ANALYSIS ===")
        print(f"PyCoeus  - Mean: {pc_stats['mean']:.6f}s, StdDev: {pc_stats['std_dev']:.6f}s")
        print(f"PyTorch  - Mean: {torch_stats['mean']:.6f}s, StdDev: {torch_stats['std_dev']:.6f}s")
        print(".2f")
        print(f"Coefficient of Variation - PyCoeus: {pc_stats['cv']:.3f}, PyTorch: {torch_stats['cv']:.3f}")

        # Performance assertion (allow up to 5x slower for now)
        assert ratio < 5.0, f"Performance regression detected: {ratio:.2f}x slower than PyTorch"

        # Statistical significance check
        # If confidence intervals don't overlap significantly, performance difference is significant
        pc_ci = pc_stats['confidence_interval']
        torch_ci = torch_stats['confidence_interval']

        if pc_ci[1] < torch_ci[0] or torch_ci[1] < pc_ci[0]:
            print("⚠️  Statistically significant performance difference detected")
        else:
            print("✅ Performance difference within statistical noise")


@pytest.mark.performance
@pytest.mark.statistical
def test_memory_usage_statistics(pycoeus_available, pytorch_available):
    """Statistical analysis of memory usage patterns"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch
    import psutil
    import os

    process = psutil.Process(os.getpid())

    def measure_memory_usage(func, iterations=10):
        """Measure memory usage statistics for a function"""
        memory_samples = []

        for _ in range(iterations):
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            result = func()

            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = peak_memory - initial_memory

            memory_samples.append(memory_used)

            # Clean up
            del result
            import gc
            gc.collect()

        return calculate_memory_statistics(memory_samples)

    def calculate_memory_statistics(samples):
        """Calculate statistics for memory usage samples"""
        if not samples:
            return {}

        mean_usage = statistics.mean(samples)
        median_usage = statistics.median(samples)
        std_dev = statistics.stdev(samples) if len(samples) > 1 else 0
        min_usage = min(samples)
        max_usage = max(samples)

        return {
            'mean': mean_usage,
            'median': median_usage,
            'std_dev': std_dev,
            'min': min_usage,
            'max': max_usage,
            'cv': std_dev / mean_usage if mean_usage > 0 else 0
        }

    # Test different tensor sizes
    sizes = [1000, 5000, 10000]

    for size in sizes:
        data = np.random.randn(size).tolist()

        # PyCoeus memory measurement
        pc_tensor = pc.PyTensor(data, [size])
        pc_memory_stats = measure_memory_usage(lambda: pc_tensor + pc_tensor)

        # PyTorch memory measurement
        torch_tensor = torch.tensor(data)
        torch_memory_stats = measure_memory_usage(lambda: torch_tensor + torch_tensor)

        print(f"\n=== MEMORY USAGE ANALYSIS (Size: {size}) ===")
        print(f"PyCoeus  - Mean: {pc_memory_stats['mean']:.2f}MB, StdDev: {pc_memory_stats['std_dev']:.2f}MB")
        print(f"PyTorch  - Mean: {torch_memory_stats['mean']:.2f}MB, StdDev: {torch_memory_stats['std_dev']:.2f}MB")

        # Memory efficiency check (PyCoeus should be within 2x of PyTorch)
        ratio = pc_memory_stats['mean'] / torch_memory_stats['mean'] if torch_memory_stats['mean'] > 0 else float('inf')
        assert ratio < 2.0, f"Memory usage {ratio:.2f}x higher than PyTorch"

        # Memory consistency check (low coefficient of variation)
        assert pc_memory_stats['cv'] < 0.5, f"PyCoeus memory usage inconsistent (CV: {pc_memory_stats['cv']:.3f})"
        assert torch_memory_stats['cv'] < 0.5, f"PyTorch memory usage inconsistent (CV: {torch_memory_stats['cv']:.3f})"


@pytest.mark.performance
@pytest.mark.statistical
def test_scalability_analysis(pycoeus_available, pytorch_available):
    """Analyze performance scalability across different tensor sizes"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    sizes = [100, 500, 1000, 5000, 10000]
    pc_times = []
    torch_times = []

    for size in sizes:
        data = np.random.randn(size).tolist()

        # PyCoeus timing
        pc_tensor = pc.PyTensor(data, [size])
        start_time = time.perf_counter()
        for _ in range(10):  # Multiple iterations for stable timing
            pc_result = pc_tensor + pc_tensor
        pc_time = (time.perf_counter() - start_time) / 10
        pc_times.append(pc_time)

        # PyTorch timing
        torch_tensor = torch.tensor(data)
        start_time = time.perf_counter()
        for _ in range(10):
            torch_result = torch_tensor + torch_tensor
        torch_time = (time.perf_counter() - start_time) / 10
        torch_times.append(torch_time)

    # Analyze scalability
    pc_ratios = [pc_times[i] / pc_times[0] for i in range(len(pc_times))]
    torch_ratios = [torch_times[i] / torch_times[0] for i in range(len(torch_times))]
    size_ratios = [sizes[i] / sizes[0] for i in range(len(sizes))]

    print("\n=== SCALABILITY ANALYSIS ===")
    for i, size in enumerate(sizes):
        print(".6f")

    # Check that scaling is roughly linear (or better)
    pc_scaling_efficiency = statistics.mean([
        pc_ratios[i] / size_ratios[i] for i in range(len(sizes))
    ])
    torch_scaling_efficiency = statistics.mean([
        torch_ratios[i] / size_ratios[i] for i in range(len(sizes))
    ])

    print(f"\nScaling Efficiency (lower is better):")
    print(f"PyCoeus: {pc_scaling_efficiency:.3f}")
    print(f"PyTorch: {torch_scaling_efficiency:.3f}")

    # PyCoeus scaling should be reasonable (not worse than 2x PyTorch scaling)
    assert pc_scaling_efficiency <= torch_scaling_efficiency * 2, \
        f"Poor scaling efficiency: PyCoeus {pc_scaling_efficiency:.3f} vs PyTorch {torch_scaling_efficiency:.3f}"


@pytest.mark.performance
@pytest.mark.statistical
def test_operation_complexity_analysis(pycoeus_available, pytorch_available):
    """Analyze computational complexity of different operations"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    def analyze_complexity(operation_name, pc_op, torch_op, sizes):
        """Analyze computational complexity for an operation"""
        pc_times = []
        torch_times = []

        for size in sizes:
            if operation_name == 'matmul':
                # For matmul, use square matrices
                matrix_size = int(math.sqrt(size))
                data_size = matrix_size * matrix_size
                data = np.random.randn(data_size).tolist()

                pc_a = pc.PyTensor(data, [matrix_size, matrix_size])
                torch_a = torch.tensor(data).reshape(matrix_size, matrix_size)

                # Time the operation
                start_time = time.perf_counter()
                pc_result = pc_op(pc_a)
                pc_time = time.perf_counter() - start_time
                pc_times.append(pc_time)

                start_time = time.perf_counter()
                torch_result = torch_op(torch_a)
                torch_time = time.perf_counter() - start_time
                torch_times.append(torch_time)
            else:
                data = np.random.randn(size).tolist()

                pc_tensor = pc.PyTensor(data, [size])
                torch_tensor = torch.tensor(data)

                start_time = time.perf_counter()
                pc_result = pc_op(pc_tensor)
                pc_time = time.perf_counter() - start_time
                pc_times.append(pc_time)

                start_time = time.perf_counter()
                torch_result = torch_op(torch_tensor)
                torch_time = time.perf_counter() - start_time
                torch_times.append(torch_time)

        return pc_times, torch_times

    # Test different operations with varying complexities
    operations = {
        'element_wise': {
            'pc_op': lambda x: x + x,
            'torch_op': lambda x: x + x,
            'complexity': 'O(n)'  # Linear
        },
        'matmul': {
            'pc_op': lambda x: x @ x,
            'torch_op': lambda x: x @ x,
            'complexity': 'O(n^3)'  # Cubic
        }
    }

    sizes = [100, 200, 500]  # Smaller sizes for complexity analysis

    for op_name, op_config in operations.items():
        pc_times, torch_times = analyze_complexity(
            op_name, op_config['pc_op'], op_config['torch_op'], sizes
        )

        print(f"\n=== {op_name.upper()} COMPLEXITY ANALYSIS ===")
        print("Size      PyCoeus(s)    PyTorch(s)     Ratio")
        print("-" * 50)

        ratios = []
        for i, size in enumerate(sizes):
            ratio = pc_times[i] / torch_times[i] if torch_times[i] > 0 else float('inf')
            ratios.append(ratio)
            print("6d")

        avg_ratio = statistics.mean(ratios)
        print(f"\nAverage performance ratio: {avg_ratio:.2f}x")

        # For element-wise operations, ratio should be more stable
        if op_name == 'element_wise':
            ratio_std = statistics.stdev(ratios) if len(ratios) > 1 else 0
            assert ratio_std < 0.5, f"Unstable element-wise performance: std={ratio_std:.3f}"


@pytest.mark.performance
@pytest.mark.statistical
def test_concurrent_performance_analysis(pycoeus_available):
    """Analyze performance under concurrent operations"""
    if not pycoeus_available:
        pytest.skip("PyCoeus not available")

    import pycoeus as pc
    import threading
    import concurrent.futures

    def worker_operation(worker_id, tensor_size, iterations):
        """Worker function for concurrent performance testing"""
        data = np.random.randn(tensor_size).tolist()
        tensor = pc.PyTensor(data, [tensor_size])

        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            result = tensor + tensor
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        return {
            'worker_id': worker_id,
            'mean_time': statistics.mean(times),
            'total_time': sum(times),
            'operations': iterations
        }

    # Test with different concurrency levels
    concurrency_levels = [1, 2, 4]
    tensor_size = 1000
    iterations_per_worker = 100

    print("\n=== CONCURRENT PERFORMANCE ANALYSIS ===")

    for num_workers in concurrency_levels:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit worker tasks
            futures = [
                executor.submit(worker_operation, i, tensor_size, iterations_per_worker)
                for i in range(num_workers)
            ]

            # Collect results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

            total_operations = sum(r['operations'] for r in results)
            total_time = max(r['total_time'] for r in results)  # Wall clock time
            avg_operation_time = statistics.mean(r['mean_time'] for r in results)

            throughput = total_operations / total_time

            print(f"Workers: {num_workers:2d} | Throughput: {throughput:6.1f} ops/sec | Avg op time: {avg_operation_time:.6f}s")

    # Verify that concurrent execution works without errors
    assert len(results) == num_workers, "Some concurrent operations failed"


@pytest.mark.performance
@pytest.mark.statistical
def test_performance_regression_detection(pycoeus_available, pytorch_available):
    """Detect performance regressions compared to baseline"""
    if not pycoeus_available or not pytorch_available:
        pytest.skip("Required libraries not available")

    import pycoeus as pc
    import torch

    # Define baseline performance expectations
    baseline_expectations = {
        'add_1000': {'max_ratio': 3.0, 'min_ops_sec': 1000},
        'mul_1000': {'max_ratio': 3.0, 'min_ops_sec': 1000},
        'matmul_100': {'max_ratio': 5.0, 'min_ops_sec': 10},
    }

    def measure_performance(operation, size):
        """Measure performance for a specific operation and size"""
        data = np.random.randn(size).tolist()

        if operation == 'matmul':
            matrix_size = int(math.sqrt(size))
            data = np.random.randn(matrix_size * matrix_size).tolist()
            pc_a = pc.PyTensor(data, [matrix_size, matrix_size])
            pc_b = pc.PyTensor(data, [matrix_size, matrix_size])
            torch_a = torch.tensor(data).reshape(matrix_size, matrix_size)
            torch_b = torch.tensor(data).reshape(matrix_size, matrix_size)

            # Time PyCoeus
            start_time = time.perf_counter()
            iterations = 10
            for _ in range(iterations):
                pc_result = pc_a @ pc_b
            pc_time = (time.perf_counter() - start_time) / iterations

            # Time PyTorch
            start_time = time.perf_counter()
            for _ in range(iterations):
                torch_result = torch_a @ torch_b
            torch_time = (time.perf_counter() - start_time) / iterations
        else:
            pc_tensor = pc.PyTensor(data, [size])
            torch_tensor = torch.tensor(data)

            if operation == 'add':
                pc_op = lambda: pc_tensor + pc_tensor
                torch_op = lambda: torch_tensor + torch_tensor
            elif operation == 'mul':
                pc_op = lambda: pc_tensor * pc_tensor
                torch_op = lambda: torch_tensor * torch_tensor

            # Time operations
            iterations = 100
            start_time = time.perf_counter()
            for _ in range(iterations):
                pc_result = pc_op()
            pc_time = (time.perf_counter() - start_time) / iterations

            start_time = time.perf_counter()
            for _ in range(iterations):
                torch_result = torch_op()
            torch_time = (time.perf_counter() - start_time) / iterations

        ratio = pc_time / torch_time if torch_time > 0 else float('inf')
        ops_per_sec = 1.0 / pc_time if pc_time > 0 else 0

        return {
            'ratio': ratio,
            'ops_per_sec': ops_per_sec,
            'pc_time': pc_time,
            'torch_time': torch_time
        }

    print("\n=== PERFORMANCE REGRESSION DETECTION ===")
    print("Operation      Ratio  Ops/sec  Status")
    print("-" * 40)

    all_passed = True

    for test_name, expectations in baseline_expectations.items():
        operation, size_str = test_name.split('_')
        size = int(size_str)

        result = measure_performance(operation, size)

        # Check against expectations
        ratio_ok = result['ratio'] <= expectations['max_ratio']
        ops_ok = result['ops_per_sec'] >= expectations['min_ops_sec']

        status = "✅ PASS" if (ratio_ok and ops_ok) else "❌ FAIL"

        if not (ratio_ok and ops_ok):
            all_passed = False

        print("15s")

    assert all_passed, "Performance regression detected - some benchmarks failed"
