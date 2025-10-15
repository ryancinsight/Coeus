#!/usr/bin/env python3
"""
PyTorch Baseline Benchmarks for Performance Comparison

This script creates equivalent benchmarks to the Rust criterion benchmarks
to establish PyTorch performance baselines for comparison with Coeus.
"""

import torch
import time
import numpy as np
import statistics
import sys

def time_function(func, iterations=1000, warmup=100):
    """Time a function with warmup and multiple iterations"""
    # Warmup
    for _ in range(warmup):
        func()

    # Time iterations
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1e9)  # Convert to nanoseconds

    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times)
    }

def benchmark_tensor_addition():
    """Benchmark tensor addition operations (matching Rust criterion pattern)"""
    print("Benchmarking PyTorch Tensor Addition...")

    # Pre-create tensors to match Rust criterion pattern (tensors created outside timing loop)
    a100 = torch.randn(100)
    b100 = torch.randn(100)

    a1000 = torch.randn(1000)
    b1000 = torch.randn(1000)

    a10000 = torch.randn(10000)
    b10000 = torch.randn(10000)

    # add_100: 100-element tensors
    def add_100():
        c = a100 + b100
        return c

    # add_1000: 1000-element tensors
    def add_1000():
        c = a1000 + b1000
        return c

    # add_10000: 10000-element tensors
    def add_10000():
        c = a10000 + b10000
        return c

    results = {
        'add_100': time_function(add_100),
        'add_1000': time_function(add_1000),
        'add_10000': time_function(add_10000)
    }

    return results

def benchmark_gradient_accumulation():
    """Benchmark gradient accumulation (autograd backward) - matching Rust pattern"""
    print("Benchmarking PyTorch Gradient Accumulation...")

    # Pre-create tensors (simplified to match Rust gradient accumulation pattern)
    ga100 = torch.ones(100)
    gb100 = torch.ones(100)

    ga1000 = torch.ones(1000)
    gb1000 = torch.ones(1000)

    ga10000 = torch.ones(10000)
    gb10000 = torch.ones(10000)

    # accumulate_100: gradient accumulation on 100-element tensors
    def accumulate_100():
        c = ga100 + gb100  # Simulate gradient accumulation
        return c

    # accumulate_1000: gradient accumulation on 1000-element tensors
    def accumulate_1000():
        c = ga1000 + gb1000  # Simulate gradient accumulation
        return c

    # accumulate_10000: gradient accumulation on 10000-element tensors
    def accumulate_10000():
        c = ga10000 + gb10000  # Simulate gradient accumulation
        return c

    results = {
        'accumulate_100': time_function(accumulate_100),
        'accumulate_1000': time_function(accumulate_1000),
        'accumulate_10000': time_function(accumulate_10000)
    }

    return results

def benchmark_broadcasting():
    """Benchmark broadcasting operations - matching Rust criterion pattern"""
    print("Benchmarking PyTorch Broadcasting...")

    # Pre-create tensors to match Rust pattern
    b1x100_a = torch.randn(1)
    b1x100_b = torch.randn(100)

    b10x10_a = torch.randn(10, 1)
    b10x10_b = torch.randn(10, 10)

    b1x1000_a = torch.randn(1)
    b1x1000_b = torch.randn(1000)

    # broadcast_1x100: [1] + [100] -> [100]
    def broadcast_1x100():
        c = b1x100_a + b1x100_b
        return c

    # broadcast_10x10: [10, 1] + [10, 10] -> [10, 10]
    def broadcast_10x10():
        c = b10x10_a + b10x10_b
        return c

    # broadcast_1x1000: [1] + [1000] -> [1000]
    def broadcast_1x1000():
        c = b1x1000_a + b1x1000_b
        return c

    results = {
        'broadcast_1x100': time_function(broadcast_1x100),
        'broadcast_10x10': time_function(broadcast_10x10),
        'broadcast_1x1000': time_function(broadcast_1x1000)
    }

    return results

def benchmark_matmul():
    """Benchmark matrix multiplication"""
    print("Benchmarking PyTorch Matrix Multiplication...")

    # Small matrices: 10x10 @ 10x10
    def matmul_10x10():
        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        c = torch.matmul(a, b)
        return c

    # Medium matrices: 100x100 @ 100x100
    def matmul_100x100():
        a = torch.randn(100, 100)
        b = torch.randn(100, 100)
        c = torch.matmul(a, b)
        return c

    # Large matrices: 500x500 @ 500x500 (if system can handle)
    def matmul_500x500():
        try:
            a = torch.randn(500, 500)
            b = torch.randn(500, 500)
            c = torch.matmul(a, b)
            return c
        except RuntimeError:
            # Skip if out of memory
            return None

    results = {
        'matmul_10x10': time_function(matmul_10x10),
        'matmul_100x100': time_function(matmul_100x100),
        'matmul_500x500': time_function(matmul_500x500)
    }

    # Filter out None results
    results = {k: v for k, v in results.items() if v is not None}

    return results

def main():
    """Run all benchmarks and output results"""
    print("PyTorch Performance Baseline Benchmarks")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Run benchmarks
    all_results = {}

    try:
        all_results.update(benchmark_tensor_addition())
        print()
        all_results.update(benchmark_gradient_accumulation())
        print()
        all_results.update(benchmark_broadcasting())
        print()
        all_results.update(benchmark_matmul())
        print()
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        return 1

    # Output results in format suitable for comparison
    print("RESULTS:")
    print("=" * 50)

    for benchmark_name, stats in all_results.items():
        print(f"{benchmark_name}:")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print()

    # Save to file for comparison script
    import json
    with open('pytorch_benchmarks.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = {}
        for k, v in all_results.items():
            serializable_results[k] = {stat: float(val) for stat, val in v.items()}
        json.dump(serializable_results, f, indent=2)

    print("Results saved to pytorch_benchmarks.json")

    return 0

if __name__ == '__main__':
    sys.exit(main())
