#!/usr/bin/env python3
"""
Test PyTorch compatibility of Coeus implementation.

This script tests the PyTorch-compatible API exposed by pycoeus.
"""

import sys
import os
import argparse
import pkgutil
import importlib

# Add the built wheel to Python path
wheel_dir = os.path.join(os.path.dirname(__file__), 'target', 'wheels')
if wheel_dir not in sys.path:
    sys.path.insert(0, wheel_dir)

pycoeus_wheel_dir = os.path.join(os.path.dirname(__file__), 'pycoeus', 'target', 'wheels')
if pycoeus_wheel_dir not in sys.path:
    sys.path.insert(0, pycoeus_wheel_dir)

try:
    import coeus as torch
    import coeus.nn as nn
    print("[OK] Successfully imported coeus as torch")
except ImportError as e:
    print(f"[ERROR] Failed to import coeus: {e}")
    sys.exit(1)

def _iter_package_modules(package):
    if not hasattr(package, "__path__"):
        return set()

    prefix = package.__name__ + "."
    modules = set()
    for module_info in pkgutil.walk_packages(package.__path__, prefix=prefix):
        modules.add(module_info.name)
    return modules

def _is_private_module_path(module_path: str) -> bool:
    parts = module_path.split(".")
    return any(part.startswith("_") for part in parts)

def compare_torch_module_surface(include_private: bool, max_list: int) -> int:
    try:
        pytorch = importlib.import_module("torch")
    except ImportError as e:
        print(f"[ERROR] Failed to import torch for comparison: {e}")
        return 2

    coeus_pkg = importlib.import_module("coeus")

    torch_modules_full = _iter_package_modules(pytorch)
    coeus_modules_full = _iter_package_modules(coeus_pkg)

    torch_rel = {name.removeprefix("torch.") for name in torch_modules_full if name.startswith("torch.")}
    coeus_rel = {name.removeprefix("coeus.") for name in coeus_modules_full if name.startswith("coeus.")}

    if not include_private:
        torch_rel = {m for m in torch_rel if not _is_private_module_path(m)}
        coeus_rel = {m for m in coeus_rel if not _is_private_module_path(m)}

    missing_in_coeus = sorted(torch_rel - coeus_rel)
    extra_in_coeus = sorted(coeus_rel - torch_rel)

    print("Module Surface Comparison: torch vs coeus")
    print("=" * 50)
    print(f"torch modules discovered: {len(torch_rel)}")
    print(f"coeus modules discovered: {len(coeus_rel)}")
    print(f"missing in coeus: {len(missing_in_coeus)}")
    print(f"extra in coeus: {len(extra_in_coeus)}")

    def group_by_root(modules):
        grouped = {}
        for module in modules:
            root = module.split(".", 1)[0]
            grouped.setdefault(root, []).append(module)
        for root in grouped:
            grouped[root].sort()
        return dict(sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])))

    grouped_missing = group_by_root(missing_in_coeus)
    grouped_extra = group_by_root(extra_in_coeus)

    print("\nMissing modules in coeus (grouped):")
    for root, items in grouped_missing.items():
        head = items[:max_list]
        suffix = "" if len(items) <= max_list else f" (+{len(items) - max_list} more)"
        print(f"- {root}: {len(items)}{suffix}")
        for module in head:
            print(f"  - {module}")

    if extra_in_coeus:
        print("\nExtra modules in coeus (grouped):")
        for root, items in grouped_extra.items():
            head = items[:max_list]
            suffix = "" if len(items) <= max_list else f" (+{len(items) - max_list} more)"
            print(f"- {root}: {len(items)}{suffix}")
            for module in head:
                print(f"  - {module}")

    return 0

def test_basic_tensor_operations():
    """Test basic tensor creation and operations."""
    print("\n[TEST] Testing basic tensor operations...")

    try:
        # Test tensor creation
        x = torch.tensor([1.0, 2.0, 3.0])
        print(f"[OK] Created tensor: {x}")

        # Test tensor operations
        y = torch.tensor([4.0, 5.0, 6.0])
        z = x + y
        print(f"[OK] Addition: {z}")

        # Test zeros, ones
        zeros = torch.zeros([3])
        ones = torch.ones([3])
        print(f"[OK] Zeros: {zeros}, Ones: {ones}")

        return True
    except Exception as e:
        print(f"[ERROR] Basic tensor operations failed: {e}")
        return False

def test_device_management():
    """Test device management."""
    print("\n[TEST] Testing device management...")

    try:
        # Test device creation
        cpu_dev = torch.device("cpu")
        print(f"[OK] CPU device: {cpu_dev}")

        cuda_dev = torch.cuda()
        print(f"[OK] CUDA device: {cuda_dev}")

        cpu_dev2 = torch.cpu()
        print(f"[OK] CPU device (alternative): {cpu_dev2}")

        return True
    except Exception as e:
        print(f"[ERROR] Device management failed: {e}")
        return False

def test_tensor_manipulation():
    """Test tensor manipulation operations."""
    print("\n[TEST] Testing tensor manipulation...")

    try:
        # Create test tensors
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([3.0, 4.0])

        # Test concatenation
        cat_result = torch.cat([t1, t2])
        print(f"[OK] Concatenation: {cat_result}")

        # Test stacking
        stack_result = torch.stack([t1, t2])
        print(f"[OK] Stacking: {stack_result}")

        # Test splitting
        split_result = torch.split(torch.tensor([1.0, 2.0, 3.0, 4.0]), 2)
        print(f"[OK] Splitting: {len(split_result)} chunks")

        # Test chunking
        chunk_result = torch.chunk(torch.tensor([1.0, 2.0, 3.0, 4.0]), 2)
        print(f"[OK] Chunking: {len(chunk_result)} chunks")

        return True
    except Exception as e:
        print(f"[ERROR] Tensor manipulation failed: {e}")
        return False

def test_neural_networks():
    """Test neural network components."""
    print("\n[TEST] Testing neural networks...")

    try:
        # Test Linear layer
        linear = nn.Linear(10, 5)
        input_tensor = torch.ones([3, 10])  # batch_size=3, input_size=10
        output = linear(input_tensor)
        print(f"[OK] Linear layer: input {input_tensor.shape} -> output {output.shape}")

        # Test activation functions
        relu = nn.ReLU()
        activated = relu(output)
        print(f"[OK] ReLU activation: {activated.shape}")

        sigmoid = nn.Sigmoid()
        sigmoid_out = sigmoid(output)
        print(f"[OK] Sigmoid activation: {sigmoid_out.shape}")

        tanh = nn.Tanh()
        tanh_out = tanh(output)
        print(f"[OK] Tanh activation: {tanh_out.shape}")

        return True
    except Exception as e:
        print(f"[ERROR] Neural networks failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_functional_api():
    """Test functional API."""
    print("\n[TEST] Testing functional API...")

    try:
        x = torch.tensor([-1.0, 0.5, 2.0])

        # Test functional activations
        relu_result = torch.nn.functional.relu(x)
        print(f"[OK] Functional ReLU: {relu_result}")

        sigmoid_result = torch.nn.functional.sigmoid(x)
        print(f"[OK] Functional Sigmoid: {sigmoid_result}")

        tanh_result = torch.nn.functional.tanh(x)
        print(f"[OK] Functional Tanh: {tanh_result}")

        return True
    except Exception as e:
        print(f"[ERROR] Functional API failed: {e}")
        return False

def test_sparse_tensors():
    """Test sparse tensor operations."""
    print("\n[TEST] Testing sparse tensors...")

    try:
        # Create CSR tensor
        # Matrix:
        # [1, 0, 2]
        # [0, 0, 3]
        # [4, 5, 0]
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        indices = [0, 2, 2, 0, 1]
        indptr = [0, 2, 3, 5]
        shape = [3, 3]

        csr = torch.sparse.sparse_csr_tensor(data, indices, indptr, shape)
        print(f"[OK] Created CSR tensor: {csr}")

        # Basic operations
        # Convert to dense (if supported, otherwise skip)
        dense = csr.to_dense()
        print(f"[OK] Converted to dense: {dense}")

        # Create COO tensor and add
        coo = torch.sparse.sparse_coo_tensor(data, [0, 0, 1, 2, 2], indices, shape)
        print(f"[OK] Created COO tensor: {coo}")
        
        # Test addition (COO + COO)
        sum_result = coo.add(coo)
        print(f"[OK] Sparse addition result (COO): {sum_result}")

        return True
    except Exception as e:
        print(f"[ERROR] Sparse tensors failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all compatibility tests."""
    print("Testing Coeus PyTorch Compatibility")
    print("=" * 50)

    tests = [
        test_basic_tensor_operations,
        test_device_management,
        test_tensor_manipulation,
        test_neural_networks,
        test_functional_api,
        test_sparse_tensors,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("SUCCESS: All PyTorch compatibility tests PASSED!")
        return 0
    else:
        print("FAILURE: Some tests failed")
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-diff", action="store_true")
    parser.add_argument("--include-private", action="store_true")
    parser.add_argument("--max-list", type=int, default=50)
    args = parser.parse_args()

    if args.module_diff:
        sys.exit(compare_torch_module_surface(args.include_private, args.max_list))

    sys.exit(main())
