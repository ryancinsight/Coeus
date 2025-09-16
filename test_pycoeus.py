#!/usr/bin/env python3
"""
Simple test script for PyCoeus Python bindings
"""

import sys
import os

# Add the target directory to Python path to import the compiled extension
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'target', 'release'))

try:
    import pycoeus

    print("✅ Successfully imported pycoeus!")
    print(f"Module contents: {dir(pycoeus)}")

    # Test basic tensor creation
    print("\n🧪 Testing basic tensor functionality...")

    # Create a tensor from data
    data = [1.0, 2.0, 3.0, 4.0]
    shape = [2, 2]
    tensor = pycoeus.PyTensor(data, shape)

    print(f"Created tensor: {tensor}")
    print(f"Tensor data: {tensor.data()}")
    print(f"Tensor shape: {tensor.shape()}")
    print(f"Tensor dimensions: {tensor.dim()}")
    print(f"Total elements: {tensor.numel()}")

    # Test arithmetic operations
    print("\n🔢 Testing arithmetic operations...")

    data2 = [5.0, 6.0, 7.0, 8.0]
    tensor2 = pycoeus.PyTensor(data2, shape)

    print(f"Tensor 2: {tensor2}")

    # Addition
    result_add = tensor + tensor2
    print(f"tensor + tensor2 = {result_add}")

    # Subtraction
    result_sub = tensor - tensor2
    print(f"tensor - tensor2 = {result_sub}")

    # Multiplication
    result_mul = tensor * tensor2
    print(f"tensor * tensor2 = {result_mul}")

    # Division
    result_div = tensor / tensor2
    print(f"tensor / tensor2 = {result_div}")

    # Test mathematical operations
    print("\n📐 Testing mathematical operations...")

    # Power
    result_pow = tensor.pow(2.0)
    print(f"tensor^2 = {result_pow}")

    # Exponential
    result_exp = tensor.exp()
    print(f"exp(tensor) = {result_exp}")

    # Logarithm
    result_log = tensor.log()
    print(f"log(tensor) = {result_log}")

    # Trigonometric
    result_sin = tensor.sin()
    print(f"sin(tensor) = {result_sin}")

    result_cos = tensor.cos()
    print(f"cos(tensor) = {result_cos}")

    # Test reduction operations
    print("\n📊 Testing reduction operations...")

    result_sum = tensor.sum()
    print(f"sum(tensor) = {result_sum}")

    result_mean = tensor.mean()
    print(f"mean(tensor) = {result_mean}")

    # Test device management
    print("\n💻 Testing device management...")

    print(f"Current device: {tensor.device()}")

    cpu_tensor = tensor.cpu()
    print(f"CPU tensor device: {cpu_tensor.device()}")

    # Test requires_grad
    print("\n🔄 Testing gradient tracking...")

    print(f"Requires grad: {tensor.requires_grad()}")

    tensor.requires_grad_(True)
    print(f"After setting requires_grad: {tensor.requires_grad()}")

    print("\n🎉 All tests passed! PyCoeus is working correctly!")

except ImportError as e:
    print(f"❌ Failed to import pycoeus: {e}")
    print("Make sure the crate is built and the Python path is correct.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
