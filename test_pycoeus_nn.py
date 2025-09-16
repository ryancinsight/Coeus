#!/usr/bin/env python3
"""
Comprehensive test script for PyCoeus neural network modules
"""

import sys
import numpy as np

try:
    import pycoeus as pc
    print("✅ PyCoeus imported successfully")

    # Test basic tensor functionality first
    print("\n=== BASIC TENSOR FUNCTIONALITY TESTS ===")

    # Create test tensors
    input_data = np.random.randn(4, 8).astype(np.float32)
    target_data = np.random.randn(4, 1).astype(np.float32)

    input_tensor = pc.PyTensor.from_numpy(input_data)
    target_tensor = pc.PyTensor.from_numpy(target_data)

    print(f"✅ Input tensor shape: {input_tensor.shape()}")
    print(f"✅ Target tensor shape: {target_tensor.shape()}")

    # Test neural network modules
    print("\n=== NEURAL NETWORK MODULE TESTS ===")

    # Test Linear layer
    print("\n--- Testing Linear Layer ---")
    try:
        linear_layer = pc.Linear(8, 4)
        output = linear_layer.forward(input_tensor)
        print(f"✅ Linear forward pass: {input_tensor.shape()} -> {output.shape()}")
        print(f"   Weight shape: {linear_layer.weight().shape()}")
        if linear_layer.bias():
            print(f"   Bias shape: {linear_layer.bias().shape()}")
    except Exception as e:
        print(f"❌ Linear layer test failed: {e}")

    # Test Conv2d layer
    print("\n--- Testing Conv2d Layer ---")
    try:
        # Reshape input for 2D convolution
        conv_input = input_tensor.reshape([2, 2, 2, 2])  # [batch, channels, height, width]
        conv_layer = pc.Conv2d(2, 4, 2, stride=1, padding=0)
        conv_output = conv_layer.forward(conv_input)
        print(f"✅ Conv2d forward pass: {conv_input.shape()} -> {conv_output.shape()}")
        print(f"   Weight shape: {conv_layer.weight().shape()}")
        if conv_layer.bias():
            print(f"   Bias shape: {conv_layer.bias().shape()}")
    except Exception as e:
        print(f"❌ Conv2d layer test failed: {e}")

    # Test ReLU activation
    print("\n--- Testing ReLU Activation ---")
    try:
        relu_layer = pc.ReLU()
        relu_output = relu_layer.forward(output)
        print(f"✅ ReLU forward pass: {output.shape()} -> {relu_output.shape()}")
    except Exception as e:
        print(f"❌ ReLU activation test failed: {e}")

    # Test MSE Loss
    print("\n--- Testing MSE Loss ---")
    try:
        mse_loss = pc.MSELoss()
        loss_output = mse_loss.forward(output, target_tensor)
        print(f"✅ MSE loss computed: shape {loss_output.shape()}")
        print(f"   Loss value: {loss_output.data()}")
    except Exception as e:
        print(f"❌ MSE loss test failed: {e}")

    # Test CrossEntropy Loss
    print("\n--- Testing CrossEntropy Loss ---")
    try:
        # Create classification targets (integer indices)
        ce_target_data = np.array([0, 1, 2, 1], dtype=np.int32)
        ce_target = pc.PyTensor(ce_target_data.tolist(), [4])

        ce_loss = pc.CrossEntropyLoss()
        ce_loss_output = ce_loss.forward(output, ce_target)
        print(f"✅ CrossEntropy loss computed: shape {ce_loss_output.shape()}")
        print(f"   Loss value: {ce_loss_output.data()}")
    except Exception as e:
        print(f"❌ CrossEntropy loss test failed: {e}")

    # Test SGD Optimizer
    print("\n--- Testing SGD Optimizer ---")
    try:
        # Create a simple model with parameters
        linear_params = [pc.PyTensor(np.random.randn(4, 8).astype(np.float32).flatten().tolist(), [4, 8])]
        sgd_optimizer = pc.SGD(linear_params, lr=0.01, momentum=0.9)

        # Test optimizer step
        sgd_optimizer.step()
        sgd_optimizer.zero_grad()
        print("✅ SGD optimizer step completed")
        print(f"   Optimizer parameters: {len(sgd_optimizer.parameters())}")
    except Exception as e:
        print(f"❌ SGD optimizer test failed: {e}")

    # Test Adam Optimizer
    print("\n--- Testing Adam Optimizer ---")
    try:
        adam_params = [pc.PyTensor(np.random.randn(4, 8).astype(np.float32).flatten().tolist(), [4, 8])]
        adam_optimizer = pc.Adam(adam_params, lr=0.001, beta1=0.9, beta2=0.999)

        # Test optimizer step
        adam_optimizer.step()
        adam_optimizer.zero_grad()
        print("✅ Adam optimizer step completed")
        print(f"   Optimizer parameters: {len(adam_optimizer.parameters())}")
    except Exception as e:
        print(f"❌ Adam optimizer test failed: {e}")

    # Test gradient computation
    print("\n=== GRADIENT COMPUTATION TESTS ===")

    # Create tensors with gradient tracking
    try:
        x = pc.PyTensor.from_data([1.0, 2.0, 3.0, 4.0], [4], True)
        y = pc.PyTensor.from_data([2.0, 3.0, 4.0, 5.0], [4], True)

        # Simple computation
        z = (x * y).sum()

        # Backward pass
        z.backward()

        x_grad = x.grad()
        y_grad = y.grad()

        if x_grad and y_grad:
            print("✅ Gradient computation successful")
            print(f"   x.grad = {x_grad.data()}")
            print(f"   y.grad = {y_grad.data()}")
        else:
            print("⚠️ Gradient computation incomplete - some gradients missing")

    except Exception as e:
        print(f"❌ Gradient computation test failed: {e}")

    print("\n🎉 PyCoeus neural network functionality test completed!")

except ImportError as e:
    print(f"❌ Failed to import PyCoeus: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
