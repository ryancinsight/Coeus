#!/usr/bin/env python3
"""
Coeus Python Usage Example

This example demonstrates using Coeus from Python with the same
API as PyTorch, showcasing seamless migration and interoperability.
"""

import numpy as np
import coeus

def main():
    print("🐍 Coeus Python API Example")
    print("==========================\n")

    # Basic tensor creation
    print("1. Creating tensors:")
    a = coeus.tensor([1.0, 2.0, 3.0])
    b = coeus.tensor([4.0, 5.0, 6.0])
    print(f"   a = {a}")
    print(f"   b = {b}")

    # Arithmetic operations
    print("\n2. Arithmetic operations:")
    c = a + b
    print(f"   a + b = {c}")

    d = c * 2.0
    print(f"   (a + b) * 2 = {d}")

    # Broadcasting
    print("\n3. Broadcasting:")
    scalar = coeus.tensor([10.0])
    broadcasted = scalar + a
    print(f"   scalar + vector = {broadcasted}")

    # Matrix operations
    print("\n4. Matrix operations:")
    m1 = coeus.tensor([[1.0, 2.0], [3.0, 4.0]])
    m2 = coeus.tensor([[5.0, 6.0], [7.0, 8.0]])
    product = m1 @ m2
    print(f"   Matrix multiplication:\n   {m1}\n   @\n   {m2}\n   =\n   {product}")

    # Neural network
    print("\n5. Neural network example:")
    # Create a simple sequential model
    model = coeus.nn.Sequential(
        coeus.nn.Linear(2, 4),
        coeus.nn.ReLU(),
        coeus.nn.Linear(4, 1),
        coeus.nn.Sigmoid()
    )

    # Sample input
    x = coeus.tensor([[0.5, 0.8]])
    output = model(x)
    print(f"   Input: {x}")
    print(f"   Model output: {output}")

    # Training example
    print("\n6. Training example:")
    # Simple binary classification data
    X = coeus.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = coeus.tensor([[0.0], [1.0], [1.0], [0.0]])  # XOR

    # Simple model for XOR
    xor_model = coeus.nn.Sequential(
        coeus.nn.Linear(2, 2),
        coeus.nn.ReLU(),
        coeus.nn.Linear(2, 1)
    )

    # Optimizer
    optimizer = coeus.optim.SGD(xor_model.parameters(), lr=0.1)
    loss_fn = coeus.nn.MSELoss()

    print("   Training XOR classifier...")
    for epoch in range(100):
        # Forward pass
        pred = xor_model(X)
        loss = loss_fn(pred, y)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 20 == 0:
            print(".4f")

    # Test the trained model
    print("\n   Testing trained model:")
    with coeus.no_grad():
        test_pred = xor_model(X)
        print(f"   Predictions: {test_pred.squeeze()}")
        print(f"   Targets:     {y.squeeze()}")
        print(f"   Accuracy:    {((test_pred > 0.5) == y).float().mean().item():.1%}")

    print("\n✅ Python usage example completed!")
    print("\n💡 Key takeaways:")
    print("   • PyTorch-compatible API for seamless migration")
    print("   • Memory-safe Rust performance under the hood")
    print("   • Zero-copy tensor operations with NumPy interoperability")
    print("   • Automatic differentiation with .backward()")
    print("   • Neural network modules with familiar PyTorch syntax")

if __name__ == "__main__":
    main()
