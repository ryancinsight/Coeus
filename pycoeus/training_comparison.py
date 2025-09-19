#!/usr/bin/env python3
"""
PyCoeus vs PyTorch Training Comparison

This script demonstrates the current limitations of PyCoeus compared to PyTorch
for actual neural network training, highlighting the need for backpropagation.
"""

import sys
import numpy as np
sys.path.insert(0, 'python')

try:
    import pycoeus as pc
    import torch
    import torch.nn as torch_nn
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure PyCoeus is built and PyTorch/matplotlib are installed")
    sys.exit(1)

def create_synthetic_data(n_samples=100, n_features=3, noise=0.1):
    """Create synthetic regression data."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # True relationship: y = 2*x1 + 3*x2 - x3 + noise
    true_weights = np.array([2.0, 3.0, -1.0])
    y = X @ true_weights + noise * np.random.randn(n_samples)
    y = y.reshape(-1, 1).astype(np.float32)
    return X, y, true_weights

def test_pycoeus_forward_only():
    """Test PyCoeus forward pass capabilities."""
    print("=" * 60)
    print("PYCOEUS TESTING (Forward Pass Only)")
    print("=" * 60)
    
    X, y, true_weights = create_synthetic_data()
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"True weights: {true_weights}")
    
    # Convert to PyCoeus tensors
    X_pc = pc.tensor(X.tolist())
    y_pc = pc.tensor(y.tolist())
    
    # Create model
    model = pc.nn.Linear(3, 1)
    loss_fn = pc.nn.MseLoss()
    
    print(f"\nInitial model weights: {model.weight().data()}")
    
    # Forward pass
    y_pred = model.forward(X_pc)
    loss = loss_fn.forward(y_pred, y_pc)
    
    print(f"Forward pass completed")
    print(f"Initial loss: {loss.data()[0]:.6f}")
    
    # Multiple forward passes (no learning)
    print(f"\nRunning 10 'epochs' (forward passes only - no learning):")
    for epoch in range(10):
        y_pred = model.forward(X_pc)
        loss = loss_fn.forward(y_pred, y_pc)
        if epoch % 2 == 0:
            print(f"  Epoch {epoch + 1}: Loss = {loss.data()[0]:.6f}")
    
    print(f"\nFinal model weights: {model.weight().data()}")
    print("Note: Weights unchanged - no backpropagation implemented")
    
    return loss.data()[0]

def test_pytorch_training():
    """Test PyTorch with actual training."""
    print("\n" + "=" * 60)
    print("PYTORCH TESTING (Full Training)")
    print("=" * 60)
    
    X, y, true_weights = create_synthetic_data()
    
    # Convert to PyTorch tensors
    X_torch = torch.tensor(X)
    y_torch = torch.tensor(y)
    
    # Create model
    model = torch_nn.Linear(3, 1)
    loss_fn = torch_nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    print(f"Initial model weights: {model.weight.data.numpy().flatten()}")
    
    # Training loop
    losses = []
    print(f"\nRunning 10 epochs with backpropagation:")
    
    for epoch in range(10):
        optimizer.zero_grad()
        y_pred = model(X_torch)
        loss = loss_fn(y_pred, y_torch)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if epoch % 2 == 0:
            print(f"  Epoch {epoch + 1}: Loss = {loss.item():.6f}")
    
    print(f"\nFinal model weights: {model.weight.data.numpy().flatten()}")
    print(f"True weights:        {true_weights}")
    
    # Calculate improvement
    initial_loss = losses[0]
    final_loss = losses[-1]
    improvement = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"\nTraining Results:")
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss:   {final_loss:.6f}")
    print(f"  Improvement:  {improvement:.1f}%")
    
    return losses

def compare_predictions():
    """Compare predictions between PyCoeus and trained PyTorch."""
    print("\n" + "=" * 60)
    print("PREDICTION COMPARISON")
    print("=" * 60)
    
    X, y, true_weights = create_synthetic_data(n_samples=10)
    
    # PyCoeus (untrained)
    X_pc = pc.tensor(X.tolist())
    model_pc = pc.nn.Linear(3, 1)
    y_pred_pc = model_pc.forward(X_pc)
    pred_pc = np.array(y_pred_pc.data()).flatten()
    
    # PyTorch (trained)
    X_torch = torch.tensor(X)
    y_torch = torch.tensor(y)
    model_torch = torch_nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model_torch.parameters(), lr=0.01)
    loss_fn = torch_nn.MSELoss()
    
    # Quick training
    for _ in range(50):
        optimizer.zero_grad()
        y_pred = model_torch(X_torch)
        loss = loss_fn(y_pred, y_torch)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        y_pred_torch = model_torch(X_torch)
    pred_torch = y_pred_torch.numpy().flatten()
    
    # True predictions using true weights
    true_pred = X @ true_weights
    
    print("Sample predictions (first 5 samples):")
    print(f"{'True values:':<20} {y.flatten()[:5]}")
    print(f"{'True function:':<20} {true_pred[:5]}")
    print(f"{'PyCoeus (untrained):':<20} {pred_pc[:5]}")
    print(f"{'PyTorch (trained):':<20} {pred_torch[:5]}")
    
    # Calculate errors
    error_pc = np.mean((pred_pc - y.flatten())**2)
    error_torch = np.mean((pred_torch - y.flatten())**2)
    error_true = np.mean((true_pred - y.flatten())**2)
    
    print(f"\nMean Squared Errors:")
    print(f"  True function:       {error_true:.6f}")
    print(f"  PyCoeus (untrained): {error_pc:.6f}")
    print(f"  PyTorch (trained):   {error_torch:.6f}")

def main():
    """Main comparison function."""
    print("PyCoeus vs PyTorch Training Comparison")
    print("This demo shows why backpropagation is essential for neural network training")
    
    # Test PyCoeus
    pycoeus_loss = test_pycoeus_forward_only()
    
    # Test PyTorch
    pytorch_losses = test_pytorch_training()
    
    # Compare predictions
    compare_predictions()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("PyCoeus Current Status:")
    print("  ✅ Forward passes work correctly")
    print("  ✅ Tensor operations implemented")
    print("  ✅ Loss functions compute correctly")
    print("  ✅ PyTorch-compatible API")
    print("  ❌ No automatic differentiation")
    print("  ❌ No backpropagation")
    print("  ❌ Optimizers don't update parameters")
    
    print("\nWhat's needed for full training:")
    print("  1. Automatic differentiation (autograd)")
    print("  2. Backward pass implementation")
    print("  3. Gradient computation")
    print("  4. Parameter update mechanisms")
    print("  5. Optimizer step() functionality")
    
    print(f"\nPyTorch achieved {((pytorch_losses[0] - pytorch_losses[-1]) / pytorch_losses[0] * 100):.1f}% loss reduction")
    print("PyCoeus loss remained constant (no learning)")
    
    print("\nPyCoeus is a solid foundation but needs backpropagation for training!")

if __name__ == "__main__":
    main()