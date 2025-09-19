# PyCoeus

PyTorch-compatible neural network library built in Rust with Python bindings.

## Features

- **PyTorch Compatibility**: Drop-in replacement for most PyTorch operations
- **High Performance**: Built in Rust for maximum speed and memory efficiency
- **Comprehensive**: 18+ loss functions, 10+ activation functions, and growing
- **GPU Support**: CUDA acceleration for training and inference
- **Type Safety**: Rust's type system prevents common ML bugs

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/coeus-ai/coeus.git
cd coeus/pycoeus

# Install with maturin
pip install maturin
maturin develop --release

# Or build wheel
maturin build --release
pip install target/wheels/pycoeus-*.whl
```

## Quick Start

```python
import pycoeus as pc
import numpy as np

# Create tensors
x = pc.PyTensor.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
y = pc.PyTensor.from_numpy(np.array([[2.0, 1.0], [1.0, 3.0]]))

# Build a model
linear1 = pc.Linear(2, 4)
relu = pc.ReLU()
linear2 = pc.Linear(4, 2)

# Forward pass
hidden = linear1.forward(x)
activated = relu.forward(hidden)
output = linear2.forward(activated)

# Compute loss
mse_loss = pc.MSELoss()
loss = mse_loss.forward(output, y)

print(f"Loss: {loss.data()}")
```

## Neural Network Components

### Layers
- `Linear` - Fully connected layer
- `Conv2d` - 2D convolution layer
- `RNN`, `LSTM`, `GRU` - Recurrent layers
- `Embedding` - Token embeddings
- `Dropout` - Regularization

### Activation Functions
- `ReLU` - Rectified Linear Unit
- `Sigmoid` - Logistic function
- `Tanh` - Hyperbolic tangent
- `Softmax` - Probability distribution

### Loss Functions
- **Regression**: `MSELoss`, `L1Loss`
- **Classification**: `CrossEntropyLoss`, `NLLLoss`, `BCELoss`
- **Ranking**: `MarginRankingLoss`, `TripletMarginLoss`
- **Specialized**: `FocalLoss`, `DiceLoss`, `IoULoss`

### Optimizers
- `SGD` - Stochastic Gradient Descent
- `Adam` - Adaptive Moment Estimation
- `AdamW` - Adam with weight decay
- `RMSprop` - Root Mean Square Propagation

## Performance Comparison

PyCoeus aims to match or exceed PyTorch performance while providing better memory safety through Rust.

## Development Status

PyCoeus is in active development. Current status:

- ✅ Core tensor operations
- ✅ Automatic differentiation
- ✅ Neural network modules (18+ implemented)
- ✅ Loss functions (18+ implemented)
- ✅ Basic optimizers
- 🚧 Advanced optimizers
- 🚧 Learning rate schedulers
- 🚧 Data loading utilities

## Contributing

We welcome contributions! Please see our [Contributing Guide](../CONTRIBUTING.md) for details.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE))
- MIT License ([LICENSE-MIT](../LICENSE-MIT))

at your option.