# Coeus Tutorial: Safe PyTorch in Rust

This tutorial provides a comprehensive introduction to Coeus, demonstrating how to build, train, and deploy neural networks with memory safety and performance guarantees.

## Table of Contents

1. [Installation](#installation)
2. [Basic Tensor Operations](#basic-tensor-operations)
3. [Automatic Differentiation](#automatic-differentiation)
4. [Building Neural Networks](#building-neural-networks)
5. [Training Loops](#training-loops)
6. [Advanced Features](#advanced-features)
7. [Python Integration](#python-integration)
8. [Production Deployment](#production-deployment)

## Installation

### Rust Installation

```bash
# Install Rust (1.70+ required)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add required components
rustup component add clippy rustfmt miri

# Install development tools
cargo install cargo-tarpaulin cargo-criterion cargo-udeps
```

### Project Setup

```bash
# Clone the repository
git clone https://github.com/ryancinsight/coeus.git
cd coeus

# Build all crates
cargo build --release

# Run tests
cargo test --workspace

# Run examples
cargo run --example basic_usage
```

## Basic Tensor Operations

Coeus provides a type-safe tensor hierarchy: `Tensor<B<S<T>>>` where:
- `T`: Data type (f32, i32, etc.)
- `S`: Storage layout (Dense, Sparse)
- `B`: Compute backend (CPU, GPU)

### Creating Tensors

```rust
use coeus_tensor::Tensor;
use coeus_backend::CpuBackend;
use coeus_storage::DenseStorage;
use coeus_dtype::float::Float32;

fn main() -> Result<(), Box<dyn<std::error::Error>>> {
    // Create tensor from vector
    let a = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[3]
    )?;

    // Create tensors with fill values
    let zeros = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[2, 3])?;
    let ones = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[2, 3])?;

    println!("a: {:?}", a.as_slice());
    println!("zeros shape: {:?}", zeros.shape().dims());
    Ok(())
}
```

### Arithmetic Operations

```rust
// Element-wise operations
let b = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
    vec![Float32::new(4.0), Float32::new(5.0), Float32::new(6.0)],
    &[3]
)?;

// Addition (supports borrowing for zero-copy)
let c = &a + &b;

// Scalar operations
let d = &c * Float32::new(2.0);

// Broadcasting
let scalar = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
    vec![Float32::new(10.0)],
    &[1]
)?;
let broadcasted = &scalar + &a; // [11.0, 12.0, 13.0]
```

### Shape Manipulation

```rust
// Reshape with dimension inference (-1 for auto-inference)
let matrix = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
    (1..=6).map(|x| Float32::new(x as f32)).collect(),
    &[2, 3]
)?;

// Reshape to 3x2 (6 elements total preserved)
let reshaped = matrix.reshape(&[3, 2])?;

// Transpose dimensions
let transposed = reshaped.transpose(0, 1)?;
```

### Matrix Operations

```rust
let m1 = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
    vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)],
    &[2, 2]
)?;

let m2 = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
    vec![Float32::new(5.0), Float32::new(6.0), Float32::new(7.0), Float32::new(8.0)],
    &[2, 2]
)?;

// Matrix multiplication
let product = m1.matmul(&m2)?;
```

## Automatic Differentiation

Coeus implements reverse-mode automatic differentiation with memory-efficient gradient computation.

### Variable Wrapping

```rust
use coeus_autograd::Variable;

// Wrap tensors in Variables for gradient tracking
let x = Variable::new(tensor_x);
let y = Variable::new(tensor_y);

// Variables track computation history
let z = &x + &y;  // Creates computation graph
```

### Gradient Computation

```rust
use coeus_autograd::backward;

// Build computation graph
let loss = &z * &z;  // loss = z²

// Compute gradients
backward(&[&loss], &[])?;

// Access gradients
if let Ok(grad) = x.grad() {
    println!("∂loss/∂x = {:?}", grad.as_slice());
}
```

### Higher-Order Operations

```rust
// Supported operations with automatic gradients
let exp_result = x.exp();       // e^x
let log_result = x.log();       // ln(x)
let sin_result = x.sin();       // sin(x)
let cos_result = x.cos();       // cos(x)
let pow_result = x.pow(&y);     // x^y

// Reductions
let sum_result = x.sum();       // sum(x)
let mean_result = x.mean();     // mean(x)
```

## Building Neural Networks

Coeus provides a modular neural network system with PyTorch-compatible APIs.

### Module System

```rust
use coeus_nn::{Module, Linear, Sequential};
use coeus_optim::SGD;

// Define a custom module
struct MyModel<B: Backend, T: DataType> {
    fc1: Linear<B, T>,
    fc2: Linear<B, T>,
}

impl<B: Backend, T: DataType> Module<B, T> for MyModel<B, T> {
    fn forward(&self, input: &Tensor<B, DenseStorage<T>, T>) -> Result<Tensor<B, DenseStorage<T>, T>> {
        let x = self.fc1.forward(input)?;
        let x = x.relu()?;  // Element-wise ReLU
        self.fc2.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter<T>> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }
}
```

### Sequential Composition

```rust
use coeus_backend::CpuBackend;
use coeus_dtype::float::Float32;

// Build network with Sequential container
// Note: Explicit type annotation required for type inference
let mut model: Sequential<CpuBackend, Float32> = Sequential::new();
model.add_module("fc1".to_string(), Linear::new(784, 128));
model.add_module("relu1".to_string(), ReLU::new());
model.add_module("fc2".to_string(), Linear::new(128, 64));
model.add_module("relu2".to_string(), ReLU::new());
model.add_module("fc3".to_string(), Linear::new(64, 10));
```

### Built-in Layers

```rust
// Linear layers
let linear = Linear::new(input_dim, output_dim)?;

// Activation functions
let relu = ReLU::new();
let sigmoid = Sigmoid::new();
let tanh = Tanh::new();

// Loss functions
let mse_loss = MSELoss::new();
let cross_entropy = CrossEntropyLoss::new();
```

## Training Loops

Complete training workflow with optimizers and loss functions.

### Optimizer Setup

```rust
use coeus_optim::{SGD, Adam};
use coeus_dtype::float::Float32;

// Stochastic Gradient Descent (basic configuration)
let optimizer: SGD<Float32> = SGD::basic(0.01)?;

// SGD with full control over hyperparameters
let optimizer: SGD<Float32> = SGD::new(
    0.01,   // learning_rate
    0.0,    // momentum
    0.0,    // weight_decay
    0.0,    // dampening
    false   // nesterov
)?;

// Adam optimizer
let optimizer: Adam<Float32> = Adam::new(
    0.001,  // learning_rate
    0.9,    // beta1
    0.999,  // beta2
    1e-8,   // epsilon
    0.0,    // weight_decay
    false   // amsgrad
)?;
```

### Training Loop

```rust
// Training hyperparameters
let epochs = 100;
let batch_size = 32;

// Training loop
for epoch in 0..epochs {
    let mut epoch_loss = 0.0;

    // Iterate over batches
    for (batch_x, batch_y) in train_loader {
        // Forward pass
        let predictions = model.forward(&batch_x)?;
        let loss = loss_fn.forward(&predictions, &batch_y)?;

        // Backward pass
        backward(&[&loss], &[])?;

        // Update parameters
        optimizer.step()?;
        optimizer.zero_grad();

        epoch_loss += loss.data().as_slice()[0].value();
    }

    println!("Epoch {}: Loss = {:.4}", epoch, epoch_loss);
}
```

### Model Persistence

```rust
use std::path::Path;

// Save model
model.save(Path::new("model.json"))?;

// Load model
let loaded_model = MyModel::load(Path::new("model.json"))?;
```

## Advanced Features

### Tracing and Observability

```rust
use tracing::{info, instrument};
use tracing_subscriber;

#[instrument(level = "trace", skip(input), fields(input_shape = ?input.shape().dims()))]
fn forward_pass(input: &Tensor) -> Result<Tensor> {
    info!("Starting forward pass");
    // ... implementation
}
```

### Performance Benchmarking

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_tensor_ops(c: &mut Criterion) {
    c.bench_function("tensor_addition", |b| {
        b.iter(|| {
            let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
            let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
            let _c = &a + &b;
        })
    });
}

criterion_group!(benches, benchmark_tensor_ops);
criterion_main!(benches);
```

## Python Integration

Coeus provides seamless Python integration via PyO3.

### Installation

```bash
# Install from source
pip install maturin
cd pycoeus
maturin develop

# Or build wheel
maturin build --release
pip install target/wheels/*.whl
```

### Python API

```python
import coeus as torch

# PyTorch-compatible API
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
z = x + y  # tensor([5., 7., 9.])

# Neural networks
model = torch.nn.Sequential(
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    # ... training code identical to PyTorch
    pass
```

### NumPy Interoperability

```python
import numpy as np
import coeus as torch

# From NumPy to Coeus (zero-copy when possible)
numpy_array = np.array([1.0, 2.0, 3.0])
coeus_tensor = torch.from_numpy(numpy_array)

# From Coeus to NumPy
numpy_result = coeus_tensor.numpy()
```

## Production Deployment

### CI/CD Pipeline

Coeus includes comprehensive CI/CD with GitHub Actions:

- **Multi-platform testing**: Ubuntu, Windows, macOS
- **Python wheel building**: Automated distribution
- **Coverage reporting**: Codecov integration
- **Security auditing**: Cargo audit integration

### Performance Characteristics

- **Memory Safety**: Zero unsafe code, Miri-validated
- **Performance**: 1.87x to 19.51x speedup vs PyTorch CPU
- **Memory Usage**: <110% of PyTorch memory usage
- **Thread Safety**: Race-free concurrent execution

### Observability

```bash
# Enable detailed tracing
RUST_LOG=coeus_tensor=trace,coeus_autograd=debug cargo run

# Production logging
RUST_LOG=warn cargo run
```

### Benchmarking

```bash
# Run performance benchmarks
cargo criterion

# Profile memory usage
cargo build --release
# Use system profiling tools
```

## Next Steps

- Explore the [examples/](../examples/) directory
- Read the [API documentation](https://docs.rs/coeus)
- Check the [ADR documentation](../docs/adr.md) for design decisions
- Join the community at [GitHub Issues](https://github.com/ryancinsight/coeus/issues)

Coeus brings the safety and performance of Rust to deep learning, enabling reliable AI systems without sacrificing usability or performance.
