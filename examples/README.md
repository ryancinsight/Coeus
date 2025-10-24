# Coeus Examples

This directory contains comprehensive examples demonstrating how to use the Coeus deep learning framework for various machine learning tasks.

## Table of Contents

- [Quick Start](#quick-start)
- [Basic Examples](#basic-examples)
- [Advanced Examples](#advanced-examples)
- [Integration Examples](#integration-examples)
- [Distributed Training](#distributed-training)
- [Performance Examples](#performance-examples)
- [Contributing](#contributing)

## Quick Start

### Prerequisites

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone the repository
git clone https://github.com/your-repo/coeus.git
cd coeus

# Run a basic example
cargo run --example basic_usage
```

### Running Examples

All examples can be run using Cargo:

```bash
# Run a specific example
cargo run --example example_name

# Run with release optimizations
cargo run --release --example example_name

# List all available examples
cargo run --example --list
```

## Basic Examples

### 1. Basic Usage (`basic_usage.rs`)

Demonstrates fundamental tensor operations and neural network components.

```bash
cargo run --example basic_usage
```

**Features:**
- Tensor creation and manipulation
- Linear layer operations
- Basic forward pass
- Loss computation

### 2. Neural Network (`neural_network.rs`)

Shows how to build and train a simple neural network for classification.

```bash
cargo run --example neural_network
```

**Features:**
- Model architecture definition
- Training loop implementation
- Loss computation and optimization
- Basic evaluation

### 3. Custom Layer (`custom_layer.rs`)

Illustrates how to create custom neural network layers.

```bash
cargo run --example custom_layer
```

**Features:**
- Custom layer implementation
- Parameter management
- Gradient computation
- Integration with existing models

## Advanced Examples

### 4. Advanced Training (`advanced_training.rs`)

Comprehensive training example with monitoring, validation, and early stopping.

```bash
cargo run --example advanced_training
```

**Features:**
- Training/validation split
- Learning rate scheduling
- Early stopping
- Training metrics tracking
- Model checkpointing

### 5. Advanced Features (`advanced_features.rs`)

Showcases advanced Coeus features like mixed precision and gradient clipping.

```bash
cargo run --example advanced_features
```

**Features:**
- Mixed precision training (FP16/FP32)
- Gradient clipping
- Advanced optimization techniques
- Performance profiling

### 6. Comprehensive Training (`comprehensive_training.rs`)

Complete machine learning workflow from data preparation to model deployment.

```bash
cargo run --example comprehensive_training
```

**Features:**
- Data preprocessing and loading
- Model training with monitoring
- Validation and early stopping
- Performance profiling
- Model checkpointing
- Evaluation and inference

## Integration Examples

### 7. GPU Basic (`gpu_basic.rs`)

Demonstrates the current GPU backend stub implementation.

```bash
cargo run --example gpu_basic
```

**Features:**
- GPU backend initialization
- Demonstrates stub operations (returns UnsupportedOperation)
- Foundation for future GPU acceleration development

### 8. Distributed Training (`distributed_training.rs`)

Multi-GPU and multi-node training examples.

```bash
cargo run --example distributed_training
```

**Features:**
- Data parallelism
- Gradient synchronization
- Multi-GPU training
- Fault tolerance

### 9. Mixed Precision Training (`mixed_precision.rs`)

Automatic mixed precision training with FP16 and gradient scaling.

```bash
cargo run --example mixed_precision
```

**Features:**
- FP16 operations for memory efficiency
- Loss scaling for numerical stability
- Gradient scaling to prevent underflow
- NaN/Inf detection and handling
- Automatic loss scale adjustment

### 9. Parallel Training (`parallel_training.rs`)

High-performance parallel training techniques.

```bash
cargo run --example parallel_training
```

**Features:**
- Parallel data loading
- Asynchronous training
- Performance optimization

## Distributed Training

### Data Parallelism

```bash
cargo run --example distributed_training
```

Demonstrates how to train models across multiple GPUs using data parallelism.

### Gradient Synchronization

The distributed training examples show how to:
- Synchronize gradients across devices
- Handle communication failures
- Scale training to multiple nodes

## Performance Examples


### Tracing (`tracing.rs`)

Demonstrates the tracing integration for debugging and monitoring.

```bash
cargo run --example tracing
```

**Features:**
- Structured logging
- Performance tracing
- Debug information collection

## Specialized Examples

### Sparse Operations (`sparse_integration_test.rs`)

Working with sparse tensors and models.

```bash
cargo run --example sparse_integration_test
```

### Python Integration (`python_usage.py`)

Using Coeus from Python via PyCoeus.

```bash
python3 examples/python_usage.py
```

### Tutorial Validation (`tutorial_validation.rs`)

Validates the tutorial examples and documentation.

```bash
cargo run --example tutorial_validation
```

## Example Structure

Each example follows a consistent structure:

1. **Imports**: Required Coeus components
2. **Setup**: Model and data initialization
3. **Training/Evaluation**: Core functionality demonstration
4. **Results**: Output and analysis
5. **Cleanup**: Resource cleanup (if needed)

## Learning Path

For new users, we recommend following this learning path:

1. **Start with basics**: `basic_usage.rs`
2. **Build networks**: `neural_network.rs`
3. **Custom components**: `custom_layer.rs`
4. **Advanced training**: `advanced_training.rs` or `comprehensive_training.rs`
5. **Mixed precision**: `mixed_precision.rs` for FP16 training
6. **Performance**: Use `tracing.rs` for monitoring and profiling
7. **Scale up**: `distributed_training.rs` or `gpu_basic.rs`

## API Documentation

All examples are thoroughly documented with inline comments explaining:
- What each section does
- Why certain design decisions were made
- Expected outputs and behaviors
- Performance considerations

## Testing Examples

Examples include integrated tests that can be run with:

```bash
cargo test --example example_name
```

## Contributing

When adding new examples:

1. Follow the existing code style and structure
2. Include comprehensive documentation
3. Add appropriate tests
4. Update this README
5. Ensure examples work with `cargo run --example`

### Example Template

```rust
//! # Example Title
//!
//! Brief description of what this example demonstrates.
//!
//! ## Features
//!
//! - Feature 1
//! - Feature 2

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Implementation
    Ok(())
}

#[cfg(test)]
mod tests {
    // Tests for the example
}
```

## Troubleshooting

### Common Issues

1. **Compilation Errors**: Ensure you have the latest Rust toolchain
2. **GPU Examples**: Requires compatible GPU and drivers
3. **Distributed Examples**: May require multiple GPUs or nodes
4. **Performance**: Use `--release` for optimized performance

### Getting Help

- Check the [main documentation](../README.md)
- Review [API documentation](../target/doc/coeus/)
- Open an issue for bugs or questions

## Performance Benchmarks

Some examples include performance benchmarks. Run with:

```bash
cargo bench --example example_name
```

This helps compare performance across different configurations and optimizations.
