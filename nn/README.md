# Coeus Neural Networks

Neural network layers and operations for the Coeus deep learning framework.

## Overview

This crate provides neural network functionality with a clear separation between stateless operations and stateful layers, following the single source of truth principle. The architecture enables zero-cost abstractions through compile-time monomorphization while maintaining PyTorch API compatibility.

## Features

- **Stateless Operations**: Pure functions in `functional/ops/` module
- **Stateful Layers**: Thin wrappers in `modules/` that delegate to operations
- **Single Source of Truth**: Each operation implemented exactly once
- **B<S<T>> Architecture**: Generic over Backend, Storage, and DataType
- **PyTorch Compatibility**: Familiar API for seamless migration
- **Automatic Differentiation**: Full autograd integration
- **Hierarchical File Structure**: Deep vertical organization for parity tracking

## Architecture Overview

The NN crate follows a clear architectural separation:

```
┌─────────────────────────────────────────────────────────────┐
│                    Stateful Layers (modules/)                │
│  - Store parameters and configuration                        │
│  - Delegate computation to functional/ops/                   │
│  - Implement Module<B, S, T> trait                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Stateless Operations (functional/ops/)       │
│  - Pure functions with no state                             │
│  - Single source of truth for all computations              │
│  - Generic over B<S<T>> architecture                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Single Source of Truth**: Each operation is implemented exactly once in `functional/ops/`
2. **Delegation Pattern**: Layers delegate all computation to operations
3. **Separation of Concerns**: State management separate from computation
4. **Domain Separation**: NN operations stay within NN crate boundaries
5. **Hierarchical Organization**: Deep vertical file structure for parity tracking

## File Structure

The NN crate uses a hierarchical file structure that enables script-based parity tracking:

### Functional Operations (`src/functional/ops/`)

Stateless pure functions organized by category:

```
nn/src/functional/ops/
├── activation/
│   ├── relu.rs             - ReLU activation function
│   ├── gelu.rs             - GELU activation function
│   ├── softmax.rs          - Softmax activation function
│   ├── sigmoid.rs          - Sigmoid activation function
│   ├── tanh.rs             - Tanh activation function
│   └── mod.rs              - Module exports
├── loss/
│   ├── mse.rs              - Mean Squared Error loss
│   ├── cross_entropy.rs    - Cross-entropy loss
│   ├── nll.rs              - Negative Log Likelihood loss
│   └── mod.rs              - Module exports
├── convolution/
│   ├── conv1d.rs           - 1D convolution operation
│   ├── conv2d.rs           - 2D convolution operation
│   ├── conv3d.rs           - 3D convolution operation
│   ├── conv_transpose.rs   - Transposed convolution
│   └── mod.rs              - Module exports
├── linear/
│   ├── dense.rs            - Dense linear transformation
│   ├── sparse.rs           - Sparse linear transformation
│   └── mod.rs              - Module exports
├── normalization/
│   ├── batch_norm.rs       - Batch normalization
│   ├── layer_norm.rs       - Layer normalization
│   ├── group_norm.rs       - Group normalization
│   └── mod.rs              - Module exports
├── pooling/
│   ├── max_pool.rs         - Max pooling operations
│   ├── avg_pool.rs         - Average pooling operations
│   ├── adaptive_pool.rs    - Adaptive pooling operations
│   └── mod.rs              - Module exports
├── attention/
│   ├── multi_head.rs       - Multi-head attention
│   ├── self_attention.rs   - Self-attention mechanism
│   ├── cross_attention.rs  - Cross-attention mechanism
│   └── mod.rs              - Module exports
└── mod.rs                  - Top-level exports
```

### Stateful Modules (`src/modules/`)

Thin wrappers that store parameters and delegate to operations:

```
nn/src/modules/
├── activation/
│   ├── relu.rs             - ReLU layer wrapper
│   ├── gelu.rs             - GELU layer wrapper
│   ├── softmax.rs          - Softmax layer wrapper
│   └── mod.rs              - Module exports
├── loss/
│   ├── mse.rs              - MSE loss wrapper
│   ├── cross_entropy.rs    - Cross-entropy loss wrapper
│   └── mod.rs              - Module exports
├── convolution/
│   ├── conv1d.rs           - Conv1D layer wrapper
│   ├── conv2d.rs           - Conv2D layer wrapper
│   ├── conv3d.rs           - Conv3D layer wrapper
│   └── mod.rs              - Module exports
├── linear/
│   ├── linear.rs           - Linear layer wrapper
│   └── mod.rs              - Module exports
├── normalization/
│   ├── batch_norm.rs       - BatchNorm layer wrapper
│   ├── layer_norm.rs       - LayerNorm layer wrapper
│   └── mod.rs              - Module exports
├── pooling/
│   ├── max_pool.rs         - MaxPool layer wrapper
│   ├── avg_pool.rs         - AvgPool layer wrapper
│   └── mod.rs              - Module exports
├── attention/
│   ├── multi_head.rs       - MultiHeadAttention layer
│   ├── transformer.rs      - Transformer layer
│   └── mod.rs              - Module exports
├── containers/
│   ├── sequential.rs       - Sequential container
│   ├── module_list.rs      - ModuleList container
│   └── mod.rs              - Module exports
└── mod.rs                  - Top-level exports
```

### Core Infrastructure (`src/`)

```
nn/src/
├── lib.rs                  - Public API and module declarations
├── error.rs                - Error types for NN operations
├── parameter.rs            - Parameter wrapper for tensors
├── module.rs               - Module trait definition
├── init.rs                 - Parameter initialization functions
└── utils.rs                - Utility functions
```

## Usage Examples

### Using Functional Operations

```rust
use coeus_nn::functional::ops::activation::relu;
use coeus_nn::functional::ops::linear::linear;
use coeus_tensor::Tensor;
use coeus_backend::CpuBackend;
use coeus_storage::DenseStorage;
use coeus_dtype::float::Float32;

type CpuTensor = Tensor<CpuBackend, DenseStorage<Float32>, Float32>;

// Create input tensor
let input = CpuTensor::from_vec(vec![1.0, -2.0, 3.0, -4.0], &[2, 2])?;

// Apply ReLU activation (stateless function)
let activated = relu(&input)?;

// Apply linear transformation
let weight = CpuTensor::from_vec(vec![0.5, 0.3, 0.2, 0.8], &[2, 2])?;
let bias = Some(CpuTensor::from_vec(vec![0.1, 0.2], &[2])?);
let output = linear(&input, &weight, bias.as_ref())?;
```

### Using Stateful Layers

```rust
use coeus_nn::modules::{Linear, ReLU, Sequential, Module};
use coeus_nn::Parameter;

// Create individual layers
let linear1 = Linear::new(784, 128, true)?;  // input_size, output_size, bias
let relu = ReLU::new();
let linear2 = Linear::new(128, 10, true)?;

// Create sequential model
let model = Sequential::new(vec![
    Box::new(linear1),
    Box::new(relu),
    Box::new(linear2),
]);

// Forward pass
let input = CpuTensor::zeros(&[32, 784])?;  // batch_size=32, features=784
let output = model.forward(&input)?;

// Access parameters
let params = model.parameters();
println!("Model has {} parameters", params.len());
```

### Training Example

```rust
use coeus_nn::modules::{Linear, ReLU, Sequential, Module};
use coeus_nn::functional::ops::loss::mse_loss;
use coeus_optim::{SGD, Optimizer};

// Create model
let mut model = Sequential::new(vec![
    Box::new(Linear::new(2, 10, true)?),
    Box::new(ReLU::new()),
    Box::new(Linear::new(10, 1, true)?),
]);

// Create optimizer
let mut optimizer = SGD::new(model.parameters(), 0.01, 0.9, 0.0)?;

// Training loop
for epoch in 0..100 {
    // Forward pass
    let predictions = model.forward(&input)?;
    
    // Compute loss
    let loss = mse_loss(&predictions, &targets, Reduction::Mean)?;
    
    // Backward pass
    loss.backward()?;
    
    // Update parameters
    optimizer.step()?;
    optimizer.zero_grad();
    
    if epoch % 10 == 0 {
        println!("Epoch {}: Loss = {:.4}", epoch, loss.item());
    }
}
```

## Module Trait

All layers implement the `Module<B, S, T>` trait:

```rust
pub trait Module<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    /// Forward pass through the module
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>;
    
    /// Get all parameters in this module
    fn parameters(&self) -> Vec<Parameter<B, S, T>>;
    
    /// Get all submodules
    fn modules(&self) -> Vec<&dyn Module<B, S, T>>;
    
    /// Zero gradients for all parameters
    fn zero_grad(&mut self);
    
    /// Set training mode
    fn train(&mut self, mode: bool);
    
    /// Get module name
    fn name(&self) -> &str;
    
    /// Clone the module (for trait objects)
    fn clone_box(&self) -> Box<dyn Module<B, S, T>>;
}
```

## Parameter Management

Parameters are tensors with gradient tracking:

```rust
use coeus_nn::Parameter;

// Create parameter
let weight = Parameter::new(
    CpuTensor::zeros(&[10, 5])?,
    "weight".to_string(),
);

// Access tensor data
let data = weight.data();

// Access gradients (after backward pass)
if let Some(grad) = weight.grad() {
    println!("Gradient norm: {:.4}", grad.norm());
}

// Zero gradients
weight.zero_grad();
```

## Hierarchical File Structure Benefits

The deep vertical file structure enables:

1. **Script-Based Parity Tracking**: Compare implementations across backends
2. **Missing Implementation Detection**: Absent files indicate missing functionality
3. **Domain Isolation**: Each operation category has its own directory
4. **Consistent Organization**: Parallel structures across all domains
5. **No Monolithic Files**: Operations split by specific function

### Parity Tracking Scripts

```bash
# Check for missing activation functions
python scripts/check_operation_parity.py --category activation

# Generate parity report
python scripts/generate_parity_report.py --output parity_report.md

# Check backend coverage
python scripts/check_backend_parity.py --backend gpu
```

## Testing

```bash
# Run all NN tests
cargo test --package nn

# Run specific test categories
cargo test --package nn --test functional_ops
cargo test --package nn --test modules
cargo test --package nn --test integration

# Run with coverage
cargo tarpaulin --package nn
```

**Test Coverage**: Comprehensive test suite covering all operations and layers

## Performance Considerations

### Zero-Cost Abstractions

The Module trait uses static dispatch for zero runtime overhead:

```rust
// Monomorphized at compile time
fn train_model<M: Module<B, S, T>>(model: &M, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
    // No virtual dispatch - direct function call
    model.forward(input)
}
```

### Single Source of Truth Benefits

- **No Code Duplication**: Each operation implemented once
- **Consistent Behavior**: Same operation always behaves identically
- **Easy Maintenance**: Changes only need to be made in one place
- **Better Testing**: Test the operation once, use everywhere

### Memory Efficiency

- **Parameter Sharing**: Parameters stored once, referenced by layers
- **Zero-Copy Views**: Layers don't copy data, only reference it
- **Lazy Evaluation**: Operations computed only when needed

## Contributing

When adding new functionality:

1. **Operations**: Add to appropriate file in `src/functional/ops/`
2. **Layers**: Add wrapper in `src/modules/` that delegates to operation
3. **Tests**: Add tests for both operation and layer
4. **Documentation**: Update this README if adding new categories

### Guidelines

- **Single Source of Truth**: Implement each operation exactly once
- **Delegation Pattern**: Layers must delegate to operations
- **File Organization**: Follow hierarchical structure
- **Generic Architecture**: Maintain B<S<T>> generics
- **Error Handling**: Use Result types for all fallible operations

## See Also

- [Coeus Tensor](../tensor/) - Tensor operations and automatic differentiation
- [Coeus Backend](../backend/) - Compute backend implementations
- [Coeus Storage](../storage/) - Memory storage abstractions
- [Coeus Optim](../optim/) - Optimization algorithms
- [Quantization Crate](../quantization/) - Quantization algorithms and operations
- [Dense Crate](../dense/) - Dense tensor operations

## License

See workspace LICENSE file.