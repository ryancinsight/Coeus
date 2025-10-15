# Coeus Optim: Neural Network Optimizers

This crate provides PyTorch-compatible optimization algorithms for training neural networks in the Coeus framework.

## Supported Optimizers

### SGD (Stochastic Gradient Descent)
```rust
use coeus_optim::{SGD, Optimizer};
use coeus_dtype::float::Float32;

let mut optimizer = SGD::<Float32>::with_momentum(0.01, 0.9).unwrap();
// Add parameters and train...
```

### Adam (Adaptive Moment Estimation)
```rust
use coeus_optim::{Adam, Optimizer};
use coeus_dtype::float::Float32;

let mut optimizer = Adam::<Float32>::default(0.001).unwrap();
// Add parameters and train...
```

### RMSprop (Root Mean Square Propagation)
```rust
use coeus_optim::{RMSprop, Optimizer};
use coeus_dtype::float::Float32;

let mut optimizer = RMSprop::<Float32>::default(0.01).unwrap();
// Add parameters and train...
```

### Adagrad (Adaptive Gradient)
```rust
use coeus_optim::{Adagrad, Optimizer};
use coeus_dtype::float::Float32;

let mut optimizer = Adagrad::<Float32>::default(0.01).unwrap();
// Add parameters and train...
```

## Usage

```rust
use coeus_optim::{Adam, Optimizer};
use coeus_autograd::Variable;
use coeus_dtype::float::Float32;

// Create optimizer
let mut optimizer = Adam::<Float32>::default(0.001).unwrap();

// Add model parameters
optimizer.add_param(weight_var, "weight".to_string()).unwrap();
optimizer.add_param(bias_var, "bias".to_string()).unwrap();

// Training loop
loop {
    // Forward pass and compute loss...

    // Zero gradients
    optimizer.zero_grad();

    // Backward pass...

    // Update parameters
    let updated = optimizer.step().unwrap();
    println!("Updated {} parameters", updated);
}
```

## Features

- **PyTorch Compatibility**: Identical API and behavior to PyTorch optimizers
- **Type Safety**: Compile-time guarantees for parameter types
- **Memory Safety**: Zero unsafe code, leverages Rust's ownership system
- **Extensible**: Easy to add custom optimizers via the `Optimizer` trait

## References

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [RMSprop and Adagrad optimization algorithms](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
- [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://jmlr.org/papers/v12/duchi11a.html)
