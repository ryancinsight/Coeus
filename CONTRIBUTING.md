# Contributing to Coeus

Welcome to the Coeus tensor library! We're excited to have you contribute to building a PyTorch-compatible tensor library in Rust. This guide will help you get started with contributing to the project.

## Table of Contents

- [Development Setup](#development-setup)
- [Code Organization](#code-organization)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Performance Guidelines](#performance-guidelines)
- [Documentation Guidelines](#documentation-guidelines)

## Development Setup

### Prerequisites

- **Rust 1.70+**: Install from [rustup.rs](https://rustup.rs/)
- **Python 3.8+**: Required for Python bindings and testing
- **Git**: Version control system

### Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/coeus.git
   cd coeus
   ```

2. **Install Rust toolchain**:
   ```bash
   rustup update stable
   rustup component add rustfmt clippy
   ```

3. **Set up Python environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install Python dependencies
   pip install -r pycoeus/requirements-dev.txt
   ```

4. **Verify setup**:
   ```bash
   # Run tests to verify everything works
   cargo test --workspace
   ```

## Code Organization

The Coeus project is organized into multiple crates following a modular architecture:

```
coeus/
├── autograd/     # Automatic differentiation engine
├── tensor/       # Core tensor operations
├── nn/          # Neural network layers and modules
├── optim/       # Optimization algorithms
├── utils/       # Data loading and preprocessing
├── fft/         # Fast Fourier Transform operations
├── hub/         # Model loading and management
├── tokenizer/   # Text tokenization
├── examples/    # Usage examples
└── pycoeus/     # Python bindings
```

### Key Architecture Principles

- **SSOT (Single Source of Truth)**: Each concept defined once, referenced everywhere
- **SOLID Principles**: Single responsibility, Open-closed, Liskov substitution, Interface segregation, Dependency inversion
- **CUPID**: Composable, Unix philosophy, Predictable, Idiomatic, Domain-specific
- **GRASP**: General Responsibility Assignment Software Patterns
- **DRY**: Don't Repeat Yourself
- **CLEAN**: Clear, Lean, Adaptable, Explicit, Nimble

## Coding Standards

### Rust Best Practices

1. **Naming Conventions**:
   - Use descriptive, neutral nouns/verbs (no adjectives)
   - Follow Rust naming conventions: `snake_case` for functions/variables, `PascalCase` for types
   - Constants: `SCREAMING_SNAKE_CASE`

2. **Error Handling**:
   - Use `Result<T, E>` for fallible operations
   - Implement custom error types using `thiserror`
   - Avoid `unwrap()` in library code

3. **Memory Safety**:
   - No unsafe code blocks
   - Proper ownership semantics
   - Zero-copy operations where possible

4. **Performance**:
   - Zero-cost abstractions
   - Iterator-based operations
   - Rayon for parallelization when beneficial

### Code Quality Tools

Run these tools before submitting changes:

```bash
# Format code
cargo fmt

# Run linter
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Run tests
cargo test --workspace

# Check documentation
cargo doc --workspace --no-deps
```

## Testing Guidelines

### Test Categories

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test interactions between components
3. **Property-Based Tests**: Use `proptest` for comprehensive testing
4. **Performance Tests**: Benchmark critical operations
5. **Edge Case Tests**: Test boundary conditions and error cases

### Test Requirements

- **100% success rate** for all tests
- **Mathematical validation** against known analytical derivatives
- **Edge case coverage** (NaN, infinity, overflow, underflow)
- **Test runtime < 30 seconds** for fast feedback

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_linear_layer_forward() -> Result<()> {
        let layer = Linear::<f32>::new(10, 5);
        let input = Tensor::from_vec(vec![1.0; 10], vec![1, 10]);
        let output = layer.forward(&input)?;

        assert_eq!(output.shape(), &[1, 5]);
        assert!(output.numel() == 5);

        Ok(())
    }

    #[test]
    fn test_gradient_correctness() -> Result<()> {
        let mut layer = Linear::<f32>::new(3, 2);
        let input = Tensor::from_vec_with_grad(vec![1.0, 2.0, 3.0], vec![3]);
        let target = Tensor::from_vec(vec![0.5, 1.5], vec![2]);

        let output = layer.forward(&input)?;
        let loss = (&output - &target)?.pow(2.0).sum()?;
        loss.backward()?;

        // Verify gradients are computed correctly
        assert_relative_eq!(
            input.grad().unwrap().as_scalar(),
            expected_gradient,
            epsilon = 1e-6
        );

        Ok(())
    }
}
```

## Pull Request Process

### Before Submitting

1. **Run full test suite**:
   ```bash
   cargo test --workspace
   ```

2. **Check code quality**:
   ```bash
   cargo clippy --workspace --all-targets --all-features -- -D warnings
   ```

3. **Format code**:
   ```bash
   cargo fmt
   ```

4. **Update documentation**:
   - Add tests for new functionality
   - Update API documentation
   - Update CHANGELOG.md if adding new features

### PR Requirements

- **Descriptive title**: Clear summary of changes
- **Detailed description**: What was changed and why
- **Tests included**: All new functionality must have tests
- **Documentation updated**: API docs and examples
- **No breaking changes**: Maintain backward compatibility

### Code Review Process

1. **Automated checks**: CI will run tests and linting
2. **Peer review**: At least one maintainer will review the code
3. **Address feedback**: Respond to review comments and make necessary changes
4. **Merge**: PR merged by maintainer after approval

## Performance Guidelines

### Benchmarking

Use `criterion` for performance benchmarking:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_tensor_addition(c: &mut Criterion) {
    let a = Tensor::from_vec(vec![1.0; 1000], vec![1000]);
    let b = Tensor::from_vec(vec![2.0; 1000], vec![1000]);

    c.bench_function("tensor_addition_1000", |bencher| {
        bencher.iter(|| black_box(&a + &b).unwrap())
    });
}

criterion_group!(benches, benchmark_tensor_addition);
criterion_main!(benches);
```

### Performance Targets

- **Memory usage**: < 2x PyTorch equivalent operations
- **Computation speed**: > 80% of PyTorch performance for CPU operations
- **Compilation time**: < 30 seconds for full workspace
- **Binary size**: < 10MB optimized release binary

## Documentation Guidelines

### API Documentation

All public APIs must be documented with:

- **Purpose**: What the function/type does
- **Parameters**: Description of each parameter
- **Return value**: What is returned
- **Errors**: What errors can be returned
- **Examples**: Usage examples
- **Mathematical background**: For mathematical operations

### Mathematical Documentation

For mathematical operations, include:

- **Mathematical formulation**: LaTeX equations when applicable
- **Algorithm references**: Citations to relevant papers/standards
- **Gradient computation**: How gradients are computed
- **Numerical stability**: Any stability considerations

### Example Documentation

```rust
/// Compute the sigmoid activation function
///
/// # Mathematical Formulation
///
/// ```math
/// σ(x) = 1 / (1 + e^(-x))
/// ```
///
/// # Gradient
///
/// ```math
/// dσ/dx = σ(x) * (1 - σ(x))
/// ```
///
/// # Example
///
/// ```rust
/// use coeus_tensor::Tensor;
///
/// let x = Tensor::from_vec(vec![0.0, 1.0, 2.0], vec![3]);
/// let sigmoid_x = x.sigmoid();
/// ```
pub fn sigmoid(&self) -> Result<Tensor<T>> {
    // Implementation
}
```

## Getting Help

- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: General questions and discussions on GitHub Discussions
- **Documentation**: Check the docs/ directory for detailed specifications
- **Examples**: See the examples/ directory for usage patterns

## License

By contributing to Coeus, you agree that your contributions will be licensed under the same license as the original project (MIT OR Apache-2.0).

---

Thank you for contributing to Coeus! Your help makes this project better for everyone. 🚀
