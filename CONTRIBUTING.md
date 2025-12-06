# Contributing to Coeus

Thank you for your interest in contributing to Coeus! This document provides comprehensive guidelines for contributors of all experience levels.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Contribution Workflow](#contribution-workflow)
- [Development Guidelines](#development-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Code Review Process](#code-review-process)
- [Community](#community)

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:

- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Show empathy towards other contributors
- Help create a positive community

## Getting Started

### Prerequisites

- **Rust**: Version 1.70+ (install via [rustup](https://rustup.rs/))
- **Git**: Version control system
- **Development Tools**: Install additional tools with:
  ```bash
  rustup component add clippy rustfmt miri
  cargo install cargo-tarpaulin cargo-criterion cargo-udeps
  ```

### Quick Setup

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/coeus.git
   cd coeus
   ```
3. **Set up upstream remote**:
   ```bash
   git remote add upstream https://github.com/ryancinsight/coeus.git
   ```
4. **Create a development branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Environment Setup

### Automated Setup (Recommended)

Run the automated setup script:

```bash
# Clone and setup in one command
git clone https://github.com/ryancinsight/coeus.git
cd coeus
./scripts/setup_dev_environment.sh
```

### Manual Setup

1. **Install Rust toolchain**:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   ```

2. **Install development dependencies**:
   ```bash
   # Core development tools
   cargo install cargo-tarpaulin cargo-criterion cargo-udeps cargo-audit

   # Documentation tools
   cargo install mdbook

   # Python bindings (optional)
   pip install maturin
   ```

3. **Verify installation**:
   ```bash
   cargo --version
   rustc --version
   cargo check --workspace
   ```

### IDE Configuration

#### VS Code / Cursor
- Install "rust-analyzer" extension
- Install "CodeLLDB" for debugging
- Use the provided `.vscode/settings.json`

#### Other Editors
- Configure rust-analyzer LSP
- Enable rustfmt on save
- Configure clippy integration

## Contribution Workflow

### 1. Choose an Issue

- Check [GitHub Issues](https://github.com/ryancinsight/coeus/issues) for open tasks
- Look for issues labeled `good-first-issue` or `help-wanted`
- Comment on the issue to indicate you're working on it

### 2. Development Process

```bash
# Ensure you're on a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... development work ...

# Run tests locally
cargo test --workspace

# Format code
cargo fmt --all

# Run linting
cargo clippy --workspace -- -D warnings

# Commit your changes
git add .
git commit -m "feat: add your feature description"
```

### 3. Testing Your Changes

```bash
# Run all tests
cargo test --workspace

# Run specific crate tests
cargo test --package coeus-tensor

# Run with coverage
cargo tarpaulin --workspace --out Html

# Run benchmarks
cargo criterion
```

### 4. Create Pull Request

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR on GitHub**:
   - Use the PR template
   - Fill out all required sections
   - Link to related issues

3. **Address review feedback**:
   - Make requested changes
   - Push additional commits
   - Request re-review when ready

## Development Guidelines

### Code Style

- **Follow Rust conventions**: Use `rustfmt` and `clippy`
- **Naming**: Use descriptive names, follow Rust naming conventions
- **Documentation**: Document all public APIs with examples
- **Error handling**: Use appropriate error types, avoid unwrap()

### Architecture Principles

#### Generic Architecture: B<S<T>>

Coeus uses a hierarchical generic architecture:

```rust
// Tensor<B, S, T> where:
// B: Backend (CpuBackend, GpuBackend)
// S: Storage (DenseStorage, SparseStorage)
// T: DataType (Float32, Float64, etc.)
let tensor: Tensor<CpuBackend, DenseStorage<Float32>, Float32> = /* ... */;
```

**Guidelines**:
- Always specify concrete types for function parameters
- Use associated types in traits
- Maintain type safety across all operations

#### Memory Safety

- Zero unsafe code in application logic
- Justified unsafe blocks only in performance-critical code
- Comprehensive testing of unsafe operations
- Memory safety audits for all changes

### Performance Considerations

- **Zero-cost abstractions**: Leverage Rust's compile-time optimizations
- **SIMD operations**: Use safe SIMD intrinsics where beneficial
- **Memory efficiency**: Minimize allocations, prefer in-place operations
- **Benchmarking**: Add benchmarks for performance-critical code

### Feature Development

#### Adding New Features

1. **Check existing issues** for similar proposals
2. **Create ADR** (Architecture Decision Record) for significant changes
3. **Start small**: Implement minimal viable feature
4. **Add comprehensive tests**
5. **Update documentation**

#### Feature Flags

Use Cargo feature flags for optional functionality:

```toml
[features]
default = ["autograd"]
autograd = ["coeus-autograd"]
distributed = ["coeus-distributed"]
```

### API Design

- **PyTorch compatibility**: Match PyTorch APIs where possible
- **Builder pattern**: Use builders for complex object construction
- **Fluent interfaces**: Chain operations where it improves readability
- **Type safety**: Leverage Rust's type system for correctness

## Testing

### Test Categories

#### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        assert_eq!(tensor.shape().dims(), &[2]);
    }
}
```

#### Integration Tests
```rust
// tests/integration_tests.rs
#[test]
fn test_end_to_end_training() {
    // Complete training workflow test
}
```

#### Property-Based Testing
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn tensor_arithmetic_commutative(a in any::<f32>(), b in any::<f32>()) {
        // Property-based test
    }
}
```

### Testing Best Practices

- **Test public APIs**: Focus on testing public interfaces
- **Edge cases**: Test boundary conditions and error cases
- **Performance regression**: Include performance assertions
- **Cross-platform**: Test on all supported platforms
- **Memory safety**: Test with Miri for undefined behavior

### Continuous Integration

All PRs must pass:
- ✅ Unit and integration tests
- ✅ Code formatting (`cargo fmt`)
- ✅ Linting (`cargo clippy`)
- ✅ Security audit (`cargo audit`)
- ✅ Coverage requirements (85%+)

## Documentation

### Code Documentation

```rust
/// Brief description of the function
///
/// # Arguments
/// * `input` - Description of input parameter
/// * `config` - Configuration options
///
/// # Returns
/// Description of return value
///
/// # Examples
/// ```
/// use coeus_nn::Linear;
/// let layer = Linear::new(10, 5).unwrap();
/// ```
pub fn complex_function(input: Tensor, config: Config) -> Result<Tensor> {
    // Implementation
}
```

### Documentation Standards

- **Comprehensive**: Document all public APIs
- **Examples**: Include runnable code examples
- **Cross-references**: Link related functions/types
- **Performance notes**: Document complexity and performance characteristics

### Updating Documentation

1. **API changes**: Update relevant doc comments
2. **New features**: Add examples and usage guides
3. **Breaking changes**: Update migration guides
4. **Build docs**: Ensure documentation builds successfully

## Code Review Process

### Review Checklist

**For Reviewers:**
- [ ] Code follows project conventions
- [ ] Tests are comprehensive and pass
- [ ] Documentation is updated
- [ ] Performance impact assessed
- [ ] Security implications reviewed
- [ ] Breaking changes properly documented

**For Contributors:**
- [ ] Self-review completed
- [ ] All tests pass locally
- [ ] Code is well-documented
- [ ] Commit messages are clear
- [ ] Related issues linked

### Review Guidelines

- **Constructive feedback**: Focus on code quality and maintainability
- **Explain rationale**: Provide context for requested changes
- **Iterative process**: Multiple review rounds are normal
- **Knowledge sharing**: Reviews are learning opportunities

### Approval Process

1. **Automated checks**: CI must pass
2. **Peer review**: At least one maintainer review required
3. **Final approval**: Maintainer merges the PR
4. **Post-merge**: Automated deployment and notifications

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General discussion and Q&A
- **Discord**: Real-time community chat (coming soon)
- **Newsletter**: Monthly updates and announcements

### Getting Help

- **Documentation**: Check docs/ and examples/
- **Issues**: Search existing issues first
- **Discussions**: Ask the community
- **Discord**: Real-time help

### Recognition

Contributors are recognized through:
- GitHub contributor statistics
- Release notes attribution
- Community spotlights
- Contributor swag program

### Governance

- **Maintainers**: Core team responsible for project direction
- **Contributors**: Community members with merge rights
- **Community**: All users and contributors
- **RFC Process**: Major changes require community consensus

## Advanced Topics

### Performance Optimization

- **Benchmarking**: Use Criterion for performance tests
- **Profiling**: Use flamegraphs and perf tools
- **Memory profiling**: Track allocations and cache efficiency
- **SIMD utilization**: Optimize for modern CPU architectures

### Research Integration

- **Reproducibility**: Ensure research code is reproducible
- **Documentation**: Research features need comprehensive docs
- **Testing**: Research code requires extensive validation
- **Stability**: Research features may have different stability guarantees

### Security

- **Audit dependencies**: Regular cargo audit runs
- **Unsafe code review**: All unsafe blocks require justification
- **Input validation**: Validate all external inputs
- **Cryptographic security**: Use audited crypto libraries

## Troubleshooting

### Common Issues

**Compilation Errors**
```bash
# Clean build
cargo clean
cargo build

# Check for missing dependencies
cargo tree
```

**Test Failures**
```bash
# Run specific test
cargo test test_name

# Run with backtrace
RUST_BACKTRACE=1 cargo test
```

**Performance Issues**
```bash
# Profile with flamegraph
cargo flamegraph --bin your_binary
```

### Getting Help

1. **Check documentation** first
2. **Search existing issues**
3. **Create minimal reproduction** case
4. **Ask the community** with clear problem description

---

Thank you for contributing to Coeus! Your contributions help make machine learning in Rust more accessible and powerful for everyone.

For questions or clarifications about these guidelines, please open a [GitHub Discussion](https://github.com/ryancinsight/coeus/discussions) or ask in our community Discord.
