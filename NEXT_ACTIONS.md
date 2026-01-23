# Coeus Architectural Enhancement - Next Actions
## Immediate Implementation Steps

**Date**: January 13, 2026  
**Priority**: Phase 1 - Architectural Cleanup  
**Timeline**: Weeks 1-4

---

## Week 1: NN Crate Restructuring - Part 1

### Day 1-2: Create ops/ Directory Structure

**Actions**:
1. Create directory structure:
```bash
mkdir nn/src/ops
touch nn/src/ops/mod.rs
touch nn/src/ops/activation.rs
touch nn/src/ops/loss.rs
touch nn/src/ops/convolution.rs
touch nn/src/ops/linear.rs
touch nn/src/ops/normalization.rs
touch nn/src/ops/pooling.rs
touch nn/src/ops/attention.rs
```

2. Define ops module structure in `nn/src/ops/mod.rs`:
```rust
//! Stateless neural network operations
//!
//! This module provides the single source of truth for all NN operations.
//! Operations are stateless functions that operate on tensors.
//! Layers in the `layers/` module wrap these operations with state.

pub mod activation;
pub mod attention;
pub mod convolution;
pub mod linear;
pub mod loss;
pub mod normalization;
pub mod pooling;

// Re-export commonly used operations
pub use activation::*;
pub use loss::*;
```

### Day 3-4: Consolidate Activation Functions

**Current State**:
- `nn/src/modules/activation/` - Stateful implementations
- `nn/src/functional/activation/` - Stateless implementations

**Target State**:
- `nn/src/ops/activation.rs` - Single source of truth (stateless)
- `nn/src/layers/activation.rs` - Thin wrappers (stateful)

**Implementation**:
1. Extract all activation logic from `functional/activation/` to `ops/activation.rs`
2. Create thin wrappers in `layers/activation.rs` that delegate to ops
3. Update `modules/activation/` to use new structure
4. Remove `functional/activation/` directory

**Example Structure**:
```rust
// nn/src/ops/activation.rs
pub fn relu<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType + PartialOrd + Zero,
{
    // Implementation
}

pub fn gelu<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType + FloatExt,
{
    // Implementation
}

// ... other activation functions
```

```rust
// nn/src/layers/activation.rs
pub struct ReLU<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> Module<B, S, T> for ReLU<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType + PartialOrd + Zero,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        crate::ops::activation::relu(input)
    }
}
```

### Day 5: Consolidate Loss Functions

**Current State**:
- `nn/src/modules/loss/` - Stateful implementations
- `nn/src/functional/loss/` - Stateless implementations

**Target State**:
- `nn/src/ops/loss.rs` - Single source of truth
- `nn/src/layers/loss.rs` - Thin wrappers (if needed)

**Implementation**:
1. Extract all loss logic to `ops/loss.rs`
2. Update `modules/loss/` to use new structure
3. Remove `functional/loss/` directory

---

## Week 2: NN Crate Restructuring - Part 2

### Day 1-2: Consolidate Convolution Operations

**Current State**:
- `nn/src/modules/convolution/` - Conv1D, Conv2D, Conv3D (separate files)
- `nn/src/functional/convolution/` - Functional implementations

**Target State**:
- `nn/src/ops/convolution.rs` - Generic convolution operations
- `nn/src/layers/convolution.rs` - Thin wrappers

**Key Insight**: Use generic dimension parameter to reduce duplication

### Day 3: Consolidate Pooling Operations

**Current State**:
- `nn/src/modules/pooling/` - MaxPool1D/2D/3D, AvgPool1D/2D/3D (9 files)
- `nn/src/functional/pooling/` - Functional implementations

**Target State**:
- `nn/src/ops/pooling.rs` - Generic pooling operations
- `nn/src/layers/pooling.rs` - Thin wrappers

### Day 4-5: Consolidate Normalization and Linear

**Actions**:
1. Consolidate normalization operations
2. Consolidate linear operations
3. Update all references

---

## Week 3: PyCoeus Optimization

### Day 1-2: Create Generic Optimizer Wrapper

**File**: `pycoeus/src/optim/base.rs`

**Implementation**:
```rust
use pyo3::prelude::*;
use optim::BaseOptimizer;
use backend::CpuBackend;
use storage::DenseStorage;
use dtype::float::Float32;

/// Generic optimizer wrapper for PyO3
pub struct PyOptimizerWrapper<O>
where
    O: BaseOptimizer<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
{
    pub inner: O,
}

impl<O> PyOptimizerWrapper<O>
where
    O: BaseOptimizer<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
{
    pub fn step(&mut self) -> PyResult<()> {
        BaseOptimizer::step(&mut self.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Step failed: {:?}", e)
            ))
    }

    pub fn zero_grad(&mut self) {
        BaseOptimizer::zero_grad(&mut self.inner);
    }

    // ... other common methods
}
```

### Day 3: Define Custom Exception Types

**File**: `pycoeus/python/coeus/exceptions.py`

**Implementation**:
```python
class CoeusError(Exception):
    """Base exception for Coeus"""
    pass

class TensorError(CoeusError):
    """Tensor operation errors"""
    pass

class BackendError(CoeusError):
    """Backend operation errors"""
    pass

class OptimizerError(CoeusError):
    """Optimizer errors"""
    pass

class NNError(CoeusError):
    """Neural network errors"""
    pass
```

### Day 4-5: Refactor Optimizer Implementations

**Actions**:
1. Update each optimizer to use `PyOptimizerWrapper`
2. Remove duplicate code
3. Test all optimizers

---

## Week 4: Testing and Validation

### Day 1-3: Unit Tests

**Actions**:
1. Write tests for new ops/ module
2. Write tests for storage traits
3. Verify all existing tests still pass

### Day 4-5: Integration Tests

**Actions**:
1. Test end-to-end training with new structure
2. Verify PyTorch parity
3. Performance benchmarks

---

## Verification Checklist

After each week, verify:

- [ ] Workspace compiles successfully (`cargo check --workspace`)
- [ ] All tests pass (`cargo test --workspace`)
- [ ] No new clippy warnings (`cargo clippy --workspace`)
- [ ] Documentation builds (`cargo doc --workspace`)
- [ ] PyCoeus builds (`cd pycoeus && maturin develop`)
- [ ] Python tests pass (`pytest pycoeus/tests/`)

---

## Success Criteria

### Week 1
- [ ] ops/ directory structure created
- [ ] Activation functions consolidated
- [ ] Loss functions consolidated
- [ ] Workspace compiles successfully

### Week 2
- [ ] Convolution operations consolidated
- [ ] Pooling operations consolidated
- [ ] Normalization and linear consolidated
- [ ] All layer implementations updated

### Week 3
- [ ] Generic optimizer wrapper created
- [ ] Custom exception types defined
- [ ] All optimizers refactored
- [ ] PyCoeus builds successfully

### Week 4
- [ ] Unit tests written and passing
- [ ] Integration tests passing
- [ ] Performance benchmarks run
- [ ] Documentation updated

---

## Commands Reference

### Build and Test
```bash
# Check compilation
cargo check --workspace

# Run tests
cargo test --workspace

# Run clippy
cargo clippy --workspace

# Build documentation
cargo doc --workspace --open

# Build PyCoeus
cd pycoeus
maturin develop
cd ..

# Run Python tests
pytest pycoeus/tests/
```

### Useful Development Commands
```bash
# Format code
cargo fmt --all

# Check for unused dependencies
cargo machete

# Run specific test
cargo test --package nn --test test_name

# Run benchmarks
cargo bench --package nn
```

---

## Notes

- Keep commits small and focused
- Write tests as you go
- Document design decisions
- Update REFACTORING_PROGRESS.md daily
- Ask for review before major changes

---

**Ready to begin Week 1 implementation!** 🚀

**Next Review**: End of Week 1 (January 20, 2026)
