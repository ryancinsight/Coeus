# Coeus Tensor

Core tensor implementation for the Coeus deep learning framework.

## Overview

Implements the foundational `Tensor<B, S, T>` type with nested backend/storage/dtype hierarchy per ADR-001.

## Features

- **Nested Type Hierarchy**: `Tensor<Backend, Storage<DataType>, DataType>`
- **Zero-Cost Abstractions**: Static dispatch via monomorphization
- **Type Safety**: Compile-time dtype/backend/storage validation
- **PyTorch API**: Compatible constructor methods

## File Structure

The tensor crate is organized following single source of truth and separation of concerns principles:

### Core Modules (`src/`)

```
tensor/src/
├── tensor_core.rs          - Core Tensor<B,S,T> type definition and trait implementations
├── error.rs                - Error types for tensor operations
├── lib.rs                  - Public API and module declarations
└── minimal_tensor.rs       - Minimal tensor implementation for testing
```

### Operations (`src/ops/`)

Stateless pure functions for tensor operations:

```
tensor/src/ops/
├── arithmetic.rs           - Element-wise arithmetic (+, -, *, /, exp, log, etc.)
├── comparison.rs           - Comparison operations (eq, ne, gt, lt, etc.)
├── creation.rs             - Tensor creation operations (zeros, ones, full, etc.)
├── matrix.rs               - Matrix operations (matmul, transpose, etc.)
├── missing_math.rs         - Additional mathematical operations
├── reduction.rs            - Reduction operations (sum, mean, max, min, etc.)
├── sparse.rs               - Sparse tensor operations
├── tensor_ops.rs           - General tensor operations
└── mod.rs                  - Module exports
```

### Implementations (`src/implementations/`)

Trait implementations organized by category:

```
tensor/src/implementations/
├── autograd.rs             - Automatic differentiation trait implementations
├── creation.rs             - Tensor creation trait implementations
├── manipulation.rs         - Tensor manipulation trait implementations
├── math.rs                 - Mathematical trait implementations
└── mod.rs                  - Module exports
```

### Sparse Storage (`src/sparse/`)

Sparse matrix format implementations:

```
tensor/src/sparse/
├── coo.rs                  - Coordinate (COO) format implementation
├── csc.rs                  - Compressed Sparse Column (CSC) format
├── csr.rs                  - Compressed Sparse Row (CSR) format
└── mod.rs                  - Module exports
```

### Additional Modules

```
tensor/src/
├── elementwise.rs          - Element-wise operation utilities
├── functions.rs            - High-level tensor functions
├── indexing.rs             - Tensor indexing and slicing operations
├── shape_ops.rs            - Shape manipulation operations (reshape, view, etc.)
├── simd_ops.rs             - SIMD-accelerated operations
├── tensor_autograd.rs      - Autograd integration
├── tensor_backend_dispatch.rs - Backend dispatch utilities
├── tensor_sparse_ops.rs    - Sparse tensor operation utilities
├── tests.rs                - Unit tests
└── zero_copy.rs            - Zero-copy tensor operations
```

### Tests (`tests/`)

Integration and property-based tests:

```
tensor/tests/
├── backend_integration.rs  - Backend integration tests (Sprint MS-44)
├── concurrency.rs          - Concurrency and thread-safety tests
├── integration_tests.rs    - General integration tests
├── integration.rs          - Additional integration tests
├── proptest_arithmetic.rs  - Property-based tests for arithmetic operations
├── proptest.rs             - General property-based tests
└── ssot_property_test.rs   - Single source of truth property tests
```

### Benchmarks (`benches/`)

Performance benchmarks:

```
tensor/benches/
└── conditional_unsafe.rs   - Sprint 2.7 conditional unsafe optimization benchmarks
```

## Architecture Principles

### 1. Single Source of Truth (SSOT)
Each operation is implemented exactly once in the appropriate module. No duplicate implementations exist across the codebase.

### 2. Separation of Concerns (SoC)
- **Operations** (`ops/`): Stateless pure functions
- **Implementations** (`implementations/`): Trait implementations
- **Core** (`tensor_core.rs`): Type definitions
- **Tests** (`tests/`): Integration and property tests

### 3. Directory Nesting Limit
Maximum 3 levels of directory nesting for maintainability:
- Level 1: `tensor/`
- Level 2: `src/`, `tests/`, `benches/`
- Level 3: `ops/`, `implementations/`, `sparse/`

### 4. Module Organization
- **Small modules** (<10 lines): Only for re-exports (e.g., `mod.rs` files)
- **Focused modules**: Each file has a clear, single responsibility
- **Logical grouping**: Related functionality grouped in subdirectories

## Navigation Guide

### Finding Operations
- **Arithmetic**: `src/ops/arithmetic.rs` - Addition, multiplication, exponential, etc.
- **Matrix**: `src/ops/matrix.rs` - Matrix multiplication, transpose
- **Reduction**: `src/ops/reduction.rs` - Sum, mean, max, min
- **Creation**: `src/ops/creation.rs` - Zeros, ones, full, random

### Finding Trait Implementations
- **Autograd**: `src/implementations/autograd.rs`
- **Creation**: `src/implementations/creation.rs`
- **Manipulation**: `src/implementations/manipulation.rs`
- **Math**: `src/implementations/math.rs`

### Finding Tests
- **Unit tests**: `src/tests.rs` and inline `#[cfg(test)]` modules
- **Integration tests**: `tests/integration_tests.rs`, `tests/integration.rs`
- **Property tests**: `tests/proptest.rs`, `tests/proptest_arithmetic.rs`
- **Backend tests**: `tests/backend_integration.rs`

### Finding Benchmarks
- **Performance**: `benches/conditional_unsafe.rs`

## Usage

```rust
use coeus_tensor::Tensor;
use coeus_backend::CpuBackend;
use coeus_storage::DenseStorage;
use coeus_dtype::float::Float32;

// Type alias for convenience
type CpuTensor<T> = Tensor<CpuBackend, DenseStorage<T>, T>;

// Create from vector
let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
let tensor = CpuTensor::<Float32>::from_vec(data, &[3]).unwrap();

// Create zeros/ones
let zeros = CpuTensor::<Float32>::zeros(&[2, 3]).unwrap();
let ones = CpuTensor::<Float32>::ones(&[4, 5]).unwrap();
```

## Testing

```bash
# Run all tests
cargo test --package tensor

# Run specific test suite
cargo test --package tensor --test backend_integration
cargo test --package tensor --test proptest_arithmetic

# Run benchmarks
cargo bench --package tensor
```

**Coverage**: 22/22 tests passing (15 integration + 7 doc)

## Type Aliases

For ergonomic usage, define type aliases:

```rust
type CpuDenseTensor<T> = Tensor<CpuBackend, DenseStorage<T>, T>;
type CpuFloat32 = CpuDenseTensor<Float32>;
type CpuFloat64 = CpuDenseTensor<Float64>;
```

## Contributing

When adding new functionality:

1. **Operations**: Add to appropriate file in `src/ops/`
2. **Trait Implementations**: Add to appropriate file in `src/implementations/`
3. **Tests**: Add integration tests to `tests/`, unit tests inline
4. **Documentation**: Update this README if adding new modules

Maintain the single source of truth principle - implement each operation exactly once.

