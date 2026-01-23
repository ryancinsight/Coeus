# Storage Operations Audit

## Overview

This document categorizes all operations currently in the storage crate to identify which operations should remain (basic) and which should be moved to higher-level crates (complex).

## Current Operations Analysis

### Basic Operations (Should Remain in Storage)

These operations are fundamental memory layout and basic arithmetic operations that belong at the storage level:

#### Creation Operations
- `from_vec()` - Create storage from vector data
- `from_slice()` - Create storage from slice data  
- `zeros()` - Create zero-filled storage
- `ones()` - Create one-filled storage
- `full()` - Create storage filled with constant value

#### Basic Arithmetic Operations
- Element-wise addition (via ArithmeticOps trait)
- Element-wise subtraction (via ArithmeticOps trait)
- Element-wise multiplication (via ArithmeticOps trait)
- Element-wise division (via ArithmeticOps trait)
- Scalar addition (via ArithmeticOps trait)
- Scalar multiplication (via ArithmeticOps trait)

#### Basic Layout Operations
- `as_slice()` - Access underlying data
- `as_mut_slice()` - Mutable access to data
- `shape()` - Get tensor shape
- `strides()` - Get memory strides
- `is_contiguous()` - Check memory contiguity
- `len()` - Get total elements
- `is_empty()` - Check if empty

#### Basic Reduction Operations
- `sum()` - Sum all elements
- `product()` - Product of all elements
- `max()` - Maximum element
- `min()` - Minimum element
- `mean()` - Mean of all elements

### Complex Operations (Should Be Moved)

These operations are higher-level mathematical operations that should be moved to tensor, nn, or other appropriate crates:

#### Activation Functions (Move to nn crate)
- `ActivationOps` trait with:
  - `relu()` - ReLU activation
  - `tanh()` - Hyperbolic tangent activation
  - `sigmoid()` - Sigmoid activation
  - `gelu()` - GELU activation
  - `swish()` - Swish activation
  - `hardsigmoid()` - Hard sigmoid activation
  - `hardswish()` - Hard swish activation

#### Linear Algebra Operations (Move to tensor crate)
- `MatMulStorage` trait with:
  - `matmul_storage()` - Matrix multiplication
- `TransposeStorage` trait with:
  - `transpose_storage()` - Matrix transpose
- Dense storage `matmul_storage()` implementation
- Dense storage `transpose_storage()` implementation

#### Sparse Format Conversions (Keep in sparse crate, but not storage)
- `to_csr()` - Convert to CSR format
- `to_csc()` - Convert to CSC format  
- `to_coo()` - Convert to COO format
- `transpose()` - Sparse matrix transpose
- `sort()` - Sort COO storage

#### Quantization Operations (Move to quantization crate)
- `unpack_and_dequantize()` - Dequantize values
- `get()` - Get dequantized element
- Quantization parameter management (`scale()`, `zero_point()`, `bits()`)

#### Distributed Operations (Keep in distributed crate)
- `gather()` - Gather distributed data
- `scatter()` - Scatter data across devices
- `all_reduce()` - Collective reduction
- Shard management operations

## Recommended Actions

### Phase 1: Split Basic Operations into Separate Files

Create hierarchical file structure for basic operations:

```
storage/src/dense/
├── arithmetic/
│   ├── add.rs      # Element-wise addition
│   ├── sub.rs      # Element-wise subtraction  
│   ├── mul.rs      # Element-wise multiplication
│   ├── div.rs      # Element-wise division
│   └── mod.rs      # Re-exports
├── layout/
│   ├── reshape.rs  # Reshape operations
│   ├── transpose.rs # Basic transpose (2D only)
│   ├── stride.rs   # Stride calculations
│   └── mod.rs      # Re-exports
├── creation/
│   ├── zeros.rs    # Zero initialization
│   ├── ones.rs     # One initialization
│   ├── from_vec.rs # Vector creation
│   └── mod.rs      # Re-exports
└── mod.rs          # Dense storage module
```

### Phase 2: Remove Complex Operations

1. **Move activation functions to nn crate**:
   - Remove `ActivationOps` trait from storage
   - Move implementations to `nn/src/functional/ops/activation/`

2. **Move linear algebra to tensor crate**:
   - Remove `MatMulStorage` and `TransposeStorage` traits
   - Move matrix multiplication to tensor crate
   - Move transpose operations to tensor crate

3. **Move quantization operations to quantization crate**:
   - Move quantization-specific operations out of storage
   - Keep only basic storage interface in storage crate

### Phase 3: Backend Delegation

Update remaining basic operations to delegate to backend primitives:

```rust
// Example: storage/src/dense/arithmetic/add.rs
pub fn add<T: DataType>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
) -> Result<DenseStorage<T>>
where
    T: core::ops::Add<Output = T> + Clone,
{
    // Delegate to backend for actual computation
    let backend = get_current_backend();
    let result_data = backend.add_primitive(lhs.as_slice(), rhs.as_slice())?;
    DenseStorage::from_vec(result_data, lhs.shape().dims())
}
```

## Requirements Mapping

- **Requirement 18.1**: Basic arithmetic operations (add, sub, mul, div) ✓
- **Requirement 18.2**: Basic layout operations (reshape, transpose, stride) ✓  
- **Requirement 18.3**: Basic creation operations (zeros, ones, from_vec) ✓
- **Requirement 18.4**: Remove complex operations (linear transformations, convolutions) ✓
- **Requirement 18.5**: Backend delegation for hardware execution ✓
- **Requirement 18.6**: Clear separation between storage and backend ✓

## File Structure Changes

### Current Structure
```
storage/src/
├── dense.rs          # Monolithic dense storage
├── sparse.rs         # Monolithic sparse storage  
├── quantized.rs      # Monolithic quantized storage
├── lib.rs           # Complex traits mixed with basic ones
└── ...
```

### Target Structure
```
storage/src/
├── dense/
│   ├── arithmetic/   # Basic arithmetic operations
│   ├── layout/       # Basic layout operations
│   ├── creation/     # Basic creation operations
│   └── mod.rs
├── sparse/           # Sparse-specific operations (basic only)
├── quantized/        # Quantized storage (basic only)
├── lib.rs           # Basic traits only
└── ...
```

This structure enables:
- Clear separation of basic vs complex operations
- Easy identification of operations that need backend delegation
- Hierarchical organization for parity tracking
- Domain-specific boundaries enforcement