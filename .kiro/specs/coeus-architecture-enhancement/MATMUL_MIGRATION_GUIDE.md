# MatMul Migration Guide

**Date:** January 16, 2026  
**Version:** 0.2.1  
**Breaking Change:** Yes (in next major version)

## Overview

Matrix multiplication operations have been moved from the `storage` crate to specialized crates (`dense` and `sparse`) to align with Requirement 18.4: "THE Storage_System SHALL NOT provide complex operations like linear transformations or convolutions."

Matrix multiplication is a linear transformation and therefore belongs in higher-level crates rather than the storage foundation layer.

## What Changed

### Before (Deprecated)

```rust
use storage::{DenseStorage, MatMulOps};

let a = DenseStorage::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
let b = DenseStorage::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;

// Old API (deprecated)
let c = a.matmul(&b, 2, 2, 2)?;
```

### After (Recommended)

```rust
use storage::DenseStorage;
use dense::DenseMatMul;
use backend::CpuBackend;

let a = DenseStorage::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
let b = DenseStorage::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
let backend = CpuBackend::default();

// New API (recommended)
let c = a.matmul(&b, 2, 2, 2, &backend)?;
```

## Migration Steps

### Step 1: Update Imports

**Old:**
```rust
use storage::MatMulOps;
```

**New:**
```rust
use dense::DenseMatMul;  // For dense storage
// OR
use sparse::SparseMatMul;  // For sparse storage
```

### Step 2: Add Backend Parameter

The new API requires a backend parameter for hardware execution:

**Old:**
```rust
let result = storage.matmul(&other, m, n, k)?;
```

**New:**
```rust
let result = storage.matmul(&other, m, n, k, &backend)?;
```

### Step 3: Update Trait Bounds

If you have generic functions using MatMulOps:

**Old:**
```rust
fn my_function<S>(storage: &S) -> Result<S>
where
    S: Storage<f32> + MatMulOps<f32>,
{
    storage.matmul(&other, m, n, k)
}
```

**New:**
```rust
use dense::DenseMatMul;

fn my_function<S, B>(storage: &S, backend: &B) -> Result<S>
where
    S: Storage<f32> + DenseMatMul<f32>,
    B: Backend<Data = f32>,
{
    storage.matmul(&other, m, n, k, backend)
}
```

## Deprecation Timeline

### Version 0.2.1 (Current)
- ✅ `MatMulOps` trait marked as deprecated
- ✅ `DenseMatMul` trait added to dense crate
- ✅ `SparseMatMul` trait already exists in sparse crate
- ⚠️ Deprecation warnings will be shown when using `MatMulOps`
- ✅ Old API still works (backward compatible)

### Version 0.3.0 (Next Major)
- ❌ `MatMulOps` trait will be removed from storage
- ✅ Only `DenseMatMul` and `SparseMatMul` will be available
- ❌ Code using old API will fail to compile

## Why This Change?

### Architectural Clarity

**Storage crate should provide:**
- ✅ Basic arithmetic (add, sub, mul, div)
- ✅ Basic layout operations (reshape, transpose, stride)
- ✅ Memory management
- ❌ Complex operations like matrix multiplication

**Dense/Sparse crates should provide:**
- ✅ Format-specific algorithms
- ✅ Complex linear algebra operations
- ✅ Matrix multiplication
- ✅ Decompositions (SVD, QR, Cholesky)

### Requirements Compliance

**Requirement 18.4 states:**
> "THE Storage_System SHALL NOT provide complex operations like linear transformations or convolutions"

Matrix multiplication IS a linear transformation, therefore it should not be in storage.

### Better Separation of Concerns

- **Storage:** Foundation layer for memory layouts
- **Dense:** Dense-specific algorithms and operations
- **Sparse:** Sparse-specific algorithms and operations
- **Tensor:** High-level tensor API
- **NN:** Neural network operations

## API Differences

### Dense MatMul

| Aspect | Old API | New API |
|--------|---------|---------|
| Trait | `storage::MatMulOps` | `dense::DenseMatMul` |
| Backend | Implicit | Explicit parameter |
| Location | storage crate | dense crate |
| Deprecation | Deprecated | Current |

### Sparse MatMul

| Aspect | Details |
|--------|---------|
| Trait | `sparse::SparseMatMul` |
| Backend | Explicit parameter (always had it) |
| Location | sparse crate (unchanged) |
| Status | ✅ Already correct - no migration needed |

**Sparse Operations Available:**
- `matmul_sparse()` - Sparse × Sparse → Sparse
- `matvec_mul()` - Sparse × Vector → Vector
- `matmul_dense()` - Sparse × Dense → Dense

**Sparse Formats Supported:**
- CSR (Compressed Sparse Row)
- CSC (Compressed Sparse Column)
- COO (Coordinate format)

## Examples

### Example 1: Dense Matrix Multiplication

**Before:**
```rust
use storage::{DenseStorage, MatMulOps};
use dtype::float::Float32;

fn multiply_matrices(
    a: &DenseStorage<Float32>,
    b: &DenseStorage<Float32>,
) -> Result<DenseStorage<Float32>> {
    a.matmul(b, 2, 2, 2)
}
```

**After:**
```rust
use storage::DenseStorage;
use dense::DenseMatMul;
use dtype::float::Float32;
use backend::{Backend, CpuBackend};

fn multiply_matrices<B: Backend<Data = Float32>>(
    a: &DenseStorage<Float32>,
    b: &DenseStorage<Float32>,
    backend: &B,
) -> Result<DenseStorage<Float32>> {
    a.matmul(b, 2, 2, 2, backend)
}
```

### Example 2: Dense Matrix-Vector Multiplication

**Before:**
```rust
use storage::{DenseStorage, MatMulOps};

let matrix = DenseStorage::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
let vector = vec![5.0, 6.0];
let result = matrix.matvec(&vector, 2, 2)?;
```

**After:**
```rust
use storage::DenseStorage;
use dense::DenseMatMul;
use backend::CpuBackend;

let matrix = DenseStorage::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
let vector = vec![5.0, 6.0];
let backend = CpuBackend::default();
let result = matrix.matvec(&vector, 2, 2, &backend)?;
```

### Example 3: Sparse Matrix Multiplication (Already Correct)

**Sparse operations already use the correct pattern:**

```rust
use sparse::SparseMatMul;
use storage::{CsrStorage, SparseFormat};
use backend::CpuBackend;
use dtype::float::Float32;

// Create sparse matrices
let csr_a = CsrStorage::from_dense(&dense_a)?;
let csr_b = CsrStorage::from_dense(&dense_b)?;
let backend = CpuBackend::default();

// Sparse × Sparse
let result = csr_a.matmul_sparse(&csr_b, SparseFormat::Csr, &backend)?;

// Sparse × Vector
let vector = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
let result = csr_a.matvec_mul(&vector, &backend)?;

// Sparse × Dense
let dense_data = vec![1.0, 2.0, 3.0, 4.0];
let result = csr_a.matmul_dense(&dense_data, 2, 2, &backend)?;
```

### Example 4: Generic Function with MatMul

**Before:**
```rust
use storage::{Storage, MatMulOps};

fn compute<T, S>(a: &S, b: &S) -> Result<S>
where
    T: DataType,
    S: Storage<T> + MatMulOps<T>,
{
    a.matmul(b, 10, 10, 10)
}
```

**After:**
```rust
use storage::Storage;
use dense::DenseMatMul;
use backend::Backend;

fn compute<T, S, B>(a: &S, b: &S, backend: &B) -> Result<S>
where
    T: DataType,
    S: Storage<T> + DenseMatMul<T>,
    B: Backend<Data = T>,
{
    a.matmul(b, 10, 10, 10, backend)
}
```

### Example 5: Mixed Dense and Sparse Operations

```rust
use storage::{DenseStorage, CsrStorage};
use dense::DenseMatMul;
use sparse::SparseMatMul;
use backend::CpuBackend;
use dtype::float::Float32;

let backend = CpuBackend::default();

// Dense × Dense
let dense_a = DenseStorage::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
let dense_b = DenseStorage::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
let dense_result = dense_a.matmul(&dense_b, 2, 2, 2, &backend)?;

// Sparse × Sparse
let sparse_a = CsrStorage::from_dense(&dense_a)?;
let sparse_b = CsrStorage::from_dense(&dense_b)?;
let sparse_result = sparse_a.matmul_sparse(&sparse_b, SparseFormat::Csr, &backend)?;

// Sparse × Dense
let mixed_result = sparse_a.matmul_dense(dense_b.as_slice(), 2, 2, &backend)?;
```

## Testing Your Migration

### Step 1: Check for Deprecation Warnings

```bash
cargo build 2>&1 | grep "deprecated"
```

Look for warnings about `MatMulOps` usage.

### Step 2: Update Code

Follow the migration steps above for each warning.

### Step 3: Verify Tests Pass

```bash
cargo test
```

### Step 4: Check for Remaining Usage

```bash
grep -r "MatMulOps" --include="*.rs" .
```

Should only find the deprecated trait definition in storage/src/traits.rs.

## FAQ

### Q: Why add a backend parameter?

**A:** The backend parameter makes hardware execution explicit and allows for:
- GPU acceleration when using GpuBackend
- TPU acceleration when using TpuBackend
- Better control over where computation happens
- Clearer separation between storage (data) and backend (computation)

### Q: Will this break my existing code?

**A:** Not immediately. The old API is deprecated but still works in version 0.2.1. You'll see deprecation warnings. In version 0.3.0, the old API will be removed and code will need to be updated.

### Q: What about sparse matrix multiplication?

**A:** Sparse matrix multiplication was already in the sparse crate and is unaffected. It already uses the correct pattern with backend parameters.

**Sparse operations available:**
- `matmul_sparse()` - Sparse × Sparse matrix multiplication
- `matvec_mul()` - Sparse × Dense vector multiplication
- `matmul_dense()` - Sparse × Dense matrix multiplication

**All sparse formats supported:**
- CSR (Compressed Sparse Row) - Primary implementation
- CSC (Compressed Sparse Column) - Converts to CSR
- COO (Coordinate format) - Converts to CSR

**Example:**
```rust
use sparse::SparseMatMul;
use storage::{CsrStorage, SparseFormat};

let result = csr_a.matmul_sparse(&csr_b, SparseFormat::Csr, &backend)?;
```

### Q: Can I use both APIs during migration?

**A:** Yes! During version 0.2.1, both APIs work. You can migrate gradually, one file at a time.

### Q: How do I suppress deprecation warnings temporarily?

**A:** Add `#[allow(deprecated)]` above the code using the old API:

```rust
#[allow(deprecated)]
use storage::MatMulOps;
```

But it's better to migrate to the new API.

## Support

If you encounter issues during migration:

1. Check this guide for examples
2. Review the API documentation: `cargo doc --open`
3. Look at test files in dense/src/linear_algebra/matmul.rs
4. Check the boundary enforcement tests in tests/boundary_enforcement_tests.rs

## Summary

- ✅ MatMul moved from storage to dense/sparse crates
- ✅ Aligns with architectural requirements
- ✅ Backward compatible in 0.2.1 (deprecated)
- ⚠️ Breaking change in 0.3.0 (removed)
- ✅ Migration is straightforward (add backend parameter)
- ✅ Better separation of concerns
- ✅ Clearer architectural boundaries

**Recommendation:** Migrate to the new API as soon as possible to avoid issues when upgrading to version 0.3.0.
