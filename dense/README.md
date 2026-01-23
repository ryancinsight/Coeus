# Coeus Dense

Dense tensor operations and algorithms for the Coeus deep learning framework.

## Overview

This crate provides dense tensor operations extracted from the tensor crate to ensure clear domain separation. It implements dense-specific algorithms and operations while maintaining the B<S<T>> generic architecture and zero-cost abstractions.

## Features

- **Dense Algorithms**: Optimized algorithms for dense tensor operations
- **Domain Separation**: Clean separation from sparse and quantized operations
- **B<S<T>> Architecture**: Generic over Backend, Storage, and DataType
- **Zero-Cost Abstractions**: Compile-time optimizations with minimal runtime overhead
- **Storage Integration**: Works seamlessly with storage crate primitives
- **Hierarchical Organization**: Clear file structure for maintainability

## Architecture Overview

The dense crate focuses exclusively on dense tensor operations:

```
┌─────────────────────────────────────────────────────────────┐
│                    Dense Tensor Operations                   │
│  - Element-wise operations (add, mul, exp, log, etc.)       │
│  - Matrix operations (matmul, transpose, etc.)              │
│  - Reduction operations (sum, mean, max, min, etc.)         │
│  - Broadcasting operations                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Storage Integration                       │
│  - Delegates to storage crate for basic operations          │
│  - Builds complex operations from storage primitives        │
│  - Maintains clear separation of concerns                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend Integration                       │
│  - Uses backend crate for hardware execution                │
│  - Supports CPU, GPU, TPU, NPU backends                     │
│  - Hardware-agnostic algorithm implementations              │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

The dense crate uses a hierarchical structure for clear organization:

### Algorithms (`src/algorithms/`)

Core dense tensor algorithms:

```
dense/src/algorithms/
├── arithmetic.rs           - Element-wise arithmetic operations
├── matrix.rs               - Matrix operations (matmul, transpose, etc.)
├── reduction.rs            - Reduction operations (sum, mean, etc.)
├── broadcasting.rs         - Broadcasting algorithms
├── indexing.rs             - Indexing and slicing operations
└── mod.rs                  - Module exports
```

### Operations (`src/ops/`)

High-level dense operations:

```
dense/src/ops/
├── elementwise.rs          - Element-wise operation implementations
├── linear_algebra.rs       - Linear algebra operations
├── statistical.rs          - Statistical operations
├── comparison.rs           - Comparison operations
├── trigonometric.rs        - Trigonometric functions
└── mod.rs                  - Module exports
```

### Utilities (`src/utils/`)

Dense tensor utilities:

```
dense/src/utils/
├── shape_utils.rs          - Shape manipulation utilities
├── stride_utils.rs         - Stride calculation utilities
├── memory_utils.rs         - Memory layout utilities
└── mod.rs                  - Module exports
```

### Core Infrastructure (`src/`)

```
dense/src/
├── lib.rs                  - Public API and module declarations
├── error.rs                - Error types for dense operations
├── traits.rs               - Dense-specific traits
└── dense_tensor.rs         - Dense tensor wrapper (if needed)
```

## Usage Examples

### Basic Dense Operations

```rust
use coeus_dense::ops::elementwise::{add, mul, exp};
use coeus_dense::ops::linear_algebra::matmul;
use coeus_storage::DenseStorage;
use coeus_backend::CpuBackend;
use coeus_dtype::float::Float32;

// Create dense storage
let a = DenseStorage::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
let b = DenseStorage::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;

// Element-wise operations
let sum = add(&a, &b)?;
let product = mul(&a, &b)?;
let exponential = exp(&a)?;

// Matrix operations
let result = matmul(&a, &b)?;
```

### Broadcasting Operations

```rust
use coeus_dense::algorithms::broadcasting::{broadcast_shapes, broadcast_add};

// Broadcast addition: [2, 3] + [3] -> [2, 3]
let a = DenseStorage::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
let b = DenseStorage::from_vec(vec![10.0, 20.0, 30.0], &[3])?;

// Check if shapes are broadcastable
let result_shape = broadcast_shapes(&[2, 3], &[3])?;
assert_eq!(result_shape, vec![2, 3]);

// Perform broadcasted addition
let result = broadcast_add(&a, &b)?;
```

### Reduction Operations

```rust
use coeus_dense::ops::statistical::{sum, mean, max, min, std_dev};

let data = DenseStorage::from_vec(
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
    &[2, 3]
)?;

// Reduce along all dimensions
let total_sum = sum(&data, None)?;
let mean_value = mean(&data, None)?;

// Reduce along specific dimension
let row_sums = sum(&data, Some(1))?;  // Sum along columns
let col_means = mean(&data, Some(0))?; // Mean along rows

// Statistical operations
let max_value = max(&data, None)?;
let min_value = min(&data, None)?;
let standard_deviation = std_dev(&data, None, 0)?; // ddof=0 for population std
```

### Matrix Operations

```rust
use coeus_dense::ops::linear_algebra::{matmul, transpose, inverse, svd};

// Matrix multiplication
let a = DenseStorage::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
let b = DenseStorage::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
let c = matmul(&a, &b)?;

// Transpose
let a_t = transpose(&a, &[1, 0])?;

// Matrix inverse (for square matrices)
let a_inv = inverse(&a)?;

// Singular Value Decomposition
let (u, s, vt) = svd(&a)?;
```

### Trigonometric Functions

```rust
use coeus_dense::ops::trigonometric::{sin, cos, tan, asin, acos, atan};

let angles = DenseStorage::from_vec(
    vec![0.0, std::f32::consts::PI / 4.0, std::f32::consts::PI / 2.0], 
    &[3]
)?;

let sin_values = sin(&angles)?;
let cos_values = cos(&angles)?;
let tan_values = tan(&angles)?;

// Inverse trigonometric functions
let asin_values = asin(&sin_values)?;
let acos_values = acos(&cos_values)?;
let atan_values = atan(&tan_values)?;
```

### Comparison Operations

```rust
use coeus_dense::ops::comparison::{eq, ne, gt, lt, ge, le};

let a = DenseStorage::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
let b = DenseStorage::from_vec(vec![1.0, 3.0, 2.0], &[3])?;

// Element-wise comparisons (return boolean tensors)
let equal = eq(&a, &b)?;        // [true, false, false]
let greater = gt(&a, &b)?;      // [false, false, true]
let less_equal = le(&a, &b)?;   // [true, true, false]
```

## Dense-Specific Traits

The dense crate defines traits specific to dense operations:

```rust
/// Dense arithmetic operations
pub trait DenseArithmetic<T: DataType> {
    fn dense_add(&self, other: &Self) -> Result<Self> where Self: Sized;
    fn dense_sub(&self, other: &Self) -> Result<Self> where Self: Sized;
    fn dense_mul(&self, other: &Self) -> Result<Self> where Self: Sized;
    fn dense_div(&self, other: &Self) -> Result<Self> where Self: Sized;
}

/// Dense matrix operations
pub trait DenseLinearAlgebra<T: DataType> {
    fn dense_matmul(&self, other: &Self) -> Result<Self> where Self: Sized;
    fn dense_transpose(&self, dims: &[usize]) -> Result<Self> where Self: Sized;
    fn dense_inverse(&self) -> Result<Self> where Self: Sized;
}

/// Dense reduction operations
pub trait DenseReduction<T: DataType> {
    fn dense_sum(&self, dim: Option<usize>) -> Result<Self> where Self: Sized;
    fn dense_mean(&self, dim: Option<usize>) -> Result<Self> where Self: Sized;
    fn dense_max(&self, dim: Option<usize>) -> Result<Self> where Self: Sized;
    fn dense_min(&self, dim: Option<usize>) -> Result<Self> where Self: Sized;
}

/// Dense broadcasting operations
pub trait DenseBroadcasting<T: DataType> {
    fn broadcast_to(&self, shape: &[usize]) -> Result<Self> where Self: Sized;
    fn broadcast_add(&self, other: &Self) -> Result<Self> where Self: Sized;
    fn broadcast_mul(&self, other: &Self) -> Result<Self> where Self: Sized;
}
```

## Integration with Storage Crate

The dense crate builds upon storage crate primitives:

```rust
use coeus_storage::{DenseStorage, Storage, StorageFromVec};
use coeus_dense::algorithms::arithmetic::dense_add_impl;

// Dense crate provides algorithms that work with storage primitives
fn add_dense_storages<T: DataType>(
    a: &DenseStorage<T>, 
    b: &DenseStorage<T>
) -> Result<DenseStorage<T>>
where
    T: core::ops::Add<Output = T> + Clone,
{
    // Use storage crate for basic operations
    // Dense crate provides the algorithm logic
    dense_add_impl(a, b)
}
```

## Performance Optimizations

### Memory Layout Optimization

```rust
use coeus_dense::utils::memory_utils::{is_contiguous, ensure_contiguous};

// Check if storage is contiguous for optimal performance
if is_contiguous(&storage) {
    // Use fast contiguous algorithms
    fast_contiguous_operation(&storage)
} else {
    // Ensure contiguous layout for better performance
    let contiguous = ensure_contiguous(&storage)?;
    fast_contiguous_operation(&contiguous)
}
```

### SIMD Acceleration

Dense operations are optimized for SIMD when possible:

```rust
// Dense operations automatically use SIMD when:
// 1. Data is contiguous
// 2. Element count is suitable for vectorization
// 3. Backend supports SIMD instructions

let result = add(&large_dense_tensor_a, &large_dense_tensor_b)?;
// Automatically uses AVX2/NEON when available
```

### Cache-Friendly Algorithms

```rust
use coeus_dense::algorithms::matrix::cache_friendly_matmul;

// Matrix multiplication with cache-friendly blocking
let result = cache_friendly_matmul(&a, &b, block_size: 64)?;
```

## Broadcasting Rules

The dense crate implements NumPy-style broadcasting:

```rust
// Broadcasting rules:
// 1. Align shapes from the right
// 2. Dimensions of size 1 can be broadcast to any size
// 3. Missing dimensions are treated as size 1

// Examples:
// [3, 1] + [4] -> [3, 4]  (broadcast [3, 1] and [1, 4])
// [2, 3] + [3] -> [2, 3]  (broadcast [1, 3] to [2, 3])
// [1, 4] + [3, 1] -> [3, 4]  (broadcast both)

use coeus_dense::algorithms::broadcasting::broadcast_shapes;

let shape_a = vec![3, 1];
let shape_b = vec![4];
let result_shape = broadcast_shapes(&shape_a, &shape_b)?;
assert_eq!(result_shape, vec![3, 4]);
```

## Error Handling

The dense crate provides comprehensive error handling:

```rust
use coeus_dense::error::DenseError;

match dense_operation(&a, &b) {
    Ok(result) => println!("Operation successful"),
    Err(DenseError::ShapeMismatch { expected, actual }) => {
        eprintln!("Shape mismatch: expected {:?}, got {:?}", expected, actual);
    }
    Err(DenseError::InvalidDimension(dim)) => {
        eprintln!("Invalid dimension: {}", dim);
    }
    Err(DenseError::SingularMatrix) => {
        eprintln!("Matrix is singular and cannot be inverted");
    }
    Err(e) => eprintln!("Dense operation error: {:?}", e),
}
```

## Testing

```bash
# Run all dense tests
cargo test --package dense

# Run specific test categories
cargo test --package dense --test algorithms
cargo test --package dense --test operations
cargo test --package dense --test integration

# Run with coverage
cargo tarpaulin --package dense

# Run benchmarks
cargo bench --package dense
```

**Test Coverage**: Comprehensive test suite covering all dense operations and algorithms

## Benchmarks

Performance benchmarks for dense operations:

```bash
# Run dense benchmarks
cargo bench --package dense

# Example results (on modern CPU):
# Dense addition:       ~8.5 GB/s throughput
# Dense multiplication: ~7.2 GB/s throughput
# Matrix multiplication: ~45 GFLOPS (depends on size)
# SIMD acceleration:    ~3.2x speedup on AVX2
```

## Integration with Tensor Crate

The tensor crate uses dense crate for dense-specific operations:

```rust
// In tensor crate:
use coeus_dense::ops::elementwise::add as dense_add;

impl<B, T> Tensor<B, DenseStorage<T>, T>
where
    B: Backend<Data = T>,
    T: DataType,
{
    pub fn add(&self, other: &Self) -> Result<Self> {
        let result_storage = dense_add(self.storage(), other.storage())?;
        Ok(Tensor::from_storage(result_storage, self.backend().clone()))
    }
}
```

## Contributing

When adding new dense functionality:

1. **Algorithms**: Add to `src/algorithms/`
2. **Operations**: Add to `src/ops/`
3. **Utilities**: Add to `src/utils/`
4. **Tests**: Add comprehensive tests for all new functionality
5. **Benchmarks**: Add performance benchmarks
6. **Documentation**: Update this README

### Guidelines

- **Domain Separation**: Keep dense-specific logic within this crate
- **Storage Integration**: Build upon storage crate primitives
- **Generic Architecture**: Maintain compatibility with B<S<T>> pattern
- **Performance**: Optimize for dense data patterns
- **Broadcasting**: Follow NumPy broadcasting rules
- **Error Handling**: Use Result types for all fallible operations

## See Also

- [Coeus Tensor](../tensor/) - Multi-dimensional tensor operations
- [Coeus Storage](../storage/) - Memory storage abstractions
- [Coeus Backend](../backend/) - Compute backend implementations
- [Coeus Sparse](../sparse/) - Sparse tensor operations
- [Coeus Quantization](../quantization/) - Quantization algorithms
- [Coeus NN](../nn/) - Neural network layers and operations

## License

See workspace LICENSE file.