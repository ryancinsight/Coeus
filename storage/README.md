# Coeus Storage

Memory storage abstractions for the Coeus deep learning framework.

## Overview

This crate provides storage primitives for multi-dimensional tensor data, separating memory layout concerns from compute backend logic. The storage system enables **zero-cost abstractions** through a flexible trait hierarchy that supports multiple memory layouts while maintaining type safety and performance.

## Features

- **DenseStorage**: Contiguous memory with row-major (C-contiguous) layout
- **Sparse Storage**: Memory-efficient CSR, CSC, and COO formats for sparse matrices
- **Quantized Storage**: Packed 4-bit, 8-bit, and 16-bit quantization for memory efficiency
- **Strided Storage**: Zero-copy views with custom strides for transpose and slicing
- **Distributed Storage**: Multi-device tensor storage for distributed computing
- **Shape Management**: Multi-dimensional shape specification with stride calculation
- **Type Safety**: Generic over `DataType` from `coeus-dtype`
- **Extensible**: Add new storage formats without modifying existing code

## Quick Start

```rust
use coeus_storage::{DenseStorage, Storage, StorageFromVec};
use coeus_dtype::float::Float32;

// Create from vector
let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
let storage = DenseStorage::from_vec(data, &[3]).unwrap();

// Create with zeros
let zeros = DenseStorage::<Float32>::zeros(&[2, 3]).unwrap();

// Access data
let slice: &[Float32] = storage.as_slice();
```

## Storage Formats

### Dense Storage

Contiguous row-major storage for dense tensors. Optimal for most operations.

```rust
use coeus_storage::{DenseStorage, Storage};
use coeus_dtype::float::Float32;

// Create 2x3 matrix
let storage = DenseStorage::<Float32>::zeros(&[2, 3]).unwrap();
assert_eq!(storage.len(), 6);
assert!(storage.is_contiguous());
```

**Use Cases:**
- General-purpose tensor operations
- Dense matrix multiplication
- Convolution operations
- When most elements are non-zero

**Memory Layout:** Row-major (C-contiguous)
```
Shape [2, 3]:
[[a, b, c],    Memory: [a, b, c, d, e, f]
 [d, e, f]]    
Stride: [3, 1]
```

### Sparse Storage

Memory-efficient storage for matrices with many zero elements. Supports three formats:

#### CSR (Compressed Sparse Row)

Efficient for row-wise operations and matrix-vector multiplication.

```rust
use coeus_storage::CsrStorage;
use coeus_dtype::float::Float32;

// 3x3 matrix: [[1, 0, 2], [0, 3, 0], [4, 0, 5]]
let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0),
                Float32::new(4.0), Float32::new(5.0)];
let indices = vec![0, 2, 1, 0, 2];  // column indices
let indptr = vec![0, 2, 3, 5];       // row pointers
let storage = CsrStorage::new(data, indices, indptr, &[3, 3]).unwrap();

assert_eq!(storage.nnz(), 5);  // 5 non-zero elements
assert_eq!(storage.sparsity(), 5.0 / 9.0);
```

**Use Cases:**
- Sparse matrix-vector multiplication
- Row-wise operations
- Graph adjacency matrices

#### CSC (Compressed Sparse Column)

Efficient for column-wise operations.

```rust
use coeus_storage::CscStorage;
use coeus_dtype::float::Float32;

// Same matrix in CSC format
let data = vec![Float32::new(1.0), Float32::new(4.0), Float32::new(3.0),
                Float32::new(2.0), Float32::new(5.0)];
let indices = vec![0, 2, 1, 0, 2];  // row indices
let indptr = vec![0, 2, 3, 5];       // column pointers
let storage = CscStorage::new(data, indices, indptr, &[3, 3]).unwrap();
```

**Use Cases:**
- Column-wise operations
- Sparse matrix-matrix multiplication
- Feature matrices in machine learning

#### COO (Coordinate Format)

Flexible format for construction and conversion.

```rust
use coeus_storage::CooStorage;
use coeus_dtype::float::Float32;

// Same matrix in COO format
let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0),
                Float32::new(4.0), Float32::new(5.0)];
let row_indices = vec![0, 0, 1, 2, 2];
let col_indices = vec![0, 2, 1, 0, 2];
let storage = CooStorage::new(data, row_indices, col_indices, &[3, 3]).unwrap();

// Convert between formats
let csr = storage.to_csr();
let csc = storage.to_csc();
```

**Use Cases:**
- Building sparse matrices incrementally
- Converting between sparse formats
- Sparse matrix construction from triplets

**Format Conversion:**
```
        COO (Coordinate)
       /   \
      /     \
    CSR     CSC
     \     /
      \   /
      Dense
```

### Quantized Storage

Packed quantized values for memory-efficient storage. Supports 4-bit, 8-bit, and 16-bit quantization.

```rust
use coeus_storage::{QuantizedStorage4, QuantizedStorage8, QuantizedStorage16};
use coeus_dtype::float::Float32;

// 4-bit quantization (2 values per byte)
let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
let storage = QuantizedStorage4::from_vec(data, &[3]).unwrap();

// 8-bit quantization (1 value per byte)
let storage = QuantizedStorage8::<Float32>::zeros(&[2, 3]).unwrap();

// 16-bit quantization (2 bytes per value)
let storage = QuantizedStorage16::<Float32>::ones(&[4]).unwrap();
```

**Use Cases:**
- Model compression
- Inference optimization
- Memory-constrained environments
- Mobile and edge deployment

**Memory Savings:**
- 4-bit: 8x compression vs Float32
- 8-bit: 4x compression vs Float32
- 16-bit: 2x compression vs Float32

### Strided Storage

Zero-copy views with custom strides for efficient transpose and slicing operations.

```rust
use coeus_storage::StridedStorage;
use coeus_dtype::float::Float32;

// Create strided storage
let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0),
                Float32::new(4.0), Float32::new(5.0), Float32::new(6.0)];
let storage = StridedStorage::new(data, &[2, 3]).unwrap();

// Transpose without copying data
let transposed = storage.transpose(None).unwrap();
assert_eq!(transposed.shape().dims(), &[3, 2]);
assert!(!transposed.is_contiguous());

// Slice without copying data
let sliced = storage.slice(&[(Some(0), Some(2), 1)]).unwrap();
```

**Use Cases:**
- Zero-copy transpose operations
- Tensor slicing and views
- Broadcasting operations
- Memory-efficient tensor manipulation

## Trait Hierarchy

The storage system uses a flexible trait hierarchy for extensibility:

### Core Traits

```rust
// Foundation trait - all storage types must implement
pub trait Storage<T: DataType> {
    fn as_slice(&self) -> &[T];
    fn as_mut_slice(&mut self) -> &mut [T];
    fn shape(&self) -> &Shape;
    fn strides(&self) -> &[usize];
    fn is_contiguous(&self) -> bool;
    fn as_storage_ref(&self) -> &Self;
    fn full(dims: &[usize], value: T) -> Result<Self>;
}

// Creation trait - enables tensor creation from vectors
pub trait StorageFromVec<T: DataType>: Storage<T> {
    fn from_vec(data: Vec<T>, dims: &[usize]) -> Result<Self>;
    fn zeros(dims: &[usize]) -> Result<Self> where T: Zero;
    fn ones(dims: &[usize]) -> Result<Self> where T: One;
}

// Conversion trait - enables gradient operations
pub trait StorageToDense<T: DataType>: Storage<T> {
    fn to_dense(&self) -> Result<DenseStorage<T>>;
}

// Matrix operations trait
pub trait MatMulStorage<T: DataType>: Storage<T> {
    fn matmul_storage(&self, other: &Self) -> Result<Self>;
}

// Transpose operations trait
pub trait TransposeStorage<T: DataType>: Storage<T> {
    fn transpose_storage(&self, dim0: usize, dim1: usize) -> Result<Self>;
}
```

### Trait Implementation Matrix

| Storage Type | Storage | FromVec | ToDense | MatMul | Transpose | AsAny |
|--------------|---------|---------|---------|--------|-----------|-------|
| Dense        | ✅      | ✅      | ✅      | ✅     | ✅        | ✅    |
| CSR          | ✅      | ✅      | ✅      | ❌     | ✅        | ✅    |
| CSC          | ✅      | ✅      | ✅      | ❌     | ✅        | ✅    |
| COO          | ✅      | ✅      | ✅      | ❌     | ✅        | ✅    |
| Quantized    | ✅      | ✅      | ✅      | ❌     | ❌        | ✅    |
| Strided      | ✅      | ✅      | ✅      | ❌     | ❌        | ✅    |

## Storage Format Trade-offs

### Dense Storage

**Advantages:**
- ✅ Fast element access (O(1))
- ✅ Cache-friendly memory layout
- ✅ Optimal for dense operations
- ✅ Simple implementation

**Disadvantages:**
- ❌ Memory inefficient for sparse data
- ❌ No compression

**Best For:** General-purpose tensors, dense matrices, convolutions

### Sparse Storage (CSR/CSC/COO)

**Advantages:**
- ✅ Memory efficient for sparse data (O(nnz) vs O(n²))
- ✅ Fast sparse operations
- ✅ Flexible format conversion

**Disadvantages:**
- ❌ Slower element access
- ❌ More complex implementation
- ❌ Overhead for dense data

**Best For:** Sparse matrices (>50% zeros), graph operations, NLP embeddings

### Quantized Storage

**Advantages:**
- ✅ Significant memory savings (2-8x)
- ✅ Faster inference on quantized hardware
- ✅ Reduced bandwidth requirements

**Disadvantages:**
- ❌ Precision loss
- ❌ Quantization/dequantization overhead
- ❌ Not suitable for training

**Best For:** Model deployment, inference, mobile/edge devices

### Strided Storage

**Advantages:**
- ✅ Zero-copy views
- ✅ Efficient transpose/slice
- ✅ Memory efficient

**Disadvantages:**
- ❌ Non-contiguous memory access
- ❌ Cache-unfriendly for some operations
- ❌ Requires stride calculations

**Best For:** Tensor views, broadcasting, zero-copy operations

## Extending the Storage System

To add a new storage format:

1. **Implement `Storage<T>` trait** (mandatory)
2. **Implement `StorageFromVec<T>` trait** (required for tensor creation)
3. **Implement `StorageToDense<T>` trait** (required for gradient operations)
4. **Optionally implement specialized traits** (`MatMulStorage`, `TransposeStorage`)

### Example: Custom Compressed Storage

```rust
use coeus_storage::{Storage, StorageFromVec, StorageToDense, DenseStorage, Shape, Result};
use coeus_dtype::DataType;

#[derive(Debug, Clone)]
pub struct CompressedStorage<T: DataType> {
    compressed_data: Vec<u8>,
    shape: Shape,
    compression_metadata: CompressionMetadata,
}

impl<T: DataType> Storage<T> for CompressedStorage<T> {
    fn as_slice(&self) -> &[T] {
        // Decompress on-the-fly or return cached slice
        unimplemented!("Requires decompression")
    }
    
    fn as_mut_slice(&mut self) -> &mut [T] {
        unimplemented!("Requires decompression")
    }
    
    fn shape(&self) -> &Shape {
        &self.shape
    }
    
    fn strides(&self) -> &[usize] {
        &[] // Compressed storage doesn't have meaningful strides
    }
    
    fn is_contiguous(&self) -> bool {
        false
    }
    
    fn as_storage_ref(&self) -> &Self {
        self
    }
    
    fn full(dims: &[usize], value: T) -> Result<Self> {
        // Implementation
    }
}

impl<T: DataType> StorageFromVec<T> for CompressedStorage<T> {
    fn from_vec(data: Vec<T>, dims: &[usize]) -> Result<Self> {
        let shape = Shape::new(dims)?;
        let compressed_data = compress(&data);
        Ok(Self {
            compressed_data,
            shape,
            compression_metadata: CompressionMetadata::default(),
        })
    }
    
    fn zeros(dims: &[usize]) -> Result<Self> {
        // Implementation
    }
    
    fn ones(dims: &[usize]) -> Result<Self> {
        // Implementation
    }
}

impl<T: DataType> StorageToDense<T> for CompressedStorage<T> {
    fn to_dense(&self) -> Result<DenseStorage<T>> {
        let decompressed = decompress(&self.compressed_data);
        DenseStorage::from_vec(decompressed, self.shape.dims())
    }
}
```

## Usage Guide

### Creating Storage

```rust
use coeus_storage::{DenseStorage, StorageFromVec};
use coeus_dtype::float::Float32;

// From vector
let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
let storage = DenseStorage::from_vec(data, &[3]).unwrap();

// Zeros
let zeros = DenseStorage::<Float32>::zeros(&[2, 3]).unwrap();

// Ones
let ones = DenseStorage::<Float32>::ones(&[4]).unwrap();

// Filled with constant
let fives = DenseStorage::full(&[2, 2], Float32::new(5.0)).unwrap();
```

### Converting Between Formats

```rust
use coeus_storage::{CooStorage, StorageFromVec, StorageToDense};
use coeus_dtype::float::Float32;

// Create sparse storage
let data = vec![Float32::new(1.0), Float32::new(2.0)];
let row_indices = vec![0, 1];
let col_indices = vec![0, 1];
let coo = CooStorage::new(data, row_indices, col_indices, &[2, 2]).unwrap();

// Convert to other sparse formats
let csr = coo.to_csr();
let csc = coo.to_csc();

// Convert to dense
let dense = coo.to_dense().unwrap();
```

### Working with Strided Views

```rust
use coeus_storage::StridedStorage;
use coeus_dtype::float::Float32;

let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0),
                Float32::new(4.0), Float32::new(5.0), Float32::new(6.0)];
let storage = StridedStorage::new(data, &[2, 3]).unwrap();

// Transpose (zero-copy)
let transposed = storage.transpose(None).unwrap();

// Slice (zero-copy)
let sliced = storage.slice(&[(Some(0), Some(2), 1)]).unwrap();

// Convert to contiguous when needed
let dense = sliced.to_dense();
```

## Testing

```bash
# Run all tests
cargo test --package coeus-storage

# Run with coverage
cargo tarpaulin --package coeus-storage

# Run benchmarks
cargo bench --package coeus-storage
```

**Test Coverage:** 100+ tests covering all storage formats and operations

## Documentation

Full API documentation available via:

```bash
cargo doc --package coeus-storage --no-deps --open
```

## Performance Considerations

### Dense Storage
- **Best Performance:** Contiguous memory access, cache-friendly
- **Use When:** Most elements are non-zero, general-purpose operations

### Sparse Storage
- **Best Performance:** When sparsity > 50%, memory-bound operations
- **Use When:** Many zero elements, graph operations, NLP

### Quantized Storage
- **Best Performance:** Inference on quantized hardware, memory-constrained
- **Use When:** Model deployment, reduced precision acceptable

### Strided Storage
- **Best Performance:** Zero-copy views, transpose without data movement
- **Use When:** Temporary views, broadcasting, slicing

## Architecture Principles

1. **Zero-Cost Abstractions:** Trait methods are monomorphized at compile time
2. **Extensibility:** New storage formats can be added without modifying existing code
3. **Type Safety:** Compiler ensures all required operations are implemented
4. **Single Source of Truth:** Each operation defined exactly once
5. **Separation of Concerns:** Storage layout separate from compute backend

## See Also

- [TRAIT_HIERARCHY.md](TRAIT_HIERARCHY.md) - Detailed trait hierarchy documentation
- [Coeus Tensor](../tensor/) - Tensor operations using storage abstractions
- [Coeus Backend](../backend/) - Compute backend implementations
- [Coeus DType](../dtype/) - Data type abstractions

