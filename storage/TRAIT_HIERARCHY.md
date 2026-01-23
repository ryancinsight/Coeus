# Storage Trait Hierarchy Documentation

**Last Updated:** January 14, 2026  
**Audit Status:** ✅ Completed - SSOT Compliance Verified

## Overview

The Coeus storage system provides a flexible, extensible trait hierarchy that enables zero-cost abstractions over different memory layouts. This document describes the trait hierarchy, implementation guidelines, and audit findings validating Single Source of Truth (SSOT) compliance.

## Audit Summary

**SSOT Compliance:** ✅ EXCELLENT (100/100)

- ✅ Zero duplicate implementations found
- ✅ Each operation defined exactly once
- ✅ Clear, explicit conversion paths between formats
- ✅ Consistent trait implementation across all storage types
- ✅ All storage formats follow Single Responsibility Principle (SRP)

See [STORAGE_AUDIT_15_1.md](../../.kiro/specs/coeus-architecture-enhancement/STORAGE_AUDIT_15_1.md) for detailed audit report.

## Core Traits

### 1. `Storage<T: DataType>`

The foundational trait that all storage types must implement. Provides basic operations for accessing and managing tensor data.

**Required Methods:**
- `as_slice(&self) -> &[T]` - Get immutable slice view
- `as_mut_slice(&mut self) -> &mut [T]` - Get mutable slice view
- `shape(&self) -> &Shape` - Get tensor shape
- `strides(&self) -> &[usize]` - Get stride information
- `is_contiguous(&self) -> bool` - Check if memory is contiguous
- `as_storage_ref(&self) -> &Self` - Get reference to self
- `full(dims: &[usize], value: T) -> Result<Self>` - Create storage filled with value

**Provided Methods:**
- `len(&self) -> usize` - Total number of elements
- `is_empty(&self) -> bool` - Check if storage is empty

**Example Implementation:**
```rust
impl<T: DataType> Storage<T> for MyCustomStorage<T> {
    fn as_slice(&self) -> &[T] {
        &self.data
    }
    
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
    
    fn shape(&self) -> &Shape {
        &self.shape
    }
    
    fn strides(&self) -> &[usize] {
        &self.strides
    }
    
    fn is_contiguous(&self) -> bool {
        true // or your custom logic
    }
    
    fn as_storage_ref(&self) -> &Self {
        self
    }
    
    fn full(dims: &[usize], value: T) -> Result<Self> {
        // Implementation
    }
}
```

### 2. `StorageFromVec<T: DataType>`

Enables creating storage from vectors and common initialization patterns. This trait is crucial for tensor creation operations.

**Required Methods:**
- `from_vec(data: Vec<T>, dims: &[usize]) -> Result<Self>` - Create from vector
- `zeros(dims: &[usize]) -> Result<Self>` where `T: Zero` - Create zero-filled storage
- `ones(dims: &[usize]) -> Result<Self>` where `T: One` - Create one-filled storage

**Example Implementation:**
```rust
impl<T: DataType> StorageFromVec<T> for MyCustomStorage<T> {
    fn from_vec(data: Vec<T>, dims: &[usize]) -> Result<Self> {
        let shape = Shape::new(dims)?;
        if data.len() != shape.size() {
            return Err(StorageError::ShapeMismatch {
                expected: shape.size(),
                actual: data.len(),
            });
        }
        Ok(Self { data, shape, /* ... */ })
    }
    
    fn zeros(dims: &[usize]) -> Result<Self>
    where
        T: num_traits::Zero,
    {
        let shape = Shape::new(dims)?;
        let data = vec![T::zero(); shape.size()];
        Ok(Self { data, shape, /* ... */ })
    }
    
    fn ones(dims: &[usize]) -> Result<Self>
    where
        T: num_traits::One,
    {
        let shape = Shape::new(dims)?;
        let data = vec![T::one(); shape.size()];
        Ok(Self { data, shape, /* ... */ })
    }
}
```

### 3. `StorageToDense<T: DataType>`

Enables conversion to dense representation, which is required for gradient operations and certain computations.

**Required Methods:**
- `to_dense(&self) -> Result<DenseStorage<T>>` - Convert to dense storage

**Example Implementation:**
```rust
impl<T: DataType> StorageToDense<T> for MyCustomStorage<T> {
    fn to_dense(&self) -> Result<DenseStorage<T>> {
        // Convert your custom format to dense
        DenseStorage::from_vec(self.data.clone(), self.shape.dims())
    }
}
```

### 4. `MatMulStorage<T: DataType>`

Provides matrix multiplication at the storage level for zero-cost abstractions.

**Required Methods:**
- `matmul_storage(&self, other: &Self) -> Result<Self>` - Matrix multiplication

**Example Implementation:**
```rust
impl<T: DataType> MatMulStorage<T> for MyCustomStorage<T>
where
    T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + num_traits::Zero + Copy,
{
    fn matmul_storage(&self, other: &Self) -> Result<Self> {
        // Implement matrix multiplication for your storage format
    }
}
```

### 5. `TransposeStorage<T: DataType>`

Provides transpose operations at the storage level.

**Required Methods:**
- `transpose_storage(&self, dim0: usize, dim1: usize) -> Result<Self>` - Transpose along dimensions

**Example Implementation:**
```rust
impl<T: DataType> TransposeStorage<T> for MyCustomStorage<T> {
    fn transpose_storage(&self, dim0: usize, dim1: usize) -> Result<Self> {
        // Implement transpose for your storage format
    }
}
```

### 6. `ActivationOps<T: DataType>`

Provides activation function operations at the storage level.

**Required Methods:**
- `relu(&self) -> Self` - ReLU activation
- `tanh(&self) -> Self` - Tanh activation
- `sigmoid(&self) -> Self` - Sigmoid activation
- `gelu(&self) -> Self` - GELU activation
- `swish(&self) -> Self` - Swish activation
- `hardsigmoid(&self) -> Self` - Hard sigmoid activation
- `hardswish(&self) -> Self` - Hard swish activation

## Storage Implementations

### Implementation Status

| Storage Type | Status | SSOT Compliance | Traits Implemented |
|--------------|--------|-----------------|-------------------|
| DenseStorage | ✅ Complete | ✅ Excellent | Storage, FromVec, ToDense, MatMul, Transpose, AsAny |
| CsrStorage | ✅ Complete | ✅ Excellent | Storage, FromVec, ToDense, Transpose, AsAny |
| CscStorage | ✅ Complete | ✅ Excellent | Storage, FromVec, ToDense, Transpose, AsAny |
| CooStorage | ✅ Complete | ✅ Excellent | Storage, FromVec, ToDense, Transpose, AsAny |
| QuantizedStorage | ✅ Complete | ✅ Excellent | Storage, FromVec, ToDense, AsAny |
| StridedStorage | ✅ Complete | ✅ Excellent | Storage, FromVec, ToDense, AsAny |
| DistributedStorage | ⚠️ Partial | Not Audited | Storage, FromVec, ToDense |

### DenseStorage<T>

Contiguous row-major storage. Implements all core traits.

**Traits Implemented:**
- `Storage<T>`
- `StorageFromVec<T>`
- `StorageToDense<T>` (returns clone)
- `MatMulStorage<T>`
- `TransposeStorage<T>`

### Sparse Storage Formats

**SSOT Validation:** ✅ All sparse formats follow single source of truth principles with no duplicate implementations.

**Format Conversion Graph:**
```
        COO (Coordinate)
       /   \
      /     \
    CSR     CSC
     \     /
      \   /
      Dense
```

**Conversion Paths:**
- COO → CSR: Direct implementation in `CooStorage::to_csr()`
- COO → CSC: Direct implementation in `CooStorage::to_csc()`
- CSR → COO: Direct implementation in `CsrStorage::to_coo()`
- CSR → CSC: Via COO in `CsrStorage::to_csc()`
- CSC → COO: Direct implementation in `CscStorage::to_coo()`
- CSC → CSR: Via COO in `CscStorage::to_csr()`
- Any → Dense: Via `StorageToDense` trait

Each conversion path is implemented exactly once with no duplication.

#### CsrStorage<T> - Compressed Sparse Row
Efficient for row-wise operations and matrix-vector multiplication.

#### CscStorage<T> - Compressed Sparse Column
Efficient for column-wise operations.

#### CooStorage<T> - Coordinate Format
Flexible format for building sparse matrices.

### QuantizedStorage<T>

Packed quantized values for memory-efficient storage.

**SSOT Validation:** ✅ Quantization and dequantization logic defined exactly once.

**Traits Implemented:**
- `Storage<T>`
- `StorageFromVec<T>`
- `StorageToDense<T>`
- `AsAny` ✅ (Added in consolidation phase)

**Variants:**
- `QuantizedStorage4<T>` - 4-bit quantization
- `QuantizedStorage8<T>` - 8-bit quantization
- `QuantizedStorage16<T>` - 16-bit quantization

### StridedStorage<T>

Custom strides for views and transpose operations without data copying.

**SSOT Validation:** ✅ All view operations defined exactly once.

**Traits Implemented:**
- `Storage<T>`
- `StorageFromVec<T>`
- `StorageToDense<T>`
- `AsAny`

### DistributedStorage<T>

Multi-device tensor storage for distributed computing.

**Status:** ⚠️ Not fully audited

**Expected Traits:**
- `Storage<T>`
- `StorageFromVec<T>`
- `StorageToDense<T>`

## SSOT Compliance Report

### Audit Findings (January 2026)

**Overall Score:** ✅ 100/100

**Metrics:**
- Storage Implementations Audited: 6
- Duplicate Implementations Found: 0
- Duplicate Implementations Removed: 0
- Consistency Fixes Applied: 1 (Added `AsAny` to `QuantizedStorage`)
- SSOT Violations: 0

**Key Findings:**
1. ✅ Each storage format implements operations exactly once
2. ✅ Format conversions follow clear, non-redundant paths
3. ✅ All trait implementations are consistent
4. ✅ No hidden duplication between implementations
5. ✅ Single source for all algorithms (quantization, strided copy, matmul)

**Recommendations:**
1. ✅ **COMPLETED:** Add `AsAny` implementation to `QuantizedStorage`
2. 📋 **FUTURE:** Audit `DistributedStorage` implementation
3. 📋 **FUTURE:** Consider consolidating `Storage<T>` and `StorageOps<T>` trait hierarchies

See [STORAGE_CONSOLIDATION_15_2.md](../../.kiro/specs/coeus-architecture-enhancement/STORAGE_CONSOLIDATION_15_2.md) for detailed consolidation report.

## Extending the Storage System

To add a new storage format:

1. **Implement `Storage<T>` trait** - This is mandatory
2. **Implement `StorageFromVec<T>` trait** - Required for tensor creation
3. **Implement `StorageToDense<T>` trait** - Required for gradient operations
4. **Optionally implement specialized traits** - `MatMulStorage`, `TransposeStorage`, `ActivationOps`

### Example: Custom Compressed Storage

```rust
use storage::{Storage, StorageFromVec, StorageToDense, DenseStorage, Shape, Result};
use dtype::DataType;

#[derive(Debug, Clone)]
pub struct CompressedStorage<T: DataType> {
    compressed_data: Vec<u8>,
    shape: Shape,
    compression_metadata: CompressionMetadata,
}

impl<T: DataType> Storage<T> for CompressedStorage<T> {
    // Implement required methods
    fn as_slice(&self) -> &[T] {
        // Decompress on-the-fly or cache
        unimplemented!("Requires decompression")
    }
    
    // ... other methods
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
    
    // ... other methods
}

impl<T: DataType> StorageToDense<T> for CompressedStorage<T> {
    fn to_dense(&self) -> Result<DenseStorage<T>> {
        let decompressed = decompress(&self.compressed_data);
        DenseStorage::from_vec(decompressed, self.shape.dims())
    }
}
```

## Trait Bounds in Operations

When writing operations that work with any storage type, use appropriate trait bounds:

```rust
pub fn my_operation<B, S, T>(tensor: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,  // Requires both traits
    T: DataType + FloatExt,
{
    // Implementation can use both Storage and StorageFromVec methods
}
```

## Benefits of This Design

1. **Zero-Cost Abstractions**: Trait methods are monomorphized at compile time
2. **Extensibility**: New storage formats can be added without modifying existing code
3. **Type Safety**: Compiler ensures all required operations are implemented
4. **Flexibility**: Operations can be generic over any storage type
5. **Performance**: Specialized implementations for each storage format

## Testing Storage Implementations

When implementing a new storage type, ensure:

1. All trait methods are tested
2. Edge cases are handled (empty tensors, single elements)
3. Shape validation is correct
4. Memory safety is guaranteed
5. Performance is acceptable for your use case

See `storage/tests/` for examples of comprehensive storage tests.
