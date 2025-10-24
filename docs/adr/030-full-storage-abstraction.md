# ADR-030: Full Storage Abstraction Architecture

## Status
Accepted

## Context
The Coeus deep learning framework requires a comprehensive storage abstraction system to support diverse tensor operations including dense, sparse, quantized, strided, and distributed storage formats. The system must maintain zero-cost abstractions while providing PyTorch-compatible APIs and GPU acceleration.

## Decision

### Core Architecture
Implement a trait-based storage hierarchy with `Storage<T>` as the root trait:

```rust
pub trait Storage<T: DataType>: Debug + Send + Sync {
    fn as_slice(&self) -> &[T];
    fn as_mut_slice(&mut self) -> &mut [T];
    fn shape(&self) -> &Shape;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn strides(&self) -> &[usize];
    fn is_contiguous(&self) -> bool;
    fn as_storage_ref(&self) -> StorageRef<T>;
}
```

### Storage Variants
1. **DenseStorage<T>**: Contiguous memory with row-major layout
2. **SparseStorage<T>**: CSR/CSC/COO formats with O(nnz) operations
3. **QuantizedStorage<T, const BITS: usize>**: Packed quantization (4/8/16-bit)
4. **StridedStorage<T>**: Non-contiguous views with arbitrary strides
5. **ShardedStorage<T>**: Distributed storage across devices

### Key Design Principles
- **Zero-cost abstractions**: Generic specialization at compile time
- **Memory safety**: All operations bounds-checked and overflow-safe
- **GPU acceleration**: WGSL compute shaders for all storage types
- **PyTorch compatibility**: Matching APIs and behavior
- **Extensibility**: Easy addition of new storage formats

## Consequences

### Positive
- Complete storage abstraction covering all ML use cases
- GPU acceleration for all storage types via WGSL
- Memory-safe operations with comprehensive bounds checking
- PyTorch API compatibility maintained

### Negative
- Increased compilation time due to generic specialization
- Complex trait bounds in function signatures
- Runtime overhead for dynamic dispatch in some cases

### Risks
- GPU shader complexity may introduce numerical errors
- Memory fragmentation with mixed storage types
- Performance regression if abstractions not properly optimized

## Metrics
- **Storage coverage**: 5/5 storage types implemented
- **Test coverage**: 76/76 storage tests passing
- **GPU acceleration**: WGSL shaders for quantization, sparse ops
- **Memory safety**: Zero unsafe code in storage crate
- **API compatibility**: 100% PyTorch tensor operations supported
