# ADR-037: Dense Crate Creation

**Status**: Accepted  
**Date**: 2026-01-16  
**Deciders**: Coeus Architecture Team  

## Context

The Coeus framework currently has dense tensor operations mixed within the `tensor` crate alongside multi-dimensional tensor operations. This creates several issues:

1. **Unclear Responsibilities**: The tensor crate handles both high-level multi-dimensional operations and low-level dense algorithms
2. **Maintenance Complexity**: Dense-specific optimizations are mixed with general tensor logic
3. **Domain Confusion**: Dense operations are not clearly separated from sparse and quantized operations
4. **Testing Complexity**: Dense-specific tests are mixed with general tensor tests

The tensor crate should focus on multi-dimensional tensor operations while dense-specific algorithms should have their own domain.

## Decision

We will create a dedicated `dense` crate that contains all dense-specific tensor operations and algorithms, with the following structure:

### New Dense Crate Structure

```
dense/
├── src/
│   ├── algorithms/
│   │   ├── arithmetic.rs       # Element-wise arithmetic operations
│   │   ├── matrix.rs           # Matrix operations (matmul, transpose, etc.)
│   │   ├── reduction.rs        # Reduction operations (sum, mean, etc.)
│   │   ├── broadcasting.rs     # Broadcasting algorithms
│   │   └── indexing.rs         # Indexing and slicing operations
│   ├── ops/
│   │   ├── elementwise.rs      # Element-wise operation implementations
│   │   ├── linear_algebra.rs   # Linear algebra operations
│   │   ├── statistical.rs      # Statistical operations
│   │   ├── comparison.rs       # Comparison operations
│   │   └── trigonometric.rs    # Trigonometric functions
│   ├── utils/
│   │   ├── shape_utils.rs      # Shape manipulation utilities
│   │   ├── stride_utils.rs     # Stride calculation utilities
│   │   └── memory_utils.rs     # Memory layout utilities
│   ├── lib.rs
│   ├── error.rs
│   ├── traits.rs               # Dense-specific traits
│   └── dense_tensor.rs         # Dense tensor wrapper (if needed)
└── Cargo.toml
```

### Dependency Hierarchy

The new hierarchy will be:
```
nn → tensor → dense → storage → backend → dtype
```

Where:
- **tensor** depends on **dense**, **sparse**, **quantization**, and **storage**
- **dense** depends only on **storage** and **dtype**
- **storage** depends only on **backend** and **dtype**

## Rationale

### Benefits

1. **Clear Domain Separation**: Dense operations have their own focused domain
2. **Improved Maintainability**: Dense-specific optimizations are isolated
3. **Better Testing**: Dedicated test suite for dense operations
4. **Cleaner Architecture**: Tensor crate focuses on multi-dimensional operations
5. **Reusability**: Dense operations can be used independently

### Design Principles Satisfied

- **Single Responsibility**: Each crate has a clear, focused responsibility
- **Domain Separation**: Dense logic contained within appropriate boundaries
- **Hierarchical Organization**: Clear dependency hierarchy
- **B<S<T>> Architecture**: Maintains generic architecture pattern

### Dense-Specific Concerns

The dense crate will handle:
- **Dense Algorithms**: Optimized for contiguous memory layout
- **Broadcasting**: NumPy-style broadcasting rules
- **Cache Optimization**: Cache-friendly algorithms for dense data
- **SIMD Optimization**: Vectorized operations for dense arrays
- **Memory Layout**: Contiguous memory access patterns

## Consequences

### Positive

- **Cleaner Architecture**: Better separation between tensor and dense concerns
- **Focused Optimization**: Dense-specific optimizations without affecting other domains
- **Better Testing**: Dedicated dense operation test suite
- **Clearer Dependencies**: Explicit dependency on storage primitives
- **Easier Maintenance**: Dense logic isolated and easier to modify

### Negative

- **Breaking Changes**: Existing code will need import updates
- **Additional Complexity**: One more crate to manage
- **Migration Effort**: Requires systematic extraction from tensor crate

### Migration Impact

**Low Impact Changes**:
- Import statement updates for dense operations
- Cargo.toml dependency additions

**Medium Impact Changes**:
- Code that directly uses dense tensor operations
- Custom dense algorithms

**High Impact Changes**:
- Code that extends dense tensor functionality
- Performance-critical dense operations

## Implementation

### Phase 1: Crate Creation
- Create new dense crate structure
- Set up basic module organization
- Define dense-specific traits

### Phase 2: Algorithm Extraction
- Move dense algorithms from tensor crate
- Organize algorithms by category
- Ensure clean separation from tensor logic

### Phase 3: Operation Implementation
- Implement high-level dense operations
- Build upon storage crate primitives
- Maintain B<S<T>> generic architecture

### Phase 4: Utility Functions
- Extract dense-specific utilities
- Shape manipulation functions
- Memory layout optimizations

### Phase 5: Integration
- Update tensor crate to use dense crate
- Ensure no circular dependencies
- Verify clean dependency hierarchy

### Phase 6: Testing and Validation
- Comprehensive test suite for dense crate
- Integration tests with tensor crate
- Performance validation

## Dense Crate API Design

### Core Traits

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
```

### Integration with Storage

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

## Alternatives Considered

### Alternative 1: Keep dense operations in tensor crate
**Rejected**: Violates domain separation and makes tensor crate too complex.

### Alternative 2: Merge dense with storage crate
**Rejected**: Storage should only handle basic operations, not complex algorithms.

### Alternative 3: Create multiple specialized dense crates
**Rejected**: Would create too much fragmentation for a cohesive domain.

## References

- [Requirements 19.1-19.7](../../.kiro/specs/coeus-architecture-enhancement/requirements.md#requirement-19-dense-crate-creation)
- [Design Document: Dense Crate](../../.kiro/specs/coeus-architecture-enhancement/design.md#dense-crate-new)
- [Clear Dependency Hierarchy](../../.kiro/specs/coeus-architecture-enhancement/requirements.md#requirement-20-clear-dependency-hierarchy)

## Status

**Accepted** - This ADR has been approved and implementation is in progress.

## Notes

- Dense crate will focus exclusively on dense tensor operations
- Integration with tensor crate will be seamless for users
- Performance benchmarks will validate no regression from extraction
- Migration guide will help users update their code