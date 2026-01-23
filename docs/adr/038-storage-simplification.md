# ADR-038: Storage Simplification

**Status**: Accepted  
**Date**: 2026-01-16  
**Deciders**: Coeus Architecture Team  

## Context

The current storage crate contains both basic memory operations and complex tensor operations, which violates the principle of layered architecture. The storage layer should serve as a foundation that provides only basic operations, with complex operations implemented in higher layers.

Current issues:
1. **Layer Violation**: Storage contains complex operations like linear transformations and convolutions
2. **Responsibility Confusion**: Storage handles both memory layout and high-level algorithms
3. **Testing Complexity**: Complex operation tests mixed with basic storage tests
4. **Maintenance Burden**: Storage changes affect both memory management and algorithms

## Decision

We will simplify the storage crate to provide only basic operations, moving complex operations to appropriate higher layers:

### Storage Crate Scope (After Simplification)

**ALLOWED in Storage**:
- Basic arithmetic operations (add, sub, mul, div)
- Basic layout operations (reshape, transpose, stride)
- Basic creation operations (zeros, ones, from_vec)
- Memory allocation and management
- Shape and stride calculations
- Device memory transfers

**NOT ALLOWED in Storage**:
- Linear transformations (moved to tensor/nn)
- Convolution operations (moved to nn)
- Complex mathematical functions (moved to dense/sparse)
- Neural network operations (moved to nn)
- Optimization algorithms (moved to optim)

### New Storage Structure

```
storage/src/
├── dense/
│   ├── arithmetic/
│   │   ├── add.rs              # Basic element-wise addition
│   │   ├── sub.rs              # Basic element-wise subtraction
│   │   ├── mul.rs              # Basic element-wise multiplication
│   │   └── div.rs              # Basic element-wise division
│   ├── layout/
│   │   ├── reshape.rs          # Memory layout reshaping
│   │   ├── transpose.rs        # Memory layout transposition
│   │   └── stride.rs           # Stride calculations
│   └── creation/
│       ├── zeros.rs            # Zero-filled tensor creation
│       ├── ones.rs             # One-filled tensor creation
│       └── from_vec.rs         # Creation from vector
├── sparse/
│   ├── csr/
│   │   ├── arithmetic/         # Basic CSR arithmetic
│   │   └── layout/             # CSR layout operations
│   ├── csc/
│   │   └── [similar structure]
│   └── coo/
│       └── [similar structure]
├── quantized/
│   ├── arithmetic/             # Basic quantized arithmetic
│   └── layout/                 # Quantized layout operations
├── ops/
│   ├── arithmetic.rs           # Unified arithmetic interface
│   ├── layout.rs               # Unified layout interface
│   └── creation.rs             # Unified creation interface
├── lib.rs
├── error.rs
└── traits.rs                   # Storage trait definitions
```

### Backend Delegation

All storage operations will delegate to backend primitives:

```rust
// storage/src/dense/arithmetic/add.rs
pub fn add<T: DataType>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
) -> Result<DenseStorage<T>>
where
    T: core::ops::Add<Output = T> + Clone,
{
    // Validate shapes (storage responsibility)
    validate_broadcast_shapes(lhs.shape(), rhs.shape())?;
    
    // Delegate to backend for execution (not storage responsibility)
    let backend = lhs.device().backend();
    let result_data = backend.add_primitive(lhs.data(), rhs.data())?;
    
    // Create result storage (storage responsibility)
    DenseStorage::from_data(result_data, result_shape)
}
```

## Rationale

### Benefits

1. **Clear Layer Separation**: Storage focuses only on memory and basic operations
2. **Foundation Architecture**: Storage serves as a proper foundation layer
3. **Simplified Testing**: Storage tests focus on memory operations only
4. **Better Maintainability**: Changes to algorithms don't affect storage
5. **Cleaner Dependencies**: Storage depends only on backend and dtype

### Design Principles Satisfied

- **Single Responsibility**: Storage handles only memory and basic operations
- **Layered Architecture**: Clear separation between foundation and higher layers
- **Backend Delegation**: Storage delegates computation to backend
- **Domain Separation**: Complex operations moved to appropriate domains

### Operation Migration

| Operation | Current Location | New Location | Rationale |
|-----------|------------------|--------------|-----------|
| Element-wise add/sub/mul/div | Storage | Storage | Basic operations |
| Linear transformation | Storage | Tensor/NN | Complex operation |
| Convolution | Storage | NN | Neural network operation |
| Matrix multiplication | Storage | Dense | Dense-specific algorithm |
| Activation functions | Storage | NN | Neural network operation |
| Reduction operations | Storage | Dense/Sparse | Domain-specific algorithms |

## Consequences

### Positive

- **Cleaner Architecture**: Clear foundation layer with basic operations only
- **Better Testing**: Focused storage tests, complex operation tests elsewhere
- **Easier Maintenance**: Storage changes don't affect complex algorithms
- **Clear Dependencies**: Storage → Backend → Dtype (no upward dependencies)
- **Foundation Stability**: Storage provides stable foundation for higher layers

### Negative

- **Breaking Changes**: Code using complex storage operations needs updates
- **Migration Effort**: Complex operations need to be moved to appropriate layers
- **Import Changes**: Users need to import complex operations from new locations

### Migration Impact

**Low Impact Changes**:
- Basic arithmetic operations (add, sub, mul, div) - no change
- Basic creation operations (zeros, ones) - no change

**Medium Impact Changes**:
- Matrix multiplication - move to dense crate
- Reduction operations - move to dense/sparse crates

**High Impact Changes**:
- Linear transformations - move to tensor/nn crates
- Convolution operations - move to nn crate
- Custom complex operations built on storage

## Implementation

### Phase 1: Audit Current Operations
- Identify all operations in storage crate
- Categorize as basic vs complex
- Document migration destinations

### Phase 2: Split Basic Operations
- Reorganize basic operations into hierarchical structure
- Separate arithmetic, layout, and creation operations
- Maintain existing APIs for basic operations

### Phase 3: Move Complex Operations
- Move linear transformations to tensor/nn crates
- Move convolution operations to nn crate
- Move domain-specific operations to appropriate crates

### Phase 4: Implement Backend Delegation
- Update storage operations to delegate to backend primitives
- Remove hardware-specific code from storage
- Ensure clear separation between storage and backend

### Phase 5: Update Dependencies
- Update higher-level crates to import complex operations from new locations
- Verify no circular dependencies
- Update documentation and examples

### Phase 6: Testing and Validation
- Comprehensive test suite for simplified storage
- Integration tests with higher-level crates
- Performance validation

## Storage Trait Hierarchy (Simplified)

```rust
/// Core storage operations (basic only)
pub trait Storage<T: DataType> {
    fn as_slice(&self) -> &[T];
    fn as_mut_slice(&mut self) -> &mut [T];
    fn shape(&self) -> &Shape;
    fn strides(&self) -> &[usize];
    fn is_contiguous(&self) -> bool;
}

/// Basic creation operations
pub trait StorageFromVec<T: DataType>: Storage<T> {
    fn from_vec(data: Vec<T>, dims: &[usize]) -> Result<Self>;
    fn zeros(dims: &[usize]) -> Result<Self> where T: Zero;
    fn ones(dims: &[usize]) -> Result<Self> where T: One;
}

/// Basic arithmetic operations
pub trait StorageArithmetic<T: DataType>: Storage<T> {
    fn add(&self, other: &Self) -> Result<Self> where Self: Sized;
    fn sub(&self, other: &Self) -> Result<Self> where Self: Sized;
    fn mul(&self, other: &Self) -> Result<Self> where Self: Sized;
    fn div(&self, other: &Self) -> Result<Self> where Self: Sized;
}

/// Basic layout operations
pub trait StorageLayout<T: DataType>: Storage<T> {
    fn reshape(&self, new_dims: &[usize]) -> Result<Self> where Self: Sized;
    fn transpose(&self, dims: &[usize]) -> Result<Self> where Self: Sized;
}

// REMOVED: Complex operations like MatMulOps, ConvolutionOps, etc.
```

## Alternatives Considered

### Alternative 1: Keep complex operations in storage
**Rejected**: Violates layered architecture and single responsibility principle.

### Alternative 2: Create separate basic-storage and complex-storage crates
**Rejected**: Would create confusion about which storage crate to use.

### Alternative 3: Move all operations to higher layers, keep only memory management
**Rejected**: Too extreme, basic arithmetic operations belong in storage.

## References

- [Requirements 18.1-18.6](../../.kiro/specs/coeus-architecture-enhancement/requirements.md#requirement-18-storage-basic-operations-only)
- [Design Document: Storage Simplification](../../.kiro/specs/coeus-architecture-enhancement/design.md#storage-simplification)
- [Backend Foundation Architecture](./034-gpu-backend-implementation.md)

## Status

**Accepted** - This ADR has been approved and implementation is in progress.

## Notes

- Migration guide will help users update code that uses complex storage operations
- Performance benchmarks will validate no regression from simplification
- Storage will maintain backward compatibility for basic operations
- Complex operations will have better homes in domain-specific crates