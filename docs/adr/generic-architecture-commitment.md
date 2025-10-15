# ADR: System-Wide B<S<T>> Generic Architecture Commitment

## Status
**APPROVED** - System-wide architectural commitment requiring ALL Coeus components to implement full `Tensor<B<S<T>>>` generics for complete backend, sparse, and datatype support - both present and future.

## Context

Coeus implements PyTorch's complete deep learning functionality in safe Rust with zero-cost abstractions. The system-wide design uses nested generics `Component<B<S<T>>>` where:

- **B (Backend)**: Compute substrate (CPU, GPU, NPU, Distributed)
- **S (Storage)**: Memory layout (Dense, Sparse CSR/CSC/COO, Quantized, Hardware-specific)
- **T (DataType)**: Element type (f32, i64, Complex<f64>, Quantized types, etc.)

This design enables maximum performance through compile-time monomorphization across the ENTIRE system while maintaining PyTorch API compatibility.

**ALL Components Implement B<S<T>> Generics:**
- Neural Networks: `Conv2D<B<S<T>>>`, `Linear<B<S<T>>>`, `Attention<B<S<T>>>`
- Optimizers: `Adam<B<S<T>>>`, `SGD<B<S<T>>>`, `RMSprop<B<S<T>>>`
- Loss Functions: `MSELoss<B<S<T>>>`, `CrossEntropyLoss<B<S<T>>>`
- Activations: `ReLU<B<S<T>>>`, `GELU<B<S<T>>>`, `Sigmoid<B<S<T>>>`
- All Components: Zero-cost specialization for any B, S, T combination

## Problem Statement

During Sprint MS-7, there was an attempt to constrain neural network modules to `DenseStorage<T>` only, reducing the generic signature from `GroupNorm<B, S, T>` to `GroupNorm<B, T>`. This was identified as a premature optimization that would:

1. **Limit Future Extensibility**: Prevent sparse neural network implementations
2. **Break Zero-Cost Abstractions**: Force runtime storage type checks instead of compile-time generics
3. **Complicate API Evolution**: Make it harder to add new storage formats (quantized, compressed)
4. **Violate Architectural Principles**: Contradict the fundamental `B<S<T>>` design commitment

## Architectural Commitment

**Coeus SHALL maintain full `Tensor<B<S<T>>>` generic hierarchy throughout all layers:**

### ✅ **Allowed: Full Generics**
```rust
// Neural network modules maintain full generics
pub struct GroupNorm<B, S, T> where S: Storage<T> + StorageFromVec<T> { ... }
pub struct Linear<B, S, T> where S: Storage<T> + StorageFromVec<T> { ... }

// Tensor operations work with any storage
pub fn add<B, S, T>(lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Tensor<B, S, T>
where S: Storage<T> + StorageFromVec<T> { ... }

// Storage abstraction enables different layouts
pub trait StorageFromVec<T: DataType>: Storage<T> {
    fn from_vec(data: Vec<T>, dims: &[usize]) -> Result<Self>;
    fn zeros(dims: &[usize]) -> Result<Self> where T: Zero;
    fn ones(dims: &[usize]) -> Result<Self> where T: One;
}
```

### ❌ **Forbidden: Storage Type Constraints**
```rust
// BREAKS: Constrains to dense only, prevents sparse implementations
pub struct GroupNorm<B, T> { /* S = DenseStorage<T> hardcoded */ }

// BREAKS: Forces runtime type checks instead of compile-time generics
pub struct GroupNorm<B, T> { storage_type: StorageType, /* runtime dispatch */ }
```

## Design Principles

### 1. System-Wide Zero-Cost Abstractions
**ALL B, S, T combinations resolved at compile-time through monomorphization:**

```rust
// Tensors: Compile-time storage specialization
let dense: Tensor<CpuBackend, DenseStorage<f32>, f32> = Tensor::zeros(&[1000, 1000]);
let sparse: Tensor<CpuBackend, CsrStorage<f32>, f32> = Tensor::zeros(&[1000, 1000]);

// NN Modules: Compile-time backend specialization
let cpu_conv = Conv2D::<CpuBackend, DenseStorage<f32>, f32>::new(3, 64, 3, 1, 1, true)?;
let gpu_conv = Conv2D::<GpuBackend, DenseStorage<f32>, f32>::new(3, 64, 3, 1, 1, true)?;

// Optimizers: Compile-time full specialization
let adam_cpu = Adam::<CpuBackend, DenseStorage<f32>, f32>::new(&[learning_rate])?;
let adam_gpu = Adam::<GpuBackend, SparseStorage<f32>, f32>::new(&[learning_rate])?;

// Zero runtime overhead - all dispatch eliminated at compile-time
```

### 2. Storage Type Extensibility
**New storage formats can be added without changing existing code:**

```rust
// Adding quantized storage requires only implementing traits
pub struct QuantizedStorage<T, ScaleT, ZeroT> { /* ... */ }

impl<T, ScaleT, ZeroT> StorageFromVec<T> for QuantizedStorage<T, ScaleT, ZeroT>
where /* bounds */ {
    fn from_vec(data: Vec<T>, dims: &[usize]) -> Result<Self> { /* ... */ }
    fn zeros(dims: &[usize]) -> Result<Self> where T: Zero { /* ... */ }
    fn ones(dims: &[usize]) -> Result<Self> where T: One { /* ... */ }
}

// Existing neural networks automatically work with quantized tensors
let quantized: Tensor<CpuBackend, QuantizedStorage<f32, f32, i8>, f32> = /* ... */;
let output = group_norm.forward(&quantized); // Works automatically
```

### 3. PyTorch API Compatibility
**Storage abstraction remains invisible to users:**

```python
# Python API identical to PyTorch
import torch
import coeus

# Dense tensors (current implementation)
x = coeus.tensor([[1, 2], [3, 4]], dtype=coeus.float32)
y = x @ x  # Matrix multiplication

# Future: Sparse tensors with same API
sparse_x = coeus.sparse_tensor([[1, 2], [3, 4]], format='csr')
sparse_y = sparse_x @ sparse_x  # Same API, different storage
```

## Implementation Strategy

### Phase 1: Storage Trait Hierarchy (Sprint MS-8)
- Define `StorageFromVec<T>` trait in `coeus_storage` crate
- Implement for `DenseStorage<T>`
- Update tensor creation methods to be generic over storage

### Phase 2: NN Module Generics (Sprint MS-9)
- Update all NN modules to maintain `B, S, T` generics
- Add `S: StorageFromVec<T>` bounds where tensor creation needed
- Ensure zero compilation errors with full generics

### Phase 3: Sparse Storage Implementation (Sprint MS-10)
- Implement `CsrStorage<T>`, `CscStorage<T>`, `CooStorage<T>`
- Add sparse tensor operations
- Test sparse neural network compatibility

### Phase 4: Storage Type Extensions (Future)
- Quantized storage formats
- Compressed storage formats
- Custom hardware-optimized layouts

## Performance Guarantees

### Zero-Cost Storage Dispatch
**Storage type selection happens at compile-time:**

```rust
// No runtime overhead - monomorphized to dense operations
fn matrix_mul<B: Backend, S: Storage<f32>, T: DataType>(
    a: &Tensor<B, S, T>,
    b: &Tensor<B, S, T>
) -> Tensor<B, S, T> {
    // Compile-time dispatch based on S
    S::matrix_mul_impl(a, b)
}
```

### Memory Layout Optimization
**Each storage type optimizes for its access patterns:**

- **DenseStorage**: Contiguous memory, optimal for BLAS operations
- **SparseStorage**: Compressed representation, optimal for sparse computations
- **StridedStorage**: Custom strides, optimal for tensor views/slices

## Risk Mitigation

### Risk: Compilation Time Increase
**Mitigation**: Use incremental compilation, separate crates for different storage types

### Risk: Code Bloat from Monomorphization
**Mitigation**: Common storage types (Dense, CSR) prioritized, others opt-in

### Risk: API Complexity for Users
**Mitigation**: Storage types remain invisible in Python API, Rust API uses type aliases

## Success Metrics

- ✅ **Zero runtime storage type dispatch overhead**
- ✅ **All storage types work with existing NN modules**
- ✅ **PyTorch API compatibility maintained**
- ✅ **Sparse neural networks supported without API changes**
- ✅ **New storage formats can be added without breaking changes**

## References

- [PyTorch Sparse Tensors](https://pytorch.org/docs/stable/sparse.html)
- [Eigen Sparse Matrix Support](https://eigen.tuxfamily.org/dox/group__SparseQuickRefPage.html)
- [JAX Array Abstractions](https://jax.readthedocs.io/en/latest/understanding-jax/arrays.html)

## Enforcement

**This ADR SHALL be enforced system-wide by:**
1. **CI Checks**: Compilation tests with multiple B, S, T combinations (CpuBackend+DenseStorage, GpuBackend+SparseStorage, etc.)
2. **Code Reviews**: Rejection of any component that doesn't implement full B<S<T>> generics
3. **API Design**: ALL public APIs maintain full `Component<B, S, T>` generic signatures
4. **Documentation**: Every component documents its B<S<T>> generic support
5. **Type System**: Generic bounds ensure compile-time enforcement
6. **Performance Tests**: Benchmarks verify zero-cost abstractions across B, S, T combinations

**Any component that doesn't implement full B<S<T>> generics SHALL be rejected. Temporary constraints require ADR approval and clear migration path to full generics.**
