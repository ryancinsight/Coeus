# Coeus Crate Interfaces and Boundaries

**Date:** January 16, 2026  
**Task:** 5.3 Create clear inter-crate interfaces  
**Requirements:** 16.5

## Overview

This document defines the clear interfaces and boundaries between Coeus framework crates. It specifies what each crate provides, what it depends on, and how crates communicate with each other.

## Dependency Hierarchy

The Coeus framework follows a strict dependency hierarchy from foundation to high-level:

```
dtype (foundation - no dependencies)
  ↑
backend (depends on: dtype)
  ↑
storage (depends on: backend, dtype)
  ↑
├─ dense (depends on: storage, dtype)
├─ sparse (depends on: storage, dtype)
└─ quantization (depends on: storage, dtype)
  ↑
tensor (depends on: dense, sparse, quantization, storage, dtype)
  ↑
nn (depends on: tensor, dense, sparse, quantization)
  ↑
pycoeus (depends on: nn, tensor, optim)
```

## Crate Interfaces

### 1. dtype Crate

**Purpose:** Pure type definitions and conversions

**Public Interface:**
```rust
// Core trait
pub trait DataType: Clone + Copy + Send + Sync + 'static {
    fn dtype() -> DType;
    fn is_floating_point() -> bool;
    fn is_integer() -> bool;
    fn is_quantized() -> bool;
}

// Concrete types
pub struct Float32(f32);
pub struct Float64(f64);
pub struct Int32(i32);
pub struct Int64(i64);
pub struct Complex32 { re: f32, im: f32 }
pub struct Complex64 { re: f64, im: f64 }

// Type enum
pub enum DType {
    Float32, Float64,
    Int32, Int64,
    Complex32, Complex64,
}
```

**Dependencies:** None (foundation)

**Provides To:**
- All crates: Type definitions and trait bounds
- Backend: Element type specifications
- Storage: Data type for storage elements

**Boundaries:**
- ✅ Contains: Pure type definitions, conversions, trait implementations
- ❌ Does NOT contain: Quantization logic, operations, algorithms

---

### 2. backend Crate

**Purpose:** Hardware execution primitives (CPU, GPU, TPU, NPU)

**Public Interface:**
```rust
// Core backend trait
pub trait Backend: Clone + Default + Send + Sync + 'static {
    type Data: DataType;
    type Device: Device;
    
    fn device(&self) -> &Self::Device;
    fn synchronize(&self) -> Result<()>;
}

// Concrete backends
pub struct CpuBackend<T: DataType> { ... }
pub struct GpuBackend<T: DataType> { ... }
pub struct TpuBackend<T: DataType> { ... }
pub struct NpuBackend<T: DataType> { ... }

// Primitive operations (organized hierarchically)
// backend/src/cpu/arithmetic/add.rs
pub fn add_primitive<T>(lhs: &[T], rhs: &[T], result: &mut [T]) -> Result<()>;

// backend/src/cpu/linear_algebra/matmul.rs
pub fn matmul_primitive<T>(a: &[T], b: &[T], c: &mut [T], m: usize, n: usize, k: usize) -> Result<()>;
```

**Dependencies:** dtype

**Provides To:**
- Storage: Hardware execution for basic operations
- Dense/Sparse: BLAS/LAPACK primitives for complex operations
- All layers: Device management and synchronization

**Boundaries:**
- ✅ Contains: Hardware-specific implementations, SIMD, GPU kernels, device management
- ❌ Does NOT contain: Storage layouts, tensor operations, neural network logic

---

### 3. storage Crate

**Purpose:** Memory layout management and basic operations

**Public Interface:**
```rust
// Core storage trait
pub trait Storage<T: DataType>: Clone + Send + Sync + 'static {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn as_slice(&self) -> &[T];
    fn as_mut_slice(&mut self) -> &mut [T];
}

// Storage creation
pub trait StorageFromVec<T: DataType>: Storage<T> {
    fn from_vec(data: Vec<T>, dims: &[usize]) -> Result<Self>;
    fn zeros(dims: &[usize]) -> Result<Self> where T: Zero;
    fn ones(dims: &[usize]) -> Result<Self> where T: One;
}

// Basic arithmetic (delegates to backend)
pub trait ArithmeticOps<T: DataType>: Storage<T> {
    fn add(&self, other: &Self) -> Result<Self>;
    fn sub(&self, other: &Self) -> Result<Self>;
    fn mul(&self, other: &Self) -> Result<Self>;
    fn div(&self, other: &Self) -> Result<Self>;
}

// Basic layout operations
pub trait LayoutOps<T: DataType>: Storage<T> {
    fn transpose(&self, rows: usize, cols: usize) -> Result<Self>;
    fn reshape(&self, old_shape: &[usize], new_shape: &[usize]) -> Result<Self>;
}

// Concrete storage types
pub struct DenseStorage<T> { data: Vec<T> }
pub struct CsrStorage<T> { data: Vec<T>, indices: Vec<usize>, indptr: Vec<usize> }
pub struct CscStorage<T> { data: Vec<T>, indices: Vec<usize>, indptr: Vec<usize> }
pub struct CooStorage<T> { data: Vec<T>, row_indices: Vec<usize>, col_indices: Vec<usize> }
pub struct StridedStorage<T> { data: Vec<T>, strides: Vec<isize> }
```

**Dependencies:** backend, dtype

**Provides To:**
- Dense/Sparse/Quantization: Storage formats and basic operations
- Tensor: Memory layout abstraction
- NN: Underlying storage for parameters

**Boundaries:**
- ✅ Contains: Memory layouts, basic arithmetic (add/sub/mul/div), basic layout ops (transpose/reshape)
- ❌ Does NOT contain: Matrix multiplication, convolutions, activations, complex algorithms
- ⚠️ **QUESTION:** Should MatMul be here? (See MatMul Placement Decision below)

**Note on MatMul:** Currently storage defines `MatMulOps` trait, but Requirement 18.4 states storage should NOT provide complex operations like linear transformations. Matrix multiplication IS a linear transformation. See "MatMul Placement Decision" section below.

---

### 4. dense Crate

**Purpose:** Dense tensor operations and algorithms

**Public Interface:**
```rust
// Dense arithmetic operations
pub trait DenseArithmetic<T: DataType> {
    fn add<B: Backend<Data = T>>(
        lhs: &DenseStorage<T>,
        rhs: &DenseStorage<T>,
        backend: &B,
    ) -> Result<DenseStorage<T>>;
    
    fn matmul<B: Backend<Data = T>>(
        lhs: &DenseStorage<T>,
        rhs: &DenseStorage<T>,
        m: usize, n: usize, k: usize,
        backend: &B,
    ) -> Result<DenseStorage<T>>;
}

// Dense linear algebra
pub trait DenseLinearAlgebra<T: DataType> {
    fn svd<B: Backend<Data = T>>(...) -> Result<(DenseStorage<T>, DenseStorage<T>, DenseStorage<T>)>;
    fn qr<B: Backend<Data = T>>(...) -> Result<(DenseStorage<T>, DenseStorage<T>)>;
    fn cholesky<B: Backend<Data = T>>(...) -> Result<DenseStorage<T>>;
}
```

**Dependencies:** storage, dtype

**Provides To:**
- Tensor: Dense tensor operations
- NN: Dense layer implementations

**Boundaries:**
- ✅ Contains: Dense-specific algorithms, BLAS/LAPACK wrappers, dense matrix operations
- ❌ Does NOT contain: Sparse operations, quantization, neural network layers

---

### 5. sparse Crate

**Purpose:** Sparse tensor operations and algorithms

**Public Interface:**
```rust
// Sparse arithmetic operations
pub trait SparseArithmetic<T: DataType> {
    fn add_sparse<B: Backend<Data = T>>(
        lhs: &CsrStorage<T>,
        rhs: &CsrStorage<T>,
        backend: &B,
    ) -> Result<CooStorage<T>>;
}

// Sparse matrix multiplication
pub trait SparseMatMul<T: DataType> {
    fn matmul_sparse<B: Backend<Data = T>>(
        lhs: &CsrStorage<T>,
        rhs: &CsrStorage<T>,
        result_format: SparseFormat,
        backend: &B,
    ) -> Result<CooStorage<T>>;
    
    fn matvec_mul<B: Backend<Data = T>>(
        matrix: &CsrStorage<T>,
        vector: &[T],
        backend: &B,
    ) -> Result<Vec<T>>;
}

// Format conversions (in storage, not sparse)
// Note: Format conversions remain in storage as they're storage-level concerns
```

**Dependencies:** storage, dtype

**Provides To:**
- Tensor: Sparse tensor operations
- NN: Sparse layer implementations

**Boundaries:**
- ✅ Contains: Sparse-specific algorithms, CSR/CSC/COO operations, sparse matrix multiplication
- ❌ Does NOT contain: Dense operations, quantization, neural network layers

---

### 6. quantization Crate

**Purpose:** Quantization algorithms and operations

**Public Interface:**
```rust
// Quantization algorithms
pub struct SymmetricQuantizer<T: DataType> {
    pub fn quantize(&self, input: &[T]) -> Result<Vec<u8>>;
    pub fn dequantize(&self, input: &[u8]) -> Result<Vec<T>>;
}

pub struct AsymmetricQuantizer<T: DataType> { ... }
pub struct DynamicQuantizer<T: DataType> { ... }

// Calibration
pub trait Calibration<T: DataType> {
    fn calibrate_entropy(...) -> Result<(f32, i32)>;
    fn calibrate_percentile(...) -> Result<(f32, i32)>;
    fn calibrate_mse(...) -> Result<(f32, i32)>;
}

// Fake quantization (for training)
pub fn fake_quantize_linear<T: DataType>(
    input: &DenseStorage<T>,
    scale: f32,
    zero_point: i32,
) -> Result<DenseStorage<T>>;
```

**Dependencies:** storage, dtype

**Provides To:**
- Tensor: Quantized tensor operations
- NN: Quantization-aware training, quantized layers

**Boundaries:**
- ✅ Contains: Quantization algorithms, calibration, fake quantization, quantized operations
- ❌ Does NOT contain: Pure type definitions (in dtype), dense/sparse operations

---

### 7. tensor Crate

**Purpose:** Multi-dimensional tensor operations and automatic differentiation

**Public Interface:**
```rust
// Core tensor type
pub struct Tensor<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    storage: S,
    shape: Shape,
    backend: B,
    grad: Option<Arc<RwLock<Tensor<B, S, T>>>>,
}

// Tensor operations (delegates to dense/sparse/quantization)
impl<B, S, T> Tensor<B, S, T> {
    // Creation
    pub fn zeros(shape: &[usize]) -> Result<Self>;
    pub fn ones(shape: &[usize]) -> Result<Self>;
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self>;
    
    // Arithmetic (delegates to storage)
    pub fn add(&self, other: &Self) -> Result<Self>;
    pub fn sub(&self, other: &Self) -> Result<Self>;
    pub fn mul(&self, other: &Self) -> Result<Self>;
    pub fn div(&self, other: &Self) -> Result<Self>;
    
    // Linear algebra (delegates to dense/sparse)
    pub fn matmul(&self, other: &Self) -> Result<Self>;
    
    // Autograd
    pub fn backward(&self) -> Result<()>;
    pub fn grad(&self) -> Option<Self>;
}

// Sparse tensor operations (thin wrappers)
impl<B, T> Tensor<B, CsrStorage<T>, T> {
    pub fn sparse_matmul(&self, other: &Self) -> Result<Tensor<B, CooStorage<T>, T>>;
}
```

**Dependencies:** dense, sparse, quantization, storage, dtype

**Provides To:**
- NN: Tensor operations for neural networks
- PyCoeus: Python tensor API

**Boundaries:**
- ✅ Contains: Multi-dimensional operations, autograd, tensor API, delegation to specialized crates
- ❌ Does NOT contain: Neural network layers, optimizers, training loops

---

### 8. nn Crate

**Purpose:** Neural network layers, operations, and training utilities

**Public Interface:**
```rust
// Stateless operations (single source of truth)
pub mod functional {
    pub mod ops {
        pub mod activation {
            pub fn relu<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>;
            pub fn gelu<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>;
        }
        
        pub mod loss {
            pub fn mse_loss<B, S, T>(...) -> Result<Tensor<B, S, T>>;
            pub fn cross_entropy_loss<B, S, T>(...) -> Result<Tensor<B, S, T>>;
        }
        
        pub mod convolution {
            pub fn conv2d<B, S, T>(...) -> Result<Tensor<B, S, T>>;
        }
    }
}

// Stateful layers (delegate to functional/ops)
pub mod modules {
    pub struct Linear<B, S, T> {
        weight: Parameter<B, S, T>,
        bias: Option<Parameter<B, S, T>>,
    }
    
    impl<B, S, T> Module<B, S, T> for Linear<B, S, T> {
        fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
            functional::ops::linear::linear(input, &self.weight, self.bias.as_ref())
        }
    }
}

// Module trait
pub trait Module<B, S, T> {
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>;
    fn parameters(&self) -> Vec<Parameter<B, S, T>>;
    fn zero_grad(&mut self);
}
```

**Dependencies:** tensor, dense, sparse, quantization

**Provides To:**
- PyCoeus: Neural network API
- Users: High-level neural network interface

**Boundaries:**
- ✅ Contains: Neural network layers, operations, training utilities, loss functions
- ❌ Does NOT contain: Tensor operations (in tensor), optimizers (in optim)

---

### 9. pycoeus Crate

**Purpose:** Python bindings for Coeus framework

**Public Interface:**
```python
# Python API (PyTorch-compatible)
import coeus

# Tensor operations
tensor = coeus.tensor([1.0, 2.0, 3.0])
result = tensor.matmul(other_tensor)

# Neural network layers
linear = coeus.nn.Linear(10, 5)
output = linear(input)

# Optimizers
optimizer = coeus.optim.Adam(model.parameters(), lr=0.001)
optimizer.step()

# Exception hierarchy
try:
    result = tensor.matmul(incompatible_tensor)
except coeus.ShapeError as e:
    print(f"Shape mismatch: {e}")
```

**Dependencies:** nn, tensor, optim

**Provides To:**
- Python users: PyTorch-compatible API

**Boundaries:**
- ✅ Contains: Python bindings, PyO3 wrappers, exception conversion
- ❌ Does NOT contain: Core algorithms (in Rust crates)

---

## Allowed Dependencies

This table shows which crates can depend on which:

| Crate | Can Depend On |
|-------|---------------|
| dtype | None (foundation) |
| backend | dtype |
| storage | backend, dtype |
| dense | storage, dtype |
| sparse | storage, dtype |
| quantization | storage, dtype |
| tensor | dense, sparse, quantization, storage, dtype |
| nn | tensor, dense, sparse, quantization |
| optim | tensor |
| pycoeus | nn, tensor, optim |

**Forbidden Dependencies:**
- ❌ Circular dependencies (e.g., storage → dense → storage)
- ❌ Skipping layers (e.g., nn → storage directly, should go through tensor)
- ❌ Reverse dependencies (e.g., dtype → backend)

---

## Communication Patterns

### 1. Delegation Pattern (Preferred)

Higher-level crates delegate to lower-level crates:

```rust
// nn/src/modules/linear.rs
impl<B, S, T> Module<B, S, T> for Linear<B, S, T> {
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Delegate to functional/ops (single source of truth)
        crate::functional::ops::linear::linear(input, &self.weight, self.bias.as_ref())
    }
}

// tensor/src/ops/sparse.rs
impl<B, T> Tensor<B, CsrStorage<T>, T> {
    pub fn sparse_matmul(&self, other: &Self) -> Result<Tensor<B, CooStorage<T>, T>> {
        // Delegate to sparse crate
        let result_storage = self.storage
            .matmul_sparse(&other.storage, SparseFormat::Csr, &self.backend)?;
        Ok(Tensor::from_storage(result_storage, self.backend.clone()))
    }
}
```

### 2. Trait Bounds Pattern

Crates use trait bounds to specify requirements:

```rust
// tensor/src/ops/arithmetic.rs
pub fn add<B, S, T>(lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T>,
    S: Storage<T> + ArithmeticOps<T>,  // Requires storage to support arithmetic
    T: DataType,
{
    let result_storage = lhs.storage.add(&rhs.storage)?;
    Ok(Tensor::from_storage(result_storage, lhs.backend.clone()))
}
```

### 3. Generic Specialization Pattern

Operations work with any storage type through generics:

```rust
// nn/src/functional/ops/activation.rs
pub fn relu<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,  // Works with any storage
    T: DataType + PartialOrd + Zero,
{
    // Implementation works for dense, sparse, quantized, etc.
}
```

---

## MatMul Placement Decision

**Current State:** Storage defines `MatMulOps` trait with matrix multiplication

**Requirement Conflict:** Requirement 18.4 states: "THE Storage_System SHALL NOT provide complex operations like linear transformations or convolutions"

**Analysis:**
- Matrix multiplication IS a linear transformation
- Requirements specify storage should have only: add, sub, mul, div, reshape, transpose, stride
- MatMul is NOT listed as a basic operation

**Options:**

### Option 1: Move MatMul to dense/sparse crates (Recommended)

**Pros:**
- ✅ Aligns with Requirement 18.4
- ✅ Separates complex operations from basic storage
- ✅ Sparse crate already has matmul implementations
- ✅ Clearer architectural boundaries

**Cons:**
- ⚠️ Breaking change for code using storage matmul directly
- ⚠️ Dense crate needs matmul implementation added

**Implementation:**
1. Add `DenseMatMul` trait to dense crate
2. Implement matmul for `DenseStorage` in dense crate
3. Deprecate `MatMulOps` in storage with warning
4. Update tensor crate to use dense/sparse matmul
5. Remove `MatMulOps` from storage in next major version

### Option 2: Keep MatMul in storage

**Pros:**
- ✅ No breaking changes
- ✅ Simpler for format-specific implementations

**Cons:**
- ❌ Violates Requirement 18.4
- ❌ Blurs boundary between basic and complex operations

### Option 3: Update requirements to include MatMul

**Pros:**
- ✅ No code changes needed
- ✅ Acknowledges MatMul as fundamental

**Cons:**
- ⚠️ Changes architectural vision
- ⚠️ May lead to scope creep (what else is "fundamental"?)

**Recommendation:** **Option 1** - Move MatMul to dense/sparse crates to align with stated requirements and maintain clear architectural boundaries.

---

## Interface Documentation Locations

Each crate should maintain interface documentation:

- `dtype/README.md` - Type system interface
- `backend/README.md` - Backend interface and primitives
- `storage/README.md` - Storage formats and basic operations
- `dense/README.md` - Dense operations interface
- `sparse/README.md` - Sparse operations interface
- `quantization/README.md` - Quantization interface
- `tensor/README.md` - Tensor API interface
- `nn/README.md` - Neural network interface
- `pycoeus/README.md` - Python API interface

---

## Verification

To verify interface boundaries are maintained:

1. **Dependency Check:**
   ```bash
   cargo tree --workspace
   ```
   Verify no circular dependencies or forbidden dependencies

2. **Boundary Tests:** (Task 5.4)
   Implement tests that verify:
   - Sparse operations only in sparse crate
   - Quantization only in quantization crate
   - Backend-specific code only in backend crate

3. **Documentation Review:**
   Ensure each crate's README documents its interface and boundaries

---

## Conclusion

The Coeus framework has well-defined crate interfaces with clear boundaries. The main architectural question is the placement of matrix multiplication operations, which should be resolved by moving MatMul to dense/sparse crates per requirements.

**Next Steps:**
1. Implement boundary enforcement tests (Task 5.4)
2. Decide on MatMul placement
3. Update crate READMEs with interface documentation
4. Verify no circular dependencies
