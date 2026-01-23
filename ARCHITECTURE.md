# Coeus Architecture

## Design Principles

### 1. Hierarchical Vertical File Tree
All crates follow a **deep vertical hierarchy** where directory structure reveals component relationships without requiring file inspection.

```
crate/
├── src/
│   ├── lib.rs              # Crate root with re-exports
│   ├── arithmetic/         # Domain: arithmetic operations
│   │   ├── mod.rs          # Module with trait + re-exports
│   │   ├── add.rs          # Single Responsibility: addition
│   │   ├── sub.rs          # Single Responsibility: subtraction
│   │   ├── mul.rs          # Single Responsibility: multiplication
│   │   └── div.rs          # Single Responsibility: division
│   ├── linear_algebra/     # Domain: linear algebra
│   │   ├── mod.rs
│   │   └── matmul.rs
│   └── creation/           # Domain: tensor creation
│       ├── mod.rs
│       └── zeros.rs
```

### 2. Single Responsibility Principle (SRP)
- Each file handles **one concern** (e.g., `add.rs` only handles addition)
- Files stay under **500 lines**
- Modules group related files by domain

### 3. Single Source of Truth (SSOT)
- Operations are defined **once** in specialized crates
- Higher-level crates delegate to lower-level implementations
- No duplicate logic between crates

### 4. Separation of Concerns (SOC)
- `storage`: Memory layout and data structures
- `dense`: Dense tensor operations
- `sparse`: Sparse tensor operations
- `backend`: Hardware dispatch (CPU/GPU)
- `tensor`: High-level API composing all above

---

## Dispatch Architecture

### Storage → Operation → Backend Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         TENSOR CRATE                            │
│  Tensor<B: Backend, S: Storage, T: DataType>                   │
│  └── Uses TensorStorageArithmetic trait for dispatch           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              TensorStorageArithmetic<T> TRAIT                   │
│  tensor/src/ops/arithmetic/traits.rs                           │
│  ├── tensor_add<B>(&self, other, backend) → Result<Self>       │
│  ├── tensor_sub<B>(&self, other, backend) → Result<Self>       │
│  ├── tensor_mul<B>(&self, other, backend) → Result<Self>       │
│  └── tensor_div<B>(&self, other, backend) → Result<Self>       │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
┌─────────────────────────┐              ┌─────────────────────────┐
│     DenseStorage<T>     │              │   CsrStorage<T> (etc)   │
│  impl TensorStorageArith │              │  impl TensorStorageArith │
│  → delegates to DENSE    │              │  → delegates to SPARSE   │
└─────────────────────────┘              └─────────────────────────┘
         │                                         │
         ▼                                         ▼
┌─────────────────────────┐              ┌─────────────────────────┐
│      DENSE CRATE        │              │      SPARSE CRATE       │
│  dense/src/arithmetic/  │              │  sparse/src/arithmetic/ │
│  ├── add.rs             │              │  ├── add.rs             │
│  ├── sub.rs             │              │  ├── sub.rs             │
│  └── mul.rs             │              │  └── mul.rs             │
│                         │              │                         │
│  Calls backend.add_dense│              │  Calls SparseAdd trait  │
└─────────────────────────┘              └─────────────────────────┘
         │                                         │
         └────────────────────┬────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       BACKEND CRATE                             │
│  backend/src/                                                   │
│  ├── lib.rs       # Backend<T> trait                           │
│  ├── cpu/         # CpuBackend implementation                  │
│  │   ├── backend.rs                                            │
│  │   ├── arithmetic/add.rs                                     │
│  │   └── linear_algebra/matmul.rs                              │
│  └── gpu/         # GpuBackend implementation                  │
│      ├── backend.rs                                            │
│      └── shaders/                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Tree Parity

Related crates maintain **parallel structure** for consistency:

### `dense` and `sparse` Crate Parity
```
dense/src/                    sparse/src/
├── lib.rs                    ├── lib.rs
├── arithmetic/               ├── arithmetic/
│   ├── mod.rs                │   ├── mod.rs
│   ├── add.rs                │   ├── add.rs
│   ├── sub.rs                │   ├── sub.rs
│   ├── mul.rs                │   ├── mul.rs
│   └── div.rs                │   └── div.rs
├── creation/                 ├── creation/
│   └── zeros.rs              │   └── zeros.rs
└── layout/                   └── conversion/
    └── transpose.rs              └── format.rs
```

### `backend` CPU/GPU Parity
```
backend/src/cpu/              backend/src/gpu/
├── backend.rs                ├── backend.rs
├── arithmetic/               ├── shaders/
│   ├── add.rs                │   ├── element_wise.wgsl
│   └── mul.rs                │   └── matmul.wgsl
└── linear_algebra/           └── (shader dispatch)
    └── matmul.rs
```

---

## Generic Backend Dispatch

### Backend Trait
```rust
// backend/src/lib.rs
pub trait Backend {
    type Data: DataType;
    type Device;
    
    // Dense operations
    fn add_dense(&self, lhs: &DenseStorage<Self::Data>, rhs: &DenseStorage<Self::Data>) 
        -> Result<DenseStorage<Self::Data>>;
    fn sub_dense(&self, ...) -> Result<DenseStorage<Self::Data>>;
    fn mul_dense(&self, ...) -> Result<DenseStorage<Self::Data>>;
    fn matmul_dense(&self, ...) -> Result<DenseStorage<Self::Data>>;
    
    // Sparse operations
    fn spmv_csr(&self, ...) -> Result<Vec<Self::Data>>;
    fn spmm_csr(&self, ...) -> Result<Vec<Self::Data>>;
}
```

### Generic Usage Pattern
```rust
// Caller passes backend instance, not hardcoded type
fn add<T: DataType, B: Backend<Data = T>>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
    backend: &B,  // Generic backend
) -> Result<DenseStorage<T>> {
    backend.add_dense(lhs, rhs)
}
```

### CPU Implementation
```rust
impl<T: DataType> Backend for CpuBackend<T> {
    type Data = T;
    type Device = CpuDevice;
    
    fn add_dense(&self, lhs: &DenseStorage<T>, rhs: &DenseStorage<T>) -> Result<DenseStorage<T>> {
        // CPU-optimized implementation
        let result: Vec<T> = lhs.iter().zip(rhs.iter()).map(|(a, b)| *a + *b).collect();
        DenseStorage::from_vec(result, lhs.shape().dims())
    }
}
```

### GPU Implementation
```rust
impl<T: DataType> Backend for GpuBackend<T> {
    type Data = T;
    type Device = GpuDevice;
    
    fn add_dense(&self, lhs: &DenseStorage<T>, rhs: &DenseStorage<T>) -> Result<DenseStorage<T>> {
        // GPU shader dispatch
        self.dispatch_shader("element_wise_add", lhs, rhs)
    }
}
```

---

## Adding New Operations

1. **Add to `backend` trait** in `backend/src/lib.rs`
2. **Implement in CPU backend** in `backend/src/cpu/`
3. **Implement in GPU backend** in `backend/src/gpu/`
4. **Add wrapper in `dense`/`sparse`** crates
5. **Update `TensorStorageArithmetic`** if needed
6. **Expose via `tensor`** crate

---

## References

- [Backend README](backend/README.md) - Detailed backend documentation
- [ARCHITECTURAL_ENHANCEMENT_PLAN.md](ARCHITECTURAL_ENHANCEMENT_PLAN.md) - Roadmap
- `tensor/src/ops/arithmetic/traits.rs` - TensorStorageArithmetic trait
- `tensor/src/ops/arithmetic/dispatch.rs` - Storage-to-crate dispatch
