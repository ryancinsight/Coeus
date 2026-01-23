# Layer Hierarchy

## 1. Tensor Layer (Highest Level)

**Location**: `tensor/src/`

**Responsibility**: High-level tensor API with ergonomic operations

**Key Characteristics**:
- User-facing API
- Operator overloading (+, -, *, /)
- Method chaining (`.relu()`, `.matmul()`, etc.)
- Automatic shape inference
- Broadcasting support

**Delegates To**: Autograd, Quantization, Dense, Sparse

**Example**:
```rust
// tensor/src/ops/arithmetic.rs
impl<T: DataType> Tensor<T> {
    pub fn add(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        // Dispatch based on tensor properties
        match (self.requires_grad(), other.requires_grad()) {
            (true, _) | (_, true) => {
                // Delegate to autograd layer
                autograd::ops::add(self, other)
            }
            _ => {
                // Delegate to dense/sparse layer
                match (self.is_sparse(), other.is_sparse()) {
                    (true, true) => sparse::ops::add(self.storage(), other.storage()),
                    _ => dense::ops::add(self.storage(), other.storage()),
                }
            }
        }
    }
}
```

## 2. Autograd Layer

**Location**: `autograd/src/`

**Responsibility**: Automatic differentiation and gradient tracking

**Key Characteristics**:
- Computational graph construction
- Backward pass implementation
- Gradient accumulation
- No direct backend calls

**Delegates To**: Dense, Sparse (for forward pass)

**Example**:
```rust
// autograd/src/ops/arithmetic.rs
pub fn add<T: DataType>(lhs: &Tensor<T>, rhs: &Tensor<T>) -> Result<Tensor<T>> {
    // Forward pass: delegate to dense/sparse
    let result = dense::ops::add(lhs.data(), rhs.data())?;
    
    // Build computational graph
    let grad_fn = AddBackward::new(lhs.clone(), rhs.clone());
    
    Ok(Tensor::from_data_with_grad(result, Some(grad_fn)))
}
```

## 3. Quantization Layer

**Location**: `quantization/src/`

**Responsibility**: Precision management and quantized operations

**Key Characteristics**:
- Quantization schemes (int8, int4, etc.)
- Dequantization when needed
- Quantized arithmetic operations
- No direct backend calls for non-quantized ops

**Delegates To**: Dense, Backend (for quantized ops)

## 4. Dense Layer

**Location**: `dense/src/`

**Responsibility**: Dense tensor operations

**Key Characteristics**:
- Dense-specific algorithms
- Shape validation
- Broadcasting logic
- Delegates ALL computation to storage/backend

**Delegates To**: Storage, Backend

**Example**:
```rust
// dense/src/arithmetic/add.rs
pub fn add<T: DataType>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
) -> Result<DenseStorage<T>> {
    // Validate shapes
    validate_broadcast_shapes(lhs.shape(), rhs.shape())?;
    
    // Delegate to storage layer for memory operations
    storage::ops::add_dense(lhs, rhs)
}
```
