# Coeus Architecture Quick Reference

A condensed reference guide for developers working with the Coeus architecture.

## Layer Responsibilities (One-Liner)

| Layer | Responsibility | Delegates To |
|-------|---------------|--------------|
| **Tensor** | User-facing API, operator overloading | Autograd, Dense, Sparse |
| **Autograd** | Gradient tracking, computational graph | Dense, Sparse |
| **Quantization** | Precision management, quantized ops | Dense, Backend |
| **Dense** | Dense tensor operations | Storage, Backend |
| **Sparse** | Sparse tensor operations | Storage, Backend |
| **Storage** | Memory layout, allocation | Backend |
| **Backend** | Device-specific primitives | Dtype, Hardware |
| **Dtype** | Type system, conversions | None (foundation) |

## File Locations

```
tensor/src/ops/          → High-level tensor operations
autograd/src/ops/        → Autograd operations
quantization/src/ops/    → Quantization operations
dense/src/               → Dense operations
sparse/src/              → Sparse operations
storage/src/ops/         → Storage operations
backend/src/{device}/    → Backend primitives (cpu, gpu, tpu, npu)
dtype/src/               → Type definitions
```

## Backend File Structure

```
backend/src/{device}/
├── arithmetic/          → +, -, *, /, pow, sqrt
├── linear_algebra/      → matmul, transpose, dot, svd, qr
├── activation/          → relu, sigmoid, tanh, gelu, softmax
├── reduction/           → sum, mean, max, min, argmax, argmin
├── convolution/         → conv1d, conv2d, conv3d
├── pooling/             → maxpool, avgpool
└── backend.rs           → Backend trait implementation
```

## Common Patterns

### Adding a New Operation

```rust
// 1. Tensor layer (tensor/src/ops/category.rs)
impl<T: DataType> Tensor<T> {
    pub fn new_op(&self) -> Result<Tensor<T>> {
        if self.requires_grad() {
            autograd::ops::new_op(self)
        } else {
            dense::ops::new_op(self.storage())
        }
    }
}

// 2. Autograd layer (autograd/src/ops/category.rs)
pub fn new_op<T: DataType>(input: &Tensor<T>) -> Result<Tensor<T>> {
    let result = dense::ops::new_op(input.data())?;
    let grad_fn = NewOpBackward::new(input.clone());
    Ok(Tensor::from_data_with_grad(result, Some(grad_fn)))
}

// 3. Dense layer (dense/src/category/new_op.rs)
pub fn new_op<T: DataType>(input: &DenseStorage<T>) -> Result<DenseStorage<T>> {
    validate_shape(input.shape())?;
    storage::ops::new_op_dense(input)
}

// 4. Storage layer (storage/src/ops/category.rs)
pub fn new_op_dense<T: DataType>(input: &DenseStorage<T>) -> Result<DenseStorage<T>> {
    let backend = input.device().backend();
    backend.new_op_primitive(input)
}

// 5. Backend layer (backend/src/cpu/category/new_op.rs)
pub fn new_op_primitive<T: DataType>(input: &[T], result: &mut [T]) -> Result<()> {
    // CPU implementation
    for i in 0..input.len() {
        result[i] = compute(input[i]);
    }
    Ok(())
}

// 6. Repeat step 5 for GPU, TPU, NPU
```

### Dispatch Decision Tree

```
Operation called on Tensor
    │
    ├─ requires_grad? ──Yes──> Autograd layer
    │                           │
    │                           └──> Dense/Sparse layer
    │
    └─ No ──> is_sparse? ──Yes──> Sparse layer
                          │
                          └─ No ──> Dense layer
                                    │
                                    └──> Storage layer
                                         │
                                         └──> Backend layer
```

## Status Markers

```rust
// Complete implementation
// STATUS: Complete
// TESTED: Yes
// OPTIMIZED: Yes

// In progress
// STATUS: In Progress
// TESTED: No
// OPTIMIZED: No
// TODO: Add SIMD optimization

// Not implemented
pub fn operation() -> Result<()> {
    unimplemented!("Not yet implemented")
}
```

## Parity Check Commands

```bash
# Check file parity across backends
bash scripts/check_backend_parity.sh

# Generate status dashboard
python3 scripts/status_dashboard.py

# Check API consistency
bash scripts/check_api_consistency.sh

# Run all checks
bash scripts/check_all.sh
```

## Testing Pattern

```rust
// tests/backend/{device}/category/test_operation.rs

#[test]
fn test_operation_basic() {
    let input = vec![1.0, 2.0, 3.0];
    let result = operation(&input).unwrap();
    assert_eq!(result, vec![expected]);
}

#[test]
fn test_operation_edge_cases() {
    // Test zeros, negatives, large values, etc.
}

#[test]
fn test_operation_shapes() {
    // Test different tensor shapes
}
```

## Common Mistakes to Avoid

❌ **Don't skip layers**
```rust
// BAD: Tensor calling Backend directly
impl Tensor {
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        backend::cpu::add(self.data(), other.data())  // ❌ Skips layers
    }
}
```

✅ **Do follow the hierarchy**
```rust
// GOOD: Tensor delegates to appropriate layer
impl Tensor {
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        if self.requires_grad() {
            autograd::ops::add(self, other)  // ✅ Correct delegation
        } else {
            dense::ops::add(self.storage(), other.storage())
        }
    }
}
```

❌ **Don't duplicate logic**
```rust
// BAD: Dense layer doing gradient tracking
pub fn add(lhs: &DenseStorage, rhs: &DenseStorage) -> Result<DenseStorage> {
    if lhs.requires_grad() {  // ❌ This is autograd's job
        // gradient tracking logic
    }
}
```

✅ **Do single responsibility**
```rust
// GOOD: Dense layer only handles dense operations
pub fn add(lhs: &DenseStorage, rhs: &DenseStorage) -> Result<DenseStorage> {
    validate_shapes(lhs.shape(), rhs.shape())?;  // ✅ Dense-specific logic
    storage::ops::add_dense(lhs, rhs)
}
```

❌ **Don't break backend parity**
```rust
// BAD: Different APIs across backends
// backend/src/cpu/arithmetic/add.rs
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> { }  // ❌

// backend/src/gpu/arithmetic/add.rs
pub fn add_gpu(lhs: &[f32], rhs: &[f32], out: &mut [f32]) { }  // ❌ Different API
```

✅ **Do maintain consistent APIs**
```rust
// GOOD: Same API across all backends
// backend/src/cpu/arithmetic/add.rs
pub fn add_primitive<T: DataType>(lhs: &[T], rhs: &[T], result: &mut [T]) -> Result<()> { }

// backend/src/gpu/arithmetic/add.rs
pub fn add_primitive<T: DataType>(lhs: &[T], rhs: &[T], result: &mut [T]) -> Result<()> { }
```

## Quick Debugging

### Operation not working?

1. **Check dispatch path**: Add println! at each layer
2. **Verify shapes**: Ensure shape validation passes
3. **Check backend**: Verify correct backend is called
4. **Test primitive**: Test backend primitive in isolation

### Performance issue?

1. **Profile each layer**: Identify bottleneck layer
2. **Check backend**: Ensure using optimized backend (GPU vs CPU)
3. **Verify no copies**: Check for unnecessary data copies
4. **Check algorithm**: Ensure using optimal algorithm for data structure

### Gradient incorrect?

1. **Check autograd layer**: Verify backward function
2. **Test forward pass**: Ensure forward pass is correct
3. **Test backward pass**: Test backward in isolation
4. **Check accumulation**: Verify gradient accumulation logic

## Useful Grep Patterns

```bash
# Find all implementations of an operation
rg "pub fn operation_name" backend/src/

# Find unimplemented operations
rg "unimplemented!" backend/src/

# Find status markers
rg "STATUS:" backend/src/

# Find TODOs
rg "TODO:" backend/src/

# Find all operations in a category
ls backend/src/cpu/arithmetic/
```

## Documentation Links

- **Full Architecture**: [ARCHITECTURE_INDEX.md](./ARCHITECTURE_INDEX.md)
- **Layer Details**: [LAYER_HIERARCHY.md](./LAYER_HIERARCHY.md)
- **Dispatch Examples**: [DISPATCH_EXAMPLES.md](./DISPATCH_EXAMPLES.md)
- **Parity Tracking**: [PARITY_TRACKING.md](./PARITY_TRACKING.md)
- **Implementation Status**: [IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md)

## Getting Help

1. Check [ARCHITECTURE_INDEX.md](./ARCHITECTURE_INDEX.md) for relevant docs
2. Look at existing implementations in the same category
3. Run parity check scripts to see what's implemented
4. Ask in project discussions with specific layer/operation

---

**Tip**: Bookmark this page for quick reference while coding!
