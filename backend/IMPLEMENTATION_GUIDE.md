# Backend Implementation Guide

## Overview

This guide provides detailed instructions for implementing new backends in the Coeus framework. It covers the Backend trait, implementation patterns, testing strategies, and integration with the adaptive selection system.

## Table of Contents

1. [Backend Trait Overview](#backend-trait-overview)
2. [Implementation Steps](#implementation-steps)
3. [Operation Categories](#operation-categories)
4. [Testing Requirements](#testing-requirements)
5. [Integration with Adaptive Selection](#integration-with-adaptive-selection)
6. [Performance Optimization](#performance-optimization)
7. [Examples](#examples)

## Backend Trait Overview

The `Backend` trait defines the interface all backends must implement:

```rust
pub trait Backend: Send + Sync + Clone {
    /// Data type supported by this backend
    type Data: DataType;
    
    /// Device type for this backend
    type Device: DeviceInfo;
    
    // Device queries
    fn device(&self) -> &Self::Device;
    fn device_name(&self) -> &str;
    fn supports(&self, operation: &str) -> bool;
    
    // Arithmetic operations
    fn add_dense(&self, lhs: &DenseStorage<Self::Data>, rhs: &DenseStorage<Self::Data>) 
        -> Result<DenseStorage<Self::Data>>;
    fn mul_dense(&self, lhs: &DenseStorage<Self::Data>, rhs: &DenseStorage<Self::Data>) 
        -> Result<DenseStorage<Self::Data>>;
    fn sub_dense(&self, lhs: &DenseStorage<Self::Data>, rhs: &DenseStorage<Self::Data>) 
        -> Result<DenseStorage<Self::Data>>;
    
    // Matrix operations
    fn matmul_dense(&self, lhs: &DenseStorage<Self::Data>, rhs: &DenseStorage<Self::Data>) 
        -> Result<DenseStorage<Self::Data>>;
    
    // Activation functions
    fn relu_dense(&self, input: &DenseStorage<Self::Data>) 
        -> Result<DenseStorage<Self::Data>>;
    fn exp_dense(&self, input: &DenseStorage<Self::Data>) 
        -> Result<DenseStorage<Self::Data>>;
    
    // Reduction operations
    fn sum_dense(&self, input: &DenseStorage<Self::Data>) -> Result<Self::Data>;
    fn mean_dense(&self, input: &DenseStorage<Self::Data>) -> Result<Self::Data>;
    
    // ... (see trait definition for complete list)
}
```

## Implementation Steps

### Step 1: Create Backend Struct

```rust
use crate::{Backend, Device, DeviceInfo};
use std::marker::PhantomData;

/// My custom backend implementation
#[derive(Debug, Clone)]
pub struct MyBackend<T: DataType> {
    device: Device,
    // Add backend-specific state here
    _phantom: PhantomData<T>,
}

impl<T: DataType> MyBackend<T> {
    /// Create a new backend instance
    pub fn new() -> Result<Self, BackendError> {
        // Initialize backend-specific resources
        let device = Device::Custom { 
            name: "My Backend".to_string() 
        };
        
        Ok(Self {
            device,
            _phantom: PhantomData,
        })
    }
}
```

### Step 2: Implement Device Queries

```rust
impl<T: DataType> Backend for MyBackend<T> {
    type Data = T;
    type Device = Device;
    
    fn device(&self) -> &Self::Device {
        &self.device
    }
    
    fn device_name(&self) -> &str {
        "my_backend"
    }
    
    fn supports(&self, operation: &str) -> bool {
        // List supported operations
        matches!(operation, 
            "arithmetic" | "matrix_multiplication" | "activation"
        )
    }
}
```

### Step 3: Implement Operations

Start with basic operations and expand:

```rust
impl<T: DataType> Backend for MyBackend<T> {
    // ... (device queries from Step 2)
    
    fn add_dense(
        &self,
        lhs: &DenseStorage<T>,
        rhs: &DenseStorage<T>,
    ) -> Result<DenseStorage<T>> {
        // Validate inputs
        if lhs.len() != rhs.len() {
            return Err(BackendError::ShapeMismatch);
        }
        
        // Perform operation
        let lhs_data = lhs.as_slice();
        let rhs_data = rhs.as_slice();
        let mut result = Vec::with_capacity(lhs_data.len());
        
        for (&a, &b) in lhs_data.iter().zip(rhs_data.iter()) {
            result.push(a + b);
        }
        
        // Create result storage
        DenseStorage::from_vec(result, lhs.shape().dims())
            .map_err(|_| BackendError::StorageError)
    }
    
    // Implement other operations...
}
```

### Step 4: Add Fallback for Unimplemented Operations

For operations not yet implemented, provide CPU fallback:

```rust
fn complex_operation(&self, input: &DenseStorage<T>) -> Result<DenseStorage<T>> {
    // Log fallback
    eprintln!("MyBackend: complex_operation not implemented, falling back to CPU");
    
    // Delegate to CPU backend
    crate::cpu::CpuBackend::new().complex_operation(input)
}
```

## Operation Categories

### 1. Arithmetic Operations

**Required**: `add_dense`, `mul_dense`, `sub_dense`

**Implementation Tips**:
- Element-wise operations
- Validate shape compatibility
- Consider SIMD optimization
- Handle broadcasting (if supported)

**Example**:
```rust
fn mul_dense(&self, lhs: &DenseStorage<T>, rhs: &DenseStorage<T>) 
    -> Result<DenseStorage<T>> 
{
    let lhs_data = lhs.as_slice();
    let rhs_data = rhs.as_slice();
    let result: Vec<T> = lhs_data.iter()
        .zip(rhs_data.iter())
        .map(|(&a, &b)| a * b)
        .collect();
    
    DenseStorage::from_vec(result, lhs.shape().dims())
        .map_err(|_| BackendError::StorageError)
}
```

### 2. Matrix Operations

**Required**: `matmul_dense`, `spmm_csr`, `spmv_csr`

**Implementation Tips**:
- Validate matrix dimensions (m×k) × (k×n) = (m×n)
- Consider blocked algorithms for cache efficiency
- Use specialized libraries (BLAS, cuBLAS, etc.)
- Optimize for specific matrix sizes

**Example**:
```rust
fn matmul_dense(&self, lhs: &DenseStorage<T>, rhs: &DenseStorage<T>) 
    -> Result<DenseStorage<T>> 
{
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    
    // Validate 2D matrices
    if lhs_shape.dims().len() != 2 || rhs_shape.dims().len() != 2 {
        return Err(BackendError::InvalidDimensions);
    }
    
    let (m, k) = (lhs_shape.dims()[0], lhs_shape.dims()[1]);
    let (k2, n) = (rhs_shape.dims()[0], rhs_shape.dims()[1]);
    
    if k != k2 {
        return Err(BackendError::ShapeMismatch);
    }
    
    // Perform matrix multiplication
    let lhs_data = lhs.as_slice();
    let rhs_data = rhs.as_slice();
    let mut result = vec![T::zero(); m * n];
    
    for i in 0..m {
        for j in 0..n {
            for l in 0..k {
                result[i * n + j] = result[i * n + j] 
                    + lhs_data[i * k + l] * rhs_data[l * n + j];
            }
        }
    }
    
    DenseStorage::from_vec(result, &[m, n])
        .map_err(|_| BackendError::StorageError)
}
```

### 3. Activation Functions

**Required**: `relu_dense`, `exp_dense`, `log_dense`, `sin_dense`, `cos_dense`

**Implementation Tips**:
- Element-wise operations
- Handle numerical stability (exp overflow, log of zero)
- Consider fused operations (e.g., gelu = x * Φ(x))
- Optimize for common activation patterns

**Example**:
```rust
fn relu_dense(&self, input: &DenseStorage<T>) -> Result<DenseStorage<T>> 
where T: PartialOrd + Default 
{
    let input_data = input.as_slice();
    let zero = T::zero();
    let result: Vec<T> = input_data.iter()
        .map(|&x| if x > zero { x } else { zero })
        .collect();
    
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|_| BackendError::StorageError)
}
```

### 4. Reduction Operations

**Required**: `sum_dense`, `mean_dense`, `max_dense`, `min_dense`, `argmax_dense`, `argmin_dense`

**Implementation Tips**:
- Handle empty tensors
- Consider numerical stability for sum/mean
- Support axis-specific reductions
- Optimize for common reduction patterns

**Example**:
```rust
fn sum_dense(&self, input: &DenseStorage<T>) -> Result<T> {
    let input_data = input.as_slice();
    let mut sum = T::zero();
    
    for &x in input_data.iter() {
        sum = sum + x;
    }
    
    Ok(sum)
}

fn mean_dense(&self, input: &DenseStorage<T>) -> Result<T> 
where T: From<f64> 
{
    let sum = self.sum_dense(input)?;
    let count = T::from(input.len() as f64);
    Ok(sum / count)
}
```

### 5. Sparse Operations

**Required**: `coo_matmul_sparse`, `coo_add_sparse`, `coo_mul_sparse`

**Implementation Tips**:
- Understand sparse formats (COO, CSR, CSC)
- Optimize for sparsity patterns
- Consider format conversion costs
- Use specialized sparse libraries

**Example**:
```rust
fn coo_matmul_sparse(
    &self,
    lhs: &CooStorage<T>,
    rhs: &CooStorage<T>,
) -> Result<CooStorage<T>> {
    // Group RHS elements by row for efficient lookup
    let mut rhs_by_row: HashMap<usize, Vec<(usize, T)>> = HashMap::new();
    for ((&val, &row), &col) in rhs.data().iter()
        .zip(rhs.row_indices().iter())
        .zip(rhs.col_indices().iter()) 
    {
        rhs_by_row.entry(row).or_default().push((col, val));
    }
    
    // Accumulate results
    let mut result_map: HashMap<(usize, usize), T> = HashMap::new();
    
    for ((&val_a, &i), &p) in lhs.data().iter()
        .zip(lhs.row_indices().iter())
        .zip(lhs.col_indices().iter()) 
    {
        if let Some(rhs_elements) = rhs_by_row.get(&p) {
            for &(j, val_b) in rhs_elements {
                *result_map.entry((i, j)).or_insert(T::zero()) += val_a * val_b;
            }
        }
    }
    
    // Convert to COO format
    let (data, rows, cols): (Vec<_>, Vec<_>, Vec<_>) = result_map.into_iter()
        .filter(|(_, val)| *val != T::zero())
        .map(|((row, col), val)| (val, row, col))
        .multiunzip();
    
    CooStorage::new(data, rows, cols, lhs.shape().dims())
        .map_err(|_| BackendError::StorageError)
}
```

### 6. Quantization Operations

**Optional**: `quantize`, `dequantize`, `quantized_matmul`

**Implementation Tips**:
- Support multiple quantization schemes (symmetric, asymmetric)
- Handle different bit widths (8-bit, 4-bit, etc.)
- Optimize for inference workloads
- Consider hardware-specific quantization

### 7. Convolution Operations

**Optional**: `conv2d_dense`

**Implementation Tips**:
- Support different padding modes (valid, same, full)
- Handle stride and dilation
- Consider im2col transformation
- Use specialized convolution libraries

## Testing Requirements

### Unit Tests

Create comprehensive unit tests for each operation:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_add_dense() {
        let backend = MyBackend::<f32>::new().unwrap();
        
        let a = DenseStorage::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = DenseStorage::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        
        let result = backend.add_dense(&a, &b).unwrap();
        let expected = vec![5.0, 7.0, 9.0];
        
        assert_eq!(result.as_slice(), &expected);
    }
    
    #[test]
    fn test_matmul_dense() {
        let backend = MyBackend::<f32>::new().unwrap();
        
        // 2x3 matrix
        let a = DenseStorage::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
            &[2, 3]
        ).unwrap();
        
        // 3x2 matrix
        let b = DenseStorage::from_vec(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 
            &[3, 2]
        ).unwrap();
        
        let result = backend.matmul_dense(&a, &b).unwrap();
        
        // Expected: 2x2 matrix
        // [1*7+2*9+3*11, 1*8+2*10+3*12]
        // [4*7+5*9+6*11, 4*8+5*10+6*12]
        let expected = vec![58.0, 64.0, 139.0, 154.0];
        
        assert_eq!(result.as_slice(), &expected);
    }
    
    #[test]
    fn test_relu_dense() {
        let backend = MyBackend::<f32>::new().unwrap();
        
        let input = DenseStorage::from_vec(
            vec![-1.0, 0.0, 1.0, -2.0, 2.0], 
            &[5]
        ).unwrap();
        
        let result = backend.relu_dense(&input).unwrap();
        let expected = vec![0.0, 0.0, 1.0, 0.0, 2.0];
        
        assert_eq!(result.as_slice(), &expected);
    }
}
```

### Integration Tests

Test backend integration with tensor operations:

```rust
#[test]
fn test_backend_integration() {
    let backend = MyBackend::<f32>::new().unwrap();
    
    // Create tensors
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], backend.clone()).unwrap();
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3], backend.clone()).unwrap();
    
    // Perform operations
    let c = a + b;
    let d = c * 2.0;
    
    // Verify results
    assert_eq!(d.to_vec(), vec![10.0, 14.0, 18.0]);
}
```

### Performance Tests

Benchmark backend performance:

```rust
#[bench]
fn bench_matmul_1000x1000(b: &mut Bencher) {
    let backend = MyBackend::<f32>::new().unwrap();
    let size = 1000;
    
    let a = DenseStorage::from_vec(vec![1.0; size * size], &[size, size]).unwrap();
    let b = DenseStorage::from_vec(vec![1.0; size * size], &[size, size]).unwrap();
    
    b.iter(|| {
        backend.matmul_dense(&a, &b).unwrap()
    });
}
```

## Integration with Adaptive Selection

### Step 1: Add Backend Type

```rust
// In backend/src/lib.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    Cpu,
    Gpu,
    Tpu,
    Npu,
    MyBackend,  // Add your backend
}
```

### Step 2: Implement Detection

```rust
impl BackendSelector {
    fn detect_available_backends() -> Vec<BackendType> {
        let mut backends = Vec::new();
        
        backends.push(BackendType::Cpu);  // Always available
        
        if Self::detect_gpu_hardware() {
            backends.push(BackendType::Gpu);
        }
        
        if Self::detect_my_backend_hardware() {
            backends.push(BackendType::MyBackend);
        }
        
        backends
    }
    
    fn detect_my_backend_hardware() -> bool {
        // Implement hardware detection
        MyBackend::<f32>::new().is_ok()
    }
}
```

### Step 3: Add Scoring Logic

```rust
impl BackendSelector {
    fn score_backend(&self, backend: BackendType, workload: &WorkloadCharacteristics) -> f32 {
        let mut score = 0.0;
        
        match workload.operation_type {
            OperationType::MatrixMultiplication => {
                score += match backend {
                    BackendType::MyBackend => {
                        // Add scoring logic for your backend
                        if workload.total_elements > 1_000_000 {
                            100.0  // Excellent for large matmul
                        } else {
                            50.0   // Moderate for small matmul
                        }
                    }
                    _ => 0.0,
                };
            }
            // Add scoring for other operation types
            _ => {}
        }
        
        score
    }
}
```

## Performance Optimization

### 1. SIMD Optimization

Use SIMD intrinsics for element-wise operations:

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn add_dense_simd(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(lhs.len());
    
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let chunks = lhs.len() / 8;
        for i in 0..chunks {
            let a = _mm256_loadu_ps(&lhs[i * 8]);
            let b = _mm256_loadu_ps(&rhs[i * 8]);
            let c = _mm256_add_ps(a, b);
            
            let mut temp = [0.0f32; 8];
            _mm256_storeu_ps(temp.as_mut_ptr(), c);
            result.extend_from_slice(&temp);
        }
        
        // Handle remainder
        for i in (chunks * 8)..lhs.len() {
            result.push(lhs[i] + rhs[i]);
        }
    }
    
    result
}
```

### 2. Parallel Processing

Use rayon for parallel operations:

```rust
use rayon::prelude::*;

fn matmul_parallel(&self, lhs: &[f32], rhs: &[f32], m: usize, k: usize, n: usize) 
    -> Vec<f32> 
{
    (0..m).into_par_iter()
        .flat_map(|i| {
            (0..n).map(move |j| {
                (0..k).map(|l| lhs[i * k + l] * rhs[l * n + j])
                    .sum::<f32>()
            })
        })
        .collect()
}
```

### 3. Memory Pooling

Reuse memory allocations:

```rust
pub struct MyBackend<T: DataType> {
    device: Device,
    memory_pool: Arc<Mutex<Vec<Vec<T>>>>,
    _phantom: PhantomData<T>,
}

impl<T: DataType> MyBackend<T> {
    fn allocate(&self, size: usize) -> Vec<T> {
        let mut pool = self.memory_pool.lock().unwrap();
        
        // Try to reuse existing allocation
        if let Some(mut buffer) = pool.pop() {
            buffer.clear();
            buffer.reserve(size);
            buffer
        } else {
            Vec::with_capacity(size)
        }
    }
    
    fn deallocate(&self, buffer: Vec<T>) {
        let mut pool = self.memory_pool.lock().unwrap();
        pool.push(buffer);
    }
}
```

## Examples

### Complete Backend Implementation

See `backend/src/cpu.rs` for a complete reference implementation.

### GPU Backend with Shaders

See `backend/src/gpu.rs` for GPU implementation using wgpu.

### Sparse Operations

See `backend/src/sparse_gpu.rs` for sparse matrix operations.

## Best Practices

1. **Start Simple**: Implement basic operations first, optimize later
2. **Test Thoroughly**: Write comprehensive unit and integration tests
3. **Profile First**: Measure before optimizing
4. **Document Assumptions**: Clearly document input requirements and edge cases
5. **Handle Errors**: Provide clear error messages for invalid inputs
6. **Fallback Gracefully**: Use CPU fallback for unimplemented operations
7. **Follow Conventions**: Match naming and style of existing backends
8. **Benchmark**: Compare performance against CPU and GPU backends

## Common Pitfalls

1. **Shape Validation**: Always validate tensor shapes before operations
2. **Memory Management**: Avoid unnecessary allocations
3. **Numerical Stability**: Handle edge cases (overflow, underflow, NaN)
4. **Thread Safety**: Ensure backend is `Send + Sync`
5. **Error Handling**: Don't panic, return `Result` types
6. **Documentation**: Document all public APIs

## Resources

- [Backend Trait Definition](src/lib.rs)
- [CPU Backend Reference](src/cpu.rs)
- [GPU Backend Reference](src/gpu.rs)
- [Testing Guide](../docs/testing.md)
- [Performance Guide](../docs/performance.md)

## Support

For questions or issues:
- Open an issue on GitHub
- Check existing backend implementations
- Review ADR documents in `docs/adr/`
