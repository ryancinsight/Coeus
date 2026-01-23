# Coeus Core Crates Implementation Plan

## Overview

This plan addresses the critical gaps identified in the audit of backend, storage, dtype, quantization, and tensor crates. The focus is on ensuring complete implementations with proper CPU/GPU dispatch while maintaining clean domain separation.

## Phase 1: Critical Fixes (Immediate - 1-2 days)

### 1.1 Fix Compilation Errors

#### Fix FFT Crate GPU Import
```rust
// Current (broken):
use backend::gpu::GpuBackend;

// Fix: Remove or conditionally import
#[cfg(feature = "gpu")]
use backend::gpu::GpuBackend;
```

#### Fix GPU Buffer Type Annotations
```rust
// Add explicit type annotation for closure parameter
buffer_slice.map_async(wgpu::MapMode::Read, move |res: Result<(), wgpu::BufferAsyncError>| {
    // ... implementation
});
```

### 1.2 Replace Placeholder Implementations

#### Backend CPU Placeholders
Replace these placeholder implementations in `backend/src/cpu/backend.rs`:

```rust
// Current placeholders that return input unchanged:
fn exp_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>> {
    // TODO: Implement actual exponential
    Ok(input.clone()) // WRONG - should compute exp
}

// Fix: Implement actual operations
fn exp_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>> {
    let input_slice = input.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    
    for (i, &val) in input_slice.iter().enumerate() {
        result[i] = val.exp(); // Use DataType::exp() method
    }
    
    DenseStorage::from_vec(result, input.shape().dims())
        .map_err(|e| BackendError::InvalidInput(format!("Storage error: {}", e)))
}
```

### 1.3 Clean Up Warnings

#### Remove Unused Imports
```rust
// storage/src/lib.rs - Remove unused imports
// use dtype::traits::FloatExt;  // Remove
// use num_traits::Zero;         // Remove

// backend/src/cpu/arithmetic/add.rs - Remove unused imports  
// use std::vec::Vec;            // Remove
```

#### Fix Unused Variables
```rust
// backend/src/cpu/backend.rs - Prefix with underscore or use
fn quantize_dense(
    &self,
    input: &DenseStorage<Self::Data>,
    _weight: &DenseStorage<Self::Data>,  // Prefix with _
    // ... other parameters
) -> Result<DenseStorage<Self::Data>> {
    // Implementation
}
```

## Phase 2: Quantization Integration (High Priority - 2-3 days)

### 2.1 Create Quantization Backend Trait

Create `backend/src/quantization.rs`:

```rust
use crate::{Backend, Result};
use dtype::DataType;
use quantization::{QuantizationBitwidth, QuantizationScheme, CalibrationStats};
use storage::DenseStorage;

/// Quantization operations for backends
pub trait QuantizationBackend<T: DataType>: Backend<Data = T> {
    /// Quantize tensor to specified bitwidth
    fn quantize(
        &self,
        input: &DenseStorage<T>,
        bitwidth: QuantizationBitwidth,
        scheme: QuantizationScheme,
        calibration_stats: Option<&CalibrationStats<T>>,
    ) -> Result<storage::QuantizedStorage<T>>;
    
    /// Dequantize tensor back to full precision
    fn dequantize(
        &self,
        input: &storage::QuantizedStorage<T>,
    ) -> Result<DenseStorage<T>>;
    
    /// Fake quantize for training (quantize then dequantize)
    fn fake_quantize(
        &self,
        input: &DenseStorage<T>,
        bitwidth: QuantizationBitwidth,
        scheme: QuantizationScheme,
    ) -> Result<DenseStorage<T>>;
}
```

### 2.2 Implement Quantization in CPU Backend

Add to `backend/src/cpu/backend.rs`:

```rust
impl<T: DataType> QuantizationBackend<T> for CpuBackend<T>
where
    T: num_traits::Float + num_traits::FromPrimitive + std::cmp::PartialOrd,
{
    fn quantize(
        &self,
        input: &DenseStorage<T>,
        bitwidth: QuantizationBitwidth,
        scheme: QuantizationScheme,
        calibration_stats: Option<&CalibrationStats<T>>,
    ) -> Result<storage::QuantizedStorage<T>> {
        use quantization::algorithms::{SymmetricQuantizer, AsymmetricQuantizer};
        
        match scheme {
            QuantizationScheme::Symmetric => {
                let quantizer = SymmetricQuantizer::new(bitwidth);
                quantizer.quantize_tensor(input, calibration_stats)
                    .map_err(|e| BackendError::QuantizationError(e.to_string()))
            }
            QuantizationScheme::Affine => {
                let quantizer = AsymmetricQuantizer::new(bitwidth);
                quantizer.quantize_tensor(input, calibration_stats)
                    .map_err(|e| BackendError::QuantizationError(e.to_string()))
            }
        }
    }
    
    // ... implement other methods
}
```

### 2.3 Add Quantization to Tensor Operations

Add to `tensor/src/ops/quantization.rs`:

```rust
use crate::{Backend, DataType, Storage, Tensor};
use backend::QuantizationBackend;
use quantization::{QuantizationBitwidth, QuantizationScheme};
use storage::{DenseStorage, StorageToDense};

/// Quantize tensor to specified bitwidth
pub fn quantize<B, S, T>(
    tensor: &Tensor<B, S, T>,
    bitwidth: QuantizationBitwidth,
    scheme: QuantizationScheme,
) -> crate::Result<Tensor<B, storage::QuantizedStorage<T>, T>>
where
    B: Backend<Data = T> + QuantizationBackend<T> + Clone,
    S: Storage<T> + StorageToDense<T>,
    T: DataType + num_traits::Float + num_traits::FromPrimitive + std::cmp::PartialOrd,
{
    let dense_storage = tensor.storage.to_dense()?;
    let quantized_storage = tensor.backend.quantize(&dense_storage, bitwidth, scheme, None)?;
    Ok(Tensor::from_storage(quantized_storage, tensor.backend.clone()))
}
```

## Phase 3: Backend Dispatch Completion (Medium Priority - 3-5 days)

### 3.1 Decision: GPU Backend Strategy

**Option A: Complete GPU Backend**
- Implement actual WGSL compute shaders
- Add GPU memory management
- Significant development effort (2-3 weeks)

**Option B: Remove GPU Backend Stub (Recommended)**
- Remove incomplete GPU backend
- Focus on CPU optimization
- Add GPU backend in future release

**Recommendation**: Choose Option B for immediate stability

### 3.2 Remove GPU Backend Stub

```rust
// backend/src/lib.rs - Remove GPU backend exports
// pub mod gpu;  // Remove this line
// pub use gpu::*;  // Remove this line

// Update Backend enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    Cpu,
    // Gpu,  // Remove until implemented
    // Tpu,  // Remove until implemented  
    // Npu,  // Remove until implemented
}
```

### 3.3 Complete CPU Backend Operations

Implement missing operations in CPU backend:

```rust
// backend/src/cpu/backend.rs
impl<T: DataType> Backend for CpuBackend<T> {
    // Implement actual exp operation
    fn exp_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        let input_slice = input.as_slice();
        let mut result = vec![T::default(); input_slice.len()];
        
        for (i, &val) in input_slice.iter().enumerate() {
            result[i] = val.exp();
        }
        
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| BackendError::InvalidInput(format!("Storage error: {}", e)))
    }
    
    // Similarly implement log, sin, cos operations
    fn log_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>> {
        // Implementation similar to exp_dense
    }
    
    // Add proper conv2d implementation or remove
    fn conv2d_dense(
        &self,
        input: &DenseStorage<Self::Data>,
        weight: &DenseStorage<Self::Data>,
        bias: Option<&DenseStorage<Self::Data>>,
        stride: &[usize],
        padding: &[usize],
    ) -> Result<DenseStorage<Self::Data>> {
        // Either implement proper convolution or return UnsupportedOperation
        Err(BackendError::UnsupportedOperation {
            operation: "conv2d_dense".to_string(),
            backend: "cpu".to_string(),
        })
    }
}
```

## Phase 4: Sparse Operations Completion (Medium Priority - 2-3 days)

### 4.1 Complete Sparse-Sparse Operations

Implement missing sparse operations in `backend/src/cpu/sparse_kernels.rs`:

```rust
// Complete COO sparse operations
pub fn coo_add_sparse<T: DataType>(
    lhs_data: &[T],
    lhs_row: &[usize],
    lhs_col: &[usize],
    rhs_data: &[T],
    rhs_row: &[usize], 
    rhs_col: &[usize],
    rows: usize,
    cols: usize,
) -> crate::Result<(Vec<T>, Vec<usize>, Vec<usize>)>
where
    T: std::ops::Add<Output = T> + Copy + Default,
{
    // Implement actual COO addition algorithm
    // 1. Merge coordinate lists
    // 2. Sum values at same coordinates
    // 3. Return result in COO format
    
    // Placeholder implementation - replace with actual algorithm
    let mut result_data = Vec::new();
    let mut result_row = Vec::new();
    let mut result_col = Vec::new();
    
    // TODO: Implement proper COO sparse addition
    // For now, return empty result to avoid compilation errors
    Ok((result_data, result_row, result_col))
}
```

## Phase 5: Tensor Operations Completion (Lower Priority - 3-4 days)

### 5.1 Complete Batch Operations

Add missing batch operations to tensor crate:

```rust
// tensor/src/ops/batch.rs
use crate::{Backend, DataType, Storage, Tensor};

/// Batch matrix multiplication
pub fn batch_matmul<B, S, T>(
    lhs: &Tensor<B, S, T>,
    rhs: &Tensor<B, S, T>,
) -> crate::Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone,
    T: DataType,
{
    // Implement batch matrix multiplication
    // Handle batch dimensions properly
    todo!("Implement batch matrix multiplication")
}

/// Batch normalization
pub fn batch_norm<B, S, T>(
    input: &Tensor<B, S, T>,
    weight: Option<&Tensor<B, S, T>>,
    bias: Option<&Tensor<B, S, T>>,
    running_mean: Option<&Tensor<B, S, T>>,
    running_var: Option<&Tensor<B, S, T>>,
    training: bool,
    momentum: T,
    eps: T,
) -> crate::Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone,
    T: DataType + num_traits::Float,
{
    // Implement batch normalization
    todo!("Implement batch normalization")
}
```

### 5.2 Complete Activation Functions

Replace placeholder activations with proper implementations:

```rust
// tensor/src/ops/activation.rs
pub fn gelu<B, S, T>(input: &Tensor<B, S, T>) -> crate::Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + Clone,
    T: DataType + num_traits::Float,
{
    // Implement actual GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    let dense_storage = input.storage.to_dense()?;
    let input_slice = dense_storage.as_slice();
    let mut result = vec![T::default(); input_slice.len()];
    
    for (i, &x) in input_slice.iter().enumerate() {
        let x_cubed = x * x * x;
        let inner = (T::from(2.0 / std::f64::consts::PI).unwrap().sqrt() * 
                    (x + T::from(0.044715).unwrap() * x_cubed)).tanh();
        result[i] = x * T::from(0.5).unwrap() * (T::one() + inner);
    }
    
    let result_storage = DenseStorage::from_vec(result, dense_storage.shape().dims())?;
    Ok(Tensor::from_storage(result_storage, input.backend.clone()))
}
```

## Phase 6: Testing and Validation (Ongoing)

### 6.1 Add Comprehensive Tests

Create test suites for each phase:

```rust
// tests/integration/quantization_tests.rs
#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use quantization::QuantizationBitwidth;
    use storage::DenseStorage;
    use tensor::Tensor;

    #[test]
    fn test_quantization_integration() {
        let backend = CpuBackend::<Float32>::new();
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let storage = DenseStorage::from_vec(data, &[3]).unwrap();
        let tensor = Tensor::from_storage(storage, backend);
        
        let quantized = tensor::ops::quantize(
            &tensor,
            QuantizationBitwidth::Bits8,
            quantization::QuantizationScheme::Symmetric,
        ).unwrap();
        
        assert_eq!(quantized.shape(), &[3]);
        // Add more assertions
    }
}
```

### 6.2 Performance Benchmarks

Add benchmarks to validate performance:

```rust
// benches/backend_dispatch.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_cpu_operations(c: &mut Criterion) {
    let backend = CpuBackend::<Float32>::new();
    let data = vec![Float32::new(1.0); 1000];
    let storage = DenseStorage::from_vec(data, &[1000]).unwrap();
    
    c.bench_function("cpu_add", |b| {
        b.iter(|| {
            backend.add_dense(black_box(&storage), black_box(&storage))
        })
    });
}

criterion_group!(benches, benchmark_cpu_operations);
criterion_main!(benches);
```

## Implementation Timeline

| Phase | Duration | Priority | Dependencies |
|-------|----------|----------|--------------|
| Phase 1: Critical Fixes | 1-2 days | CRITICAL | None |
| Phase 2: Quantization Integration | 2-3 days | HIGH | Phase 1 |
| Phase 3: Backend Dispatch | 3-5 days | MEDIUM | Phase 1 |
| Phase 4: Sparse Operations | 2-3 days | MEDIUM | Phase 3 |
| Phase 5: Tensor Operations | 3-4 days | LOW | Phase 2, 3 |
| Phase 6: Testing | Ongoing | HIGH | All phases |

**Total Estimated Time**: 11-17 days

## Success Criteria

1. **Compilation**: All crates compile without errors or warnings
2. **Quantization**: Quantization algorithms integrated and usable in tensor operations
3. **Backend Dispatch**: CPU backend complete, incomplete backends removed or documented
4. **Domain Separation**: Clean boundaries maintained between crates
5. **Testing**: Comprehensive test coverage for all implemented features
6. **Performance**: No regression in CPU backend performance

## Risk Mitigation

1. **Scope Creep**: Focus on critical fixes first, defer nice-to-have features
2. **Breaking Changes**: Maintain backward compatibility where possible
3. **Performance**: Benchmark before and after changes
4. **Testing**: Add tests for each fix to prevent regressions

This implementation plan provides a clear roadmap to address the critical gaps while maintaining the architectural integrity of the Coeus framework.