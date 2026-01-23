# Enhanced Sparse and Dense Tensor Architecture

## Overview

This document specifies the enhanced architecture for sparse and dense tensors with zero-cost abstractions, complete generic support, and deep hierarchical file organization for easy comparison and maintenance.

## Core Design Principles

1. **Zero-Cost Abstractions**: All generic dispatch resolved at compile-time
2. **Universal Backend Support**: Any datatype on any backend (CPU/GPU/TPU/NPU)
3. **Deep Hierarchical Organization**: Mirror file structures for easy comparison
4. **Single Source of Truth**: Each operation implemented exactly once
5. **Domain Separation**: Clear boundaries between crates and responsibilities

## Enhanced Architecture

### 1. Unified Tensor Type with Storage Abstraction

```rust
/// Unified tensor type supporting any storage format
pub struct Tensor<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    storage: S,
    backend: B,
    grad_fn: Option<Arc<dyn Function<B, S, T>>>,
    requires_grad: bool,
    grad: Arc<RwLock<Option<Box<Tensor<B, DenseStorage<T>, T>>>>>,
}

/// Storage trait hierarchy for zero-cost dispatch
pub trait Storage<T: DataType>: Clone + Send + Sync + 'static {
    fn shape(&self) -> &Shape;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
}

/// Dense storage implementation
pub struct DenseStorage<T: DataType> {
    data: Vec<T>,
    shape: Shape,
}

/// Sparse storage implementations
pub struct CsrStorage<T: DataType> {
    data: Vec<T>,
    indices: Vec<usize>,
    indptr: Vec<usize>,
    shape: Shape,
}

pub struct CscStorage<T: DataType> {
    data: Vec<T>,
    indices: Vec<usize>,
    indptr: Vec<usize>,
    shape: Shape,
}

pub struct CooStorage<T: DataType> {
    data: Vec<T>,
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    shape: Shape,
}
```

### 2. Deep Hierarchical File Structure

The enhanced architecture uses deep vertical hierarchies that mirror across domains:

```
coeus/
├── dtype/                          # Foundation: Pure type definitions
│   ├── src/
│   │   ├── float/
│   │   │   ├── f16.rs
│   │   │   ├── f32.rs
│   │   │   ├── f64.rs
│   │   │   └── bfloat16.rs
│   │   ├── int/
│   │   │   ├── i8.rs
│   │   │   ├── i16.rs
│   │   │   ├── i32.rs
│   │   │   └── i64.rs
│   │   ├── complex/
│   │   │   ├── complex32.rs
│   │   │   └── complex64.rs
│   │   └── traits.rs
│   └── Cargo.toml
│
├── backend/                        # Foundation: Hardware execution
│   ├── src/
│   │   ├── cpu/
│   │   │   ├── arithmetic/
│   │   │   │   ├── dense/
│   │   │   │   │   ├── add.rs
│   │   │   │   │   ├── sub.rs
│   │   │   │   │   ├── mul.rs
│   │   │   │   │   └── div.rs
│   │   │   │   └── sparse/
│   │   │   │       ├── csr/
│   │   │   │       │   ├── add.rs
│   │   │   │       │   ├── sub.rs
│   │   │   │       │   ├── mul.rs
│   │   │   │       │   └── div.rs
│   │   │   │       ├── csc/
│   │   │   │       │   ├── add.rs
│   │   │   │       │   ├── sub.rs
│   │   │   │       │   ├── mul.rs
│   │   │   │       │   └── div.rs
│   │   │   │       └── coo/
│   │   │   │           ├── add.rs
│   │   │   │           ├── sub.rs
│   │   │   │           ├── mul.rs
│   │   │   │           └── div.rs
│   │   │   ├── linear_algebra/
│   │   │   │   ├── dense/
│   │   │   │   │   ├── matmul.rs
│   │   │   │   │   ├── transpose.rs
│   │   │   │   │   ├── qr.rs
│   │   │   │   │   ├── lu.rs
│   │   │   │   │   └── cholesky.rs
│   │   │   │   └── sparse/
│   │   │   │       ├── csr/
│   │   │   │       │   ├── matmul.rs
│   │   │   │       │   ├── transpose.rs
│   │   │   │       │   └── spmv.rs
│   │   │   │       ├── csc/
│   │   │   │       │   ├── matmul.rs
│   │   │   │       │   ├── transpose.rs
│   │   │   │       │   └── spmv.rs
│   │   │   │       └── coo/
│   │   │   │           ├── matmul.rs
│   │   │   │           ├── transpose.rs
│   │   │   │           └── to_csr.rs
│   │   │   ├── activation/
│   │   │   │   ├── dense/
│   │   │   │   │   ├── relu.rs
│   │   │   │   │   ├── sigmoid.rs
│   │   │   │   │   ├── tanh.rs
│   │   │   │   │   ├── gelu.rs
│   │   │   │   │   └── softmax.rs
│   │   │   │   └── sparse/
│   │   │   │       ├── csr/
│   │   │   │       │   ├── relu.rs
│   │   │   │       │   ├── sigmoid.rs
│   │   │   │       │   └── tanh.rs
│   │   │   │       ├── csc/
│   │   │   │       │   ├── relu.rs
│   │   │   │       │   ├── sigmoid.rs
│   │   │   │       │   └── tanh.rs
│   │   │   │       └── coo/
│   │   │   │           ├── relu.rs
│   │   │   │           ├── sigmoid.rs
│   │   │   │           └── tanh.rs
│   │   │   └── reduction/
│   │   │       ├── dense/
│   │   │       │   ├── sum.rs
│   │   │       │   ├── mean.rs
│   │   │       │   ├── max.rs
│   │   │       │   └── min.rs
│   │   │       └── sparse/
│   │   │           ├── csr/
│   │   │           │   ├── sum.rs
│   │   │           │   ├── mean.rs
│   │   │           │   ├── max.rs
│   │   │           │   └── min.rs
│   │   │           ├── csc/
│   │   │           │   ├── sum.rs
│   │   │           │   ├── mean.rs
│   │   │           │   ├── max.rs
│   │   │           │   └── min.rs
│   │   │           └── coo/
│   │   │               ├── sum.rs
│   │   │               ├── mean.rs
│   │   │               ├── max.rs
│   │   │               └── min.rs
│   │   ├── gpu/                    # Mirror structure of cpu/
│   │   │   ├── arithmetic/
│   │   │   │   ├── dense/
│   │   │   │   └── sparse/
│   │   │   │       ├── csr/
│   │   │   │       ├── csc/
│   │   │   │       └── coo/
│   │   │   ├── linear_algebra/
│   │   │   ├── activation/
│   │   │   └── reduction/
│   │   ├── tpu/                    # Mirror structure of cpu/
│   │   │   ├── arithmetic/
│   │   │   ├── linear_algebra/
│   │   │   ├── activation/
│   │   │   └── reduction/
│   │   └── npu/                    # Mirror structure of cpu/
│   │       ├── arithmetic/
│   │       ├── linear_algebra/
│   │       ├── activation/
│   │       └── reduction/
│   └── Cargo.toml
│
├── storage/                        # Basic storage operations
│   ├── src/
│   │   ├── dense/
│   │   │   ├── storage.rs
│   │   │   ├── layout.rs
│   │   │   └── indexing.rs
│   │   ├── sparse/
│   │   │   ├── csr/
│   │   │   │   ├── storage.rs
│   │   │   │   ├── layout.rs
│   │   │   │   └── indexing.rs
│   │   │   ├── csc/
│   │   │   │   ├── storage.rs
│   │   │   │   ├── layout.rs
│   │   │   │   └── indexing.rs
│   │   │   └── coo/
│   │   │       ├── storage.rs
│   │   │       ├── layout.rs
│   │   │       └── indexing.rs
│   │   ├── quantized/
│   │   │   ├── q4.rs
│   │   │   ├── q8.rs
│   │   │   └── q16.rs
│   │   └── traits.rs
│   └── Cargo.toml
│
├── dense/                          # Dense-specific algorithms
│   ├── src/
│   │   ├── arithmetic/
│   │   │   ├── cpu/
│   │   │   │   ├── add.rs
│   │   │   │   ├── sub.rs
│   │   │   │   ├── mul.rs
│   │   │   │   └── div.rs
│   │   │   ├── gpu/
│   │   │   │   ├── add.rs
│   │   │   │   ├── sub.rs
│   │   │   │   ├── mul.rs
│   │   │   │   └── div.rs
│   │   │   ├── tpu/
│   │   │   │   ├── add.rs
│   │   │   │   ├── sub.rs
│   │   │   │   ├── mul.rs
│   │   │   │   └── div.rs
│   │   │   └── npu/
│   │   │       ├── add.rs
│   │   │       ├── sub.rs
│   │   │       ├── mul.rs
│   │   │       └── div.rs
│   │   ├── linear_algebra/
│   │   │   ├── cpu/
│   │   │   │   ├── matmul.rs
│   │   │   │   ├── transpose.rs
│   │   │   │   ├── qr.rs
│   │   │   │   ├── lu.rs
│   │   │   │   └── cholesky.rs
│   │   │   ├── gpu/
│   │   │   ├── tpu/
│   │   │   └── npu/
│   │   ├── activation/
│   │   │   ├── cpu/
│   │   │   ├── gpu/
│   │   │   ├── tpu/
│   │   │   └── npu/
│   │   └── reduction/
│   │       ├── cpu/
│   │       ├── gpu/
│   │       ├── tpu/
│   │       └── npu/
│   └── Cargo.toml
│
├── sparse/                         # Sparse-specific algorithms
│   ├── src/
│   │   ├── formats/
│   │   │   ├── csr/
│   │   │   │   ├── arithmetic/
│   │   │   │   │   ├── cpu/
│   │   │   │   │   │   ├── add.rs
│   │   │   │   │   │   ├── sub.rs
│   │   │   │   │   │   ├── mul.rs
│   │   │   │   │   │   └── div.rs
│   │   │   │   │   ├── gpu/
│   │   │   │   │   ├── tpu/
│   │   │   │   │   └── npu/
│   │   │   │   ├── linear_algebra/
│   │   │   │   │   ├── cpu/
│   │   │   │   │   │   ├── matmul.rs
│   │   │   │   │   │   ├── spmv.rs
│   │   │   │   │   │   └── transpose.rs
│   │   │   │   │   ├── gpu/
│   │   │   │   │   ├── tpu/
│   │   │   │   │   └── npu/
│   │   │   │   ├── activation/
│   │   │   │   │   ├── cpu/
│   │   │   │   │   ├── gpu/
│   │   │   │   │   ├── tpu/
│   │   │   │   │   └── npu/
│   │   │   │   └── reduction/
│   │   │   │       ├── cpu/
│   │   │   │       ├── gpu/
│   │   │   │       ├── tpu/
│   │   │   │       └── npu/
│   │   │   ├── csc/                # Mirror structure of csr/
│   │   │   │   ├── arithmetic/
│   │   │   │   ├── linear_algebra/
│   │   │   │   ├── activation/
│   │   │   │   └── reduction/
│   │   │   └── coo/                # Mirror structure of csr/
│   │   │       ├── arithmetic/
│   │   │       ├── linear_algebra/
│   │   │       ├── activation/
│   │   │       └── reduction/
│   │   ├── conversion/
│   │   │   ├── csr_to_csc.rs
│   │   │   ├── csr_to_coo.rs
│   │   │   ├── csc_to_csr.rs
│   │   │   ├── csc_to_coo.rs
│   │   │   ├── coo_to_csr.rs
│   │   │   └── coo_to_csc.rs
│   │   └── algorithms/
│   │       ├── sparsity_analysis.rs
│   │       ├── pattern_optimization.rs
│   │       └── memory_layout.rs
│   └── Cargo.toml
│
├── quantization/                   # Quantization algorithms
│   ├── src/
│   │   ├── algorithms/
│   │   │   ├── symmetric.rs
│   │   │   ├── asymmetric.rs
│   │   │   └── dynamic.rs
│   │   ├── calibration/
│   │   │   ├── entropy.rs
│   │   │   ├── percentile.rs
│   │   │   └── mse.rs
│   │   ├── fake_quantize/
│   │   │   ├── linear.rs
│   │   │   └── conv.rs
│   │   └── types/
│   │       ├── qint4.rs
│   │       ├── qint8.rs
│   │       └── quint8.rs
│   └── Cargo.toml
│
├── tensor/                         # Multi-dimensional tensor operations
│   ├── src/
│   │   ├── ops/
│   │   │   ├── arithmetic/
│   │   │   │   ├── dense/
│   │   │   │   │   ├── add.rs
│   │   │   │   │   ├── sub.rs
│   │   │   │   │   ├── mul.rs
│   │   │   │   │   └── div.rs
│   │   │   │   └── sparse/
│   │   │   │       ├── csr/
│   │   │   │       ├── csc/
│   │   │   │       └── coo/
│   │   │   ├── layout/
│   │   │   │   ├── dense/
│   │   │   │   │   ├── reshape.rs
│   │   │   │   │   ├── transpose.rs
│   │   │   │   │   └── permute.rs
│   │   │   │   └── sparse/
│   │   │   │       ├── csr/
│   │   │   │       ├── csc/
│   │   │   │       └── coo/
│   │   │   ├── indexing/
│   │   │   │   ├── dense/
│   │   │   │   └── sparse/
│   │   │   └── creation/
│   │   │       ├── dense/
│   │   │       └── sparse/
│   │   ├── autograd/
│   │   │   ├── functions/
│   │   │   └── engine.rs
│   │   └── tensor_core.rs
│   └── Cargo.toml
│
└── nn/                             # Neural network operations
    ├── src/
    │   ├── functional/
    │   │   └── ops/
    │   │       ├── activation/
    │   │       │   ├── dense/
    │   │       │   │   ├── relu.rs
    │   │       │   │   ├── sigmoid.rs
    │   │       │   │   ├── tanh.rs
    │   │       │   │   ├── gelu.rs
    │   │       │   │   └── softmax.rs
    │   │       │   └── sparse/
    │   │       │       ├── csr/
    │   │       │       ├── csc/
    │   │       │       └── coo/
    │   │       ├── loss/
    │   │       │   ├── dense/
    │   │       │   │   ├── mse.rs
    │   │       │   │   ├── cross_entropy.rs
    │   │       │   │   └── nll.rs
    │   │       │   └── sparse/
    │   │       ├── convolution/
    │   │       │   ├── dense/
    │   │       │   │   ├── conv1d.rs
    │   │       │   │   ├── conv2d.rs
    │   │       │   │   └── conv3d.rs
    │   │       │   └── sparse/
    │   │       ├── linear/
    │   │       │   ├── dense/
    │   │       │   │   └── linear.rs
    │   │       │   └── sparse/
    │   │       │       ├── csr/
    │   │       │       ├── csc/
    │   │       │       └── coo/
    │   │       ├── normalization/
    │   │       │   ├── dense/
    │   │       │   │   ├── batch_norm.rs
    │   │       │   │   ├── layer_norm.rs
    │   │       │   │   └── group_norm.rs
    │   │       │   └── sparse/
    │   │       ├── pooling/
    │   │       │   ├── dense/
    │   │       │   │   ├── max_pool.rs
    │   │       │   │   ├── avg_pool.rs
    │   │       │   │   └── adaptive_pool.rs
    │   │       │   └── sparse/
    │   │       └── attention/
    │   │           ├── dense/
    │   │           │   ├── self_attention.rs
    │   │           │   ├── multi_head.rs
    │   │           │   └── scaled_dot_product.rs
    │   │           └── sparse/
    │   └── modules/
    │       ├── activation/
    │       ├── loss/
    │       ├── convolution/
    │       ├── linear/
    │       ├── normalization/
    │       ├── pooling/
    │       └── attention/
    └── Cargo.toml
```

### 3. Zero-Cost Generic Dispatch System

```rust
/// Backend trait with associated types for zero-cost dispatch
pub trait Backend: Clone + Send + Sync + 'static {
    type Data: DataType;
    type Device: Clone + Send + Sync;

    fn device(&self) -> &Self::Device;
    fn supports(&self, operation: &str) -> bool;

    // Dense operations
    fn add_dense<S>(&self, lhs: &S, rhs: &S) -> Result<S>
    where
        S: DenseStorage<Self::Data>;

    fn matmul_dense<S>(&self, lhs: &S, rhs: &S) -> Result<S>
    where
        S: DenseStorage<Self::Data>;

    // Sparse operations
    fn add_csr<S>(&self, lhs: &S, rhs: &S) -> Result<S>
    where
        S: CsrStorage<Self::Data>;

    fn matmul_csr<S>(&self, lhs: &S, rhs: &S) -> Result<S>
    where
        S: CsrStorage<Self::Data>;

    // Similar for CSC and COO...
}

/// Storage-specific traits for compile-time dispatch
pub trait DenseStorage<T: DataType>: Storage<T> {
    fn as_slice(&self) -> &[T];
    fn as_mut_slice(&mut self) -> &mut [T];
}

pub trait CsrStorage<T: DataType>: Storage<T> {
    fn data(&self) -> &[T];
    fn indices(&self) -> &[usize];
    fn indptr(&self) -> &[usize];
}

pub trait CscStorage<T: DataType>: Storage<T> {
    fn data(&self) -> &[T];
    fn indices(&self) -> &[usize];
    fn indptr(&self) -> &[usize];
}

pub trait CooStorage<T: DataType>: Storage<T> {
    fn data(&self) -> &[T];
    fn row_indices(&self) -> &[usize];
    fn col_indices(&self) -> &[usize];
}
```

### 4. Operation Dispatch Architecture

```rust
/// Generic operation dispatcher using compile-time resolution
pub trait TensorOps<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    fn add(&self, other: &Self) -> Result<Self>;
    fn matmul(&self, other: &Self) -> Result<Self>;
    fn relu(&self) -> Result<Self>;
    fn sum(&self) -> Result<T>;
}

/// Dense tensor operations
impl<B, T> TensorOps<B, DenseStorage<T>, T> for Tensor<B, DenseStorage<T>, T>
where
    B: Backend<Data = T>,
    T: DataType,
{
    fn add(&self, other: &Self) -> Result<Self> {
        let result_storage = self.backend.add_dense(&self.storage, &other.storage)?;
        Ok(Tensor::from_storage(result_storage, self.backend.clone()))
    }

    fn matmul(&self, other: &Self) -> Result<Self> {
        let result_storage = self.backend.matmul_dense(&self.storage, &other.storage)?;
        Ok(Tensor::from_storage(result_storage, self.backend.clone()))
    }

    // ... other operations
}

/// CSR sparse tensor operations
impl<B, T> TensorOps<B, CsrStorage<T>, T> for Tensor<B, CsrStorage<T>, T>
where
    B: Backend<Data = T>,
    T: DataType,
{
    fn add(&self, other: &Self) -> Result<Self> {
        let result_storage = self.backend.add_csr(&self.storage, &other.storage)?;
        Ok(Tensor::from_storage(result_storage, self.backend.clone()))
    }

    fn matmul(&self, other: &Self) -> Result<Self> {
        let result_storage = self.backend.matmul_csr(&self.storage, &other.storage)?;
        Ok(Tensor::from_storage(result_storage, self.backend.clone()))
    }

    // ... other operations
}

// Similar implementations for CSC and COO...
```

### 5. Parity Tracking System

The deep hierarchical structure enables automatic parity tracking:

```bash
#!/bin/bash
# Script to identify missing implementations

echo "=== Backend Parity Analysis ==="
for backend in cpu gpu tpu npu; do
    echo "Backend: $backend"
    for category in arithmetic linear_algebra activation reduction; do
        echo "  Category: $category"
        for format in dense sparse/csr sparse/csc sparse/coo; do
            echo "    Format: $format"
            cpu_ops=$(find backend/src/cpu/$category/$format -name "*.rs" 2>/dev/null | wc -l)
            backend_ops=$(find backend/src/$backend/$category/$format -name "*.rs" 2>/dev/null | wc -l)
            if [ $cpu_ops -gt $backend_ops ]; then
                echo "      MISSING: $((cpu_ops - backend_ops)) operations"
                # List specific missing files
                comm -23 <(find backend/src/cpu/$category/$format -name "*.rs" | sort) \
                         <(find backend/src/$backend/$category/$format -name "*.rs" | sort) | \
                sed 's/.*\//      - /'
            else
                echo "      COMPLETE: $backend_ops operations"
            fi
        done
    done
done

echo "=== Dense vs Sparse Parity Analysis ==="
for category in arithmetic linear_algebra activation reduction; do
    echo "Category: $category"
    dense_ops=$(find dense/src/$category -name "*.rs" 2>/dev/null | wc -l)
    csr_ops=$(find sparse/src/formats/csr/$category -name "*.rs" 2>/dev/null | wc -l)
    csc_ops=$(find sparse/src/formats/csc/$category -name "*.rs" 2>/dev/null | wc -l)
    coo_ops=$(find sparse/src/formats/coo/$category -name "*.rs" 2>/dev/null | wc -l)
    
    echo "  Dense: $dense_ops operations"
    echo "  CSR: $csr_ops operations"
    echo "  CSC: $csc_ops operations"
    echo "  COO: $coo_ops operations"
    
    if [ $dense_ops -gt $csr_ops ]; then
        echo "  CSR MISSING: $((dense_ops - csr_ops)) operations"
    fi
    if [ $dense_ops -gt $csc_ops ]; then
        echo "  CSC MISSING: $((dense_ops - csc_ops)) operations"
    fi
    if [ $dense_ops -gt $coo_ops ]; then
        echo "  COO MISSING: $((dense_ops - coo_ops)) operations"
    fi
done
```

### 6. Benefits of Enhanced Architecture

1. **Zero-Cost Abstractions**: All dispatch resolved at compile-time
2. **Universal Support**: Any datatype on any backend with any storage format
3. **Easy Comparison**: Parallel file structures enable quick identification of gaps
4. **Maintainability**: Single source of truth with clear domain separation
5. **Extensibility**: New backends/formats/operations easily added
6. **Performance**: No runtime overhead for generic dispatch
7. **Type Safety**: Compile-time guarantees for all operations

### 7. Implementation Strategy

1. **Phase 1**: Create enhanced file structure
2. **Phase 2**: Implement zero-cost generic dispatch
3. **Phase 3**: Migrate existing operations to new structure
4. **Phase 4**: Add parity tracking scripts
5. **Phase 5**: Fill in missing implementations
6. **Phase 6**: Performance optimization and testing

This architecture provides a solid foundation for sparse and dense tensors with complete generic support, zero-cost abstractions, and easy maintenance through hierarchical organization.