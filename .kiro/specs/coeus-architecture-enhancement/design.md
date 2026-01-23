# Design Document

## Overview

This design document specifies the architectural enhancement of the Coeus deep learning framework and PyCoeus Python bindings. The enhancement establishes a clean separation between stateless operations and stateful layers, eliminates code duplication through single source of truth principles, optimizes the file structure hierarchy, and systematically addresses PyTorch API parity gaps.

The design maintains Coeus's core principles:
- **Zero-cost abstractions** through compile-time monomorphization
- **Memory safety** guaranteed by Rust's ownership system
- **B<S<T>> generic architecture** supporting any backend, storage, and datatype combination
- **PyTorch API compatibility** for seamless migration

### Comprehensive Architecture Documentation

A complete set of architecture documentation has been created in the `docs/` directory to support this design:

- **[docs/ARCHITECTURE_INDEX.md](../../../docs/ARCHITECTURE_INDEX.md)** - Complete navigation guide and documentation index
- **[docs/QUICK_REFERENCE.md](../../../docs/QUICK_REFERENCE.md)** - Quick reference for developers
- **[docs/LAYER_HIERARCHY.md](../../../docs/LAYER_HIERARCHY.md)** - Detailed explanation of the 8-layer architecture
- **[docs/DISPATCH_EXAMPLES.md](../../../docs/DISPATCH_EXAMPLES.md)** - Concrete examples of operation dispatch flow
- **[docs/PARITY_TRACKING.md](../../../docs/PARITY_TRACKING.md)** - Backend parity tracking methodology
- **[docs/IMPLEMENTATION_STATUS.md](../../../docs/IMPLEMENTATION_STATUS.md)** - Implementation status tracking

These documents provide:
1. **Layer-by-layer architecture explanation** showing how operations flow from Tensor → Autograd → Quantization → Dense/Sparse → Storage → Backend → Dtype
2. **Concrete dispatch examples** demonstrating how operations like addition, matrix multiplication, and neural network forward passes flow through the architecture
3. **Backend parity tracking methodology** using hierarchical file structures to identify missing implementations across CPU, GPU, TPU, and NPU backends
4. **Implementation status tracking** with scripts and tools to monitor progress
5. **Quick reference guides** for developers working on the codebase

**For implementation details, refer to the comprehensive documentation in the `docs/` directory.**

## Architecture

## Architecture

### Corrected High-Level Architecture

The enhanced architecture follows a clear dependency hierarchy with proper domain separation:

```
┌─────────────────────────────────────────────────────────────┐
│                     Python API (PyCoeus)                     │
│  - PyTorch-compatible interface                              │
│  - Generic optimizer wrappers                                │
│  - Custom exception hierarchy                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Neural Network (nn/)                      │
│  - Stateful layers (modules/)                               │
│  - Stateless operations (functional/ops/)                   │
│  - Depends on: tensor, dense, sparse, quantization          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Tensor (tensor/)                        │
│  - Multi-dimensional tensor operations                       │
│  - Automatic differentiation support                         │
│  - Depends on: dense, sparse, quantization, storage         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│     Dense (dense/)    │    Sparse (sparse/)    │ Quantization │
│  - Dense algorithms   │  - Sparse algorithms   │ (quantization/) │
│  - Dense operations   │  - CSR/CSC/COO ops    │ - Quant algorithms │
│  - Depends on:        │  - Depends on:        │ - Depends on:      │
│    storage, dtype     │    storage, dtype     │   storage, dtype   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Storage (storage/)                       │
│  - Basic arithmetic operations (add, sub, mul, div)          │
│  - Memory layout management                                  │
│  - Depends on: backend, dtype                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Backend (backend/)                      │
│  - Hardware execution primitives (foundation)               │
│  - CPU, GPU, TPU, NPU implementations                       │
│  - Depends on: dtype                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Types (dtype/)                     │
│  - Pure type definitions and conversions                     │
│  - No dependencies (foundation types)                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Architectural Principles

1. **Single Source of Truth**: Each operation is implemented exactly once in `nn/src/functional/ops/`, with layers delegating to these implementations

2. **Proper Domain Separation**: 
   - `backend/` provides hardware execution primitives (foundation layer)
   - `dtype/` provides pure type definitions and conversions (no quantization)
   - `storage/` provides basic arithmetic operations (add, sub, mul, div) and memory layout
   - `sparse/` provides sparse-specific operations and formats
   - `quantization/` (new crate) provides quantization logic extracted from nn and dtype
   - `tensor/` provides dense tensor operations using storage
   - `nn/functional/ops/` provides neural network operations using storage primitives
   - `nn/modules/` provides stateful wrappers with parameters

3. **Hierarchical File Organization**: 
   - Deep vertical hierarchies mirror implementation domains
   - Parallel file structures across backends enable parity comparison
   - Script-based identification of missing implementations
   - No monolithic files - operations split by category and type

4. **Foundation-Up Architecture**: 
   - Backend is the foundation providing hardware primitives
   - Storage builds on backend for basic operations
   - Higher layers build on storage primitives
   - Clear dependency hierarchy prevents circular dependencies

5. **Generic Architecture**: All components maintain `<B, S, T>` generics for compile-time specialization

6. **Trait-Based Abstraction**: Storage operations unified through trait hierarchy enabling extensibility

## Components and Interfaces

### 1. Hierarchical File Structure for Parity Tracking

The framework uses deep vertical hierarchies organized by domain and implementation type:

```
backend/
├── src/
│   ├── cpu/
│   │   ├── arithmetic/
│   │   │   ├── add.rs
│   │   │   ├── sub.rs
│   │   │   ├── mul.rs
│   │   │   └── div.rs
│   │   ├── linear_algebra/
│   │   │   ├── matmul.rs
│   │   │   ├── transpose.rs
│   │   │   └── decomposition.rs
│   │   ├── activation/
│   │   │   ├── relu.rs
│   │   │   ├── sigmoid.rs
│   │   │   └── tanh.rs
│   │   └── reduction/
│   │       ├── sum.rs
│   │       ├── mean.rs
│   │       └── max.rs
│   ├── gpu/
│   │   └── [parallel structure to cpu/]
│   ├── tpu/
│   │   └── [parallel structure to cpu/]
│   └── npu/
       └── [parallel structure to cpu/]

storage/
├── src/
│   ├── dense/
│   │   ├── arithmetic/
│   │   │   ├── add.rs      # Basic element-wise addition
│   │   │   ├── sub.rs      # Basic element-wise subtraction
│   │   │   ├── mul.rs      # Basic element-wise multiplication
│   │   │   └── div.rs      # Basic element-wise division
│   │   ├── layout/
│   │   │   ├── reshape.rs
│   │   │   ├── transpose.rs
│   │   │   └── stride.rs
│   │   └── creation/
│   │       ├── zeros.rs
│   │       ├── ones.rs
│   │       └── from_vec.rs
│   ├── quantized/         # Basic quantized storage (moved from dtype)
│   │   ├── arithmetic/
│   │   └── layout/
│   └── strided/
│       ├── arithmetic/
│       └── layout/

sparse/
├── src/
│   ├── formats/
│   │   ├── csr/
│   │   │   ├── arithmetic/
│   │   │   │   ├── add.rs
│   │   │   │   ├── mul.rs
│   │   │   │   └── matmul.rs
│   │   │   ├── conversion/
│   │   │   └── indexing/
│   │   ├── csc/
│   │   │   └── [parallel structure]
│   │   └── coo/
│   │       └── [parallel structure]
│   └── ops/
│       ├── linear_algebra/
│       ├── reduction/
│       └── conversion/

quantization/              # NEW CRATE - extracted from nn and dtype
├── src/
│   ├── algorithms/
│   │   ├── symmetric.rs
│   │   ├── asymmetric.rs
│   │   └── dynamic.rs
│   ├── calibration/
│   │   ├── entropy.rs
│   │   ├── percentile.rs
│   │   └── mse.rs
│   ├── fake_quantize/
│   │   ├── linear.rs
│   │   └── conv.rs
│   └── kernels/
│       ├── quantize.rs
│       ├── dequantize.rs
│       └── quantized_ops.rs

nn/
├── src/
│   ├── functional/
│   │   └── ops/
│   │       ├── activation/
│   │       │   ├── relu.rs
│   │       │   ├── gelu.rs
│   │       │   ├── softmax.rs
│   │       │   └── mod.rs
│   │       ├── loss/
│   │       │   ├── mse.rs
│   │       │   ├── cross_entropy.rs
│   │       │   └── mod.rs
│   │       ├── convolution/
│   │       │   ├── conv1d.rs
│   │       │   ├── conv2d.rs
│   │       │   ├── conv3d.rs
│   │       │   └── mod.rs
│   │       ├── linear/
│   │       │   ├── dense.rs
│   │       │   ├── sparse.rs
│   │       │   └── mod.rs
│   │       └── [other operation categories]
│   └── modules/
│       ├── activation/
│       │   ├── relu.rs
│       │   ├── gelu.rs
│       │   └── mod.rs
│       ├── loss/
│       │   └── [parallel structure]
│       └── [other module categories]
```

This structure enables:
- **Script-based parity comparison**: Compare file presence across backends
- **Missing implementation identification**: Absent files indicate missing functionality
- **Domain isolation**: Each crate maintains its own hierarchy
- **Consistent organization**: Parallel structures across all domains
- **No monolithic files**: Operations split by category and specific function

### 2. NN Operations Module (`nn/src/functional/ops/`)

The operations module provides stateless pure functions organized by category:

#### functional/ops/activation.rs
```rust
/// ReLU activation function
pub fn relu<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + PartialOrd + Zero,
{
    // Implementation: element-wise max(0, x)
}

/// GELU activation function
pub fn gelu<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + FloatExt,
{
    // Implementation: x * Φ(x) where Φ is Gaussian CDF
}

/// Softmax activation with dimension support
pub fn softmax<B, S, T>(input: &Tensor<B, S, T>, dim: isize) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + FloatExt,
{
    // Implementation: exp(x_i) / sum(exp(x_j))
}
```

#### functional/ops/loss.rs
```rust
/// Mean Squared Error loss
pub fn mse_loss<B, S, T>(
    predictions: &Tensor<B, S, T>,
    targets: &Tensor<B, S, T>,
    reduction: Reduction,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + FloatExt,
{
    // Implementation: mean((predictions - targets)^2)
}

/// Cross-entropy loss
pub fn cross_entropy_loss<B, S, T>(
    logits: &Tensor<B, S, T>,
    targets: &Tensor<B, S, T>,
    reduction: Reduction,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + FloatExt,
{
    // Implementation: -sum(targets * log(softmax(logits)))
}
```

#### functional/ops/convolution.rs
```rust
/// Generic N-dimensional convolution
pub fn conv_nd<B, S, T, const N: usize>(
    input: &Tensor<B, S, T>,
    weight: &Tensor<B, S, T>,
    bias: Option<&Tensor<B, S, T>>,
    stride: &[usize; N],
    padding: &[usize; N],
    dilation: &[usize; N],
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + FloatExt,
{
    // Implementation: N-dimensional convolution operation
}

/// 2D convolution (specialization of conv_nd)
pub fn conv2d<B, S, T>(
    input: &Tensor<B, S, T>,
    weight: &Tensor<B, S, T>,
    bias: Option<&Tensor<B, S, T>>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + FloatExt,
{
    conv_nd(input, weight, bias, &[stride.0, stride.1], &[padding.0, padding.1], &[dilation.0, dilation.1])
}
```

#### functional/ops/pooling.rs
```rust
/// Generic max pooling
pub fn max_pool_nd<B, S, T, const N: usize>(
    input: &Tensor<B, S, T>,
    kernel_size: &[usize; N],
    stride: &[usize; N],
    padding: &[usize; N],
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + PartialOrd,
{
    // Implementation: sliding window maximum
}

/// Generic average pooling
pub fn avg_pool_nd<B, S, T, const N: usize>(
    input: &Tensor<B, S, T>,
    kernel_size: &[usize; N],
    stride: &[usize; N],
    padding: &[usize; N],
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + FloatExt,
{
    // Implementation: sliding window average
}
```

### 3. NN Layers Module (`nn/src/modules/`)

Layers are thin stateful wrappers that delegate computation to functional/ops:

#### modules/activation.rs
```rust
/// ReLU activation layer
#[derive(Debug, Clone)]
pub struct ReLU<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> ReLU<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + PartialOrd + Zero,
{
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<B, S, T> Module<B, S, T> for ReLU<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        crate::functional::ops::activation::relu(input)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![] // No learnable parameters
    }

    fn modules(&self) -> Vec<&dyn Module<B, S, T>> {
        vec![] // No submodules
    }

    fn zero_grad(&mut self) {
        // No parameters to zero
    }

    fn train(&mut self, _mode: bool) {
        // No training-specific behavior
    }

    fn name(&self) -> &str {
        "ReLU"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}
```

#### modules/linear.rs
```rust
/// Linear transformation layer (fully connected)
#[derive(Debug, Clone)]
pub struct Linear<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    weight: Parameter<B, S, T>,
    bias: Option<Parameter<B, S, T>>,
    in_features: usize,
    out_features: usize,
}

impl<B, S, T> Linear<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType + FloatExt,
{
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Result<Self> {
        // Initialize weight with Xavier/Glorot initialization
        let weight_data = init::xavier_uniform(in_features, out_features);
        let weight = Parameter::new(
            Tensor::from_vec(weight_data, &[out_features, in_features])?,
            "weight".to_string(),
        );

        let bias = if bias {
            let bias_data = vec![T::zero(); out_features];
            Some(Parameter::new(
                Tensor::from_vec(bias_data, &[out_features])?,
                "bias".to_string(),
            ))
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }
}

impl<B, S, T> Module<B, S, T> for Linear<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        crate::functional::ops::linear::linear(input, self.weight.data(), self.bias.as_ref().map(|b| b.data()))
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn modules(&self) -> Vec<&dyn Module<B, S, T>> {
        vec![]
    }

    fn zero_grad(&mut self) {
        self.weight.zero_grad();
        if let Some(ref mut bias) = self.bias {
            bias.zero_grad();
        }
    }

    fn train(&mut self, _mode: bool) {
        // No training-specific behavior for linear layer
    }

    fn name(&self) -> &str {
        "Linear"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}
```

### 4. Domain-Separated Storage Implementation

The storage abstraction maintains proper domain separation with basic operations only:

#### Dense Storage (storage/ crate)
```rust
// storage/src/dense/arithmetic/add.rs
pub fn add<T: DataType>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
) -> Result<DenseStorage<T>>
where
    T: core::ops::Add<Output = T> + Clone,
{
    // Basic element-wise addition - delegates to backend for execution
}

// storage/src/dense/arithmetic/mul.rs  
pub fn mul<T: DataType>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
) -> Result<DenseStorage<T>>
where
    T: core::ops::Mul<Output = T> + Clone,
{
    // Basic element-wise multiplication - delegates to backend for execution
}
```

#### Sparse Storage (sparse/ crate)
```rust
// sparse/src/formats/csr/arithmetic/add.rs
pub fn add<T: DataType>(
    lhs: &CsrStorage<T>,
    rhs: &CsrStorage<T>,
) -> Result<CsrStorage<T>>
where
    T: core::ops::Add<Output = T> + Clone + Zero,
{
    // CSR-specific addition algorithm
}

// sparse/src/formats/csr/arithmetic/matmul.rs
pub fn matmul<T: DataType>(
    lhs: &CsrStorage<T>,
    rhs: &CsrStorage<T>,
) -> Result<CsrStorage<T>>
where
    T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Clone + Zero,
{
    // CSR matrix multiplication - NOT a basic operation, specific to sparse
}
```

#### Quantization Crate (quantization/ - NEW)
```rust
// quantization/src/algorithms/symmetric.rs
pub struct SymmetricQuantizer<T: DataType> {
    scale: f32,
    zero_point: i32,
    _phantom: PhantomData<T>,
}

impl<T: DataType> SymmetricQuantizer<T> {
    pub fn quantize(&self, input: &[T]) -> Result<Vec<u8>> {
        // Quantization algorithm implementation
    }
    
    pub fn dequantize(&self, input: &[u8]) -> Result<Vec<T>> {
        // Dequantization algorithm implementation
    }
}

// quantization/src/fake_quantize/linear.rs
pub fn fake_quantize_linear<T: DataType>(
    input: &storage::DenseStorage<T>,
    scale: f32,
    zero_point: i32,
) -> Result<storage::DenseStorage<T>> {
    // Fake quantization for training
}
```

#### Backend Foundation (backend/ crate)
```rust
// backend/src/cpu/arithmetic/add.rs
pub fn add_primitive<T: DataType>(
    lhs: &[T],
    rhs: &[T],
    result: &mut [T],
) -> Result<()>
where
    T: core::ops::Add<Output = T> + Copy,
{
    // SIMD-optimized element-wise addition
    // This is the foundation that storage builds upon
}

// backend/src/gpu/arithmetic/add.rs  
pub fn add_primitive<T: DataType>(
    lhs: &[T],
    rhs: &[T],
    result: &mut [T],
) -> Result<()>
where
    T: core::ops::Add<Output = T> + Copy,
{
    // GPU kernel for element-wise addition
    // Parallel structure to CPU implementation
}
```

### 5. Unified Storage Trait Hierarchy

The storage abstraction provides a unified interface for different memory layouts:

```rust
/// Core storage operations
pub trait StorageOps<T: DataType> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn as_slice(&self) -> &[T];
    fn as_mut_slice(&mut self) -> &mut [T];
}

/// Tensor creation from vectors
pub trait StorageFromVec<T: DataType>: Storage<T> {
    fn from_vec(data: Vec<T>, dims: &[usize]) -> Result<Self>;
    fn zeros(dims: &[usize]) -> Result<Self> where T: Zero;
    fn ones(dims: &[usize]) -> Result<Self> where T: One;
    fn full(dims: &[usize], value: T) -> Result<Self>;
}

/// Matrix multiplication operations
pub trait MatMulOps<T: DataType>: StorageOps<T> {
    fn matmul(&self, other: &Self, m: usize, n: usize, k: usize) -> Result<Self>
    where
        Self: Sized;
}

/// Layout transformation operations
pub trait LayoutOps<T: DataType>: StorageOps<T> {
    fn transpose(&self, dims: &[usize]) -> Result<Self> where Self: Sized;
    fn reshape(&self, new_dims: &[usize]) -> Result<Self> where Self: Sized;
}

/// Element-wise arithmetic operations
pub trait ArithmeticOps<T: DataType>: StorageOps<T> {
    fn add(&self, other: &Self) -> Result<Self> where Self: Sized;
    fn sub(&self, other: &Self) -> Result<Self> where Self: Sized;
    fn mul(&self, other: &Self) -> Result<Self> where Self: Sized;
    fn div(&self, other: &Self) -> Result<Self> where Self: Sized;
}

/// Reduction operations
pub trait ReductionOps<T: DataType>: StorageOps<T> {
    fn sum(&self) -> T;
    fn mean(&self) -> T where T: FloatExt;
    fn max(&self) -> T where T: PartialOrd;
    fn min(&self) -> T where T: PartialOrd;
}

/// Sparse storage operations
pub trait SparseOps<T: DataType>: StorageOps<T> {
    fn to_csr(&self) -> Result<(Vec<T>, Vec<usize>, Vec<usize>)>;
    fn to_csc(&self) -> Result<(Vec<T>, Vec<usize>, Vec<usize>)>;
    fn to_coo(&self) -> Result<(Vec<T>, Vec<usize>, Vec<usize>)>;
    fn nnz(&self) -> usize; // Number of non-zeros
}

/// Full storage implementation (all operations)
pub trait FullStorage<T: DataType>:
    StorageOps<T>
    + StorageFromVec<T>
    + MatMulOps<T>
    + LayoutOps<T>
    + ArithmeticOps<T>
    + ReductionOps<T>
{
}
```

### 6. PyCoeus Generic Optimizer Wrapper

Eliminates duplicate code across optimizer implementations:

```rust
// pycoeus/src/optim/base.rs
use pyo3::prelude::*;
use optim::BaseOptimizer;
use backend::CpuBackend;
use storage::DenseStorage;
use dtype::float::Float32;

/// Generic optimizer wrapper for PyO3
pub struct PyOptimizerWrapper<O>
where
    O: BaseOptimizer<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
{
    pub inner: O,
}

impl<O> PyOptimizerWrapper<O>
where
    O: BaseOptimizer<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
{
    pub fn new(inner: O) -> Self {
        Self { inner }
    }

    pub fn step(&mut self) -> PyResult<()> {
        BaseOptimizer::step(&mut self.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Optimizer step failed: {:?}", e)
            ))
    }

    pub fn zero_grad(&mut self) {
        BaseOptimizer::zero_grad(&mut self.inner);
    }

    pub fn get_lr(&self) -> Vec<f32> {
        BaseOptimizer::get_lr(&self.inner)
    }

    pub fn set_lr(&mut self, lr: Vec<f32>) {
        BaseOptimizer::set_lr(&mut self.inner, lr);
    }

    pub fn state_dict(&self) -> PyResult<HashMap<String, Vec<f32>>> {
        BaseOptimizer::state_dict(&self.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to get state dict: {:?}", e)
            ))
    }

    pub fn load_state_dict(&mut self, state: HashMap<String, Vec<f32>>) -> PyResult<()> {
        BaseOptimizer::load_state_dict(&mut self.inner, state)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to load state dict: {:?}", e)
            ))
    }
}

// Usage in specific optimizers:
#[pyclass(name = "Adam")]
pub struct PyAdam {
    wrapper: PyOptimizerWrapper<Adam<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>,
}

#[pymethods]
impl PyAdam {
    #[new]
    fn new(params: Vec<PyTensor>, lr: f32, betas: (f32, f32), eps: f32) -> PyResult<Self> {
        let rust_params = params.into_iter().map(|p| p.inner).collect();
        let adam = Adam::new(rust_params, lr, betas, eps)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to create Adam optimizer: {:?}", e)
            ))?;
        Ok(Self {
            wrapper: PyOptimizerWrapper::new(adam),
        })
    }

    fn step(&mut self) -> PyResult<()> {
        self.wrapper.step()
    }

    fn zero_grad(&mut self) {
        self.wrapper.zero_grad()
    }

    // ... other methods delegate to wrapper
}
```

### 7. PyCoeus Exception Hierarchy

```python
# pycoeus/python/coeus/exceptions.py

class CoeusError(Exception):
    """Base exception for all Coeus errors"""
    pass

class TensorError(CoeusError):
    """Raised when tensor operations fail"""
    pass

class BackendError(CoeusError):
    """Raised when backend operations fail"""
    pass

class OptimizerError(CoeusError):
    """Raised when optimizer operations fail"""
    pass

class NNError(CoeusError):
    """Raised when neural network operations fail"""
    pass

class StorageError(CoeusError):
    """Raised when storage operations fail"""
    pass

class ShapeError(TensorError):
    """Raised when tensor shapes are incompatible"""
    pass

class DeviceError(BackendError):
    """Raised when device operations fail"""
    pass
```

```rust
// pycoeus/src/error.rs
use pyo3::prelude::*;
use pyo3::exceptions::PyException;

// Create custom Python exception types
pyo3::create_exception!(coeus, CoeusError, PyException);
pyo3::create_exception!(coeus, TensorError, CoeusError);
pyo3::create_exception!(coeus, BackendError, CoeusError);
pyo3::create_exception!(coeus, OptimizerError, CoeusError);
pyo3::create_exception!(coeus, NNError, CoeusError);
pyo3::create_exception!(coeus, StorageError, CoeusError);
pyo3::create_exception!(coeus, ShapeError, TensorError);
pyo3::create_exception!(coeus, DeviceError, BackendError);

/// Convert Rust errors to Python exceptions
pub fn convert_error(err: impl std::fmt::Display) -> PyErr {
    let err_str = err.to_string();
    
    // Pattern match on error message to determine exception type
    if err_str.contains("shape") || err_str.contains("dimension") {
        PyErr::new::<ShapeError, _>(err_str)
    } else if err_str.contains("backend") || err_str.contains("device") {
        PyErr::new::<BackendError, _>(err_str)
    } else if err_str.contains("optimizer") {
        PyErr::new::<OptimizerError, _>(err_str)
    } else if err_str.contains("storage") {
        PyErr::new::<StorageError, _>(err_str)
    } else if err_str.contains("tensor") {
        PyErr::new::<TensorError, _>(err_str)
    } else {
        PyErr::new::<CoeusError, _>(err_str)
    }
}
```

## Data Models

### Module State

```rust
/// Module state for serialization
pub struct ModuleState<T: DataType> {
    /// Parameter name to tensor data mapping
    pub parameters: HashMap<String, Vec<T>>,
    
    /// Buffer name to tensor data mapping (non-learnable state)
    pub buffers: HashMap<String, Vec<T>>,
    
    /// Module metadata
    pub metadata: ModuleMetadata,
}

pub struct ModuleMetadata {
    /// Module type name
    pub module_type: String,
    
    /// Module configuration
    pub config: HashMap<String, serde_json::Value>,
    
    /// Framework version
    pub version: String,
}
```

### Optimizer State

```rust
/// Optimizer state for checkpointing
pub struct OptimizerState<T: DataType> {
    /// Parameter-specific state (e.g., momentum, variance)
    pub param_state: HashMap<String, HashMap<String, Vec<T>>>,
    
    /// Global optimizer state (e.g., step count)
    pub global_state: HashMap<String, serde_json::Value>,
    
    /// Hyperparameters
    pub hyperparameters: HashMap<String, f64>,
}
```

### PyTorch Parity Report

```rust
/// PyTorch API parity analysis
pub struct ParityReport {
    /// Total PyTorch API surface
    pub total_pytorch_items: usize,
    
    /// Implemented items
    pub implemented_items: Vec<String>,
    
    /// Missing items by category
    pub missing_items: HashMap<Category, Vec<MissingItem>>,
    
    /// Parity percentage
    pub parity_percentage: f64,
}

pub enum Category {
    Tensor,
    NN,
    Optim,
    Autograd,
    Utils,
    Distributed,
    JIT,
    Other,
}

pub struct MissingItem {
    pub name: String,
    pub category: Category,
    pub priority: Priority,
    pub implementable: bool,
    pub reason: Option<String>,
}

pub enum Priority {
    Critical,  // Core functionality
    Important, // Commonly used
    Optional,  // Rarely used
}
```

## Error Handling

### Error Types

```rust
/// NN crate errors
#[derive(Debug, thiserror::Error)]
pub enum NNError {
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },
    
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    
    #[error("Operation not supported: {operation}")]
    UnsupportedOperation { operation: String },
    
    #[error("Serialization error: {message}")]
    SerializationError { message: String },
    
    #[error("Tensor error: {0}")]
    TensorError(#[from] tensor::TensorError),
    
    #[error("Backend error: {0}")]
    BackendError(#[from] backend::BackendError),
    
    #[error("Storage error: {0}")]
    StorageError(#[from] storage::StorageError),
}

/// PyCoeus errors
#[derive(Debug, thiserror::Error)]
pub enum PyError {
    #[error("Conversion error: {message}")]
    ConversionError { message: String },
    
    #[error("NN error: {0}")]
    NNError(#[from] nn::NNError),
    
    #[error("Optimizer error: {0}")]
    OptimizerError(#[from] optim::OptimError),
}
```

### Error Handling Strategy

1. **Rust Layer**: Use `Result<T, E>` for all fallible operations
2. **Python Layer**: Convert Rust errors to appropriate Python exceptions
3. **Error Context**: Include relevant context (shapes, types, operation names)
4. **Recovery**: Provide clear error messages with actionable guidance

## Testing Strategy

### Unit Testing

**Operations Testing** (`nn/src/functional/ops/`):
- Test each operation in isolation
- Verify mathematical correctness
- Test edge cases (empty tensors, single elements, large tensors)
- Test different data types (f32, f64)

**Layers Testing** (`nn/src/modules/`):
- Test parameter initialization
- Test forward pass delegation to functional/ops
- Test parameter management (zero_grad, state_dict)
- Test Module trait implementation

**Storage Testing** (`storage/src/`):
- Test trait implementations for each storage type
- Test storage creation (from_vec, zeros, ones)
- Test storage operations (arithmetic, reductions)
- Test storage conversions (dense ↔ sparse)

### Integration Testing

**End-to-End Training**:
- Create simple network (Linear → ReLU → Linear)
- Train on synthetic data
- Verify loss decreases
- Verify gradients flow correctly

**PyCoeus Integration**:
- Test Python API matches PyTorch
- Test error handling and exceptions
- Test optimizer state management
- Test model serialization/deserialization

### Property-Based Testing

Property-based tests will be defined in the Correctness Properties section below.

### Performance Testing

**Benchmarks**:
- Compare operation performance before/after refactoring
- Verify zero-cost abstractions (no runtime overhead)
- Compare with PyTorch on standard benchmarks
- Profile memory usage and allocation patterns

**Targets**:
- CPU operations within 2x of PyTorch
- GPU operations within 1.5x of PyTorch
- Zero allocation overhead from abstraction layers
- Compile-time optimization preserved


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Single Source of Truth for Operations

*For any* neural network operation (activation, loss, convolution, pooling, normalization, attention), there SHALL exist exactly one implementation in the `nn/src/functional/ops/` module, and all other references to that operation SHALL delegate to this single implementation.

**Validates: Requirements 1.2, 1.4**

### Property 2: Layer Delegation to Operations

*For any* neural network layer in `nn/src/modules/`, the `forward()` method SHALL call the corresponding function in `nn/src/functional/ops/` and SHALL NOT reimplement the operation logic.

**Validates: Requirements 1.3, 3.2**

### Property 3: B<S<T>> Architecture Compliance

*For any* component (operation, layer, optimizer) in the framework, the type signature SHALL include generic parameters `<B, S, T>` where B is Backend, S is Storage, and T is DataType, enabling compile-time specialization for any valid combination.

**Validates: Requirements 1.5, 10.1, 10.2, 10.3, 10.4, 10.5**

### Property 4: Generic Dimension Parameter Usage

*For any* set of related operations that differ only in dimensionality (e.g., conv1d, conv2d, conv3d), there SHALL exist a generic implementation using const generic dimension parameters, and the specialized versions SHALL delegate to this generic implementation.

**Validates: Requirements 2.8**

### Property 5: Module Trait Implementation

*For any* layer in `nn/src/modules/`, the layer SHALL implement the `Module<B, S, T>` trait with all required methods (forward, parameters, modules, zero_grad, train, name, clone_box).

**Validates: Requirements 3.3**

### Property 6: Parameter Management Abstraction

*For any* layer with learnable parameters, the parameters SHALL be stored as `Parameter<B, S, T>` instances and SHALL be accessible through the `parameters()` method.

**Validates: Requirements 3.4**

### Property 7: Serialization Round-Trip

*For any* layer implementing the Module trait, serializing the layer's state to a state dictionary and then deserializing it SHALL produce a layer with equivalent parameter values (within floating-point precision).

**Validates: Requirements 3.5**

### Property 8: StorageFromVec Trait Bounds

*For any* operation in `nn/src/functional/ops/` that creates new tensors, the function signature SHALL include `S: StorageFromVec<T>` in its trait bounds.

**Validates: Requirements 4.4**

### Property 9: Storage Format Extensibility

*For any* new storage format that implements the required storage traits (`StorageOps`, `StorageFromVec`, `MatMulOps`, etc.), all existing tensor operations and neural network layers SHALL compile and function correctly without modification.

**Validates: Requirements 4.5**

### Property 10: Optimizer Wrapper Usage

*For any* optimizer exposed in PyCoeus, the implementation SHALL use `PyOptimizerWrapper<O>` and SHALL NOT duplicate the implementation of `step()`, `zero_grad()`, `get_lr()`, `set_lr()`, `state_dict()`, or `load_state_dict()` methods.

**Validates: Requirements 5.2, 5.3**

### Property 11: Consistent Optimizer Error Handling

*For any* optimizer operation that can fail, the error SHALL be converted using the `convert_error()` function and SHALL result in an appropriate Python exception type (OptimizerError or its subclasses).

**Validates: Requirements 5.4**

### Property 12: PyTorch Optimizer API Compatibility

*For any* optimizer in PyCoeus, the public method signatures SHALL match the corresponding PyTorch optimizer's method signatures (method names, parameter names, parameter types, return types).

**Validates: Requirements 5.5**

### Property 13: Rust Error to Python Exception Mapping

*For any* Rust error converted to Python, the error SHALL be mapped to the most specific appropriate exception type based on error content (ShapeError for shape issues, BackendError for backend issues, etc.).

**Validates: Requirements 6.6**

### Property 14: Error Message Context

*For any* error raised in PyCoeus, the error message SHALL include relevant context such as tensor shapes, operation names, or parameter values that caused the error.

**Validates: Requirements 6.7**

### Property 15: Hierarchical File Structure for Parity Tracking

*For any* backend implementation (CPU, GPU, TPU, NPU), there SHALL exist parallel directory structures organized by storage type (dense, sparse, quantized) and operation category, enabling script-based comparison of implementation coverage.

**Validates: Requirements 8.2, 8.4, 8.5**

### Property 16: Domain Separation Enforcement

*For any* sparse tensor operation, the implementation SHALL reside exclusively within the sparse crate, and SHALL NOT be implemented in the tensor or storage crates.

**Validates: Requirements 16.1, 16.2, 16.4**

### Property 17: Directory Nesting Depth Limit

*For any* directory in the framework, the nesting depth from the crate root SHALL be at most 4 levels, accommodating the deep vertical hierarchy while maintaining navigability.

**Validates: Requirements 8.6**

### Property 18: Empty File Elimination

*For any* file in the framework, the file SHALL contain at least 10 lines of non-comment, non-whitespace code, or SHALL be explicitly marked as a placeholder with a TODO comment.

**Validates: Requirements 8.7**

### Property 19: Compilation Success After Changes

*For any* architectural change made to the framework, running `cargo check --workspace` SHALL complete successfully with zero compilation errors.

**Validates: Requirements 9.1**

### Property 20: Test Suite Success After Changes

*For any* architectural change made to the framework, running `cargo test --workspace` SHALL complete with all previously passing tests still passing.

**Validates: Requirements 9.2**

### Property 21: Zero Clippy Warnings

*For any* code in the framework, running `cargo clippy --workspace -- -D warnings` SHALL complete successfully with zero warnings.

**Validates: Requirements 9.3**

### Property 22: Documentation Build Success

*For any* code in the framework, running `cargo doc --workspace` SHALL complete successfully with zero errors and zero warnings.

**Validates: Requirements 9.4**

### Property 23: PyCoeus Build Success

*For any* change to PyCoeus or its dependencies, running `maturin develop` in the pycoeus directory SHALL complete successfully.

**Validates: Requirements 9.5**

### Property 24: Public API Backward Compatibility

*For any* existing public API function or method in core modules (tensor, nn, optim), the function signature SHALL remain unchanged or SHALL be marked with a deprecation warning before removal.

**Validates: Requirements 13.1, 13.2, 13.3**

### Property 25: Deprecation Warning Presence

*For any* API marked for deprecation, calling the deprecated function SHALL emit a compiler warning or runtime warning indicating the deprecation and suggesting the replacement API.

**Validates: Requirements 13.4**

### Property 26: Rust Naming Convention Compliance

*For any* public function, the name SHALL use snake_case, and for any public type, the name SHALL use PascalCase, following Rust naming conventions.

**Validates: Requirements 14.1**

### Property 27: Result Type Error Handling

*For any* operation that can fail, the function SHALL return a `Result<T, E>` type rather than panicking or returning an Option.

**Validates: Requirements 14.2**

### Property 28: Unsafe Block Documentation

*For any* unsafe block in the codebase, there SHALL be a comment immediately preceding the block explaining why the unsafe code is necessary and why it is safe.

**Validates: Requirements 14.3**

### Property 29: Rustfmt Compliance

*For any* code in the framework, running `cargo fmt --check` SHALL complete successfully with no formatting changes required.

**Validates: Requirements 14.5**

### Property 30: Operations Unit Test Coverage

*For any* operation in `nn/src/functional/ops/`, there SHALL exist at least one unit test that verifies the operation's correctness on valid inputs.

**Validates: Requirements 15.2**

### Property 31: Test Coverage Threshold

*For any* core crate (dtype, storage, tensor, autograd, backend, nn, optim), the line coverage percentage SHALL be at least 90% as measured by a coverage tool.

**Validates: Requirements 15.1**

### Property 32: Zero-Cost Abstraction Preservation

*For any* architectural change, benchmark measurements SHALL show that the abstraction layers (ops delegation, storage traits, generic wrappers) introduce zero runtime overhead compared to direct implementation, as verified by comparing assembly output or benchmark timings.

**Validates: Requirements 12.1, 12.4**

### Property 33: SIMD and GPU Acceleration Preservation

*For any* operation that previously used SIMD or GPU acceleration, the refactored implementation SHALL continue to use the same acceleration mechanisms, as verified by checking for SIMD intrinsics or GPU kernel calls in the generated code.

**Validates: Requirements 12.2, 12.3**

### Property 34: Rustdoc Coverage

*For any* public function, type, trait, or module, there SHALL exist rustdoc documentation that includes at minimum a summary sentence and, for functions, a description of parameters and return values.

**Validates: Requirements 11.1**

### Property 35: Crate Boundary Enforcement

*For any* functionality implemented in a domain-specific crate (sparse, dtype, backend), the functionality SHALL NOT be duplicated or reimplemented in other crates.

**Validates: Requirements 16.4, 16.6**

### Property 36: Quantization Crate Separation

*For any* quantization-related functionality, the implementation SHALL reside exclusively within the quantization crate and SHALL NOT be implemented in the nn or dtype crates.

**Validates: Requirements 17.1, 17.2, 17.3, 17.6**

### Property 37: Storage Basic Operations Boundary

*For any* operation in storage crates, the operation SHALL be limited to basic arithmetic (add, subtract, multiply, divide), basic layout (reshape, transpose), or basic creation (zeros, ones, from_vec), and SHALL NOT implement complex operations like linear transformations or convolutions.

**Validates: Requirements 18.1, 18.2, 18.3, 18.4**

### Property 38: Backend Foundation Dependency

*For any* storage operation that requires hardware execution, the storage implementation SHALL delegate to backend primitives rather than implementing hardware-specific code directly.

**Validates: Requirements 18.5, 18.6**

## Testing Strategy

The testing strategy employs a dual approach combining unit tests for specific examples and property-based tests for universal properties:

### Unit Tests

Unit tests focus on:
- **Specific examples**: Concrete test cases demonstrating correct behavior
- **Edge cases**: Empty tensors, single elements, boundary values
- **Error conditions**: Invalid inputs, shape mismatches, unsupported operations
- **Integration points**: Layer composition, optimizer integration, serialization

Unit tests are valuable for:
- Catching concrete bugs in specific scenarios
- Documenting expected behavior through examples
- Testing error handling paths
- Validating integration between components

### Property-Based Tests

Property-based tests focus on:
- **Universal properties**: Properties that hold for all valid inputs
- **Comprehensive input coverage**: Randomized testing across input space
- **Mathematical correctness**: Verifying algebraic properties
- **Invariant preservation**: Ensuring system invariants are maintained

Each property test will:
- Run a minimum of 100 iterations with randomized inputs
- Reference its corresponding design document property
- Use the tag format: **Feature: coeus-architecture-enhancement, Property {number}: {property_text}**

### Property Test Configuration

```rust
use proptest::prelude::*;

// Example property test structure
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]
    
    /// Feature: coeus-architecture-enhancement, Property 7: Serialization Round-Trip
    /// For any layer, serializing then deserializing SHALL produce equivalent parameters
    #[test]
    fn test_layer_serialization_round_trip(
        in_features in 1usize..100,
        out_features in 1usize..100,
    ) {
        let layer = Linear::<CpuBackend, DenseStorage<Float32>, Float32>::new(
            in_features,
            out_features,
            true
        ).unwrap();
        
        // Serialize
        let state_dict = layer.state_dict();
        
        // Deserialize
        let mut new_layer = Linear::<CpuBackend, DenseStorage<Float32>, Float32>::new(
            in_features,
            out_features,
            true
        ).unwrap();
        new_layer.load_state_dict(&state_dict).unwrap();
        
        // Verify equivalence
        let original_params = layer.parameters();
        let loaded_params = new_layer.parameters();
        
        for (orig, loaded) in original_params.iter().zip(loaded_params.iter()) {
            let orig_data = orig.data().as_slice();
            let loaded_data = loaded.data().as_slice();
            
            for (o, l) in orig_data.iter().zip(loaded_data.iter()) {
                prop_assert!((o.get() - l.get()).abs() < 1e-6);
            }
        }
    }
}
```

### Testing Balance

The framework maintains a balance between unit and property tests:

- **Unit tests** provide concrete examples and catch specific bugs
- **Property tests** provide comprehensive coverage and catch edge cases
- Both are necessary and complementary
- Property tests handle the "lots of inputs" case, reducing the need for many similar unit tests
- Unit tests focus on specific scenarios, integration points, and error conditions

### Test Organization

```
nn/
├── src/
│   ├── functional/
│   │   └── ops/
│   │       ├── activation.rs
│   │       └── tests/
│   │           └── activation_tests.rs  # Unit tests for activation ops
│   └── modules/
│       ├── activation.rs
│       └── tests/
│           └── activation_tests.rs  # Unit tests for activation layers
└── tests/
    ├── property_tests.rs            # Property-based tests
    ├── integration_tests.rs         # Integration tests
    └── benchmark_tests.rs           # Performance benchmarks
```

### Continuous Integration

All tests run in CI on every commit:
1. Unit tests: `cargo test --workspace`
2. Property tests: `cargo test --workspace --release` (for performance)
3. Clippy: `cargo clippy --workspace -- -D warnings`
4. Format: `cargo fmt --check`
5. Documentation: `cargo doc --workspace`
6. PyCoeus: `cd pycoeus && maturin develop && pytest`
7. Coverage: `cargo tarpaulin --workspace --out Xml`

### Performance Validation

Benchmarks run before and after architectural changes:
1. Operation benchmarks: Individual op performance
2. Layer benchmarks: Forward pass timing
3. End-to-end benchmarks: Full training loop
4. Memory benchmarks: Allocation patterns

Performance regression is detected by comparing benchmark results with a threshold (e.g., no more than 5% slowdown).
