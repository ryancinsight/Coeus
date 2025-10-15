# Coeus Tensor

Core tensor implementation for the Coeus deep learning framework.

## Overview

Implements the foundational `Tensor<B, S, T>` type with nested backend/storage/dtype hierarchy per ADR-001.

## Features

- **Nested Type Hierarchy**: `Tensor<Backend, Storage<DataType>, DataType>`
- **Zero-Cost Abstractions**: Static dispatch via monomorphization
- **Type Safety**: Compile-time dtype/backend/storage validation
- **PyTorch API**: Compatible constructor methods

## Usage

```rust
use coeus_tensor::Tensor;
use coeus_backend::CpuBackend;
use coeus_storage::DenseStorage;
use coeus_dtype::float::Float32;

// Type alias for convenience
type CpuTensor<T> = Tensor<CpuBackend, DenseStorage<T>, T>;

// Create from vector
let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
let tensor = CpuTensor::<Float32>::from_vec(data, &[3]).unwrap();

// Create zeros/ones
let zeros = CpuTensor::<Float32>::zeros(&[2, 3]).unwrap();
let ones = CpuTensor::<Float32>::ones(&[4, 5]).unwrap();
```

## Testing

```bash
cargo test --package coeus-tensor
```

**Coverage**: 22/22 tests passing (15 integration + 7 doc)

## Type Aliases

For ergonomic usage, define type aliases:

```rust
type CpuDenseTensor<T> = Tensor<CpuBackend, DenseStorage<T>, T>;
type CpuFloat32 = CpuDenseTensor<Float32>;
type CpuFloat64 = CpuDenseTensor<Float64>;
```

