# Coeus Storage

Memory storage abstractions for the Coeus deep learning framework.

## Overview

This crate provides storage primitives for multi-dimensional tensor data, separating memory layout concerns from compute backend logic.

## Features

- **DenseStorage**: Contiguous memory with row-major (C-contiguous) layout
- **Shape Management**: Multi-dimensional shape specification with stride calculation
- **Zero-Copy Views**: Slice-based access for efficient memory usage
- **Type Safety**: Generic over `DataType` from `coeus-dtype`

## Usage

```rust
use coeus_storage::{DenseStorage, Storage, Shape};
use coeus_dtype::float::Float32;

// Create from vector
let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
let storage = DenseStorage::from_vec(data, &[3]).unwrap();

// Create with zeros
let zeros = DenseStorage::<Float32>::zeros(&[2, 3]).unwrap();

// Access data
let slice: &[Float32] = storage.as_slice();
```

## Architecture

Storage abstractions follow the trait hierarchy:

```
Storage<T: DataType>
├── DenseStorage<T>     // Implemented ✅
├── StridedStorage<T>   // Future
└── SparseStorage<T>    // Future
```

## Memory Layout

Row-major (C-contiguous) by default:

```
Shape [2, 3]:
[[a, b, c],    Memory: [a, b, c, d, e, f]
 [d, e, f]]    
Stride: [3, 1]
```

## Testing

```bash
cargo test --package coeus-storage
```

**Coverage**: 23/23 tests passing (14 unit + 9 doc tests)

## Documentation

Full API documentation available via:

```bash
cargo doc --package coeus-storage --no-deps --open
```

