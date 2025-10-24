//! # Coeus Tensor
//!
//! PyTorch-like tensor library with automatic differentiation, backend abstraction, and higher-order derivatives.
//!
//! This crate provides a unified tensor architecture that consolidates all tensor forms
//! into a single type with generic dtype and backend abstraction.
//!
//! ## Architecture
//!
//! The tensor library provides a single unified tensor type that works across all backends
//! and supports both regular operations and automatic differentiation:
//!
//! ### UnifiedTensor (Single Tensor Type)
//! ```rust
//! use coeus_tensor::{Tensor, CpuBackend};
//! use coeus_dtype::FloatDtype;
//!
//! // Create CPU tensor with explicit backend and dtype
//! let backend = CpuBackend::new();
//! let tensor = Tensor::<f32, CpuBackend>::from_vec(
//!     backend,
//!     vec![1.0, 2.0, 3.0],
//!     vec![3]
//! ).unwrap();
//!
//! // Operations work like PyTorch with zero-copy semantics
//! let result = tensor.add(&tensor).unwrap();
//!
//! // Enable autograd for gradient computation
//! let autograd_tensor = tensor.with_autograd(true);
//! let grad_result = autograd_tensor.neg(); // Supports gradient computation
//! ```
//!
//! ## Key Design Principles
//!
//! - **Single Source of Truth (SSOT)**: One tensor type handles both backend and autograd
//! - **Generic Backend Abstraction**: Works with any backend (CPU, GPU, custom)
//! - **Generic Dtype Support**: Full support for all numeric types through trait system
//! - **Zero-Copy Operations**: Copy-on-write semantics minimize memory allocations
//! - **Optional Autograd**: Can work with or without gradient tracking
//! - **Type Safety**: Compile-time guarantees for all operations
//!
//! ## Backend Architecture
//!
//! The unified tensor uses a backend abstraction system:
//! - `B: Backend<T>`: Generic backend trait for device-agnostic operations
//! - `T: Dtype`: Generic data type trait for numeric operations
//! - Zero unsafe code with proper trait bounds and memory safety

// Core infrastructure
pub mod error;
pub mod tensor_impl;
pub mod tensor_sparse;
pub mod tensor_core;
pub mod tensor_autograd;
pub mod tensor_dense_ops;
pub mod tensor_dense_ops_ext;
pub mod tensor_sparse_ops;

// Tensor operations
pub mod creation;
pub mod indexing;
pub mod shape_ops;

// Mathematical operations
pub mod arithmetic;
pub mod elementwise;
pub mod matrix;
pub mod reduction;

// Re-export core types and traits
pub use tensor_core::{AsAny, DifferentiableFunction, Function, Backend, CpuBackend};
pub use tensor_core::Tensor;
pub use tensor_core::Device;

// Re-export storage and dtype traits
pub use coeus_dtype::traits::FloatExt;
pub use coeus_dtype::DataType;
pub use coeus_storage::{DenseStorage, Shape, Storage, StorageFromVec, StorageToDense};

// Error handling
pub use error::TensorError;

// PyTorch-style type aliases for ergonomic API
/// CPU dense tensor type alias
pub type TensorCpuDense<T> = Tensor<CpuBackend<T>, DenseStorage<T>, T>;

// Re-export AutoGradTensor from the autograd module
pub use tensor_autograd::AutoGradTensor;
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]

