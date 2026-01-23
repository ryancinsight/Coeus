//! # Coeus Tensor
//!
//! PyTorch-like tensor library with automatic differentiation, backend abstraction, and higher-order derivatives.
//!
//! This crate provides a unified tensor architecture that consolidates all tensor forms
//! into a single type with simplified generic backend abstraction.
//!
//! ## Architecture
//!
//! The tensor library provides a single unified tensor type that works across all backends
//! and supports both regular operations and automatic differentiation:
//!
//! ### Simplified Tensor API
//! ```rust
//! use tensor::Tensor;
//! use backend::CpuBackend;
//! use dtype::float::Float32;
//! use storage::DenseStorage;
//!
//! let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
//!     vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
//!     &[3]
//! ).unwrap();
//!
//! // Operations work like PyTorch with zero-copy semantics
//! let result = &tensor + &tensor;
//!
//! // Enable autograd for gradient computation
//! let autograd_tensor = tensor.requires_grad_(true);
//! let grad_result = -&autograd_tensor; // Supports gradient computation
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
//! The simplified tensor uses a backend abstraction with associated types:
//! - `B: Backend<Data = T, Device = D>`: Backend trait with associated data and device types
//! - Generic methods support any storage type (dense/sparse) through trait bounds
//! - Zero unsafe code with proper trait bounds and memory safety
//! - Complete sparse tensor support through generic operations

// Module declarations
extern crate alloc;

pub mod ops;

pub mod functions;
pub mod minimal_tensor;

pub mod tensor_backend_dispatch;
pub mod tensor_core;
pub use implementations::*;
pub mod implementations;

// Additional modules
pub mod elementwise;
pub mod error;
pub mod indexing;
pub mod shape_ops;

// Re-export full tensor implementation
pub use tensor_core::{AsAny, DifferentiableFunction, Function, OperationName, Tensor};

// Re-export error types and utilities

// Re-export storage utilities
pub use storage::{Shape, StorageFromVec, StorageToDense};

// Minimal API for testing - full API to be implemented later

// Advanced zero-copy optimizations - temporarily disabled
// pub mod zero_copy;
// pub mod simd_ops;

// Re-export dtype traits for convenience
pub use backend::{Backend, BackendError, Device};
pub use dtype::float::Float32;
pub use dtype::{traits::FloatExt, DataType};
pub use storage::{CsrStorage, CscStorage, CooStorage, DenseStorage, SparseFormat, SparseStorage, Storage};
pub use dense;

// Re-export CpuBackend with Float32 default for convenience
pub use backend::CpuBackend;

// Result type for tensor operations
pub type Result<T> = std::result::Result<T, TensorError>;

// Re-export convenience functions from ops::creation
pub use ops::creation::{cat, randn};

/// Creates a thread-safe gradient storage container
pub fn grad_rwlock<T>(value: T) -> std::sync::RwLock<T> {
    std::sync::RwLock::new(value)
}

// Re-export error types
pub use error::TensorError;

// Sparse tensor support
pub mod sparse;

#[cfg(test)]
mod tests;
