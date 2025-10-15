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
//! use coeus_tensor::Tensor;
//! use coeus_backend::CpuBackend;
//! use coeus_storage::DenseStorage;
//! use coeus_dtype::float::Float32;
//!
//! // Create CPU tensor with explicit backend, storage, and dtype
//! let tensor = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
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
//! The unified tensor uses a backend abstraction system:
//! - `B: Backend<T>`: Generic backend trait for device-agnostic operations
//! - `T: Dtype`: Generic data type trait for numeric operations
//! - Zero unsafe code with proper trait bounds and memory safety


// Module declarations
pub mod ops {
    pub mod arithmetic;
    pub mod creation;
    pub mod matrix;
    pub mod reduction;
}

// Temporarily disabled due to alloc issues
pub mod tensor_core;
pub mod tensor_impl;
pub mod minimal_tensor;

// Additional modules
pub mod error;
pub mod shape_ops;
pub mod elementwise;

// Re-export full tensor implementation
pub use tensor_core::{AsAny, DifferentiableFunction, Function, Tensor};

// Re-export error types and utilities

// Re-export storage utilities
pub use coeus_storage::{Shape, StorageFromVec, StorageToDense};

// Minimal API for testing - full API to be implemented later

// Advanced zero-copy optimizations - temporarily disabled
// pub mod zero_copy;
// pub mod simd_ops;

// Re-export dtype traits for convenience
pub use coeus_dtype::{DataType, traits::FloatExt};
pub use coeus_backend::{Backend, BackendError, CpuBackend, Device};
pub use coeus_storage::{DenseStorage, Storage};

// Result type for tensor operations
pub type Result<T> = std::result::Result<T, TensorError>;

/// Creates a thread-safe gradient storage container
pub fn grad_rwlock<T>(value: T) -> std::sync::RwLock<T> {
    std::sync::RwLock::new(value)
}

// Re-export error types
pub use error::TensorError;
