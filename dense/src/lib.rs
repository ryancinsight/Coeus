//! # Dense Tensor Operations
//!
//! This crate provides dense tensor operations and algorithms for the Coeus deep learning framework.
//! It focuses on operations specific to dense (contiguous) memory layouts, building on the storage
//! foundation to provide higher-level dense tensor functionality.
//!
//! ## Architecture
//!
//! The dense crate sits between storage and tensor in the dependency hierarchy:
//! - Depends on: storage, dtype, backend
//! - Used by: tensor, nn
//!
//! ## Organization
//!
//! Operations are organized into hierarchical modules:
//! - `arithmetic/` - Basic element-wise operations (add, sub, mul, div)
//! - `reduction/` - Reduction operations (sum, max, min, mean)
//! - `activation/` - Activation functions (relu, sigmoid, tanh, gelu)
//! - `layout/` - Memory layout operations (reshape, transpose, stride)
//! - `creation/` - Tensor creation operations (zeros, ones, from_vec)
//! - `linear_algebra/` - Matrix operations (matmul)
//!
//! This structure enables:
//! - Clear separation of concerns
//! - Easy identification of missing operations
//! - Hierarchical parity tracking across backends

#![no_std]
#![warn(missing_docs, clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

extern crate alloc;

// Re-export core dependencies
pub use storage::{DenseStorage, Result, StorageError};
pub use dtype::DataType;

// Core modules
pub mod arithmetic;
pub mod reduction;
pub mod activation;
pub mod layout;
pub mod creation;
pub mod linear_algebra;

// Re-export commonly used items
pub use arithmetic::DenseArithmetic;
pub use layout::DenseLayout;
pub use creation::DenseCreation;
pub use linear_algebra::DenseMatMul;
pub use reduction::DenseReduce;
pub use activation::DenseActivation;
