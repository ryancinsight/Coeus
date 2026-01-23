//! Storage-level arithmetic dispatch traits.
//!
//! This module provides `TensorStorageArithmetic`, the core trait enabling the `tensor` crate
//! to delegate arithmetic operations to specialized crates (`dense`, `sparse`) while supporting
//! generic backend dispatch (CPU, GPU).
//!
//! ## Architecture
//!
//! The dispatch follows a hierarchical pattern enforcing SRP, SSOT, and SOC:
//!
//! ```text
//! Tensor<B, S, T>
//!   └── calls S::tensor_add(&self, other, backend)
//!         └── DenseStorage delegates to dense::arithmetic::add(self, other, backend)
//!               └── Backend::add_dense() → CPU/GPU dispatch
//!         └── CsrStorage delegates to sparse::SparseAdd::add_sparse()
//!               └── Converts result, no backend dispatch (pure CPU for sparse)
//! ```
//!
//! ## File Tree Parity
//!
//! This trait mirrors the structure in `dense` and `sparse` crates:
//! - `dense/src/arithmetic/{add,sub,mul,div}.rs`
//! - `sparse/src/arithmetic/{add,sub,mul,div}.rs`
//!
//! ## GPU/CPU Dispatch
//!
//! Dense operations pass the `backend` parameter to enable GPU execution.
//! Sparse operations currently use CPU-only implementations but accept backend
//! for future GPU sparse kernel support.

use crate::Result;
use backend::Backend;
use dtype::DataType;

/// Trait for storage-level arithmetic operations.
/// This allows the tensor crate to delegate arithmetic logic to the underlying storage implementation,
/// whether it be dense (delegating to `dense` crate) or sparse (delegating to `sparse` crate).
pub trait TensorStorageArithmetic<T: DataType> {
    /// Element-wise addition
    fn tensor_add<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise subtraction
    fn tensor_sub<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise multiplication
    fn tensor_mul<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise division
    fn tensor_div<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise negation
    fn tensor_neg<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        Self: Sized;
}

