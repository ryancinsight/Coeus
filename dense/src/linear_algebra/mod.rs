//! Linear algebra operations for dense tensors
//!
//! This module provides linear algebra operations that are specific to dense storage,
//! including matrix multiplication, which is a complex operation that should not be
//! in the storage foundation layer.

pub mod matmul;

pub use matmul::DenseMatMul;
