//! Tensor operations following PyTorch-like API structure
//!
//! This module provides a clean, flat API for tensor operations organized by category.

pub mod activations;
pub mod arithmetic;
pub mod bitwise;
// pub mod comparison; // TODO: Implement proper comparison operations
pub mod creation;
pub mod elementwise;
pub mod indexing;
// pub mod logical; // TODO: Implement proper logical operations
pub mod matrix;
pub mod reduction;
// pub mod storage_test; // Temporarily disabled
