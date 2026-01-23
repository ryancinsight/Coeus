//! Stateless neural network operations
//!
//! This module provides the single source of truth for all NN operations.
//! Operations are stateless functions that operate on tensors.
//! Layers in the `layers/` module wrap these operations with state.

pub mod activation;
pub mod loss;
// pub mod convolution;
// pub mod linear;
// pub mod normalization;
// pub mod pooling;
// pub mod attention;

// Re-export commonly used operations
pub use activation::*;
pub use loss::*;
