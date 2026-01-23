//! Fake quantization module
//!
//! This module provides fake quantization operations for quantization-aware training.
//! It includes separate implementations for different operation types.

pub mod linear;
pub mod conv;

// Re-export commonly used types
pub use linear::*;
pub use conv::*;