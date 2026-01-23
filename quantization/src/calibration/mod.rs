//! Calibration module
//!
//! This module provides calibration methods for determining optimal quantization parameters.
//! It includes separate implementations for different calibration techniques.

pub mod entropy;
pub mod percentile;
pub mod mse;

// Re-export commonly used types
pub use entropy::*;
pub use percentile::*;
pub use mse::*;

// Re-export from algorithms for backward compatibility
pub use crate::algorithms::calibration::*;