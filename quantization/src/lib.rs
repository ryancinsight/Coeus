//! # Quantization Crate
//!
//! This crate provides quantization algorithms and operations for the Coeus deep learning framework.
//! It includes support for various quantization schemes, calibration methods, and fake quantization
//! for training quantized models.
//!
//! ## Features
//!
//! - **Quantization Algorithms**: Symmetric, asymmetric, and dynamic quantization
//! - **Calibration Methods**: Entropy, percentile, and MSE-based calibration
//! - **Fake Quantization**: Training-time quantization simulation
//! - **Type System**: Quantized data types and storage formats
//!
//! ## Architecture
//!
//! The crate is organized into several modules:
//! - `algorithms/`: Core quantization algorithms
//! - `calibration/`: Calibration methods for determining quantization parameters
//! - `fake_quantize/`: Fake quantization for training
//! - `types/`: Quantized data types and storage formats

#![deny(missing_docs)]
#![warn(clippy::all)]

pub mod algorithms;
pub mod calibration;
pub mod fake_quantize;
pub mod types;

// Re-export commonly used types and functions
pub use algorithms::{
    CalibrationConfig, CalibrationMethod, CalibrationStats, MixedPrecisionConfig,
    QuantizationBitwidth, QuantizationGranularity, QuantizationScheme, QuantizedWeights,
    SerializableQuantizedWeights,
};
pub use calibration::*;
pub use fake_quantize::*;
pub use types::{QInt4, QInt8, QUInt4, QUInt8};