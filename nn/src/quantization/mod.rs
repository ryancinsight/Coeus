//! # Quantization-Aware Training (QAT), Quantized Inference, and Post-Training Quantization (PTQ)
//!
//! Implements full B<S<T>> generic quantization support for neural networks.
//! This module now uses the dedicated quantization crate for core functionality.
//!
//! ## Features
//!
//! - **Fake Quantization (QAT)**: Simulates quantization effects during training
//! - **Quantized Inference**: Optimized quantized operations for inference
//! - **Post-Training Quantization (PTQ)**: Quantize pre-trained models without retraining
//! - **Multi-Bitwidth Support**: 4-bit, 8-bit, and 16-bit quantization
//! - **Quantization Schemes**: Affine and symmetric quantization
//! - **Full Generic Support**: Works with any B<S<T>> tensor configuration
//!
//! ## Quantization Schemes
//!
//! ### Affine Quantization
//! ```text
//! q = round((x - zero_point) / scale)
//! x_dq = (q - zero_point) * scale
//! ```
//!
//! ### Symmetric Quantization
//! ```text
//! q = round(x / scale)
//! x_dq = q * scale
//! ```
//!
//! ## Post-Training Quantization (PTQ)
//!
//! PTQ enables quantization of pre-trained models without retraining:
//!
//! 1. **Calibration Data Collection**: Run representative samples through the model
//! 2. **Parameter Estimation**: Compute optimal scales and zero-points for each layer
//! 3. **Model Conversion**: Replace FP32 weights with quantized equivalents
//! 4. **Accuracy Validation**: Verify model accuracy meets requirements
//!
//! ### PTQ Algorithms
//!
//! - **Min-Max**: Uses absolute min/max values from calibration data
//! - **Percentile**: Uses percentile-based clipping for outlier robustness
//! - **MSE Minimization**: Finds parameters that minimize quantization error
//! - **Entropy Minimization**: Optimizes for information preservation
//!
//! ## PTQ Layer Fusion
//!
//! Layer fusion combines multiple neural network operations into optimized fused kernels:
//!
//! ### Fusion Patterns
//!
//! - **Conv + BatchNorm**: Fused convolution with batch normalization parameters
//! - **Linear + Activation**: Fused linear layer with activation function
//! - **Conv + ReLU**: Fused convolution with ReLU activation
//! - **Multi-Head Attention Fusion**: Optimized attention computation with fused projections
//!
//! ### Benefits
//!
//! - **Reduced Kernel Launches**: Fewer GPU/TPU kernel invocations
//! - **Better Memory Access**: Optimized memory layouts and access patterns
//! - **Parameter Sharing**: Quantization parameters shared across fused operations
//! - **Improved Performance**: Lower latency and higher throughput
//!
//! ## Hardware-Specific Quantization Optimization
//!
//! Specialized optimizations for different hardware backends:
//!
//! ### TPU Optimization
//! - **BFloat16 Native Support**: Direct bfloat16 operations without conversion overhead
//! - **Fused Quantization Kernels**: TPU-specific fused operations for quantized inference
//! - **Matrix Unit Optimization**: Leveraging TPU matrix units for quantized matrix operations
//!
//! ### NPU Optimization
//! - **Int4 Acceleration**: Native 4-bit integer operations for maximum compression
//! - **Hardware-Specific Quantization**: NPU-optimized quantization formats and ranges
//! - **Low-Power Inference**: Energy-efficient quantized operations for edge devices
//!
//! ### GPU Optimization
//! - **Tensor Core Acceleration**: Leveraging tensor cores for quantized matrix operations
//! - **Mixed Precision Compute**: FP16/INT8 fused operations with automatic precision selection
//! - **Memory Hierarchy Optimization**: Efficient use of GPU memory hierarchies for quantized data
//!
//! ### CPU Optimization
//! - **SIMD Quantization**: AVX-512/AVX2 vectorized quantization operations
//! - **Cache-Aware Kernels**: Optimized memory access patterns for CPU cache hierarchies
//! - **Multi-Core Scaling**: Efficient parallelization across CPU cores for quantization tasks

pub mod fusion;
pub mod quantization_ops;
pub mod quantized_layers;
pub mod serialization;

// Re-export from quantization crate
pub use quantization::{
    // Core types and algorithms
    QuantizationScheme, QuantizationGranularity, QuantizationBitwidth,
    CalibrationMethod, CalibrationConfig, CalibrationStats,
    MixedPrecisionConfig, QuantizedWeights, SerializableQuantizedWeights,
    
    // Calibration
    CalibrationPipeline, SerializableCalibrationPipeline,
    SerializableCalibrationStats,
    
    // Fake quantization
    LinearFakeQuantize, ConvFakeQuantize,
    
    // Types
    QuantizationParams, QuantizationError, QuantizationNoiseAnalysis,
    QuantizationResult, QInt8, QUInt8, QInt4, QUInt4, QuantizedType,
    
    // Utilities
    MinMaxQuantizer, SymmetricQuantizer, PercentileQuantizer,
    CalibrationData,
    
    // Calibration methods
    EntropyCalibrator, PercentileCalibrator, MseCalibrator,
    
    // Algorithms
    SymmetricQuantizer as AlgorithmicSymmetricQuantizer,
    AsymmetricQuantizer, DynamicQuantizer,
};

// Re-exports for convenience from local modules
pub use fusion::*;
pub use quantization_ops::*;
pub use quantized_layers::*;
pub use serialization::*;
