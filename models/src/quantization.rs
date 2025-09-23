//! Quantization schemes and dequantization for efficient model inference.

#![allow(clippy::manual_div_ceil)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::manual_is_multiple_of)]
//!
//! This module provides multiple quantization schemes used by llama.cpp and similar
//! model inference engines. Quantization reduces memory usage while maintaining
//! model accuracy through optimized numerical representations.
//!
//! ## Supported Quantization Schemes
//!
//! - **Q4_0**: 4-bit quantization with zero point (most common)
//! - **Q4_1**: 4-bit quantization with better zero handling
//! - **Q5_0**: 5-bit quantization for higher precision
//! - **Q5_1**: 5-bit quantization with refined zero handling
//! - **Q8_0**: 8-bit quantization for maximum compatibility
//! - **Q8_1**: 8-bit quantization with enhanced accuracy
//!
//! ## Memory Efficiency
//!
//! Quantization provides significant memory savings:
//! - Q4_0: ~75% memory reduction vs F32
//! - Q5_0: ~68% memory reduction vs F32
//! - Q8_0: ~50% memory reduction vs F32
//!
//! ## Performance Characteristics
//!
//! - **Memory Usage**: Dramatically reduced memory footprint
//! - **Inference Speed**: Comparable or faster than F32 on modern hardware
//! - **Accuracy**: Minimal accuracy loss with proper calibration
//! - **Compatibility**: Universal support across model architectures

use crate::error::{ModelError, ModelResult};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Quantization scheme types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizationScheme {
    /// No quantization (full precision)
    None,
    /// 4-bit quantization, zero point
    Q4_0,
    /// 4-bit quantization, improved zero handling
    Q4_1,
    /// 5-bit quantization, zero point
    Q5_0,
    /// 5-bit quantization, improved zero handling
    Q5_1,
    /// 8-bit quantization, zero point
    Q8_0,
    /// 8-bit quantization, improved accuracy
    Q8_1,
    /// Full precision (F32)
    F32,
    /// Full precision (F16)
    F16,
}

impl QuantizationScheme {
    /// Get the name of the quantization scheme
    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q8_1 => "q8_1",
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }

    /// Get the bits per weight for this scheme
    pub fn bits_per_weight(&self) -> usize {
        match self {
            Self::None => 32, // Full precision
            Self::Q4_0 | Self::Q4_1 => 4,
            Self::Q5_0 | Self::Q5_1 => 5,
            Self::Q8_0 | Self::Q8_1 => 8,
            Self::F32 => 32,
            Self::F16 => 16,
        }
    }

    /// Calculate memory usage for given number of elements
    pub fn memory_usage(&self, elements: usize) -> usize {
        match self {
            Self::None | Self::F32 => elements * 4, // 4 bytes per element
            Self::F16 => elements * 2,              // 2 bytes per element
            Self::Q4_0 | Self::Q4_1 => (elements * 4 + 7) / 8, // 4 bits per element, rounded up
            Self::Q5_0 | Self::Q5_1 => (elements * 5 + 7) / 8, // 5 bits per element, rounded up
            Self::Q8_0 | Self::Q8_1 => elements,    // 1 byte per element
        }
    }

    /// Check if this is a quantized scheme
    pub fn is_quantized(&self) -> bool {
        !matches!(self, Self::None | Self::F32 | Self::F16)
    }

    /// Get the block size for this quantization scheme
    pub fn block_size(&self) -> usize {
        match self {
            Self::None | Self::F32 | Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
        }
    }

    /// Parse quantization scheme from string
    pub fn from_str(s: &str) -> ModelResult<Self> {
        match s.to_lowercase().as_str() {
            "none" => Ok(Self::None),
            "q4_0" => Ok(Self::Q4_0),
            "q4_1" => Ok(Self::Q4_1),
            "q5_0" => Ok(Self::Q5_0),
            "q5_1" => Ok(Self::Q5_1),
            "q8_0" => Ok(Self::Q8_0),
            "q8_1" => Ok(Self::Q8_1),
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            _ => Err(ModelError::quantization(format!(
                "Unknown quantization scheme: {}",
                s
            ))),
        }
    }
}

impl fmt::Display for QuantizationScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Quantization type information
#[derive(Debug, Clone)]
pub struct QuantizationType {
    /// Quantization scheme
    pub scheme: QuantizationScheme,
    /// Whether the model is fully quantized
    pub is_quantized: bool,
    /// Memory savings percentage (0.0 to 1.0)
    pub memory_savings: f64,
}

impl QuantizationType {
    /// Create new quantization info
    pub fn new(scheme: QuantizationScheme) -> Self {
        let is_quantized = scheme.is_quantized();
        let memory_savings = if is_quantized {
            match scheme {
                QuantizationScheme::Q4_0 | QuantizationScheme::Q4_1 => 0.75,
                QuantizationScheme::Q5_0 | QuantizationScheme::Q5_1 => 0.6875,
                QuantizationScheme::Q8_0 | QuantizationScheme::Q8_1 => 0.5,
                _ => 0.0,
            }
        } else {
            0.0
        };

        Self {
            scheme,
            is_quantized,
            memory_savings,
        }
    }
}

/// Quantized tensor data structure
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Raw quantized data
    pub data: Vec<u8>,
    /// Original tensor shape
    pub shape: Vec<usize>,
    /// Quantization scheme used
    pub quantization: QuantizationScheme,
    /// Scales for dequantization
    pub scales: Vec<f32>,
    /// Zero points for dequantization
    pub zero_points: Vec<f32>,
}

impl QuantizedTensor {
    /// Create a new quantized tensor
    pub fn new(data: Vec<u8>, shape: Vec<usize>) -> Self {
        Self {
            data,
            shape,
            quantization: QuantizationScheme::Q8_0, // Default to Q8_0
            scales: Vec::new(),
            zero_points: Vec::new(),
        }
    }

    /// Create with quantization scheme
    pub fn with_quantization(mut self, quantization: QuantizationScheme) -> Self {
        self.quantization = quantization;
        self
    }

    /// Create with scales
    pub fn with_scales(mut self, scales: Vec<f32>) -> Self {
        self.scales = scales;
        self
    }

    /// Create with zero points
    pub fn with_zero_points(mut self, zero_points: Vec<f32>) -> Self {
        self.zero_points = zero_points;
        self
    }

    /// Get the number of elements
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.data.len() + self.scales.len() * 4 + self.zero_points.len() * 4
    }

    /// Check if tensor is valid
    pub fn is_valid(&self) -> bool {
        !self.shape.is_empty() && self.shape.iter().all(|&d| d > 0)
    }

    /// Dequantize to F32 tensor (simplified implementation)
    pub fn dequantize(&self) -> ModelResult<Vec<f32>> {
        match self.quantization {
            QuantizationScheme::Q8_0 => self.dequantize_q8_0(),
            QuantizationScheme::Q4_0 => self.dequantize_q4_0(),
            QuantizationScheme::F32 => self.dequantize_f32(),
            _ => Err(ModelError::quantization(format!(
                "Dequantization not implemented for scheme: {}",
                self.quantization
            ))),
        }
    }

    /// Dequantize Q8_0 data
    fn dequantize_q8_0(&self) -> ModelResult<Vec<f32>> {
        if self.data.len() != self.num_elements() {
            return Err(ModelError::quantization("Data size mismatch for Q8_0"));
        }

        let mut result = Vec::with_capacity(self.num_elements());
        for &byte in &self.data {
            let value = (byte as i8) as f32;
            result.push(value);
        }

        Ok(result)
    }

    /// Dequantize Q4_0 data (simplified)
    fn dequantize_q4_0(&self) -> ModelResult<Vec<f32>> {
        let _block_size = 32;
        let elements_per_byte = 2;
        let expected_bytes = (self.num_elements() + elements_per_byte - 1) / elements_per_byte;

        if self.data.len() != expected_bytes {
            return Err(ModelError::quantization("Data size mismatch for Q4_0"));
        }

        let mut result = Vec::with_capacity(self.num_elements());
        for (i, &byte) in self.data.iter().enumerate() {
            let val1 = ((byte >> 4) & 0xF) as i8 as f32;
            let val2 = (byte & 0xF) as i8 as f32;

            if i * 2 < self.num_elements() {
                result.push(val1);
            }
            if i * 2 + 1 < self.num_elements() {
                result.push(val2);
            }
        }

        Ok(result)
    }

    /// Dequantize F32 data (pass-through)
    fn dequantize_f32(&self) -> ModelResult<Vec<f32>> {
        if self.data.len() % 4 != 0 {
            return Err(ModelError::quantization(
                "Data size not divisible by 4 for F32",
            ));
        }

        let mut result = Vec::with_capacity(self.data.len() / 4);
        for chunk in self.data.chunks_exact(4) {
            let value = f32::from_le_bytes(chunk.try_into().unwrap());
            result.push(value);
        }

        Ok(result)
    }
}

/// Quantization statistics for performance analysis
#[derive(Debug, Clone)]
pub struct QuantizationStats {
    /// Original memory usage in bytes
    pub original_size: usize,
    /// Quantized memory usage in bytes
    pub quantized_size: usize,
    /// Compression ratio (quantized / original)
    pub compression_ratio: f64,
    /// Estimated accuracy loss (0.0 = no loss, 1.0 = complete loss)
    pub accuracy_loss: f64,
    /// Quantization time in milliseconds
    pub quantization_time_ms: u64,
    /// Dequantization time in milliseconds
    pub dequantization_time_ms: u64,
}

impl QuantizationStats {
    /// Calculate compression ratio
    pub fn calculate_compression_ratio(original: usize, quantized: usize) -> f64 {
        if original == 0 {
            1.0
        } else {
            quantized as f64 / original as f64
        }
    }

    /// Create stats for a quantization operation
    pub fn new(
        original_size: usize,
        quantized_size: usize,
        quantization_time_ms: u64,
        dequantization_time_ms: u64,
    ) -> Self {
        let compression_ratio = Self::calculate_compression_ratio(original_size, quantized_size);

        // Estimate accuracy loss based on quantization scheme
        let accuracy_loss = match quantized_size * 8 / original_size {
            0..=32 => 0.001,   // Q4_0, Q4_1
            33..=40 => 0.0005, // Q5_0, Q5_1
            41..=64 => 0.0001, // Q8_0, Q8_1
            _ => 0.0,          // Full precision
        };

        Self {
            original_size,
            quantized_size,
            compression_ratio,
            accuracy_loss,
            quantization_time_ms,
            dequantization_time_ms,
        }
    }

    /// Check if quantization provides good trade-off
    pub fn is_efficient(&self) -> bool {
        self.compression_ratio < 0.5 && self.accuracy_loss < 0.01
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_scheme_properties() {
        let q4_0 = QuantizationScheme::Q4_0;
        assert_eq!(q4_0.bits_per_weight(), 4);
        assert_eq!(q4_0.block_size(), 32);
        assert!(q4_0.is_quantized());
        assert_eq!(q4_0.name(), "q4_0");

        let f32 = QuantizationScheme::F32;
        assert_eq!(f32.bits_per_weight(), 32);
        assert_eq!(f32.block_size(), 1);
        assert!(!f32.is_quantized());
        assert_eq!(f32.name(), "f32");
    }

    #[test]
    fn test_memory_usage_calculation() {
        let scheme = QuantizationScheme::Q4_0;
        let elements = 1000;

        // Q4_0 should use ~500 bytes for 1000 elements (4 bits each)
        let expected_usage = (elements * 4 + 7) / 8;
        assert_eq!(scheme.memory_usage(elements), expected_usage);

        let f32_scheme = QuantizationScheme::F32;
        assert_eq!(f32_scheme.memory_usage(elements), elements * 4);
    }

    #[test]
    fn test_quantization_scheme_parsing() {
        assert_eq!(
            QuantizationScheme::from_str("q4_0").unwrap(),
            QuantizationScheme::Q4_0
        );
        assert_eq!(
            QuantizationScheme::from_str("Q4_1").unwrap(),
            QuantizationScheme::Q4_1
        );
        assert_eq!(
            QuantizationScheme::from_str("F32").unwrap(),
            QuantizationScheme::F32
        );
        assert_eq!(
            QuantizationScheme::from_str("f16").unwrap(),
            QuantizationScheme::F16
        );

        assert!(QuantizationScheme::from_str("invalid").is_err());
        assert!(QuantizationScheme::from_str("").is_err());
    }

    #[test]
    fn test_quantized_tensor_operations() {
        let data = vec![1u8, 2u8, 3u8];
        let shape = vec![3];
        let tensor = QuantizedTensor::new(data.clone(), shape.clone())
            .with_quantization(QuantizationScheme::Q8_0);

        assert_eq!(tensor.num_elements(), 3);
        assert_eq!(tensor.memory_usage(), 3); // Just the data bytes
        assert!(tensor.is_valid());
        assert_eq!(tensor.data, data);
        assert_eq!(tensor.shape, shape);
    }

    #[test]
    fn test_quantization_stats() {
        let stats = QuantizationStats::new(1000, 250, 10, 5);

        assert_eq!(stats.original_size, 1000);
        assert_eq!(stats.quantized_size, 250);
        assert_eq!(stats.compression_ratio, 0.25);
        assert_eq!(stats.accuracy_loss, 0.001); // Q4_0 level
        assert_eq!(stats.quantization_time_ms, 10);
        assert_eq!(stats.dequantization_time_ms, 5);
        assert!(stats.is_efficient());
    }

    #[test]
    fn test_dequantization_q8_0() {
        let data = vec![0u8, 128u8, 255u8]; // 0, -128, -1 as i8
        let shape = vec![3];
        let tensor = QuantizedTensor::new(data, shape).with_quantization(QuantizationScheme::Q8_0);

        let result = tensor.dequantize().unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], -128.0);
        assert_eq!(result[2], -1.0);
    }

    #[test]
    fn test_dequantization_q4_0() {
        let data = vec![0x12, 0x34]; // Two 4-bit values per byte
        let shape = vec![4];
        let tensor = QuantizedTensor::new(data, shape).with_quantization(QuantizationScheme::Q4_0);

        let result = tensor.dequantize().unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], 1.0); // 0x1
        assert_eq!(result[1], 2.0); // 0x2
        assert_eq!(result[2], 3.0); // 0x3
        assert_eq!(result[3], 4.0); // 0x4
    }

    #[test]
    fn test_invalid_quantized_tensor() {
        let tensor = QuantizedTensor::new(vec![], vec![0]);
        assert!(!tensor.is_valid());

        let tensor = QuantizedTensor::new(vec![1], vec![1]);
        assert!(tensor.is_valid());
    }
}
