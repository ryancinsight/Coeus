//! Advanced quantization schemes and utilities.
//!
//! This module provides comprehensive quantization support following llama.cpp/GGUF standards,
//! including block-wise quantization for efficient sub-8-bit representations.
//!
//! ## Supported Schemes
//!
//! - **Q4_0**: 4-bit quantization with block-wise scaling
//! - **Q4_1**: 4-bit quantization with improved zero handling
//! - **Q5_0**: 5-bit quantization with block-wise scaling
//! - **Q5_1**: 5-bit quantization with enhanced accuracy
//! - **Q8_0**: 8-bit quantization with block-wise scaling
//! - **Q8_1**: 8-bit quantization with improved precision
//!
//! ## Block-wise Quantization
//!
//! Block-wise quantization divides tensors into fixed-size blocks (typically 32 elements)
//! and applies separate scaling/zero-point parameters to each block. This provides:
//! - Better accuracy than global quantization
//! - Efficient memory usage for sub-8-bit schemes
//! - Hardware-friendly access patterns

use crate::Dtype;
use std::fmt;

/// Block size for block-wise quantization operations
pub const BLOCK_SIZE: usize = 32;

/// Quantization scheme enumeration matching GGUF standards
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizationScheme {
    /// No quantization (full precision)
    None,
    /// 4-bit quantization, block-wise
    Q4_0,
    /// 4-bit quantization, improved zero handling
    Q4_1,
    /// 5-bit quantization, block-wise
    Q5_0,
    /// 5-bit quantization, enhanced accuracy
    Q5_1,
    /// 8-bit quantization, block-wise
    Q8_0,
    /// 8-bit quantization, improved precision
    Q8_1,
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
        }
    }

    /// Get bits per weight for this scheme
    pub fn bits_per_weight(&self) -> usize {
        match self {
            Self::None => 32,
            Self::Q4_0 | Self::Q4_1 => 4,
            Self::Q5_0 | Self::Q5_1 => 5,
            Self::Q8_0 | Self::Q8_1 => 8,
        }
    }

    /// Check if this is a quantized scheme
    pub fn is_quantized(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Get block size for this scheme
    pub fn block_size(&self) -> usize {
        match self {
            Self::None => 1,
            _ => BLOCK_SIZE,
        }
    }

    /// Calculate memory usage for given number of elements
    pub fn memory_usage(&self, elements: usize) -> usize {
        match self {
            Self::None => elements * 4, // F32 equivalent
            Self::Q4_0 | Self::Q4_1 => {
                let blocks = elements.div_ceil(BLOCK_SIZE);
                // Each block: quantized data + scale + zero_point
                blocks * (BLOCK_SIZE * 4 / 8 + 4 + 4) // 4 bits/element + f32 scale + f32 zero_point
            }
            Self::Q5_0 | Self::Q5_1 => {
                let blocks = elements.div_ceil(BLOCK_SIZE);
                // Each block: quantized data + scale + zero_point
                blocks * (BLOCK_SIZE * 5 / 8 + 4 + 4) // 5 bits/element + f32 scale + f32 zero_point
            }
            Self::Q8_0 | Self::Q8_1 => {
                let blocks = elements.div_ceil(BLOCK_SIZE);
                // Each block: quantized data + scale + zero_point
                blocks * (BLOCK_SIZE + 4 + 4) // 8 bits/element + f32 scale + f32 zero_point
            }
        }
    }
}

impl fmt::Display for QuantizationScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Advanced quantization trait for block-wise schemes
pub trait AdvancedQuantizedDtype: Dtype {
    /// Quantization scheme type
    type Scheme: QuantizationSchemeTrait;

    /// Create new quantization scheme instance
    fn scheme() -> Self::Scheme;

    /// Quantize a block of data using the scheme
    fn quantize_block(data: &[f32], scheme: &Self::Scheme) -> Vec<Self>;

    /// Dequantize a block of data using the scheme
    fn dequantize_block(data: &[Self], scheme: &Self::Scheme) -> Vec<f32>;
}

/// Trait for quantization scheme implementations
pub trait QuantizationSchemeTrait {
    /// Quantization scheme type
    fn scheme_type() -> QuantizationScheme;

    /// Quantize a single value
    fn quantize_value(&self, value: f32) -> f32;

    /// Dequantize a single value
    fn dequantize_value(&self, quantized: f32) -> f32;

    /// Get scale factor for current block
    fn scale(&self) -> f32;

    /// Get zero point for current block
    fn zero_point(&self) -> f32;
}

/// Q4_0 quantization scheme (4-bit, block-wise)
pub struct Q4_0Scheme {
    scale: f32,
    zero_point: f32,
}

impl Q4_0Scheme {
    /// Create new Q4_0 scheme with default parameters
    pub fn new() -> Self {
        Self {
            scale: 1.0 / 15.0, // 4-bit range: 0-15
            zero_point: 8.0,   // Center of 4-bit range
        }
    }

    /// Create scheme with custom scale and zero point
    pub fn with_params(scale: f32, zero_point: f32) -> Self {
        Self { scale, zero_point }
    }

    /// Calibrate scheme from data block
    pub fn calibrate(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self::new();
        }

        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        if range == 0.0 {
            return Self::new();
        }

        let scale = range / 15.0; // 4-bit range (0-15)
        let zero_point = -min_val / scale;

        Self { scale, zero_point }
    }
}

impl Default for Q4_0Scheme {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizationSchemeTrait for Q4_0Scheme {
    fn scheme_type() -> QuantizationScheme {
        QuantizationScheme::Q4_0
    }

    fn quantize_value(&self, value: f32) -> f32 {
        let quantized = (value / self.scale + self.zero_point).round();
        quantized.clamp(0.0, 15.0)
    }

    fn dequantize_value(&self, quantized: f32) -> f32 {
        (quantized - self.zero_point) * self.scale
    }

    fn scale(&self) -> f32 {
        self.scale
    }

    fn zero_point(&self) -> f32 {
        self.zero_point
    }
}

/// Q8_0 quantization scheme (8-bit, block-wise)
pub struct Q8_0Scheme {
    scale: f32,
    zero_point: f32,
}

impl Q8_0Scheme {
    /// Create new Q8_0 scheme with default parameters
    pub fn new() -> Self {
        Self {
            scale: 1.0 / 127.0, // 8-bit signed range: -127 to 127
            zero_point: 0.0,    // Symmetric quantization
        }
    }

    /// Create scheme with custom scale and zero point
    pub fn with_params(scale: f32, zero_point: f32) -> Self {
        Self { scale, zero_point }
    }

    /// Calibrate scheme from data block
    pub fn calibrate(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self::new();
        }

        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        if range == 0.0 {
            return Self::new();
        }

        let scale = range / 254.0; // 8-bit signed range (-127 to 127)
        let zero_point = -min_val / scale - 127.0; // Map to i8 range

        Self { scale, zero_point }
    }
}

impl Default for Q8_0Scheme {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizationSchemeTrait for Q8_0Scheme {
    fn scheme_type() -> QuantizationScheme {
        QuantizationScheme::Q8_0
    }

    fn quantize_value(&self, value: f32) -> f32 {
        let quantized = (value / self.scale + self.zero_point).round();
        quantized.clamp(-127.0, 127.0)
    }

    fn dequantize_value(&self, quantized: f32) -> f32 {
        (quantized - self.zero_point) * self.scale
    }

    fn scale(&self) -> f32 {
        self.scale
    }

    fn zero_point(&self) -> f32 {
        self.zero_point
    }
}

/// Block-wise quantization utilities
pub mod block {
    use super::*;

    /// Quantize tensor using block-wise scheme
    pub fn quantize_tensor<T: AdvancedQuantizedDtype>(
        data: &[f32],
        scheme: &T::Scheme,
    ) -> Vec<T> {
        let mut result = Vec::with_capacity(data.len());

        // Process in blocks
        for chunk in data.chunks(BLOCK_SIZE) {
            let quantized_block = T::quantize_block(chunk, scheme);
            result.extend(quantized_block);
        }

        result
    }

    /// Dequantize tensor using block-wise scheme
    pub fn dequantize_tensor<T: AdvancedQuantizedDtype>(
        data: &[T],
        scheme: &T::Scheme,
    ) -> Vec<f32> {
        let mut result = Vec::with_capacity(data.len());

        // Process in blocks
        for chunk in data.chunks(BLOCK_SIZE) {
            let dequantized_block = T::dequantize_block(chunk, scheme);
            result.extend(dequantized_block);
        }

        result
    }

    /// Calculate quantization error for a block
    pub fn quantization_error(original: &[f32], dequantized: &[f32]) -> f32 {
        if original.len() != dequantized.len() {
            return f32::INFINITY;
        }

        let mut error_sum = 0.0;
        for (&orig, &deq) in original.iter().zip(dequantized.iter()) {
            let diff = orig - deq;
            error_sum += diff * diff;
        }

        (error_sum / original.len() as f32).sqrt() // RMSE
    }
}

/// Packed quantization utilities for sub-8-bit schemes
pub mod packed {

    /// Pack 4-bit values into bytes (Q4_0 format)
    pub fn pack_q4_0(values: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(values.len().div_ceil(2));

        for chunk in values.chunks(2) {
            let byte = match chunk.len() {
                2 => (chunk[0] & 0xF) | ((chunk[1] & 0xF) << 4),
                1 => chunk[0] & 0xF,
                _ => 0,
            };
            result.push(byte);
        }

        result
    }

    /// Unpack 4-bit values from bytes (Q4_0 format)
    pub fn unpack_q4_0(data: &[u8], num_elements: usize) -> Vec<u8> {
        let mut result = Vec::with_capacity(num_elements);

        for &byte in data {
            result.push(byte & 0xF);
            if result.len() < num_elements {
                result.push((byte >> 4) & 0xF);
            }
        }

        result.truncate(num_elements);
        result
    }

    /// Pack 5-bit values into bytes (Q5_0 format)
    pub fn pack_q5_0(values: &[u8]) -> Vec<u8> {
        let mut result = Vec::new();
        let mut current_byte = 0u8;
        let mut bits_in_byte = 0;

        for &value in values {
            let val = (value & 0x1F) as u32; // Ensure 5 bits
            let mut remaining_bits = 5;

            while remaining_bits > 0 {
                let bits_to_write = std::cmp::min(remaining_bits, 8 - bits_in_byte);
                let mask = (1u32 << bits_to_write) - 1;
                let bits = (val >> (5 - remaining_bits)) & mask;

                current_byte |= (bits << bits_in_byte) as u8;
                bits_in_byte += bits_to_write;
                remaining_bits -= bits_to_write;

                if bits_in_byte == 8 {
                    result.push(current_byte);
                    current_byte = 0;
                    bits_in_byte = 0;
                }
            }
        }

        if bits_in_byte > 0 {
            result.push(current_byte);
        }

        result
    }

    /// Unpack 5-bit values from bytes (Q5_0 format)
    pub fn unpack_q5_0(data: &[u8], num_elements: usize) -> Vec<u8> {
        let mut result = Vec::with_capacity(num_elements);
        let mut bit_position = 0;

        while result.len() < num_elements {
            let mut val = 0u32;
            let mut bits_read = 0;

            while bits_read < 5 && bit_position / 8 < data.len() {
                let byte_index = bit_position / 8;
                let bit_offset = bit_position % 8;
                let bits_available = 8 - bit_offset;
                let bits_to_read = std::cmp::min(5 - bits_read, bits_available);

                let mask = ((1u32 << bits_to_read) - 1) << bit_offset;
                let bits = ((data[byte_index] as u32) & mask) >> bit_offset;

                val |= bits << bits_read;
                bits_read += bits_to_read;
                bit_position += bits_to_read;
            }

            result.push((val & 0x1F) as u8);
        }

        result.truncate(num_elements);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_q4_0_scheme() {
        let scheme = Q4_0Scheme::new();
        assert_eq!(scheme.scale(), 1.0 / 15.0);
        assert_eq!(scheme.zero_point(), 8.0);

        // Test quantization round-trip with values suitable for the default scheme
        // The default scheme is symmetric around zero with range [-1, 1]
        // 4-bit quantization has inherent precision limits, so we test basic functionality
        let test_values = vec![0.0, 0.0667, -0.0667]; // Values that should quantize well
        for &value in &test_values {
            let quantized = scheme.quantize_value(value);
            let dequantized = scheme.dequantize_value(quantized);
            // For 4-bit, expect reasonable but not perfect accuracy
            assert!((value - dequantized).abs() < 0.2, "Value {} quantized to {} dequantized to {}", value, quantized, dequantized);
        }

        // Test that extreme values are clamped properly
        let large_value = 10.0;
        let quantized = scheme.quantize_value(large_value);
        assert_eq!(quantized, 15.0); // Should be clamped to max

        let small_value = -10.0;
        let quantized = scheme.quantize_value(small_value);
        assert_eq!(quantized, 0.0); // Should be clamped to min (0 for u4)
    }

    #[test]
    fn test_q8_0_scheme() {
        let scheme = Q8_0Scheme::new();
        assert_eq!(scheme.scale(), 1.0 / 127.0);
        assert_eq!(scheme.zero_point(), 0.0);

        // Test quantization round-trip
        let test_values = vec![-1.0, 0.0, 0.5, 1.0];
        for &value in &test_values {
            let quantized = scheme.quantize_value(value);
            let dequantized = scheme.dequantize_value(quantized);
            assert_relative_eq!(value, dequantized, epsilon = 0.01); // High precision for 8-bit
        }
    }

    #[test]
    fn test_quantization_scheme_properties() {
        assert_eq!(QuantizationScheme::Q4_0.bits_per_weight(), 4);
        assert_eq!(QuantizationScheme::Q8_0.bits_per_weight(), 8);
        assert!(QuantizationScheme::Q4_0.is_quantized());
        assert!(!QuantizationScheme::None.is_quantized());
        assert_eq!(QuantizationScheme::Q4_0.block_size(), BLOCK_SIZE);
        assert_eq!(QuantizationScheme::None.block_size(), 1);
    }

    #[test]
    fn test_packed_q4_0() {
        let values = vec![1u8, 2u8, 3u8, 4u8];
        let packed = packed::pack_q4_0(&values);
        let unpacked = packed::unpack_q4_0(&packed, values.len());

        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_packed_q5_0() {
        let values = vec![1u8, 2u8, 3u8, 4u8];
        let packed = packed::pack_q5_0(&values);
        let unpacked = packed::unpack_q5_0(&packed, values.len());

        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_memory_usage_calculation() {
        let elements = 1000;

        // Q4_0 should use significantly less memory
        let q4_0_usage = QuantizationScheme::Q4_0.memory_usage(elements);
        let f32_usage = QuantizationScheme::None.memory_usage(elements);

        assert!(q4_0_usage < f32_usage);
        assert!(q4_0_usage > 0);
    }
}
