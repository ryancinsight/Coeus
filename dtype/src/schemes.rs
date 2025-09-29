//! Quantization scheme implementations for different data types.
//!
//! This module provides concrete implementations of advanced quantization schemes
//! for different integer types, following GGUF/llama.cpp standards.

use crate::quantization::{AdvancedQuantizedDtype, QuantizationSchemeTrait, Q4_0Scheme, Q8_0Scheme, BLOCK_SIZE, packed};

/// Implement AdvancedQuantizedDtype for i8 (Q8_0, Q4_0 support)
impl AdvancedQuantizedDtype for i8 {
    type Scheme = Q8_0Scheme;

    fn scheme() -> Self::Scheme {
        Q8_0Scheme::new()
    }

    fn quantize_block(data: &[f32], scheme: &Self::Scheme) -> Vec<Self> {
        data.iter()
            .map(|&x| {
                let quantized = scheme.quantize_value(x);
                quantized as i8
            })
            .collect()
    }

    fn dequantize_block(data: &[Self], scheme: &Self::Scheme) -> Vec<f32> {
        data.iter()
            .map(|&x| scheme.dequantize_value(x as f32))
            .collect()
    }
}

/// Implement AdvancedQuantizedDtype for u8 (Q8_0, Q4_0 support)
impl AdvancedQuantizedDtype for u8 {
    type Scheme = Q8_0Scheme;

    fn scheme() -> Self::Scheme {
        Q8_0Scheme::new()
    }

    fn quantize_block(data: &[f32], scheme: &Self::Scheme) -> Vec<Self> {
        data.iter()
            .map(|&x| {
                let quantized = scheme.quantize_value(x);
                quantized as u8
            })
            .collect()
    }

    fn dequantize_block(data: &[Self], scheme: &Self::Scheme) -> Vec<f32> {
        data.iter()
            .map(|&x| scheme.dequantize_value(x as f32))
            .collect()
    }
}

/// 4-bit quantization storage using packed representation
pub struct Q4Tensor {
    /// Packed 4-bit data (2 values per byte)
    pub data: Vec<u8>,
    /// Scales for each block
    pub scales: Vec<f32>,
    /// Zero points for each block
    pub zero_points: Vec<f32>,
    /// Original shape
    pub shape: Vec<usize>,
}

impl Q4Tensor {
    /// Create new Q4 tensor with calibration
    pub fn quantize(data: &[f32], shape: &[usize]) -> Self {
        let block_size = BLOCK_SIZE;
        let num_blocks = data.len().div_ceil(block_size);

        let mut packed_data = Vec::new();
        let mut scales = Vec::with_capacity(num_blocks);
        let mut zero_points = Vec::with_capacity(num_blocks);

        // Process each block
        for chunk in data.chunks(block_size) {
            let scheme = Q4_0Scheme::calibrate(chunk);
            scales.push(scheme.scale());
            zero_points.push(scheme.zero_point());

            // Quantize block to 4-bit values
            let quantized_values: Vec<u8> = chunk
                .iter()
                .map(|&x| {
                    let q = scheme.quantize_value(x);
                    q as u8 // 4-bit value stored in u8
                })
                .collect();

            // Pack 4-bit values into bytes
            let packed_block = packed::pack_q4_0(&quantized_values);
            packed_data.extend(packed_block);
        }

        Self {
            data: packed_data,
            scales,
            zero_points,
            shape: shape.to_vec(),
        }
    }

    /// Dequantize back to f32
    pub fn dequantize(&self) -> Vec<f32> {
        let mut result = Vec::new();
        let block_size = BLOCK_SIZE;

        for (block_idx, chunk) in self.data.chunks(block_size / 2).enumerate() {
            if block_idx >= self.scales.len() {
                break;
            }

            let scheme = Q4_0Scheme::with_params(
                self.scales[block_idx],
                self.zero_points[block_idx],
            );

            // Unpack 4-bit values
            let unpacked = packed::unpack_q4_0(chunk, block_size.min(self.shape.iter().product::<usize>() - block_idx * block_size));

            // Dequantize
            let dequantized: Vec<f32> = unpacked
                .iter()
                .map(|&x| scheme.dequantize_value(x as f32))
                .collect();

            result.extend(dequantized);
        }

        result
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.data.len() + self.scales.len() * 4 + self.zero_points.len() * 4
    }

    /// Get compression ratio vs f32
    pub fn compression_ratio(&self) -> f64 {
        let original_size: usize = self.shape.iter().product::<usize>() * 4;
        self.memory_usage() as f64 / original_size as f64
    }
}

/// 5-bit quantization storage using packed representation
pub struct Q5Tensor {
    /// Packed 5-bit data
    pub data: Vec<u8>,
    /// Scales for each block
    pub scales: Vec<f32>,
    /// Zero points for each block
    pub zero_points: Vec<f32>,
    /// Original shape
    pub shape: Vec<usize>,
}

impl Q5Tensor {
    /// Create new Q5 tensor with calibration
    pub fn quantize(data: &[f32], shape: &[usize]) -> Self {
        let block_size = BLOCK_SIZE;
        let num_blocks = data.len().div_ceil(block_size);

        let mut packed_data = Vec::new();
        let mut scales = Vec::with_capacity(num_blocks);
        let mut zero_points = Vec::with_capacity(num_blocks);

        // Process each block
        for chunk in data.chunks(block_size) {
            // Use Q4_0 scheme as base (can be extended to Q5_0)
            let scheme = Q4_0Scheme::calibrate(chunk);
            scales.push(scheme.scale());
            zero_points.push(scheme.zero_point());

            // Quantize block to 5-bit values (using 4-bit for now, can be extended)
            let quantized_values: Vec<u8> = chunk
                .iter()
                .map(|&x| {
                    let q = scheme.quantize_value(x);
                    (q as u8).min(31) // 5-bit range
                })
                .collect();

            // Pack 5-bit values
            let packed_block = packed::pack_q5_0(&quantized_values);
            packed_data.extend(packed_block);
        }

        Self {
            data: packed_data,
            scales,
            zero_points,
            shape: shape.to_vec(),
        }
    }

    /// Dequantize back to f32
    pub fn dequantize(&self) -> Vec<f32> {
        let mut result = Vec::new();
        let block_size = BLOCK_SIZE;

        for (block_idx, chunk) in self.data.chunks(block_size * 5 / 8).enumerate() {
            if block_idx >= self.scales.len() {
                break;
            }

            let scheme = Q4_0Scheme::with_params(
                self.scales[block_idx],
                self.zero_points[block_idx],
            );

            // Unpack 5-bit values
            let unpacked = packed::unpack_q5_0(chunk, block_size.min(self.shape.iter().product::<usize>() - block_idx * block_size));

            // Dequantize
            let dequantized: Vec<f32> = unpacked
                .iter()
                .map(|&x| scheme.dequantize_value(x as f32))
                .collect();

            result.extend(dequantized);
        }

        result
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.data.len() + self.scales.len() * 4 + self.zero_points.len() * 4
    }

    /// Get compression ratio vs f32
    pub fn compression_ratio(&self) -> f64 {
        let original_size: usize = self.shape.iter().product::<usize>() * 4;
        self.memory_usage() as f64 / original_size as f64
    }
}

/// 8-bit quantization storage
pub struct Q8Tensor {
    /// 8-bit quantized data
    pub data: Vec<i8>,
    /// Scales for each block
    pub scales: Vec<f32>,
    /// Zero points for each block
    pub zero_points: Vec<f32>,
    /// Original shape
    pub shape: Vec<usize>,
}

impl Q8Tensor {
    /// Create new Q8 tensor with calibration
    pub fn quantize(data: &[f32], shape: &[usize]) -> Self {
        let block_size = BLOCK_SIZE;
        let num_blocks = data.len().div_ceil(block_size);

        let mut quantized_data = Vec::with_capacity(data.len());
        let mut scales = Vec::with_capacity(num_blocks);
        let mut zero_points = Vec::with_capacity(num_blocks);

        // Process each block
        for chunk in data.chunks(block_size) {
            let scheme = Q8_0Scheme::calibrate(chunk);
            scales.push(scheme.scale());
            zero_points.push(scheme.zero_point());

            // Quantize block
            let quantized_block: Vec<i8> = chunk
                .iter()
                .map(|&x| scheme.quantize_value(x) as i8)
                .collect();

            quantized_data.extend(quantized_block);
        }

        Self {
            data: quantized_data,
            scales,
            zero_points,
            shape: shape.to_vec(),
        }
    }

    /// Dequantize back to f32
    pub fn dequantize(&self) -> Vec<f32> {
        let mut result = Vec::new();
        let block_size = BLOCK_SIZE;

        for (block_idx, chunk) in self.data.chunks(block_size).enumerate() {
            if block_idx >= self.scales.len() {
                break;
            }

            let scheme = Q8_0Scheme::with_params(
                self.scales[block_idx],
                self.zero_points[block_idx],
            );

            // Dequantize block
            let dequantized: Vec<f32> = chunk
                .iter()
                .map(|&x| scheme.dequantize_value(x as f32))
                .collect();

            result.extend(dequantized);
        }

        result
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.data.len() + self.scales.len() * 4 + self.zero_points.len() * 4
    }

    /// Get compression ratio vs f32
    pub fn compression_ratio(&self) -> f64 {
        let original_size: usize = self.shape.iter().product::<usize>() * 4;
        self.memory_usage() as f64 / original_size as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_tensor_round_trip() {
        let original_data = vec![0.0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -1.0];
        let shape = vec![8];

        let q4_tensor = Q4Tensor::quantize(&original_data, &shape);
        let dequantized = q4_tensor.dequantize();

        // Check basic properties
        assert_eq!(dequantized.len(), original_data.len());
        assert!(q4_tensor.memory_usage() < original_data.len() * 4);
        assert!(q4_tensor.compression_ratio() < 1.0);

        // Check reasonable accuracy (4-bit quantization has limited precision)
        for (&orig, &deq) in original_data.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            assert!(error < 0.2, "Large error: {} for {} -> {}", error, orig, deq);
        }
    }

    #[test]
    fn test_q8_tensor_round_trip() {
        let original_data = vec![0.0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -1.0];
        let shape = vec![8];

        let q8_tensor = Q8Tensor::quantize(&original_data, &shape);
        let dequantized = q8_tensor.dequantize();

        // Check basic properties
        assert_eq!(dequantized.len(), original_data.len());
        assert!(q8_tensor.memory_usage() < original_data.len() * 4);

        // Check high accuracy (8-bit quantization)
        for (&orig, &deq) in original_data.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            assert!(error < 0.01, "Large error: {} for {} -> {}", error, orig, deq);
        }
    }

    #[test]
    fn test_compression_ratios() {
        let data = vec![0.0; 1024];
        let shape = vec![1024];

        let q4_tensor = Q4Tensor::quantize(&data, &shape);
        let q8_tensor = Q8Tensor::quantize(&data, &shape);

        // Q4 should compress better than Q8
        assert!(q4_tensor.compression_ratio() < q8_tensor.compression_ratio());
        assert!(q4_tensor.compression_ratio() < 0.5); // At least 50% compression
        assert!(q8_tensor.compression_ratio() < 0.75); // At least 25% compression
    }

    #[test]
    fn test_advanced_quantized_dtype_trait() {
        let scheme = i8::scheme();
        assert_eq!(scheme.scale(), 1.0 / 127.0);

        let test_data = vec![0.0, 0.5, -0.5];
        let quantized = i8::quantize_block(&test_data, &scheme);
        let dequantized = i8::dequantize_block(&quantized, &scheme);

        assert_eq!(quantized.len(), test_data.len());
        assert_eq!(dequantized.len(), test_data.len());

        // Check reasonable accuracy
        for (&orig, &deq) in test_data.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            assert!(error < 0.1);
        }
    }
}
