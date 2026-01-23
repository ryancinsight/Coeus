//! Dynamic quantization algorithms
//!
//! Dynamic quantization computes quantization parameters at runtime
//! based on the actual data distribution.

use coeus_error::Result;
use dtype::DataType;
use num_traits::{Float, Zero};

use super::{AsymmetricQuantizer, SymmetricQuantizer};

/// Dynamic quantizer that computes parameters at runtime
#[derive(Debug, Clone)]
pub struct DynamicQuantizer<T: DataType> {
    /// Number of bits for quantization
    pub bits: usize,
    /// Whether to use symmetric quantization
    pub symmetric: bool,
    _phantom: core::marker::PhantomData<T>,
}

impl<T> DynamicQuantizer<T>
where
    T: DataType + Float + Zero + Clone + PartialOrd,
{
    /// Create a new dynamic quantizer
    pub fn new(bits: usize, symmetric: bool) -> Self {
        Self {
            bits,
            symmetric,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Quantize data with dynamically computed parameters
    pub fn quantize(&self, input: &[T]) -> Result<(Vec<u8>, T, i32)> {
        if input.is_empty() {
            return Err(coeus_error::Error::Tensor(coeus_error::TensorError::OperationFailed(
                "Cannot quantize empty data".to_string(),
            )));
        }

        let min_val = input.iter().cloned().fold(input[0], T::min);
        let max_val = input.iter().cloned().fold(input[0], T::max);

        if self.symmetric {
            let scale = SymmetricQuantizer::compute_scale(min_val, max_val, self.bits);
            let quantizer = SymmetricQuantizer::new(scale, self.bits);
            let quantized = quantizer.quantize(input);
            // Convert i32 to u8 for consistency (this is a simplification)
            let quantized_u8: Vec<u8> = quantized.iter().map(|&x| (x + 128) as u8).collect();
            Ok((quantized_u8, scale, 0))
        } else {
            let (scale, zero_point) = AsymmetricQuantizer::compute_params(min_val, max_val, self.bits);
            let quantizer = AsymmetricQuantizer::new(scale, zero_point, self.bits);
            let quantized = quantizer.quantize(input);
            Ok((quantized, scale, zero_point))
        }
    }

    /// Dequantize data with given parameters
    pub fn dequantize(&self, quantized: &[u8], scale: T, zero_point: i32) -> Vec<T> {
        if self.symmetric {
            let quantizer = SymmetricQuantizer::new(scale, self.bits);
            // Convert u8 back to i32 (reverse the conversion from quantize)
            let quantized_i32: Vec<i32> = quantized.iter().map(|&x| x as i32 - 128).collect();
            quantizer.dequantize(&quantized_i32)
        } else {
            let quantizer = AsymmetricQuantizer::new(scale, zero_point, self.bits);
            quantizer.dequantize(quantized)
        }
    }

    /// Compute quantization error metrics
    pub fn compute_error(&self, original: &[T], quantized: &[T]) -> (T, T, T) {
        if original.len() != quantized.len() {
            return (T::zero(), T::zero(), T::zero());
        }

        let mut max_error = T::zero();
        let mut sum_squared_error = T::zero();
        let mut sum_abs_error = T::zero();

        for (orig, quant) in original.iter().zip(quantized.iter()) {
            let error = (*orig - *quant).abs();
            max_error = T::max(max_error, error);
            sum_squared_error = sum_squared_error + error * error;
            sum_abs_error = sum_abs_error + error;
        }

        let mse = sum_squared_error / T::from(original.len()).unwrap();
        let mae = sum_abs_error / T::from(original.len()).unwrap();

        (max_error, mse, mae)
    }
}