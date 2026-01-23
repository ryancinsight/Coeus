//! Symmetric quantization algorithms
//!
//! Symmetric quantization uses a single scale parameter and assumes zero_point = 0.
//! This is simpler and often more efficient than asymmetric quantization.


use dtype::DataType;
use num_traits::{Float, Zero};

/// Symmetric quantizer that uses only a scale parameter
#[derive(Debug, Clone)]
pub struct SymmetricQuantizer<T: DataType> {
    /// Scale factor for quantization
    pub scale: T,
    /// Number of bits for quantization
    pub bits: usize,
}

impl<T> SymmetricQuantizer<T>
where
    T: DataType + Float + Zero + Clone,
{
    /// Create a new symmetric quantizer
    pub fn new(scale: T, bits: usize) -> Self {
        Self { scale, bits }
    }

    /// Compute scale from min/max values
    pub fn compute_scale(min_val: T, max_val: T, bits: usize) -> T {
        let abs_max = T::max(min_val.abs(), max_val.abs());
        let qmax = T::from((1i64 << (bits - 1)) - 1).unwrap();
        if abs_max > T::zero() {
            abs_max / qmax
        } else {
            T::one()
        }
    }

    /// Quantize a single value
    pub fn quantize_value(&self, value: T) -> i32 {
        let quantized = (value / self.scale).round();
        let qmin = -(1i32 << (self.bits - 1));
        let qmax = (1i32 << (self.bits - 1)) - 1;
        
        quantized.to_i32().unwrap_or(0).clamp(qmin, qmax)
    }

    /// Dequantize a single value
    pub fn dequantize_value(&self, quantized: i32) -> T {
        T::from(quantized).unwrap() * self.scale
    }

    /// Quantize a slice of values
    pub fn quantize(&self, input: &[T]) -> Vec<i32> {
        input.iter().map(|&x| self.quantize_value(x)).collect()
    }

    /// Dequantize a slice of values
    pub fn dequantize(&self, input: &[i32]) -> Vec<T> {
        input.iter().map(|&x| self.dequantize_value(x)).collect()
    }
}