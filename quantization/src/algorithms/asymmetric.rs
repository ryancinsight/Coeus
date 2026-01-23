//! Asymmetric quantization algorithms
//!
//! Asymmetric quantization uses both scale and zero_point parameters,
//! allowing for better representation of asymmetric data distributions.


use dtype::DataType;
use num_traits::{Float, Zero};

/// Asymmetric quantizer that uses scale and zero_point parameters
#[derive(Debug, Clone)]
pub struct AsymmetricQuantizer<T: DataType> {
    /// Scale factor for quantization
    pub scale: T,
    /// Zero point offset
    pub zero_point: i32,
    /// Number of bits for quantization
    pub bits: usize,
}

impl<T> AsymmetricQuantizer<T>
where
    T: DataType + Float + Zero + Clone,
{
    /// Create a new asymmetric quantizer
    pub fn new(scale: T, zero_point: i32, bits: usize) -> Self {
        Self { scale, zero_point, bits }
    }

    /// Compute scale and zero_point from min/max values
    pub fn compute_params(min_val: T, max_val: T, bits: usize) -> (T, i32) {
        let qmin = 0i32;
        let qmax = (1i32 << bits) - 1;
        
        let scale = if max_val > min_val {
            (max_val - min_val) / T::from(qmax - qmin).unwrap()
        } else {
            T::one()
        };
        
        let zero_point_real = qmin as f64 - (min_val / scale).to_f64().unwrap_or(0.0);
        let zero_point = zero_point_real.round().clamp(qmin as f64, qmax as f64) as i32;
        
        (scale, zero_point)
    }

    /// Quantize a single value
    pub fn quantize_value(&self, value: T) -> u8 {
        let quantized = (value / self.scale).round().to_i32().unwrap_or(0) + self.zero_point;
        let qmin = 0i32;
        let qmax = (1i32 << self.bits) - 1;
        
        quantized.clamp(qmin, qmax) as u8
    }

    /// Dequantize a single value
    pub fn dequantize_value(&self, quantized: u8) -> T {
        (T::from(quantized as i32 - self.zero_point).unwrap()) * self.scale
    }

    /// Quantize a slice of values
    pub fn quantize(&self, input: &[T]) -> Vec<u8> {
        input.iter().map(|&x| self.quantize_value(x)).collect()
    }

    /// Dequantize a slice of values
    pub fn dequantize(&self, input: &[u8]) -> Vec<T> {
        input.iter().map(|&x| self.dequantize_value(x)).collect()
    }
}