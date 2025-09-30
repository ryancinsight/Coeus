//! Quantization utilities

/// Operations for quantized types
pub struct QuantOps;

impl QuantOps {
    /// Generic quantization operation
    pub fn quantize<T: crate::QuantizedDtype>(value: f32) -> T {
        T::quantize(value, T::scale(), T::zero_point())
    }

    /// Generic dequantization operation
    pub fn dequantize<T: crate::QuantizedDtype>(value: T) -> f32 {
        value.dequantize(T::scale(), T::zero_point())
    }
}
