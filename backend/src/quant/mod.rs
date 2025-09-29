use coeus_dtype::{Dtype, Float};
use num_traits::cast::NumCast;

/// Quantization operations.
pub trait QuantOps<D: Dtype + Float> {
    fn quantize(&self, data: &[D], scale: D, zero: i32) -> Vec<i8>;
    fn dequantize(&self, data: &[i8], scale: D, zero: i32) -> Vec<D>;
    fn quantized_mul(&self, a: &[i8], b: &[i8], scale_a: D, zero_a: i32, scale_b: D, zero_b: i32) -> Vec<i8>;
}

// Impl for f32 (extend to other D)
impl QuantOps<f32> for Backend {
    fn quantize(&self, data: &[f32], scale: f32, zero: i32) -> Vec<i8> {
        data.iter().map(|&x| ((x / scale) + zero as f32).clamp(-128.0, 127.0) as i8).collect()
    }

    fn dequantize(&self, data: &[i8], scale: f32, zero: i32) -> Vec<f32> {
        data.iter().map(|&x| (x as f32 - zero as f32) * scale).collect()
    }

    // ...quantized_mul via int ops...
}
