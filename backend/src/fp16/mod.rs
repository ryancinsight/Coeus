use half::f16;
use coeus_dtype::{Dtype, Float};

pub trait HalfOps<D: Dtype + Float> {
    fn to_half(&self, data: &[D]) -> Vec<f16>;
    fn from_half(&self, data: &[f16]) -> Vec<D>;
    fn half_mul(&self, a: &[f16], b: &[f16]) -> Vec<f16>;
}

// Impl for f32
impl HalfOps<f32> {
    fn to_half(data: &[f32]) -> Vec<f16> {
        data.iter().map(|&x| f16::from_f32(x)).collect()
    }

    fn from_half(data: &[f16]) -> Vec<f32> {
        data.iter().map(|&x| f32::from(x)).collect()
    }

    // ...half_mul via f16 ops...
}
