//! Half-precision floating point operations

use half::{f16, bf16};

/// Operations for f16 and bf16 types
pub struct HalfOps;

impl HalfOps {
    /// Convert f32 to f16
    pub fn f32_to_f16(value: f32) -> f16 {
        f16::from_f32(value)
    }

    /// Convert f16 to f32
    pub fn f16_to_f32(value: f16) -> f32 {
        value.to_f32()
    }

    /// Convert f32 to bf16
    pub fn f32_to_bf16(value: f32) -> bf16 {
        bf16::from_f32(value)
    }

    /// Convert bf16 to f32
    pub fn bf16_to_f32(value: bf16) -> f32 {
        value.to_f32()
    }
}
