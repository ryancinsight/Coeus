//! # Quantized Data Types
//!
//! Affine quantization types for model compression and efficient inference.
//!
//! ## Affine Quantization Formula
//!
//! Quantization: `q = round((x - zero_point) / scale)`
//! Dequantization: `x = q * scale + zero_point`
//!
//! Where:
//! - `q`: quantized value (stored in memory)
//! - `x`: original floating-point value
//! - `scale`: quantization scale factor
//! - `zero_point`: quantization zero point offset

use core::fmt;
use serde::{Deserialize, Serialize};

/// Affine quantized 8-bit signed integer with scale and zero point
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QInt8 {
    /// The quantized value stored in memory (typically -128 to 127)
    pub value: i8,
    /// Quantization scale factor
    pub scale: f32,
    /// Quantization zero point offset
    pub zero_point: i8,
}

impl QInt8 {
    /// Create a new `QInt8` from a quantized value
    #[must_use]
    pub const fn new(value: i8, scale: f32, zero_point: i8) -> Self {
        Self {
            value,
            scale,
            zero_point,
        }
    }

    /// Quantize a floating-point value to `QInt8`
    ///
    /// # Arguments
    /// * `x` - The floating-point value to quantize
    /// * `scale` - Quantization scale factor
    /// * `zero_point` - Quantization zero point offset
    ///
    /// # Returns
    /// Quantized `QInt8` value
    #[must_use]
    pub fn quantize(x: f32, scale: f32, zero_point: i8) -> Self {
        // q = round((x - zero_point) / scale)
        let quantized = ((x - f32::from(zero_point)) / scale).round();
        let clamped = quantized.clamp(f32::from(i8::MIN), f32::from(i8::MAX));
        #[allow(clippy::cast_possible_truncation)]
        Self::new(clamped as i8, scale, zero_point)
    }

    /// Dequantize to floating-point value
    ///
    /// # Returns
    /// Dequantized floating-point value
    #[must_use]
    pub fn dequantize(self) -> f32 {
        // x = q * scale + zero_point
        f32::from(self.value) * self.scale + f32::from(self.zero_point)
    }

    /// Get the quantized value range for this type
    #[must_use]
    pub const fn quantized_range() -> (i8, i8) {
        (i8::MIN, i8::MAX)
    }

    /// Get the representable floating-point range
    ///
    /// # Returns
    /// (`min_value`, `max_value`) tuple
    #[must_use]
    pub fn float_range(self) -> (f32, f32) {
        let min_q = f32::from(i8::MIN);
        let max_q = f32::from(i8::MAX);
        (
            min_q * self.scale + f32::from(self.zero_point),
            max_q * self.scale + f32::from(self.zero_point),
        )
    }
}

/// Affine quantized 8-bit unsigned integer with scale and zero point
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QUInt8 {
    /// The quantized value stored in memory (typically 0 to 255)
    pub value: u8,
    /// Quantization scale factor
    pub scale: f32,
    /// Quantization zero point offset
    pub zero_point: u8,
}

impl QUInt8 {
    /// Create a new `QUInt8` from a quantized value
    #[must_use]
    pub const fn new(value: u8, scale: f32, zero_point: u8) -> Self {
        Self {
            value,
            scale,
            zero_point,
        }
    }

    /// Quantize a floating-point value to `QUInt8`
    ///
    /// # Arguments
    /// * `x` - The floating-point value to quantize
    /// * `scale` - Quantization scale factor
    /// * `zero_point` - Quantization zero point offset
    ///
    /// # Returns
    /// Quantized `QUInt8` value
    #[must_use]
    pub fn quantize(x: f32, scale: f32, zero_point: u8) -> Self {
        // q = round((x - zero_point) / scale)
        let quantized = ((x - f32::from(zero_point)) / scale).round();
        let clamped = quantized.clamp(f32::from(u8::MIN), f32::from(u8::MAX));
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Self::new(clamped as u8, scale, zero_point)
    }

    /// Dequantize to floating-point value
    ///
    /// # Returns
    /// Dequantized floating-point value
    #[must_use]
    pub fn dequantize(self) -> f32 {
        // x = q * scale + zero_point
        f32::from(self.value) * self.scale + f32::from(self.zero_point)
    }

    /// Get the quantized value range for this type
    #[must_use]
    pub const fn quantized_range() -> (u8, u8) {
        (u8::MIN, u8::MAX)
    }

    /// Get the representable floating-point range
    ///
    /// # Returns
    /// (`min_value`, `max_value`) tuple
    #[must_use]
    pub fn float_range(self) -> (f32, f32) {
        let min_q = f32::from(u8::MIN);
        let max_q = f32::from(u8::MAX);
        (
            min_q * self.scale + f32::from(self.zero_point),
            max_q * self.scale + f32::from(self.zero_point),
        )
    }
}

/// 4-bit signed integer quantization with packing
///
/// Since most hardware doesn't support 4-bit types natively, we pack
/// two `QInt4` values into a single byte for efficient storage.
/// The first value occupies bits 0-3, the second value occupies bits 4-7.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QInt4 {
    /// Packed 4-bit values (two `QInt4` values per byte)
    pub packed_value: i8,
    /// Which 4-bit slot this value occupies (0 = low 4 bits, 1 = high 4 bits)
    pub slot: u8,
    /// Quantization scale factor
    pub scale: f32,
    /// Quantization zero point offset
    pub zero_point: i8,
}

impl QInt4 {
    /// Create a new `QInt4` from a packed value and slot
    #[must_use]
    pub const fn new(packed_value: i8, slot: u8, scale: f32, zero_point: i8) -> Self {
        Self {
            packed_value,
            slot,
            scale,
            zero_point,
        }
    }

    /// Quantize a floating-point value to `QInt4`
    ///
    /// # Arguments
    /// * `x` - The floating-point value to quantize
    /// * `scale` - Quantization scale factor
    /// * `zero_point` - Quantization zero point offset
    ///
    /// # Returns
    /// Quantized `QInt4` value (packed into low 4 bits, slot 0)
    #[must_use]
    pub fn quantize(x: f32, scale: f32, zero_point: i8) -> Self {
        #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
        let quantized = ((x - f32::from(zero_point)) / scale).round() as i8;
        let clamped = quantized.clamp(-8, 7); // 4-bit signed range: -8 to 7
        Self::new(clamped, 0, scale, zero_point)
    }

    /// Dequantize this `QInt4` value back to floating-point
    #[must_use]
    pub fn dequantize(&self) -> f32 {
        let value = if self.slot == 0 {
            self.packed_value & 0x0F
        } else {
            (self.packed_value >> 4) & 0x0F
        };

        // Sign extend 4-bit value to 8-bit
        let signed_value = if value & 0x08 != 0 {
            value | !0x0F
        } else {
            value
        };

        f32::from(signed_value) * self.scale + f32::from(self.zero_point)
    }

    /// Pack two `QInt4` values into a single byte
    ///
    /// # Arguments
    /// * `low` - The low 4-bit value (slot 0)
    /// * `high` - The high 4-bit value (slot 1)
    ///
    /// # Returns
    /// Packed byte containing both values
    #[must_use]
    pub fn pack(low: i8, high: i8) -> i8 {
        ((high & 0x0F) << 4) | (low & 0x0F)
    }

    /// Unpack a byte into two `QInt4` values
    ///
    /// # Arguments
    /// * `packed` - The packed byte
    ///
    /// # Returns
    /// Tuple of (`low_4bit_value`, `high_4bit_value`)
    #[must_use]
    pub fn unpack(packed: i8) -> (i8, i8) {
        let low = packed & 0x0F;
        let high = (packed >> 4) & 0x0F;
        (low, high)
    }
}

/// 4-bit unsigned integer quantization with packing
///
/// Similar to `QInt4` but for unsigned values, packing two `QUInt4`
/// values into a single byte for efficient storage.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QUInt4 {
    /// Packed 4-bit values (two `QUInt4` values per byte)
    pub packed_value: u8,
    /// Which 4-bit slot this value occupies (0 = low 4 bits, 1 = high 4 bits)
    pub slot: u8,
    /// Quantization scale factor
    pub scale: f32,
    /// Quantization zero point offset
    pub zero_point: u8,
}

impl QUInt4 {
    /// Create a new `QUInt4` from a packed value and slot
    #[must_use]
    pub const fn new(packed_value: u8, slot: u8, scale: f32, zero_point: u8) -> Self {
        Self {
            packed_value,
            slot,
            scale,
            zero_point,
        }
    }

    /// Quantize a floating-point value to `QUInt4`
    ///
    /// # Arguments
    /// * `x` - The floating-point value to quantize
    /// * `scale` - Quantization scale factor
    /// * `zero_point` - Quantization zero point offset
    ///
    /// # Returns
    /// Quantized `QUInt4` value (packed into low 4 bits, slot 0)
    #[must_use]
    pub fn quantize(x: f32, scale: f32, zero_point: u8) -> Self {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_lossless
        )]
        let quantized = ((x - f32::from(zero_point)) / scale).round() as u8;
        let clamped = quantized.min(15); // 4-bit unsigned range: 0 to 15
        Self::new(clamped, 0, scale, zero_point)
    }

    /// Dequantize this `QUInt4` value back to floating-point
    #[must_use]
    pub fn dequantize(&self) -> f32 {
        let value = if self.slot == 0 {
            self.packed_value & 0x0F
        } else {
            (self.packed_value >> 4) & 0x0F
        };

        f32::from(value) * self.scale + f32::from(self.zero_point)
    }

    /// Pack two `QUInt4` values into a single byte
    ///
    /// # Arguments
    /// * `low` - The low 4-bit value (slot 0)
    /// * `high` - The high 4-bit value (slot 1)
    ///
    /// # Returns
    /// Packed byte containing both values
    #[must_use]
    pub fn pack(low: u8, high: u8) -> u8 {
        ((high & 0x0F) << 4) | (low & 0x0F)
    }

    /// Unpack a byte into two `QUInt4` values
    ///
    /// # Arguments
    /// * `packed` - The packed byte
    ///
    /// # Returns
    /// Tuple of (`low_4bit_value`, `high_4bit_value`)
    #[must_use]
    pub fn unpack(packed: u8) -> (u8, u8) {
        let low = packed & 0x0F;
        let high = (packed >> 4) & 0x0F;
        (low, high)
    }
}

/// Trait for types that support quantization operations
///
/// This trait provides the basic interface for affine quantization,
/// which converts floating-point values to quantized representations
/// and back using scale and zero point parameters.
pub trait QuantizedType {
    /// Quantize a floating-point value
    fn quantize(x: f32, scale: f32, zero_point: Self) -> Self
    where
        Self: Sized;

    /// Dequantize to floating-point value
    fn dequantize(&self) -> f32;

    /// Get the representable floating-point range
    fn float_range(&self) -> (f32, f32);
}

impl QuantizedType for QInt8 {
    fn quantize(x: f32, scale: f32, zero_point: Self) -> Self {
        // q = round((x - zero_point) / scale)
        let zero_point_f32 = f32::from(zero_point.value);
        let quantized = ((x - zero_point_f32) / scale).round();
        let clamped = quantized.clamp(f32::from(i8::MIN), f32::from(i8::MAX));
        #[allow(clippy::cast_possible_truncation)]
        Self::new(clamped as i8, scale, zero_point.value)
    }

    fn dequantize(&self) -> f32 {
        // x = q * scale + zero_point
        f32::from(self.value) * self.scale + f32::from(self.zero_point)
    }

    fn float_range(&self) -> (f32, f32) {
        let min_q = f32::from(i8::MIN);
        let max_q = f32::from(i8::MAX);
        (
            min_q * self.scale + f32::from(self.zero_point),
            max_q * self.scale + f32::from(self.zero_point),
        )
    }
}

impl QuantizedType for QUInt8 {
    fn quantize(x: f32, scale: f32, zero_point: Self) -> Self {
        // q = round((x - zero_point) / scale)
        let zero_point_f32 = f32::from(zero_point.value);
        let quantized = ((x - zero_point_f32) / scale).round();
        let clamped = quantized.clamp(f32::from(u8::MIN), f32::from(u8::MAX));
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Self::new(clamped as u8, scale, zero_point.value)
    }

    fn dequantize(&self) -> f32 {
        // x = q * scale + zero_point
        f32::from(self.value) * self.scale + f32::from(self.zero_point)
    }

    fn float_range(&self) -> (f32, f32) {
        let min_q = f32::from(u8::MIN);
        let max_q = f32::from(u8::MAX);
        (
            min_q * self.scale + f32::from(self.zero_point),
            max_q * self.scale + f32::from(self.zero_point),
        )
    }
}

// Display implementations

impl fmt::Display for QInt8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QInt8({}, scale={}, zero_point={})",
            self.value, self.scale, self.zero_point
        )
    }
}

impl fmt::Display for QUInt8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QUInt8({}, scale={}, zero_point={})",
            self.value, self.scale, self.zero_point
        )
    }
}

impl fmt::Display for QInt4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QInt4({}, slot={}, scale={}, zero_point={})",
            self.packed_value, self.slot, self.scale, self.zero_point
        )
    }
}

impl fmt::Display for QUInt4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QUInt4({}, slot={}, scale={}, zero_point={})",
            self.packed_value, self.slot, self.scale, self.zero_point
        )
    }
}