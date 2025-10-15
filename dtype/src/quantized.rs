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

/// Affine quantized 8-bit signed integer with scale and zero point
#[derive(Debug, Clone, Copy, PartialEq)]
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
#[derive(Debug, Clone, Copy, PartialEq)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_qint8_quantize_dequantize() {
        // Test perfect quantization/dequantization cycle
        let original = 3.5_f32;
        let scale = 0.1_f32;
        let zero_point = 0_i8;

        let quantized = QInt8::quantize(original, scale, zero_point);
        let dequantized = quantized.dequantize();

        // Should be close to original (within quantization error)
        assert!((original - dequantized).abs() < scale / 2.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_quint8_quantize_dequantize() {
        // Test perfect quantization/dequantization cycle
        let original = 2.7_f32;
        let scale = 0.05_f32;
        let zero_point = 0_u8;

        let quantized = QUInt8::quantize(original, scale, zero_point);
        let dequantized = quantized.dequantize();

        // Should be close to original (within quantization error)
        assert!((original - dequantized).abs() < scale / 2.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_qint8_quantized_trait() {
        let q = QInt8::new(10, 0.1, 0);
        assert_eq!(q.dequantize(), 1.0);

        let quantized = QInt8::quantize(2.5, 0.1, 0);
        assert_eq!(quantized.value, 25);
        assert_eq!(quantized.dequantize(), 2.5);

        let (min_val, max_val) = q.float_range();
        assert_eq!(min_val, -12.8);
        assert_eq!(max_val, 12.7);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_quint8_quantized_trait() {
        let q = QUInt8::new(10, 0.1, 0);
        assert_eq!(q.dequantize(), 1.0);

        let quantized = QUInt8::quantize(2.5, 0.05, 0);
        assert_eq!(quantized.value, 50);
        assert_eq!(quantized.dequantize(), 2.5);

        let (min_val, max_val) = q.float_range();
        assert_eq!(min_val, 0.0);
        assert_eq!(max_val, 25.5);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_qint8_ranges() {
        let q = QInt8::new(0, 0.1, 0);
        let (min_val, max_val) = q.float_range();

        // For scale=0.1, zero_point=0:
        // min = -128 * 0.1 + 0 = -12.8
        // max = 127 * 0.1 + 0 = 12.7
        assert_eq!(min_val, -12.8);
        assert_eq!(max_val, 12.7);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_quint8_ranges() {
        let q = QUInt8::new(0, 0.05, 0);
        let (min_val, max_val) = q.float_range();

        // For scale=0.05, zero_point=0:
        // min = 0 * 0.05 + 0 = 0.0
        // max = 255 * 0.05 + 0 = 12.75
        assert_eq!(min_val, 0.0);
        assert_eq!(max_val, 12.75);
    }

    #[test]
    fn test_qint8_clamping() {
        // Test that quantization properly clamps to i8 range
        let q1 = QInt8::quantize(1000.0, 1.0, 0); // Would be 1000, but clamped to 127
        assert_eq!(q1.value, 127);

        let q2 = QInt8::quantize(-1000.0, 1.0, 0); // Would be -1000, but clamped to -128
        assert_eq!(q2.value, -128);
    }

    #[test]
    fn test_quint8_clamping() {
        // Test that quantization properly clamps to u8 range
        let q1 = QUInt8::quantize(1000.0, 1.0, 0); // Would be 1000, but clamped to 255
        assert_eq!(q1.value, 255);

        let q2 = QUInt8::quantize(-100.0, 1.0, 0); // Would be -100, but clamped to 0
        assert_eq!(q2.value, 0);
    }
}

/// 4-bit signed integer quantization with packing
///
/// Since most hardware doesn't support 4-bit types natively, we pack
/// two `QInt4` values into a single byte for efficient storage.
/// The first value occupies bits 0-3, the second value occupies bits 4-7.
#[derive(Debug, Clone, Copy, PartialEq)]
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
#[derive(Debug, Clone, Copy, PartialEq)]
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

#[cfg(test)]
mod tests_4bit {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_qint4_quantize_dequantize() {
        let x = 3.5;
        let scale = 0.5;
        let zero_point = 0;

        let quantized = QInt4::quantize(x, scale, zero_point);
        let dequantized = quantized.dequantize();

        // Should be close to original value
        assert!((x - dequantized).abs() < scale); // Within one quantization step
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_quint4_quantize_dequantize() {
        let x = 7.5;
        let scale = 0.5;
        let zero_point = 0;

        let quantized = QUInt4::quantize(x, scale, zero_point);
        let dequantized = quantized.dequantize();

        // Should be close to original value
        assert!((x - dequantized).abs() < scale); // Within one quantization step
    }

    #[test]
    fn test_qint4_packing() {
        let low = 5i8; // 0101
        let high = 10i8; // 1010

        let packed = QInt4::pack(low, high);
        let (unpacked_low, unpacked_high) = QInt4::unpack(packed);

        assert_eq!(unpacked_low, low & 0x0F);
        assert_eq!(unpacked_high, high & 0x0F);
    }

    #[test]
    fn test_quint4_packing() {
        let low = 5u8; // 0101
        let high = 10u8; // 1010

        let packed = QUInt4::pack(low, high);
        let (unpacked_low, unpacked_high) = QUInt4::unpack(packed);

        assert_eq!(unpacked_low, low & 0x0F);
        assert_eq!(unpacked_high, high & 0x0F);
    }

    #[test]
    fn test_qint4_clamping() {
        // Test clamping to valid 4-bit signed range
        let q1 = QInt4::quantize(1000.0, 1.0, 0); // Would be 1000, clamped to 7
        assert_eq!(q1.packed_value, 7);

        let q2 = QInt4::quantize(-1000.0, 1.0, 0); // Would be -1000, clamped to -8
        assert_eq!(q2.packed_value, -8);
    }

    #[test]
    fn test_quint4_clamping() {
        // Test clamping to valid 4-bit unsigned range
        let q1 = QUInt4::quantize(1000.0, 1.0, 0); // Would be 1000, clamped to 15
        assert_eq!(q1.packed_value, 15);

        let q2 = QUInt4::quantize(-100.0, 1.0, 0); // Would be negative, clamped to 0
        assert_eq!(q2.packed_value, 0);
    }
}

/// Symmetric quantization utilities
///
/// Symmetric quantization sets `zero_point` = 0, which simplifies the
/// quantization formula and is commonly used in practice.
pub mod symmetric {
    use super::{QInt4, QInt8, QUInt4, QUInt8};

    /// Symmetric quantization for `QInt8`
    ///
    /// Uses `zero_point` = 0 and scale = `max_abs_value` / 127
    #[must_use]
    pub fn quantize_i8(x: f32, scale: f32) -> QInt8 {
        QInt8::quantize(x, scale, 0)
    }

    /// Symmetric quantization for `QUInt8`
    ///
    /// Uses `zero_point` = 0 and scale = `max_abs_value` / 127
    #[must_use]
    pub fn quantize_u8(x: f32, scale: f32) -> QUInt8 {
        QUInt8::quantize(x, scale, 0)
    }

    /// Symmetric quantization for `QInt4`
    ///
    /// Uses `zero_point` = 0 and scale = `max_abs_value` / 7
    #[must_use]
    pub fn quantize_i4(x: f32, scale: f32) -> QInt4 {
        QInt4::quantize(x, scale, 0)
    }

    /// Symmetric quantization for `QUInt4`
    ///
    /// Uses `zero_point` = 0 and scale = `max_abs_value` / 15
    #[must_use]
    pub fn quantize_u4(x: f32, scale: f32) -> QUInt4 {
        QUInt4::quantize(x, scale, 0)
    }

    /// Calculate symmetric scale for `QInt8`
    ///
    /// scale = `max_abs_value` / 127
    #[must_use]
    pub fn scale_i8(max_abs_value: f32) -> f32 {
        max_abs_value / 127.0
    }

    /// Calculate symmetric scale for `QUInt8`
    ///
    /// scale = `max_abs_value` / 127
    #[must_use]
    pub fn scale_u8(max_abs_value: f32) -> f32 {
        max_abs_value / 127.0
    }

    /// Calculate symmetric scale for `QInt4`
    ///
    /// scale = `max_abs_value` / 7
    #[must_use]
    pub fn scale_i4(max_abs_value: f32) -> f32 {
        max_abs_value / 7.0
    }

    /// Calculate symmetric scale for `QUInt4`
    ///
    /// scale = `max_abs_value` / 15
    #[must_use]
    pub fn scale_u4(max_abs_value: f32) -> f32 {
        max_abs_value / 15.0
    }

    /// Find the maximum absolute value in a slice for symmetric quantization
    #[must_use]
    pub fn max_abs_value(data: &[f32]) -> f32 {
        data.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[cfg(feature = "std")]
        use std::vec;

        #[test]
        fn test_symmetric_quantize_i8() {
            let scale = scale_i8(10.0); // max_abs = 10.0, scale = 10/127
            let quantized = quantize_i8(5.0, scale);
            let dequantized = quantized.dequantize();

            // Should be close to original value
            assert!((5.0 - dequantized).abs() < 2.0 * scale);
        }

        #[test]
        fn test_symmetric_quantize_u8() {
            let scale = scale_u8(10.0); // max_abs = 10.0, scale = 10/127
            let quantized = quantize_u8(5.0, scale);
            let dequantized = quantized.dequantize();

            // Should be close to original value
            assert!((5.0 - dequantized).abs() < 2.0 * scale);
        }

        #[test]
        fn test_symmetric_quantize_i4() {
            let scale = scale_i4(10.0); // max_abs = 10.0, scale = 10/7
            let quantized = quantize_i4(5.0, scale);
            let dequantized = quantized.dequantize();

            // Should be close to original value
            assert!((5.0 - dequantized).abs() < 2.0 * scale);
        }

        #[test]
        fn test_symmetric_quantize_u4() {
            let scale = scale_u4(10.0); // max_abs = 10.0, scale = 10/15
            let quantized = quantize_u4(5.0, scale);
            let dequantized = quantized.dequantize();

            // Should be close to original value
            assert!((5.0 - dequantized).abs() < 2.0 * scale);
        }

        #[test]
        #[allow(clippy::float_cmp)]
        fn test_max_abs_value() {
            let data = vec![1.0, -3.0, 2.0, -5.0, 0.0];
            assert_eq!(max_abs_value(&data), 5.0);

            let data_empty = vec![];
            assert_eq!(max_abs_value(&data_empty), 0.0);

            let data_positive = vec![1.0, 2.0, 3.0];
            assert_eq!(max_abs_value(&data_positive), 3.0);
        }

        #[test]
        #[allow(clippy::float_cmp)]
        fn test_symmetric_scales() {
            assert_eq!(scale_i8(10.0), 10.0 / 127.0);
            assert_eq!(scale_u8(10.0), 10.0 / 127.0);
            assert_eq!(scale_i4(10.0), 10.0 / 7.0);
            assert_eq!(scale_u4(10.0), 10.0 / 15.0);
        }
    }
}
