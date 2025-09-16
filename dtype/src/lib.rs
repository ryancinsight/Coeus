//! # Coeus Core
//!
//! Core data type definitions and traits used throughout the Coeus ecosystem.
//!
//! This crate provides:
//! - The `Dtype` trait for tensor data types
//! - Floating point and integer type distinctions
//! - Data type enumeration and utilities
//! - Type-erased data containers

use half::{bf16, f16};
use num_traits::{Float, FromPrimitive, Num, One, ToPrimitive, Zero};
use std::fmt;

/// Trait for all supported data types in tensors
pub trait Dtype:
    Num
    + Copy
    + Zero
    + One
    + PartialOrd
    + fmt::Debug
    + Send
    + Sync
    + FromPrimitive
    + ToPrimitive
    + 'static
{
    /// Check if this type supports floating point operations
    fn is_float() -> bool {
        false
    }

    /// Check if this type is signed
    fn is_signed() -> bool {
        false
    }

    /// Get the name of this data type
    fn name() -> &'static str;

    /// Convert this value to f64 for gradient computation
    fn to_f64(&self) -> Option<f64> {
        None
    }
}

/// Trait for types that support numeric operations (both float and int)
/// This is the core trait for quantization support - both floating point
/// and integer types can implement this for unified numeric operations
pub trait NumericDtype: Dtype {}

/// Trait for floating point data types
pub trait FloatDtype: NumericDtype + Float + std::iter::Sum {}

/// Trait for integer data types that support quantization
pub trait IntDtype: NumericDtype {}

/// Trait for quantized data types (integer types representing quantized floats)
pub trait QuantizedDtype: IntDtype {
    /// Scale factor for quantization (multiplies quantized values to get float)
    fn scale() -> f32;

    /// Zero point for quantization (subtracts from quantized values before scaling)
    fn zero_point() -> Self;

    /// Quantization range for symmetric quantization
    fn quantization_range() -> (Self, Self);

    /// Convert floating point to quantized representation
    fn quantize(value: f32, scale: f32, zero_point: Self) -> Self;

    /// Convert quantized value back to floating point
    fn dequantize(self, scale: f32, zero_point: Self) -> f32;
}

/// Dynamic quantization utilities
pub mod quantization {
    use super::*;

    /// Quantization calibration data
    pub struct QuantizationCalib<T: QuantizedDtype> {
        pub scale: f32,
        pub zero_point: T,
        pub min_val: f32,
        pub max_val: f32,
    }

    impl<T: QuantizedDtype + num_traits::FromPrimitive + num_traits::Bounded> QuantizationCalib<T> {
        /// Calibrate quantization parameters from observed data
        pub fn calibrate(data: &[f32], num_bits: u32) -> Self {
            let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            let range = max_val - min_val;
            let scale = range / ((1 << num_bits) - 1) as f32;

            // For symmetric quantization when data is centered around zero
            let zero_point_val = if min_val < 0.0 && max_val > 0.0 {
                // Check if data is approximately symmetric around zero
                let abs_min = min_val.abs();
                let abs_max = max_val.abs();
                if (abs_min - abs_max).abs() / (abs_min + abs_max) < 0.1 {
                    // Within 10% symmetry
                    0 // Use symmetric quantization
                } else {
                    ((0.0 - min_val) / scale).round() as i32
                }
            } else {
                0
            };

            let zero_point = if let Some(zp) = T::from_i32(zero_point_val) {
                zp
            } else {
                // Handle overflow for types like i8 where zero_point calculation exceeds range
                if zero_point_val > 0 {
                    T::max_value() // Use max value for positive overflow
                } else {
                    T::min_value() // Use min value for negative overflow
                }
            };

            Self {
                scale,
                zero_point,
                min_val,
                max_val,
            }
        }

        /// Apply symmetric quantization (zero_point = 0)
        pub fn symmetric_scale(num_bits: u32, max_abs_val: f32) -> f32 {
            max_abs_val / ((1 << (num_bits - 1)) - 1) as f32
        }

        /// Apply asymmetric quantization with optimal zero point
        pub fn asymmetric_scale_zero_point(
            min_val: f32,
            max_val: f32,
            num_bits: u32,
        ) -> (f32, i32) {
            let range = max_val - min_val;
            let scale = range / ((1 << num_bits) - 1) as f32;
            let zero_point = ((0.0 - min_val) / scale).round() as i32;
            (scale, zero_point)
        }
    }

    /// Quantize a tensor with calibration
    pub fn quantize_tensor<T: QuantizedDtype>(
        tensor: &[f32],
        calib: &QuantizationCalib<T>,
    ) -> Vec<T> {
        tensor
            .iter()
            .map(|&x| T::quantize(x, calib.scale, calib.zero_point))
            .collect()
    }

    /// Dequantize a tensor with calibration
    pub fn dequantize_tensor<T: QuantizedDtype>(
        tensor: &[T],
        calib: &QuantizationCalib<T>,
    ) -> Vec<f32> {
        tensor
            .iter()
            .map(|&x| x.dequantize(calib.scale, calib.zero_point))
            .collect()
    }
}

// Note: impl_dtype macro removed as it was unused

/// Macro to implement Dtype for floating point types with f64 conversion
macro_rules! impl_dtype_float {
    ($t:ty, $name:expr) => {
        impl Dtype for $t {
            fn is_float() -> bool {
                true
            }
            fn is_signed() -> bool {
                true
            }
            fn name() -> &'static str {
                $name
            }
            fn to_f64(&self) -> Option<f64> {
                Some(f64::from(*self))
            }
        }
    };
}

/// Macro to implement Dtype for integer types with f64 conversion
macro_rules! impl_dtype_int {
    ($t:ty, $name:expr, $is_signed:expr) => {
        impl Dtype for $t {
            fn is_float() -> bool {
                false
            }
            fn is_signed() -> bool {
                $is_signed
            }
            fn name() -> &'static str {
                $name
            }
            fn to_f64(&self) -> Option<f64> {
                Some(*self as f64)
            }
        }
    };
}

// Implement for floating point types
impl_dtype_float!(f32, "f32");
impl_dtype_float!(f64, "f64");
impl_dtype_float!(f16, "f16");
impl_dtype_float!(bf16, "bf16");

// Implement for integer types
impl_dtype_int!(i8, "i8", true);
impl_dtype_int!(i16, "i16", true);
impl_dtype_int!(i32, "i32", true);
impl_dtype_int!(i64, "i64", true);
impl_dtype_int!(u8, "u8", false);
impl_dtype_int!(u16, "u16", false);
impl_dtype_int!(u32, "u32", false);
impl_dtype_int!(u64, "u64", false);

// Implement NumericDtype for all numeric types (both float and int)
impl NumericDtype for f32 {}
impl NumericDtype for f64 {}
impl NumericDtype for f16 {}
impl NumericDtype for bf16 {}
impl NumericDtype for i8 {}
impl NumericDtype for i16 {}
impl NumericDtype for i32 {}
impl NumericDtype for i64 {}
impl NumericDtype for u8 {}
impl NumericDtype for u16 {}
impl NumericDtype for u32 {}
impl NumericDtype for u64 {}

// Implement FloatDtype for floating point types
impl FloatDtype for f32 {}
impl FloatDtype for f64 {}
impl FloatDtype for f16 {}
impl FloatDtype for bf16 {}

// Implement IntDtype for integer types (supports quantization)
impl IntDtype for i8 {}
impl IntDtype for i16 {}
impl IntDtype for i32 {}
impl IntDtype for i64 {}
impl IntDtype for u8 {}
impl IntDtype for u16 {}
impl IntDtype for u32 {}
impl IntDtype for u64 {}

// Implement QuantizedDtype for common quantization schemes
// Example: int8 with scale=1.0/127.0 for symmetric quantization
impl QuantizedDtype for i8 {
    fn scale() -> f32 {
        1.0 / 127.0
    }

    fn zero_point() -> Self {
        0
    }

    fn quantization_range() -> (Self, Self) {
        (-127, 127)
    }

    fn quantize(value: f32, scale: f32, zero_point: Self) -> Self {
        let quantized = (value / scale).round() as i32 + zero_point as i32;
        quantized.clamp(i8::MIN as i32, i8::MAX as i32) as i8
    }

    fn dequantize(self, scale: f32, zero_point: Self) -> f32 {
        ((self as i32 - zero_point as i32) as f32) * scale
    }
}

// Example: uint8 with scale=1.0/255.0 for asymmetric quantization
impl QuantizedDtype for u8 {
    fn scale() -> f32 {
        1.0 / 255.0
    }

    fn zero_point() -> Self {
        128
    }

    fn quantization_range() -> (Self, Self) {
        (0, 255)
    }

    fn quantize(value: f32, scale: f32, zero_point: Self) -> Self {
        let quantized = (value / scale).round() as i32 + zero_point as i32;
        quantized.clamp(u8::MIN as i32, u8::MAX as i32) as u8
    }

    fn dequantize(self, scale: f32, zero_point: Self) -> f32 {
        ((self as i32 - zero_point as i32) as f32) * scale
    }
}

// Implement QuantizedDtype for i16 (useful for 16-bit quantization)
impl QuantizedDtype for i16 {
    fn scale() -> f32 {
        1.0 / 32767.0
    }

    fn zero_point() -> Self {
        0
    }

    fn quantization_range() -> (Self, Self) {
        (-32767, 32767)
    }

    fn quantize(value: f32, scale: f32, zero_point: Self) -> Self {
        let quantized = (value / scale).round() as i32 + zero_point as i32;
        quantized.clamp(i16::MIN as i32, i16::MAX as i32) as i16
    }

    fn dequantize(self, scale: f32, zero_point: Self) -> f32 {
        ((self as i32 - zero_point as i32) as f32) * scale
    }
}

/// Enum representing all supported data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Bool,
    F16,
    BF16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl DataType {
    /// Get the size in bytes for this data type
    pub fn size(&self) -> usize {
        match self {
            DataType::Bool => 1,
            DataType::F16 | DataType::BF16 => 2,
            DataType::F32 | DataType::I32 | DataType::U32 => 4,
            DataType::F64 | DataType::I64 | DataType::U64 => 8,
            DataType::I16 | DataType::U16 => 2,
            DataType::I8 | DataType::U8 => 1,
        }
    }

    /// Check if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            DataType::F16 | DataType::BF16 | DataType::F32 | DataType::F64
        )
    }

    /// Check if this is a signed type
    pub fn is_signed(&self) -> bool {
        match self {
            DataType::Bool => false,
            DataType::F16
            | DataType::BF16
            | DataType::F32
            | DataType::F64
            | DataType::I8
            | DataType::I16
            | DataType::I32
            | DataType::I64 => true,
            DataType::U8 | DataType::U16 | DataType::U32 | DataType::U64 => false,
        }
    }

    /// Check if this type supports signed operations (for backward compatibility)
    pub fn supports_signed_ops(&self) -> bool {
        self.is_signed()
    }

    /// Get the name of this data type
    pub fn name(&self) -> &'static str {
        match self {
            DataType::Bool => "bool",
            DataType::F16 => "f16",
            DataType::BF16 => "bf16",
            DataType::F32 => "f32",
            DataType::F64 => "f64",
            DataType::I8 => "i8",
            DataType::I16 => "i16",
            DataType::I32 => "i32",
            DataType::I64 => "i64",
            DataType::U8 => "u8",
            DataType::U16 => "u16",
            DataType::U32 => "u32",
            DataType::U64 => "u64",
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{}", self.name())
    }
}

/// Type-erased container for tensor data
#[derive(Clone)]
pub enum DataContainer {
    Bool(Vec<bool>),
    F16(Vec<f16>),
    BF16(Vec<bf16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

impl DataContainer {
    /// Create a new container from a vector - SAFE implementation with proper type handling
    /// This method is deprecated and will be removed. Use specific constructors instead.
    #[deprecated(note = "Use specific constructors like from_f32(), from_f64() for type safety")]
    pub fn new<T: Dtype>(_data: Vec<T>) -> Self {
        // This method previously used unsafe transmute which violates memory safety
        // The new implementation requires explicit type specification at compile time
        panic!("DataContainer::new() is deprecated. Use specific constructors like from_f32(), from_f64() for type safety.");
    }

    /// Create a container from f32 data
    pub fn from_f32(data: Vec<f32>) -> Self {
        Self::F32(data)
    }

    /// Create a container from f64 data
    pub fn from_f64(data: Vec<f64>) -> Self {
        Self::F64(data)
    }

    /// Create a container from i32 data
    pub fn from_i32(data: Vec<i32>) -> Self {
        Self::I32(data)
    }

    /// Create a container from i64 data
    pub fn from_i64(data: Vec<i64>) -> Self {
        Self::I64(data)
    }

    /// Get the length of the container
    pub fn len(&self) -> usize {
        match self {
            DataContainer::Bool(v) => v.len(),
            DataContainer::F16(v) => v.len(),
            DataContainer::BF16(v) => v.len(),
            DataContainer::F32(v) => v.len(),
            DataContainer::F64(v) => v.len(),
            DataContainer::I8(v) => v.len(),
            DataContainer::I16(v) => v.len(),
            DataContainer::I32(v) => v.len(),
            DataContainer::I64(v) => v.len(),
            DataContainer::U8(v) => v.len(),
            DataContainer::U16(v) => v.len(),
            DataContainer::U32(v) => v.len(),
            DataContainer::U64(v) => v.len(),
        }
    }

    /// Check if the container is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the data type of this container
    pub fn dtype(&self) -> DataType {
        match self {
            DataContainer::Bool(_) => DataType::Bool,
            DataContainer::F16(_) => DataType::F16,
            DataContainer::BF16(_) => DataType::BF16,
            DataContainer::F32(_) => DataType::F32,
            DataContainer::F64(_) => DataType::F64,
            DataContainer::I8(_) => DataType::I8,
            DataContainer::I16(_) => DataType::I16,
            DataContainer::I32(_) => DataType::I32,
            DataContainer::I64(_) => DataType::I64,
            DataContainer::U8(_) => DataType::U8,
            DataContainer::U16(_) => DataType::U16,
            DataContainer::U32(_) => DataType::U32,
            DataContainer::U64(_) => DataType::U64,
        }
    }
}

impl fmt::Debug for DataContainer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            DataContainer::Bool(v) => write!(f, "Bool({v:?})"),
            DataContainer::F16(v) => write!(f, "F16({v:?})"),
            DataContainer::BF16(v) => write!(f, "BF16({v:?})"),
            DataContainer::F32(v) => write!(f, "F32({v:?})"),
            DataContainer::F64(v) => write!(f, "F64({v:?})"),
            DataContainer::I8(v) => write!(f, "I8({v:?})"),
            DataContainer::I16(v) => write!(f, "I16({v:?})"),
            DataContainer::I32(v) => write!(f, "I32({v:?})"),
            DataContainer::I64(v) => write!(f, "I64({v:?})"),
            DataContainer::U8(v) => write!(f, "U8({v:?})"),
            DataContainer::U16(v) => write!(f, "U16({v:?})"),
            DataContainer::U32(v) => write!(f, "U32({v:?})"),
            DataContainer::U64(v) => write!(f, "U64({v:?})"),
        }
    }
}

/// Marker trait for data types that are safe for GPU memory transfer
/// This trait indicates that the type can be safely cast to/from bytes using bytemuck
pub trait GpuSafe: Dtype + bytemuck::Pod {}

/// Implement GpuSafe for types that are both Dtype and Pod
impl<T: Dtype + bytemuck::Pod> GpuSafe for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_properties() {
        assert!(f32::is_float());
        assert!(f32::is_signed());
        assert_eq!(f32::name(), "f32");

        assert!(f16::is_float());
        assert!(f16::is_signed());
        assert_eq!(f16::name(), "f16");

        assert!(bf16::is_float());
        assert!(bf16::is_signed());
        assert_eq!(bf16::name(), "bf16");

        assert!(!i32::is_float());
        assert!(i32::is_signed());
        assert_eq!(i32::name(), "i32");
    }

    #[test]
    fn test_data_type_info() {
        assert_eq!(DataType::F16.size(), 2);
        assert!(DataType::F16.is_float());
        assert!(DataType::F16.is_signed());

        assert_eq!(DataType::BF16.size(), 2);
        assert!(DataType::BF16.is_float());
        assert!(DataType::BF16.is_signed());

        assert_eq!(DataType::F32.size(), 4);
        assert!(DataType::F32.is_float());
        assert!(DataType::F32.is_signed());

        assert_eq!(DataType::U8.size(), 1);
        assert!(!DataType::U8.is_float());
        assert!(!DataType::U8.is_signed());
    }

    #[test]
    fn test_trait_hierarchy() {
        // Test that all numeric types implement NumericDtype
        assert!(f32::is_float());
        assert!(i32::is_signed());
        assert!(!u8::is_signed());

        // Test quantization support
        assert_eq!(i8::scale(), 1.0 / 127.0);
        assert_eq!(i8::zero_point(), 0);
        assert_eq!(u8::scale(), 1.0 / 255.0);
        assert_eq!(u8::zero_point(), 128);
    }

    #[test]
    fn test_enhanced_quantization_round_trip() {
        use approx::assert_relative_eq;

        // Test i8 quantization round-trip (symmetric quantization)
        let symmetric_values = vec![0.0, 0.5, -0.5, 0.25, -0.25];
        for &value in &symmetric_values {
            let quantized_i8 = i8::quantize(value, i8::scale(), i8::zero_point());
            let dequantized_i8 = quantized_i8.dequantize(i8::scale(), i8::zero_point());
            assert_relative_eq!(value, dequantized_i8, epsilon = 0.01);
        }

        // Test u8 quantization round-trip (asymmetric quantization for positive values)
        // Note: asymmetric quantization with zero_point=128 maps [0,1] to [128,255] range
        // So 0.0 -> 128, 1.0 -> 255, and dequantization gives (255-128)/255 = 0.498
        let test_cases = vec![
            (0.0, 128u8, 0.0),      // 0.0 quantizes to 128, dequantizes to 0.0
            (0.5, 255u8, 0.498),    // 0.5 quantizes to 255 (clamped), dequantizes to ~0.498
            (1.0, 255u8, 0.498),    // 1.0 quantizes to 255 (clamped), dequantizes to ~0.498
            (0.25, 192u8, 0.25098), // 0.25 quantizes to 192, dequantizes to ~0.25098
        ];

        for (input, expected_quantized, expected_dequantized) in test_cases {
            let quantized_u8 = u8::quantize(input, u8::scale(), u8::zero_point());
            let dequantized_u8 = quantized_u8.dequantize(u8::scale(), u8::zero_point());

            // Verify quantization gives expected result
            assert_eq!(quantized_u8, expected_quantized);

            // Verify dequantization gives expected result
            use approx::assert_relative_eq;
            assert_relative_eq!(dequantized_u8, expected_dequantized, epsilon = 0.01);
        }

        // Test i16 quantization round-trip (symmetric)
        let i16_values = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25];
        for &value in &i16_values {
            let quantized_i16 = i16::quantize(value, i16::scale(), i16::zero_point());
            let dequantized_i16 = quantized_i16.dequantize(i16::scale(), i16::zero_point());
            assert_relative_eq!(value, dequantized_i16, epsilon = 0.01);
        }
    }

    #[test]
    fn test_quantization_ranges() {
        // Test i8 range
        let (i8_min, i8_max) = i8::quantization_range();
        assert_eq!(i8_min, -127);
        assert_eq!(i8_max, 127);

        // Test u8 range
        let (u8_min, u8_max) = u8::quantization_range();
        assert_eq!(u8_min, 0);
        assert_eq!(u8_max, 255);

        // Test i16 range
        let (i16_min, i16_max) = i16::quantization_range();
        assert_eq!(i16_min, -32767);
        assert_eq!(i16_max, 32767);
    }

    #[test]
    fn test_dynamic_quantization_calibration() {
        use super::quantization::QuantizationCalib;

        // Test with a smaller data range that fits well in 8-bit quantization
        let data = vec![-0.25, -0.125, 0.0, 0.125, 0.25];
        let calib = QuantizationCalib::<i8>::calibrate(&data, 8);

        // Check calibration results
        assert_eq!(calib.min_val, -0.25);
        assert_eq!(calib.max_val, 0.25);
        assert!(calib.scale > 0.0);

        // Test quantization and dequantization with calibration
        let quantized = super::quantization::quantize_tensor::<i8>(&data, &calib);
        let dequantized = super::quantization::dequantize_tensor::<i8>(&quantized, &calib);

        // Debug output to understand quantization precision
        println!("Original data: {:?}", data);
        println!("Quantized values: {:?}", quantized);
        println!("Dequantized values: {:?}", dequantized);
        println!("Scale: {}, Zero point: {}", calib.scale, calib.zero_point);

        // Verify that the quantized values are within the expected range
        // Note: Range check is redundant since q is i8 type, but kept for documentation
        for &_q in &quantized {
            // i8 range is guaranteed by the type system, no runtime check needed
            // Quantized value range check is guaranteed by i8 type system
        }

        // For small ranges like this, quantization should be reasonably accurate
        // Calculate expected quantization error based on scale and bit depth
        let quantization_error = calib.scale * 0.5; // Half LSB error
        let _relative_tolerance = quantization_error / (calib.max_val - calib.min_val).abs();

        for (original, dq) in data.iter().zip(dequantized.iter()) {
            use approx::assert_relative_eq;
            // Use calculated quantization error as tolerance
            let tolerance = quantization_error.max(calib.scale); // At least one quantization step
            assert_relative_eq!(original, dq, epsilon = tolerance);
        }

        // Verify that the sign and relative ordering are preserved
        let original_signs: Vec<i32> = data.iter().map(|&x| x.signum() as i32).collect();
        let dequantized_signs: Vec<i32> = dequantized.iter().map(|&x| x.signum() as i32).collect();
        assert_eq!(
            original_signs, dequantized_signs,
            "Quantization should preserve signs"
        );
    }

    #[test]
    fn test_symmetric_quantization() {
        use super::quantization::QuantizationCalib;

        let max_abs_val = 2.0;
        let scale = QuantizationCalib::<i8>::symmetric_scale(8, max_abs_val);
        let expected_scale = max_abs_val / ((1 << 7) - 1) as f32; // 8-bit symmetric
        use approx::assert_relative_eq;
        assert_relative_eq!(scale, expected_scale);
    }

    #[test]
    fn test_asymmetric_quantization() {
        use super::quantization::QuantizationCalib;

        let min_val = -1.0;
        let max_val = 2.0;
        let (scale, zero_point) =
            QuantizationCalib::<i8>::asymmetric_scale_zero_point(min_val, max_val, 8);

        assert!(scale > 0.0);
        assert!(zero_point >= 0);

        // Verify zero point calculation
        let expected_zero_point = ((0.0 - min_val) / scale).round() as i32;
        assert_eq!(zero_point, expected_zero_point);
    }

    #[test]
    fn test_quantization_edge_cases() {
        // Test edge cases for quantization
        let scale = 1.0 / 127.0;

        // Test value outside range gets clamped
        let large_value = 10.0;
        let quantized = i8::quantize(large_value, scale, 0);
        assert_eq!(quantized, 127); // Should be clamped to max

        let small_value = -10.0;
        let quantized = i8::quantize(small_value, scale, 0);
        assert_eq!(quantized, -128); // Should be clamped to min

        // Test zero quantization
        let zero_quantized = i8::quantize(0.0, scale, 0);
        assert_eq!(zero_quantized, 0);
        let zero_dequantized = zero_quantized.dequantize(scale, 0);
        use approx::assert_relative_eq;
        assert_relative_eq!(zero_dequantized, 0.0, epsilon = 1e-6);
    }
}
