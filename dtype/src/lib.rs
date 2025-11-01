//! # Coeus Data Types
//!
//! Core data type abstractions for the Coeus deep learning framework.
//! Provides a complete set of numeric types with safe, efficient operations.
//!
//! ## Architecture
//!
//! The dtype system is built around the `DataType` trait, which defines
//! the interface for all numeric types supported by Coeus tensors.
//!
//! ## Supported Types
//!
//! - **Floating Point**: f16, f32, f64, bfloat16
//! - **Integer**: i8, i16, i32, i64, u8, u16, u32, u64
//! - **Complex**: `Complex<f32>`, `Complex<f64>`
//! - **Quantized**: 4-bit and 8-bit affine quantized types (`QInt4`, `QUInt4`, `QInt8`, `QUInt8`)
//!
//! ## Safety
//!
//! All operations are memory-safe with no unsafe code. Numerical operations
//! are validated for correctness and numerical stability.

#![no_std]
#![warn(missing_docs, clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

#[cfg(feature = "std")]
extern crate std;

// Re-export key dependencies for user convenience
pub use num_traits;

// Core traits and types
pub mod error;
pub mod traits;

// Primitive data types
pub mod complex;
pub mod float;
pub mod int;
pub mod quantized;

// Type promotion rules
pub mod promotion;

// Quantization utilities (feature-gated)
#[cfg(all(feature = "quantized", feature = "std"))]
pub mod quantization;

#[cfg(all(feature = "quantized", feature = "std"))]
pub use quantization::{QuantizationError, QuantizationNoiseAnalysis};

// Type aliases for convenience
pub use error::DtypeError;
pub use traits::{DataType, FloatExt};

// Quantized types (feature-gated)
#[cfg(feature = "quantized")]
pub use quantized::{symmetric, QInt4, QInt8, QUInt4, QUInt8, QuantizedType};

/// Result type for dtype operations
pub type Result<T> = core::result::Result<T, DtypeError>;

/// Core numeric types supported by Coeus
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Dtype {
    /// 16-bit half precision floating point
    Half,
    /// 16-bit brain floating point
    BFloat16,
    /// 32-bit single precision floating point
    Float32,
    /// 64-bit double precision floating point
    Float64,
    /// 8-bit signed integer
    Int8,
    /// 16-bit signed integer
    Int16,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 8-bit unsigned integer
    UInt8,
    /// 16-bit unsigned integer
    UInt16,
    /// 32-bit unsigned integer
    UInt32,
    /// 64-bit unsigned integer
    UInt64,
    /// 32-bit complex floating point
    Complex32,
    /// 64-bit complex floating point
    Complex64,
    /// 4-bit quantized (signed, packed)
    QInt4,
    /// 4-bit quantized (unsigned, packed)
    QUInt4,
    /// 8-bit quantized (signed)
    QInt8,
    /// 8-bit quantized (unsigned)
    QUInt8,
}

impl Dtype {
    /// Returns true if this dtype is a floating point type
    #[must_use]
    pub const fn is_floating_point(self) -> bool {
        matches!(
            self,
            Self::Half | Self::BFloat16 | Self::Float32 | Self::Float64
        )
    }

    /// Returns true if this dtype is an integer type
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::Int8
                | Self::Int16
                | Self::Int32
                | Self::Int64
                | Self::UInt8
                | Self::UInt16
                | Self::UInt32
                | Self::UInt64
        )
    }

    /// Returns true if this dtype is a complex type
    #[must_use]
    pub const fn is_complex(self) -> bool {
        matches!(self, Self::Complex32 | Self::Complex64)
    }

    /// Returns true if this dtype is quantized
    #[must_use]
    pub const fn is_quantized(self) -> bool {
        matches!(
            self,
            Self::QInt4 | Self::QUInt4 | Self::QInt8 | Self::QUInt8
        )
    }

    /// Returns the size in bytes of this dtype
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Int8 | Self::UInt8 | Self::QInt4 | Self::QUInt4 | Self::QInt8 | Self::QUInt8 => 1,
            Self::Half | Self::BFloat16 | Self::Int16 | Self::UInt16 => 2,
            Self::Float32 | Self::Int32 | Self::UInt32 => 4,
            Self::Float64 | Self::Int64 | Self::UInt64 | Self::Complex32 => 8,
            Self::Complex64 => 16,
        }
    }

    /// Returns the name of this dtype as a string
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Half => "half",
            Self::BFloat16 => "bfloat16",
            Self::Float32 => "float32",
            Self::Float64 => "float64",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::UInt8 => "uint8",
            Self::UInt16 => "uint16",
            Self::UInt32 => "uint32",
            Self::UInt64 => "uint64",
            Self::Complex32 => "complex32",
            Self::Complex64 => "complex64",
            Self::QInt4 => "qint4",
            Self::QUInt4 => "quint4",
            Self::QInt8 => "qint8",
            Self::QUInt8 => "quint8",
        }
    }
}

#[cfg(feature = "std")]
impl core::fmt::Display for Dtype {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;

    #[test]
    fn test_dtype_properties() {
        assert!(Dtype::Float32.is_floating_point());
        assert!(Dtype::Int32.is_integer());
        assert!(Dtype::Complex64.is_complex());
        assert!(Dtype::QInt8.is_quantized());
        assert!(!Dtype::Float32.is_integer());
        assert!(!Dtype::Int32.is_floating_point());
    }

    #[test]
    fn test_dtype_sizes() {
        assert_eq!(Dtype::Float32.size_bytes(), 4);
        assert_eq!(Dtype::Float64.size_bytes(), 8);
        assert_eq!(Dtype::Int8.size_bytes(), 1);
        assert_eq!(Dtype::Complex64.size_bytes(), 16);
    }

    #[test]
    fn test_dtype_names() {
        assert_eq!(Dtype::Float32.name(), "float32");
        assert_eq!(Dtype::Int64.name(), "int64");
        assert_eq!(Dtype::Complex32.name(), "complex32");
    }

    #[cfg(feature = "complex")]
    mod complex_tests {
        use super::*;
        use crate::complex::{Complex32, Complex64};
        use crate::traits::ComplexExt;

        #[test]
        fn test_complex32_creation() {
            let c = Complex32::new(3.0, 4.0);
            assert_eq!(c.re(), 3.0);
            assert_eq!(c.im(), 4.0);
        }

        #[test]
        fn test_complex64_creation() {
            let c = Complex64::new(3.0, 4.0);
            assert_eq!(c.re(), 3.0);
            assert_eq!(c.im(), 4.0);
        }

        #[test]
        fn test_complex32_operations() {
            let c1 = Complex32::new(3.0, 4.0);
            let c2 = Complex32::new(1.0, 2.0);

            // Addition
            let sum = c1 + c2;
            assert_eq!(sum.re(), 4.0);
            assert_eq!(sum.im(), 6.0);

            // Conjugate
            let conj = c1.conj();
            assert_eq!(conj.re(), 3.0);
            assert_eq!(conj.im(), -4.0);

            // Norm squared
            assert_eq!(c1.norm_sqr(), 25.0);

            // Norm (magnitude)
            assert_eq!(c1.norm(), 5.0);
        }

        #[test]
        fn test_complex64_operations() {
            let c1 = Complex64::new(3.0, 4.0);
            let c2 = Complex64::new(1.0, 2.0);

            // Addition
            let sum = c1 + c2;
            assert_eq!(sum.re(), 4.0);
            assert_eq!(sum.im(), 6.0);

            // Conjugate
            let conj = c1.conj();
            assert_eq!(conj.re(), 3.0);
            assert_eq!(conj.im(), -4.0);

            // Norm squared
            assert_eq!(c1.norm_sqr(), 25.0);

            // Norm (magnitude)
            assert_eq!(c1.norm(), 5.0);
        }

        #[test]
        fn test_complex32_datatype_trait() {
            assert_eq!(Complex32::dtype(), Dtype::Complex32);
            assert_eq!(Complex32::size_bytes(), 8);
            assert_eq!(Complex32::name(), "complex32");
            assert!(Complex32::is_complex());
            assert!(!Complex32::is_floating_point());
            assert!(!Complex32::is_integer());
            assert!(!Complex32::is_quantized());
        }

        #[test]
        fn test_complex64_datatype_trait() {
            assert_eq!(Complex64::dtype(), Dtype::Complex64);
            assert_eq!(Complex64::size_bytes(), 16);
            assert_eq!(Complex64::name(), "complex64");
            assert!(Complex64::is_complex());
            assert!(!Complex64::is_floating_point());
            assert!(!Complex64::is_integer());
            assert!(!Complex64::is_quantized());
        }

        #[test]
        fn test_complex32_complex_ext_trait() {
            use crate::traits::ComplexExt;

            let c = Complex32::new(3.0, 4.0);
            assert_eq!(c.re(), 3.0);
            assert_eq!(c.im(), 4.0);

            let conj = c.conj();
            assert_eq!(conj.re(), 3.0);
            assert_eq!(conj.im(), -4.0);

            assert_eq!(c.norm_sqr(), 25.0);
            assert_eq!(c.norm(), 5.0);
            assert_eq!(c.arg(), 4.0_f32.atan2(3.0));
        }

        #[test]
        fn test_complex64_complex_ext_trait() {
            use crate::traits::ComplexExt;

            let c = Complex64::new(3.0, 4.0);
            assert_eq!(c.re(), 3.0);
            assert_eq!(c.im(), 4.0);

            let conj = c.conj();
            assert_eq!(conj.re(), 3.0);
            assert_eq!(conj.im(), -4.0);

            assert_eq!(c.norm_sqr(), 25.0);
            assert_eq!(c.norm(), 5.0);
            assert_eq!(c.arg(), 4.0_f64.atan2(3.0));
        }

        #[test]
        fn test_complex32_basic_math() {
            let c1 = Complex32::new(3.0, 4.0);
            let c2 = Complex32::new(1.0, 2.0);

            // Test addition (should work through num-complex Num trait)
            let sum = c1 + c2;
            assert_eq!(sum.re(), 4.0);
            assert_eq!(sum.im(), 6.0);
        }

        #[test]
        fn test_complex64_basic_math() {
            let c1 = Complex64::new(3.0, 4.0);
            let c2 = Complex64::new(1.0, 2.0);

            // Test addition (should work through num-complex Num trait)
            let sum = c1 + c2;
            assert_eq!(sum.re(), 4.0);
            assert_eq!(sum.im(), 6.0);
        }
    }
}
