//! Core traits for data types
//!
//! Defines the `DataType` trait that all numeric types in Coeus must implement.
//! This trait provides a common interface for arithmetic operations, conversions,
//! and type introspection.

use core::fmt::Debug;
use num_traits::{Num, NumCast, NumOps, One, Zero};

/// Core trait that all data types in Coeus must implement.
///
/// This trait defines the minimal interface required for a type to be used
/// as a tensor element type. It combines numerical operations with type
/// introspection and safe conversion capabilities.
///
/// # Safety
///
/// Implementors must ensure that all operations are memory-safe and
/// numerically correct. No unsafe code is permitted in trait implementations.
pub trait DataType:
    Copy
    + Clone
    + Debug
    + Default
    + PartialEq
    + Num
    + NumCast
    + NumOps
    + Zero
    + One
    + Sized
    + Send
    + Sync
    + 'static
{
    /// Returns the dtype enum variant for this type
    #[must_use]
    fn dtype() -> crate::Dtype;

    /// Returns the size in bytes of this type
    #[must_use]
    fn size_bytes() -> usize {
        Self::dtype().size_bytes()
    }

    /// Returns the name of this type as a string
    #[must_use]
    fn name() -> &'static str {
        Self::dtype().name()
    }

    /// Returns true if this type is a floating point type
    #[must_use]
    fn is_floating_point() -> bool {
        Self::dtype().is_floating_point()
    }

    /// Returns true if this type is an integer type
    #[must_use]
    fn is_integer() -> bool {
        Self::dtype().is_integer()
    }

    /// Returns true if this type is a complex type
    #[must_use]
    fn is_complex() -> bool {
        Self::dtype().is_complex()
    }

    /// Returns true if this type is quantized
    #[must_use]
    fn is_quantized() -> bool {
        Self::dtype().is_quantized()
    }

    /// Safe cast from this type to another `DataType`
    ///
    /// Returns `None` if the conversion would lose precision or overflow.
    #[must_use]
    fn cast_to<T: DataType>(self) -> Option<T> {
        T::from(self)
    }

    /// Checked cast from this type to another `DataType`
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion would lose precision or overflow.
    fn checked_cast_to<T: DataType>(self) -> crate::Result<T> {
        self.cast_to::<T>().ok_or_else(|| {
            crate::DtypeError::CastError {
                from: Self::dtype(),
                to: T::dtype(),
                value: "value", // TODO: Add proper string formatting when std is available
            }
        })
    }
}

/// Extension trait for floating point operations
///
/// Provides common mathematical functions that are only meaningful
/// for floating point types.
pub trait FloatExt: DataType {
    /// Compute the natural logarithm
    #[must_use]
    fn ln(self) -> Self;

    /// Compute the base-2 logarithm
    #[must_use]
    fn log2(self) -> Self;

    /// Compute the base-10 logarithm
    #[must_use]
    fn log10(self) -> Self;

    /// Compute the exponential function
    #[must_use]
    fn exp(self) -> Self;

    /// Compute the exponential function minus 1
    #[must_use]
    fn exp_m1(self) -> Self;

    /// Compute the 2^x function
    #[must_use]
    fn exp2(self) -> Self;

    /// Compute the power function
    #[must_use]
    fn powf(self, exp: Self) -> Self;

    /// Compute the square root
    #[must_use]
    fn sqrt(self) -> Self;

    /// Compute the cube root
    #[must_use]
    fn cbrt(self) -> Self;

    /// Compute the sine
    #[must_use]
    fn sin(self) -> Self;

    /// Compute the cosine
    #[must_use]
    fn cos(self) -> Self;

    /// Compute the tangent
    #[must_use]
    fn tan(self) -> Self;

    /// Compute the arcsine
    #[must_use]
    fn asin(self) -> Self;

    /// Compute the arccosine
    #[must_use]
    fn acos(self) -> Self;

    /// Compute the arctangent
    #[must_use]
    fn atan(self) -> Self;

    /// Compute the hyperbolic sine
    #[must_use]
    fn sinh(self) -> Self;

    /// Compute the hyperbolic cosine
    #[must_use]
    fn cosh(self) -> Self;

    /// Compute the hyperbolic tangent
    #[must_use]
    fn tanh(self) -> Self;

    /// Compute the inverse hyperbolic sine
    #[must_use]
    fn asinh(self) -> Self;

    /// Compute the inverse hyperbolic cosine
    #[must_use]
    fn acosh(self) -> Self;

    /// Compute the inverse hyperbolic tangent
    #[must_use]
    fn atanh(self) -> Self;

    /// Compute the error function
    #[must_use]
    fn erf(self) -> Self;

    /// Compute the complementary error function
    #[must_use]
    fn erfc(self) -> Self;

    /// Round to nearest integer
    #[must_use]
    fn round(self) -> Self;

    /// Truncate towards zero
    #[must_use]
    fn trunc(self) -> Self;

    /// Round towards negative infinity
    #[must_use]
    fn floor(self) -> Self;

    /// Round towards positive infinity
    #[must_use]
    fn ceil(self) -> Self;

    /// Fractional part
    #[must_use]
    fn fract(self) -> Self;

    /// Absolute value
    #[must_use]
    fn abs(self) -> Self;

    /// Sign function (+1, 0, or -1)
    #[must_use]
    fn signum(self) -> Self;

    /// Check if value is NaN
    #[must_use]
    fn is_nan(self) -> bool;

    /// Check if value is infinite
    #[must_use]
    fn is_infinite(self) -> bool;

    /// Check if value is finite
    #[must_use]
    fn is_finite(self) -> bool;
}

/// Extension trait for integer operations
///
/// Provides operations that are specific to integer types,
/// such as bitwise operations and overflow detection.
pub trait IntExt: DataType {
    /// Checked addition that returns None on overflow
    #[must_use]
    fn checked_add(self, rhs: Self) -> Option<Self>;

    /// Checked subtraction that returns None on overflow
    #[must_use]
    fn checked_sub(self, rhs: Self) -> Option<Self>;

    /// Checked multiplication that returns None on overflow
    #[must_use]
    fn checked_mul(self, rhs: Self) -> Option<Self>;

    /// Checked division that returns None on division by zero
    #[must_use]
    fn checked_div(self, rhs: Self) -> Option<Self>;

    /// Checked remainder that returns None on division by zero
    #[must_use]
    fn checked_rem(self, rhs: Self) -> Option<Self>;

    /// Checked negation that returns None on overflow
    #[must_use]
    fn checked_neg(self) -> Option<Self>;

    /// Checked absolute value that returns None on overflow
    #[must_use]
    fn checked_abs(self) -> Option<Self>;

    /// Bitwise AND
    #[must_use]
    fn bitand(self, rhs: Self) -> Self;

    /// Bitwise OR
    #[must_use]
    fn bitor(self, rhs: Self) -> Self;

    /// Bitwise XOR
    #[must_use]
    fn bitxor(self, rhs: Self) -> Self;

    /// Bitwise NOT
    #[must_use]
    fn bitnot(self) -> Self;

    /// Left shift (checked)
    #[must_use]
    fn checked_shl(self, rhs: u32) -> Option<Self>;

    /// Right shift (checked)
    #[must_use]
    fn checked_shr(self, rhs: u32) -> Option<Self>;

    /// Count leading zeros
    #[must_use]
    fn leading_zeros(self) -> u32;

    /// Count trailing zeros
    #[must_use]
    fn trailing_zeros(self) -> u32;

    /// Count set bits (population count)
    #[must_use]
    fn count_ones(self) -> u32;

    /// Count unset bits
    #[must_use]
    fn count_zeros(self) -> u32;
}

/// Extension trait for complex number operations
///
/// Provides operations specific to complex numbers, building on
/// the floating point operations.
#[cfg(feature = "complex")]
pub trait ComplexExt: DataType {
    /// The real component type (f32 or f64)
    type Real;

    /// Returns the real part
    #[must_use]
    fn re(self) -> Self::Real;

    /// Returns the imaginary part
    #[must_use]
    fn im(self) -> Self::Real;

    /// Returns the complex conjugate
    #[must_use]
    fn conj(self) -> Self;

    /// Returns the magnitude squared
    #[must_use]
    fn norm_sqr(self) -> Self::Real;

    /// Returns the magnitude
    #[must_use]
    fn norm(self) -> Self::Real;

    /// Returns the argument (phase angle)
    #[must_use]
    fn arg(self) -> Self::Real;
}

// Blanket implementations are provided in individual type implementations

// Implement Dtype conversions for primitive types
impl From<f32> for crate::Dtype {
    fn from(_: f32) -> Self {
        Self::Float32
    }
}

impl From<f64> for crate::Dtype {
    fn from(_: f64) -> Self {
        Self::Float64
    }
}

impl From<i8> for crate::Dtype {
    fn from(_: i8) -> Self {
        Self::Int8
    }
}

impl From<i16> for crate::Dtype {
    fn from(_: i16) -> Self {
        Self::Int16
    }
}

impl From<i32> for crate::Dtype {
    fn from(_: i32) -> Self {
        Self::Int32
    }
}

impl From<i64> for crate::Dtype {
    fn from(_: i64) -> Self {
        Self::Int64
    }
}

impl From<u8> for crate::Dtype {
    fn from(_: u8) -> Self {
        Self::UInt8
    }
}

impl From<u16> for crate::Dtype {
    fn from(_: u16) -> Self {
        Self::UInt16
    }
}

impl From<u32> for crate::Dtype {
    fn from(_: u32) -> Self {
        Self::UInt32
    }
}

impl From<u64> for crate::Dtype {
    fn from(_: u64) -> Self {
        Self::UInt64
    }
}

// Complex type conversions
#[cfg(feature = "complex")]
mod complex_conversions {
    // Note: super::* not needed as we only use num_complex types directly

    impl From<num_complex::Complex<f32>> for crate::Dtype {
        fn from(_: num_complex::Complex<f32>) -> Self {
            Self::Complex32
        }
    }

    impl From<num_complex::Complex<f64>> for crate::Dtype {
        fn from(_: num_complex::Complex<f64>) -> Self {
            Self::Complex64
        }
    }
}

// Half/BFloat16 type conversions deferred to Sprint 2
// #[cfg(feature = "half")]
// mod half_conversions {
//     use super::*;
//     impl From<half::f16> for crate::Dtype {
//         fn from(_: half::f16) -> Self {
//             Self::Half
//         }
//     }
//     impl From<half::bf16> for crate::Dtype {
//         fn from(_: half::bf16) -> Self {
//             Self::BFloat16
//         }
//     }
// }

#[cfg(feature = "complex")]
mod complex_datatype_impls {
    use super::{ComplexExt, DataType};
    use crate::complex::{Complex32, Complex64};

    impl DataType for Complex32 {
        fn dtype() -> crate::Dtype {
            crate::Dtype::Complex32
        }
    }

    impl DataType for Complex64 {
        fn dtype() -> crate::Dtype {
            crate::Dtype::Complex64
        }
    }

    impl ComplexExt for Complex32 {
        type Real = f32;

        fn re(self) -> Self::Real {
            self.re
        }

        fn im(self) -> Self::Real {
            self.im
        }

        fn conj(self) -> Self {
            Complex32::new(self.re, -self.im)
        }

        fn norm_sqr(self) -> Self::Real {
            self.re * self.re + self.im * self.im
        }

        fn norm(self) -> Self::Real {
            self.norm_sqr().sqrt()
        }

        fn arg(self) -> Self::Real {
            self.im.atan2(self.re)
        }
    }

    impl ComplexExt for Complex64 {
        type Real = f64;

        fn re(self) -> Self::Real {
            self.re
        }

        fn im(self) -> Self::Real {
            self.im
        }

        fn conj(self) -> Self {
            Complex64::new(self.re, -self.im)
        }

        fn norm_sqr(self) -> Self::Real {
            self.re * self.re + self.im * self.im
        }

        fn norm(self) -> Self::Real {
            self.norm_sqr().sqrt()
        }

        fn arg(self) -> Self::Real {
            self.im.atan2(self.re)
        }
    }
}
