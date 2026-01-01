//! Floating point data types
//!
//! Implements floating point types supported by Coeus.
//! Starting with Float32 and Float64 - Half/BFloat16 planned for later.

use core::fmt;
use num_traits::{Bounded, Float, Num, NumCast, One, Zero};

use crate::traits::{DataType, FloatExt};
use crate::Dtype;

// Re-export libm functions for no_std compatibility
#[cfg(not(feature = "std"))]
use libm;

/// 32-bit single precision floating point type
#[derive(Clone, Copy, Default, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Float32(pub f32);

impl Float32 {
    /// Create a new Float32
    #[must_use]
    pub const fn new(value: f32) -> Self {
        Self(value)
    }

    /// Get the inner f32 value
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl fmt::Debug for Float32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Float32({})", self.0)
    }
}

impl From<Float32> for f64 {
    fn from(value: Float32) -> f64 {
        core::convert::From::from(value.0)
    }
}

impl fmt::Display for Float32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Zero for Float32 {
    fn zero() -> Self {
        Self(0.0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl One for Float32 {
    fn one() -> Self {
        Self(1.0)
    }
}

impl Bounded for Float32 {
    fn min_value() -> Self {
        Self(f32::MIN)
    }

    fn max_value() -> Self {
        Self(f32::MAX)
    }
}

impl Num for Float32 {
    type FromStrRadixErr = core::num::ParseFloatError;

    fn from_str_radix(str: &str, radix: u32) -> core::result::Result<Self, Self::FromStrRadixErr> {
        if radix != 10 {
            // ParseFloatError doesn't have a public constructor
            return Err("1.0".parse::<f32>().unwrap_err());
        }
        str.parse::<f32>().map(Self)
    }
}

impl NumCast for Float32 {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        n.to_f32().map(Self::new)
    }
}

impl num_traits::ToPrimitive for Float32 {
    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }

    fn to_f64(&self) -> Option<f64> {
        Some(<f64 as core::convert::From<f32>>::from(self.0))
    }
}

/// Conversion from primitive types to Float32.
///
/// This implementation allows converting from various primitive numeric types
/// to Float32. All conversions preserve special values (NaN, infinity) and
/// handle precision loss gracefully.
///
/// # Precision Loss
///
/// - Integer to f32: Large integers may lose precision due to f32's 24-bit mantissa
/// - f64 to f32: Values outside f32 range become infinity, precision is reduced
///
/// # Examples
///
/// ```
/// use num_traits::FromPrimitive;
/// use num_traits::Float;
/// use dtype::float::Float32;
///
/// let from_int = Float32::from_i64(42).unwrap();
/// assert_eq!(from_int.get(), 42.0);
///
/// let from_float = Float32::from_f64(3.14159).unwrap();
/// assert!((from_float.get() - 3.14159).abs() < 1e-5);
///
/// // Special values are preserved
/// let inf = Float32::from_f64(f64::INFINITY).unwrap();
/// assert!(inf.is_infinite());
/// ```
impl num_traits::FromPrimitive for Float32 {
    /// Convert from i64 to Float32.
    ///
    /// Large integers may lose precision due to f32's 24-bit mantissa.
    /// Always returns Some(value) as all i64 values can be represented
    /// (though possibly with precision loss).
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self(n as f32))
    }

    /// Convert from u64 to Float32.
    ///
    /// Large integers may lose precision due to f32's 24-bit mantissa.
    /// Always returns Some(value) as all u64 values can be represented
    /// (though possibly with precision loss).
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    fn from_u64(n: u64) -> Option<Self> {
        Some(Self(n as f32))
    }

    /// Convert from f32 to Float32.
    ///
    /// This is a zero-cost conversion that preserves all values including
    /// NaN and infinity.
    #[inline]
    fn from_f32(n: f32) -> Option<Self> {
        Some(Self(n))
    }

    /// Convert from f64 to Float32.
    ///
    /// Values outside f32 range become infinity. Precision is reduced from
    /// 53-bit to 24-bit mantissa. Special values (NaN, infinity) are preserved.
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_precision_loss)]
    fn from_f64(n: f64) -> Option<Self> {
        Some(Self(n as f32))
    }
}

impl Float for Float32 {
    fn nan() -> Self {
        Self(f32::NAN)
    }

    fn infinity() -> Self {
        Self(f32::INFINITY)
    }

    fn neg_infinity() -> Self {
        Self(f32::NEG_INFINITY)
    }

    fn neg_zero() -> Self {
        Self(-0.0)
    }

    fn min_value() -> Self {
        Self(f32::MIN)
    }

    fn min_positive_value() -> Self {
        Self(f32::MIN_POSITIVE)
    }

    fn max_value() -> Self {
        Self(f32::MAX)
    }

    fn is_nan(self) -> bool {
        self.0.is_nan()
    }

    fn is_infinite(self) -> bool {
        self.0.is_infinite()
    }

    fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    fn is_normal(self) -> bool {
        self.0.is_normal()
    }

    fn classify(self) -> core::num::FpCategory {
        self.0.classify()
    }

    fn floor(self) -> Self {
        Self(self.0.floor())
    }

    fn ceil(self) -> Self {
        Self(self.0.ceil())
    }

    fn round(self) -> Self {
        Self(self.0.round())
    }

    fn trunc(self) -> Self {
        Self(self.0.trunc())
    }

    fn fract(self) -> Self {
        Self(self.0.fract())
    }

    fn abs(self) -> Self {
        Self(self.0.abs())
    }

    fn signum(self) -> Self {
        Self(self.0.signum())
    }

    fn is_sign_positive(self) -> bool {
        self.0.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.0.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0.mul_add(a.0, b.0))
    }

    fn recip(self) -> Self {
        Self(1.0 / self.0)
    }

    fn powi(self, n: i32) -> Self {
        Self(self.0.powi(n))
    }

    fn powf(self, n: Self) -> Self {
        Self(self.0.powf(n.0))
    }

    fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    fn exp(self) -> Self {
        Self(self.0.exp())
    }

    fn exp2(self) -> Self {
        Self(self.0.exp2())
    }

    fn ln(self) -> Self {
        Self(self.0.ln())
    }

    fn log(self, base: Self) -> Self {
        Self(self.0.log(base.0))
    }

    fn log2(self) -> Self {
        Self(self.0.log2())
    }

    fn log10(self) -> Self {
        Self(self.0.log10())
    }

    fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    fn abs_sub(self, other: Self) -> Self {
        Self((self.0 - other.0).abs())
    }

    fn cbrt(self) -> Self {
        Self(self.0.cbrt())
    }

    fn hypot(self, other: Self) -> Self {
        Self(self.0.hypot(other.0))
    }

    fn sin(self) -> Self {
        Self(self.0.sin())
    }

    fn cos(self) -> Self {
        Self(self.0.cos())
    }

    fn tan(self) -> Self {
        Self(self.0.tan())
    }

    fn asin(self) -> Self {
        Self(self.0.asin())
    }

    fn acos(self) -> Self {
        Self(self.0.acos())
    }

    fn atan(self) -> Self {
        Self(self.0.atan())
    }

    fn atan2(self, other: Self) -> Self {
        Self(self.0.atan2(other.0))
    }

    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = self.0.sin_cos();
        (Self(sin), Self(cos))
    }

    fn exp_m1(self) -> Self {
        Self(self.0.exp_m1())
    }

    fn ln_1p(self) -> Self {
        Self(self.0.ln_1p())
    }

    fn sinh(self) -> Self {
        Self(self.0.sinh())
    }

    fn cosh(self) -> Self {
        Self(self.0.cosh())
    }

    fn tanh(self) -> Self {
        Self(self.0.tanh())
    }

    fn asinh(self) -> Self {
        Self(self.0.asinh())
    }

    fn acosh(self) -> Self {
        Self(self.0.acosh())
    }

    fn atanh(self) -> Self {
        Self(self.0.atanh())
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        let bits = self.0.to_bits();
        let sign = ((bits >> 31) & 1) as i8;
        let exponent = ((bits >> 23) & 0xff) as i16;
        let mantissa = bits & 0x007f_ffff;
        #[allow(clippy::cast_lossless)]
        (mantissa as u64, exponent - 127, sign)
    }

    fn epsilon() -> Self {
        Self(f32::EPSILON)
    }

    fn to_degrees(self) -> Self {
        Self(self.0.to_degrees())
    }

    fn to_radians(self) -> Self {
        Self(self.0.to_radians())
    }
}

impl From<f32> for Float32 {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

impl From<f64> for Float32 {
    fn from(value: f64) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Self(value as f32)
    }
}

impl Float for Float64 {
    fn nan() -> Self {
        Self(f64::NAN)
    }

    fn infinity() -> Self {
        Self(f64::INFINITY)
    }

    fn neg_infinity() -> Self {
        Self(f64::NEG_INFINITY)
    }

    fn neg_zero() -> Self {
        Self(-0.0)
    }

    fn min_value() -> Self {
        Self(f64::MIN)
    }

    fn min_positive_value() -> Self {
        Self(f64::MIN_POSITIVE)
    }

    fn max_value() -> Self {
        Self(f64::MAX)
    }

    fn is_nan(self) -> bool {
        self.0.is_nan()
    }

    fn is_infinite(self) -> bool {
        self.0.is_infinite()
    }

    fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    fn is_normal(self) -> bool {
        self.0.is_normal()
    }

    fn classify(self) -> core::num::FpCategory {
        self.0.classify()
    }

    fn floor(self) -> Self {
        Self(self.0.floor())
    }

    fn ceil(self) -> Self {
        Self(self.0.ceil())
    }

    fn round(self) -> Self {
        Self(self.0.round())
    }

    fn trunc(self) -> Self {
        Self(self.0.trunc())
    }

    fn fract(self) -> Self {
        Self(self.0.fract())
    }

    fn abs(self) -> Self {
        Self(self.0.abs())
    }

    fn signum(self) -> Self {
        Self(self.0.signum())
    }

    fn is_sign_positive(self) -> bool {
        self.0.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.0.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0.mul_add(a.0, b.0))
    }

    fn recip(self) -> Self {
        Self(1.0 / self.0)
    }

    fn powi(self, n: i32) -> Self {
        Self(self.0.powi(n))
    }

    fn powf(self, n: Self) -> Self {
        Self(self.0.powf(n.0))
    }

    fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    fn exp(self) -> Self {
        Self(self.0.exp())
    }

    fn exp2(self) -> Self {
        Self(self.0.exp2())
    }

    fn ln(self) -> Self {
        Self(self.0.ln())
    }

    fn log(self, base: Self) -> Self {
        Self(self.0.log(base.0))
    }

    fn log2(self) -> Self {
        Self(self.0.log2())
    }

    fn log10(self) -> Self {
        Self(self.0.log10())
    }

    fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    fn abs_sub(self, other: Self) -> Self {
        Self((self.0 - other.0).abs())
    }

    fn cbrt(self) -> Self {
        Self(self.0.cbrt())
    }

    fn hypot(self, other: Self) -> Self {
        Self(self.0.hypot(other.0))
    }

    fn sin(self) -> Self {
        Self(self.0.sin())
    }

    fn cos(self) -> Self {
        Self(self.0.cos())
    }

    fn tan(self) -> Self {
        Self(self.0.tan())
    }

    fn asin(self) -> Self {
        Self(self.0.asin())
    }

    fn acos(self) -> Self {
        Self(self.0.acos())
    }

    fn atan(self) -> Self {
        Self(self.0.atan())
    }

    fn atan2(self, other: Self) -> Self {
        Self(self.0.atan2(other.0))
    }

    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = self.0.sin_cos();
        (Self(sin), Self(cos))
    }

    fn exp_m1(self) -> Self {
        Self(self.0.exp_m1())
    }

    fn ln_1p(self) -> Self {
        Self(self.0.ln_1p())
    }

    fn sinh(self) -> Self {
        Self(self.0.sinh())
    }

    fn cosh(self) -> Self {
        Self(self.0.cosh())
    }

    fn tanh(self) -> Self {
        Self(self.0.tanh())
    }

    fn asinh(self) -> Self {
        Self(self.0.asinh())
    }

    fn acosh(self) -> Self {
        Self(self.0.acosh())
    }

    fn atanh(self) -> Self {
        Self(self.0.atanh())
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        let bits = self.0.to_bits();
        let sign = ((bits >> 63) & 1) as i8;
        let exponent = ((bits >> 52) & 0x7ff) as i16;
        let mantissa = bits & 0x000f_ffff_ffff_ffff;
        (mantissa, exponent - 1023, sign)
    }

    fn epsilon() -> Self {
        Self(f64::EPSILON)
    }

    fn to_degrees(self) -> Self {
        Self(self.0.to_degrees())
    }

    fn to_radians(self) -> Self {
        Self(self.0.to_radians())
    }
}

impl core::ops::Add for Float32 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl core::ops::Sub for Float32 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl core::ops::Mul for Float32 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl core::ops::Div for Float32 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl core::ops::Rem for Float32 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        Self(self.0 % rhs.0)
    }
}

impl core::ops::Neg for Float32 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl core::ops::AddAssign for Float32 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl core::ops::SubAssign for Float32 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl core::ops::MulAssign for Float32 {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl core::ops::DivAssign for Float32 {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl core::iter::Sum for Float32 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a> core::iter::Sum<&'a Self> for Float32 {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + *x)
    }
}

// NumOps is implemented through the Add/Sub/Mul/Div/Rem trait implementations above

impl DataType for Float32 {
    fn dtype() -> Dtype {
        Dtype::Float32
    }
}

unsafe impl bytemuck::Pod for Float32 {}

unsafe impl bytemuck::Zeroable for Float32 {}

impl FloatExt for Float32 {
    fn erf(self) -> Self {
        Self(libm::erff(self.0))
    }

    fn erfc(self) -> Self {
        Self(libm::erfcf(self.0))
    }
}

/// 64-bit double precision floating point type
#[derive(Clone, Copy, Default, PartialEq, PartialOrd)]
pub struct Float64(pub f64);

impl Float64 {
    /// Create a new Float64
    #[must_use]
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// Get the inner f64 value
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl fmt::Debug for Float64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Float64({})", self.0)
    }
}

impl fmt::Display for Float64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Zero for Float64 {
    fn zero() -> Self {
        Self(0.0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl One for Float64 {
    fn one() -> Self {
        Self(1.0)
    }
}

impl Bounded for Float64 {
    fn min_value() -> Self {
        Self(f64::MIN)
    }

    fn max_value() -> Self {
        Self(f64::MAX)
    }
}

impl Num for Float64 {
    type FromStrRadixErr = core::num::ParseFloatError;

    fn from_str_radix(str: &str, radix: u32) -> core::result::Result<Self, Self::FromStrRadixErr> {
        if radix != 10 {
            // ParseFloatError doesn't have a public constructor
            return Err("1.0".parse::<f32>().unwrap_err());
        }
        str.parse::<f64>().map(Self)
    }
}

impl NumCast for Float64 {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        n.to_f64().map(Self::new)
    }
}

impl num_traits::ToPrimitive for Float64 {
    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.0)
    }
}

/// Conversion from primitive types to Float64.
///
/// This implementation allows converting from various primitive numeric types
/// to Float64. All conversions preserve special values (NaN, infinity) and
/// handle precision loss gracefully.
///
/// # Precision Loss
///
/// - Integer to f64: Very large integers (>2^53) may lose precision due to f64's 53-bit mantissa
/// - f32 to f64: Lossless conversion (f32 mantissa fits in f64)
///
/// # Examples
///
/// ```
/// use num_traits::FromPrimitive;
/// use num_traits::Float;
/// use dtype::float::Float64;
///
/// let from_int = Float64::from_i64(42).unwrap();
/// assert_eq!(from_int.get(), 42.0);
///
/// let from_float = Float64::from_f32(3.14159).unwrap();
/// assert!((from_float.get() - 3.14159).abs() < 1e-6);
///
/// // Special values are preserved
/// let inf = Float64::from_f64(f64::INFINITY).unwrap();
/// assert!(inf.is_infinite());
/// ```
impl num_traits::FromPrimitive for Float64 {
    /// Convert from i64 to Float64.
    ///
    /// Integers larger than 2^53 may lose precision due to f64's 53-bit mantissa.
    /// Always returns Some(value) as all i64 values can be represented
    /// (though possibly with precision loss for very large values).
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self(n as f64))
    }

    /// Convert from u64 to Float64.
    ///
    /// Integers larger than 2^53 may lose precision due to f64's 53-bit mantissa.
    /// Always returns Some(value) as all u64 values can be represented
    /// (though possibly with precision loss for very large values).
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    fn from_u64(n: u64) -> Option<Self> {
        Some(Self(n as f64))
    }

    /// Convert from f32 to Float64.
    ///
    /// This is a lossless conversion as f32's 24-bit mantissa fits entirely
    /// within f64's 53-bit mantissa. Special values (NaN, infinity) are preserved.
    #[inline]
    #[allow(clippy::cast_lossless)]
    fn from_f32(n: f32) -> Option<Self> {
        Some(Self(n as f64))
    }

    /// Convert from f64 to Float64.
    ///
    /// This is a zero-cost conversion that preserves all values including
    /// NaN and infinity.
    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        Some(Self(n))
    }
}

impl core::ops::Add for Float64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl core::ops::Sub for Float64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl core::ops::Mul for Float64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl core::ops::Div for Float64 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl core::ops::Rem for Float64 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        Self(self.0 % rhs.0)
    }
}

impl core::ops::Neg for Float64 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl core::ops::AddAssign for Float64 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl core::ops::SubAssign for Float64 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl core::ops::MulAssign for Float64 {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl core::ops::DivAssign for Float64 {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl core::iter::Sum for Float64 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a> core::iter::Sum<&'a Self> for Float64 {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + *x)
    }
}

// NumOps is implemented through the Add/Sub/Mul/Div/Rem trait implementations above

impl DataType for Float64 {
    fn dtype() -> Dtype {
        Dtype::Float64
    }
}

impl FloatExt for Float64 {
    fn erf(self) -> Self {
        Self(libm::erf(self.0))
    }

    fn erfc(self) -> Self {
        Self(libm::erfc(self.0))
    }
}

/// 16-bit half precision floating point type
#[derive(Clone, Copy, Default, PartialEq, PartialOrd)]
#[cfg(feature = "half")]
pub struct Half(pub half::f16);

#[cfg(feature = "half")]
impl Half {
    /// Create a new Half
    #[must_use]
    pub fn new(value: f32) -> Self {
        Self(half::f16::from_f32(value))
    }

    /// Get the inner f16 value as f32
    #[must_use]
    pub fn get(self) -> f32 {
        self.0.to_f32()
    }
}

#[cfg(feature = "half")]
impl fmt::Debug for Half {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Half({})", self.get())
    }
}

#[cfg(feature = "half")]
impl fmt::Display for Half {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.get())
    }
}

#[cfg(feature = "half")]
impl Zero for Half {
    fn zero() -> Self {
        Self(half::f16::ZERO)
    }

    fn is_zero(&self) -> bool {
        self.0.to_bits() == 0
    }
}

#[cfg(feature = "half")]
impl One for Half {
    fn one() -> Self {
        Self(half::f16::ONE)
    }
}

#[cfg(feature = "half")]
impl Num for Half {
    type FromStrRadixErr = core::num::ParseFloatError;

    #[allow(clippy::redundant_closure)]
    fn from_str_radix(str: &str, radix: u32) -> core::result::Result<Self, Self::FromStrRadixErr> {
        if radix != 10 {
            return Err("invalid radix".parse::<f32>().unwrap_err());
        }
        str.parse::<f32>().map(|v| Self::new(v))
    }
}

#[cfg(feature = "half")]
impl NumCast for Half {
    #[allow(clippy::redundant_closure)]
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        n.to_f32().map(|v| Self::new(v))
    }
}

#[cfg(feature = "half")]
impl num_traits::ToPrimitive for Half {
    fn to_i64(&self) -> Option<i64> {
        self.get().to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.get().to_u64()
    }

    fn to_f32(&self) -> Option<f32> {
        Some(self.get())
    }

    #[allow(clippy::cast_lossless)]
    fn to_f64(&self) -> Option<f64> {
        self.to_f32().map(|v| v as f64)
    }
}

#[cfg(feature = "half")]
impl core::ops::Add for Half {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(half::f16::from_f32(self.get() + rhs.get()))
    }
}

#[cfg(feature = "half")]
impl core::ops::Sub for Half {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(half::f16::from_f32(self.get() - rhs.get()))
    }
}

#[cfg(feature = "half")]
impl core::ops::Mul for Half {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(half::f16::from_f32(self.get() * rhs.get()))
    }
}

#[cfg(feature = "half")]
impl core::ops::Div for Half {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self(half::f16::from_f32(self.get() / rhs.get()))
    }
}

#[cfg(feature = "half")]
impl core::ops::Rem for Half {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        Self(half::f16::from_f32(self.get() % rhs.get()))
    }
}

#[cfg(feature = "half")]
impl core::ops::Neg for Half {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(half::f16::from_f32(-self.get()))
    }
}

#[cfg(feature = "half")]
impl core::ops::AddAssign for Half {
    fn add_assign(&mut self, rhs: Self) {
        *self = Self(half::f16::from_f32(self.get() + rhs.get()));
    }
}

#[cfg(feature = "half")]
impl DataType for Half {
    fn dtype() -> Dtype {
        Dtype::Half
    }
}

// Half does not implement FloatExt as it doesn't implement num_traits::Float

/// 16-bit brain floating point type
#[derive(Clone, Copy, Default, PartialEq, PartialOrd)]
#[cfg(feature = "half")]
pub struct BFloat16(pub half::bf16);

#[cfg(feature = "half")]
impl BFloat16 {
    /// Create a new `BFloat16`
    #[must_use]
    pub fn new(value: f32) -> Self {
        Self(half::bf16::from_f32(value))
    }

    /// Get the inner bf16 value as f32
    #[must_use]
    pub fn get(self) -> f32 {
        self.0.to_f32()
    }
}

#[cfg(feature = "half")]
impl fmt::Debug for BFloat16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BFloat16({})", self.get())
    }
}

#[cfg(feature = "half")]
impl fmt::Display for BFloat16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.get())
    }
}

#[cfg(feature = "half")]
impl Zero for BFloat16 {
    fn zero() -> Self {
        Self(half::bf16::ZERO)
    }

    fn is_zero(&self) -> bool {
        self.0.to_bits() == 0
    }
}

#[cfg(feature = "half")]
impl One for BFloat16 {
    fn one() -> Self {
        Self(half::bf16::ONE)
    }
}

#[cfg(feature = "half")]
impl Num for BFloat16 {
    type FromStrRadixErr = core::num::ParseFloatError;

    #[allow(clippy::redundant_closure)]
    fn from_str_radix(str: &str, radix: u32) -> core::result::Result<Self, Self::FromStrRadixErr> {
        if radix != 10 {
            return Err("invalid radix".parse::<f32>().unwrap_err());
        }
        str.parse::<f32>().map(|v| Self::new(v))
    }
}

#[cfg(feature = "half")]
impl NumCast for BFloat16 {
    #[allow(clippy::redundant_closure)]
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        n.to_f32().map(|v| Self::new(v))
    }
}

#[cfg(feature = "half")]
impl num_traits::ToPrimitive for BFloat16 {
    fn to_i64(&self) -> Option<i64> {
        self.get().to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.get().to_u64()
    }

    fn to_f32(&self) -> Option<f32> {
        Some(self.get())
    }

    #[allow(clippy::cast_lossless)]
    fn to_f64(&self) -> Option<f64> {
        self.to_f32().map(|v| v as f64)
    }
}

#[cfg(feature = "half")]
impl core::ops::Add for BFloat16 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(half::bf16::from_f32(self.get() + rhs.get()))
    }
}

#[cfg(feature = "half")]
impl core::ops::Sub for BFloat16 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(half::bf16::from_f32(self.get() - rhs.get()))
    }
}

#[cfg(feature = "half")]
impl core::ops::Mul for BFloat16 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(half::bf16::from_f32(self.get() * rhs.get()))
    }
}

#[cfg(feature = "half")]
impl core::ops::Div for BFloat16 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self(half::bf16::from_f32(self.get() / rhs.get()))
    }
}

#[cfg(feature = "half")]
impl core::ops::Rem for BFloat16 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        Self(half::bf16::from_f32(self.get() % rhs.get()))
    }
}

#[cfg(feature = "half")]
impl core::ops::Neg for BFloat16 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(half::bf16::from_f32(-self.get()))
    }
}

#[cfg(feature = "half")]
impl core::ops::AddAssign for BFloat16 {
    fn add_assign(&mut self, rhs: Self) {
        *self = Self(half::bf16::from_f32(self.get() + rhs.get()));
    }
}

#[cfg(feature = "half")]
impl DataType for BFloat16 {
    fn dtype() -> Dtype {
        Dtype::BFloat16
    }
}

#[cfg(feature = "half")]
// BFloat16 does not implement FloatExt as it doesn't implement num_traits::Float
// Re-exports for convenience
pub use self::{BFloat16 as BF16, Half as F16};
pub use self::{Float32 as F32, Float64 as F64};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_float32_arithmetic() {
        let a = Float32(3.0);
        let b = Float32(2.0);

        assert_eq!(a + b, Float32(5.0));
        assert_eq!(a - b, Float32(1.0));
        assert_eq!(a * b, Float32(6.0));
        assert_relative_eq!((a / b).get(), 1.5, epsilon = 1e-6);
    }

    #[test]
    fn test_float64_arithmetic() {
        let a = Float64(3.0);
        let b = Float64(2.0);

        assert_eq!(a + b, Float64(5.0));
        assert_eq!(a - b, Float64(1.0));
        assert_eq!(a * b, Float64(6.0));
        assert_relative_eq!((a / b).get(), 1.5, epsilon = 1e-12);
    }

    #[test]
    fn test_float32_math_functions() {
        let x = Float32(1.0);

        assert_relative_eq!(
            num_traits::Float::exp(x).get(),
            core::f32::consts::E,
            epsilon = 1e-5
        );
        assert_relative_eq!(num_traits::Float::ln(x).get(), 0.0, epsilon = 1e-6);
        assert_relative_eq!(num_traits::Float::sqrt(x).get(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(
            num_traits::Float::sin(x).get(),
            0.841_470_96,
            epsilon = 1e-6
        );
        assert_relative_eq!(num_traits::Float::cos(x).get(), 0.540_302_3, epsilon = 1e-6);
    }

    #[test]
    fn test_float64_math_functions() {
        let x = Float64(1.0);

        assert_relative_eq!(x.exp().get(), core::f64::consts::E, epsilon = 1e-12);
        assert_relative_eq!(x.ln().get(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(x.sqrt().get(), 1.0, epsilon = 1e-12);
        assert_relative_eq!(x.sin().get(), 1.0_f64.sin(), epsilon = 1e-12);
        assert_relative_eq!(x.cos().get(), 1.0_f64.cos(), epsilon = 1e-12);
    }

    #[test]
    fn test_float32_special_values() {
        let zero = Float32::zero();
        let one = Float32::one();
        let inf = Float32(f32::INFINITY);
        let neg_inf = Float32(f32::NEG_INFINITY);
        let nan = Float32(f32::NAN);

        assert!(zero.is_zero());
        assert!(one.is_one());
        assert!(num_traits::Float::is_infinite(inf));
        assert!(num_traits::Float::is_infinite(neg_inf));
        assert!(num_traits::Float::is_nan(nan));
    }

    #[test]
    fn test_float64_special_values() {
        let zero = Float64::zero();
        let one = Float64::one();
        let inf = Float64(f64::INFINITY);
        let neg_inf = Float64(f64::NEG_INFINITY);
        let nan = Float64(f64::NAN);

        assert!(zero.is_zero());
        assert!(one.is_one());
        assert!(num_traits::Float::is_infinite(inf));
        assert!(num_traits::Float::is_infinite(neg_inf));
        assert!(num_traits::Float::is_nan(nan));
    }

    #[test]
    fn test_float32_dtype() {
        assert_eq!(Float32::dtype(), Dtype::Float32);
        assert!(Float32::dtype().is_floating_point());
        assert!(!Float32::dtype().is_integer());
    }

    #[test]
    fn test_float64_dtype() {
        assert_eq!(Float64::dtype(), Dtype::Float64);
        assert!(Float64::dtype().is_floating_point());
        assert!(!Float64::dtype().is_integer());
    }

    #[test]
    fn test_half_arithmetic() {
        #[cfg(feature = "half")]
        {
            let a = Half(half::f16::from_f32(3.0));
            let b = Half(half::f16::from_f32(2.0));

            assert_eq!(a + b, Half(half::f16::from_f32(5.0)));
            assert_eq!(a - b, Half(half::f16::from_f32(1.0)));
            assert_eq!(a * b, Half(half::f16::from_f32(6.0)));
            assert_relative_eq!((a / b).get(), 1.5, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_bfloat16_arithmetic() {
        #[cfg(feature = "half")]
        {
            let a = BFloat16(half::bf16::from_f32(3.0));
            let b = BFloat16(half::bf16::from_f32(2.0));

            assert_eq!(a + b, BFloat16(half::bf16::from_f32(5.0)));
            assert_eq!(a - b, BFloat16(half::bf16::from_f32(1.0)));
            assert_eq!(a * b, BFloat16(half::bf16::from_f32(6.0)));
            assert_relative_eq!((a / b).get(), 1.5, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_half_dtype() {
        #[cfg(feature = "half")]
        {
            assert_eq!(Half::dtype(), Dtype::Half);
            assert!(Half::dtype().is_floating_point());
            assert!(!Half::dtype().is_integer());
        }
    }

    #[test]
    fn test_bfloat16_dtype() {
        #[cfg(feature = "half")]
        {
            assert_eq!(BFloat16::dtype(), Dtype::BFloat16);
            assert!(BFloat16::dtype().is_floating_point());
            assert!(!BFloat16::dtype().is_integer());
        }
    }

    // FromPrimitive tests for Float32
    #[test]
    fn test_float32_from_i64() {
        use num_traits::FromPrimitive;
        use num_traits::ToPrimitive;

        // Normal values
        assert_eq!(
            Float32::from_i64(0).unwrap().get().to_bits(),
            0.0f32.to_bits()
        );
        assert_eq!(
            Float32::from_i64(42).unwrap().get().to_bits(),
            42.0f32.to_bits()
        );
        assert_eq!(
            Float32::from_i64(-42).unwrap().get().to_bits(),
            (-42.0f32).to_bits()
        );
        assert_eq!(
            Float32::from_i64(i64::MAX).unwrap().get().to_bits(),
            i64::MAX.to_f32().unwrap().to_bits()
        );
        assert_eq!(
            Float32::from_i64(i64::MIN).unwrap().get().to_bits(),
            i64::MIN.to_f32().unwrap().to_bits()
        );

        // Large values (precision loss expected)
        let large = 1_000_000_000_000_i64;
        let converted = Float32::from_i64(large).unwrap();
        assert!((converted.get() - large.to_f32().unwrap()).abs() < 1e6);
    }

    #[test]
    fn test_float32_from_u64() {
        use num_traits::FromPrimitive;
        use num_traits::ToPrimitive;

        // Normal values
        assert_eq!(
            Float32::from_u64(0).unwrap().get().to_bits(),
            0.0f32.to_bits()
        );
        assert_eq!(
            Float32::from_u64(42).unwrap().get().to_bits(),
            42.0f32.to_bits()
        );
        assert_eq!(
            Float32::from_u64(u64::MAX).unwrap().get().to_bits(),
            u64::MAX.to_f32().unwrap().to_bits()
        );

        // Large values (precision loss expected)
        let large = 1_000_000_000_000_u64;
        let converted = Float32::from_u64(large).unwrap();
        assert!((converted.get() - large.to_f32().unwrap()).abs() < 1e6);
    }

    #[test]
    fn test_float32_from_f32() {
        use num_traits::FromPrimitive;

        // Normal values
        let normal_pos = 3.25_f32;
        let normal_neg = -2.75_f32;
        assert_eq!(
            Float32::from_f32(0.0).unwrap().get().to_bits(),
            0.0f32.to_bits()
        );
        assert_eq!(
            Float32::from_f32(normal_pos).unwrap().get().to_bits(),
            normal_pos.to_bits()
        );
        assert_eq!(
            Float32::from_f32(normal_neg).unwrap().get().to_bits(),
            normal_neg.to_bits()
        );

        // Special values
        assert!(num_traits::Float::is_infinite(
            Float32::from_f32(f32::INFINITY).unwrap()
        ));
        assert!(num_traits::Float::is_infinite(
            Float32::from_f32(f32::NEG_INFINITY).unwrap()
        ));
        assert!(num_traits::Float::is_nan(
            Float32::from_f32(f32::NAN).unwrap()
        ));

        // Edge cases
        assert_eq!(
            Float32::from_f32(f32::MIN).unwrap().get().to_bits(),
            f32::MIN.to_bits()
        );
        assert_eq!(
            Float32::from_f32(f32::MAX).unwrap().get().to_bits(),
            f32::MAX.to_bits()
        );
    }

    #[test]
    fn test_float32_from_f64() {
        use num_traits::FromPrimitive;

        // Normal values
        assert_eq!(
            Float32::from_f64(0.0).unwrap().get().to_bits(),
            0.0f32.to_bits()
        );
        assert_relative_eq!(Float32::from_f64(3.25).unwrap().get(), 3.25, epsilon = 1e-5);
        assert_relative_eq!(
            Float32::from_f64(-2.75).unwrap().get(),
            -2.75,
            epsilon = 1e-5
        );

        // Special values
        assert!(num_traits::Float::is_infinite(
            Float32::from_f64(f64::INFINITY).unwrap()
        ));
        assert!(num_traits::Float::is_infinite(
            Float32::from_f64(f64::NEG_INFINITY).unwrap()
        ));
        assert!(num_traits::Float::is_nan(
            Float32::from_f64(f64::NAN).unwrap()
        ));

        // Values outside f32 range become infinity
        assert!(num_traits::Float::is_infinite(
            Float32::from_f64(f64::MAX).unwrap()
        ));
        assert!(num_traits::Float::is_infinite(
            Float32::from_f64(-f64::MAX).unwrap()
        ));

        // Precision loss from f64 to f32
        let precise = 1.234_567_890_123_456_7_f64;
        let converted = Float32::from_f64(precise).unwrap();
        #[allow(clippy::cast_possible_truncation)]
        let expected = precise as f32;
        assert_relative_eq!(converted.get(), expected, epsilon = 1e-6);
    }

    // FromPrimitive tests for Float64
    #[test]
    fn test_float64_from_i64() {
        use num_traits::FromPrimitive;
        use num_traits::ToPrimitive;

        // Normal values
        assert_eq!(
            Float64::from_i64(0).unwrap().get().to_bits(),
            0.0f64.to_bits()
        );
        assert_eq!(
            Float64::from_i64(42).unwrap().get().to_bits(),
            42.0f64.to_bits()
        );
        assert_eq!(
            Float64::from_i64(-42).unwrap().get().to_bits(),
            (-42.0f64).to_bits()
        );
        assert_eq!(
            Float64::from_i64(i64::MAX).unwrap().get().to_bits(),
            i64::MAX.to_f64().unwrap().to_bits()
        );
        assert_eq!(
            Float64::from_i64(i64::MIN).unwrap().get().to_bits(),
            i64::MIN.to_f64().unwrap().to_bits()
        );

        // Large values (precision loss for values > 2^53)
        let large = 1_000_000_000_000_000_i64;
        let converted = Float64::from_i64(large).unwrap();
        assert_relative_eq!(converted.get(), large.to_f64().unwrap(), epsilon = 1.0);
    }

    #[test]
    fn test_float64_from_u64() {
        use num_traits::FromPrimitive;
        use num_traits::ToPrimitive;

        // Normal values
        assert_eq!(
            Float64::from_u64(0).unwrap().get().to_bits(),
            0.0f64.to_bits()
        );
        assert_eq!(
            Float64::from_u64(42).unwrap().get().to_bits(),
            42.0f64.to_bits()
        );
        assert_eq!(
            Float64::from_u64(u64::MAX).unwrap().get().to_bits(),
            u64::MAX.to_f64().unwrap().to_bits()
        );

        // Large values (precision loss for values > 2^53)
        let large = 1_000_000_000_000_000_u64;
        let converted = Float64::from_u64(large).unwrap();
        assert_relative_eq!(converted.get(), large.to_f64().unwrap(), epsilon = 1.0);
    }

    #[test]
    fn test_float64_from_f32() {
        use num_traits::FromPrimitive;

        // Normal values (lossless conversion)
        let normal_pos = 3.25_f32;
        let normal_neg = -2.75_f32;
        assert_eq!(
            Float64::from_f32(0.0).unwrap().get().to_bits(),
            0.0f64.to_bits()
        );
        assert_relative_eq!(
            Float64::from_f32(normal_pos).unwrap().get(),
            <f64 as core::convert::From<f32>>::from(normal_pos),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            Float64::from_f32(normal_neg).unwrap().get(),
            <f64 as core::convert::From<f32>>::from(normal_neg),
            epsilon = 1e-12
        );

        // Special values
        assert!(Float64::from_f32(f32::INFINITY).unwrap().is_infinite());
        assert!(Float64::from_f32(f32::NEG_INFINITY).unwrap().is_infinite());
        assert!(Float64::from_f32(f32::NAN).unwrap().is_nan());

        // Edge cases
        assert_eq!(
            Float64::from_f32(f32::MIN).unwrap().get().to_bits(),
            <f64 as core::convert::From<f32>>::from(f32::MIN).to_bits()
        );
        assert_eq!(
            Float64::from_f32(f32::MAX).unwrap().get().to_bits(),
            <f64 as core::convert::From<f32>>::from(f32::MAX).to_bits()
        );
    }

    #[test]
    fn test_float64_from_f64() {
        use num_traits::FromPrimitive;

        // Normal values
        let normal_pos = 3.25_f64;
        let normal_neg = -2.75_f64;
        assert_eq!(
            Float64::from_f64(0.0).unwrap().get().to_bits(),
            0.0f64.to_bits()
        );
        assert_eq!(
            Float64::from_f64(normal_pos).unwrap().get().to_bits(),
            normal_pos.to_bits()
        );
        assert_eq!(
            Float64::from_f64(normal_neg).unwrap().get().to_bits(),
            normal_neg.to_bits()
        );

        // Special values
        assert!(Float64::from_f64(f64::INFINITY).unwrap().is_infinite());
        assert!(Float64::from_f64(f64::NEG_INFINITY).unwrap().is_infinite());
        assert!(Float64::from_f64(f64::NAN).unwrap().is_nan());

        // Edge cases
        assert_eq!(
            Float64::from_f64(f64::MIN).unwrap().get().to_bits(),
            f64::MIN.to_bits()
        );
        assert_eq!(
            Float64::from_f64(f64::MAX).unwrap().get().to_bits(),
            f64::MAX.to_bits()
        );
    }

    #[test]
    fn test_from_primitive_zero_cost() {
        use num_traits::FromPrimitive;

        // Verify that FromPrimitive conversions are zero-cost
        // (compiler should optimize these to simple wrapping)
        let f32_val = Float32::from_f32(1.0).unwrap();
        let f64_val = Float64::from_f64(1.0).unwrap();

        assert_eq!(f32_val.get().to_bits(), 1.0f32.to_bits());
        assert_eq!(f64_val.get().to_bits(), 1.0f64.to_bits());
    }
}
