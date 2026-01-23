//! Half-precision floating point types
//!
//! Provides Half (f16) and BFloat16 type wrappers with full trait implementations
//! for use in mixed-precision training and inference.
//!
//! # Features
//!
//! - `Half`: IEEE 754 half-precision (f16) - more precision, good for inference
//! - `BFloat16`: Brain floating point - more range, good for training gradients
//!
//! # Example
//!
//! ```rust
//! use dtype::half::{Half, BFloat16};
//! use dtype::traits::DataType;
//!
//! let h = Half::new(1.5);
//! let bf = BFloat16::new(2.0);
//!
//! assert!(Half::is_floating_point());
//! assert!(BFloat16::is_floating_point());
//! ```

use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Rem, Sub};
use core::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use num_traits::{Bounded, Float, FromPrimitive, Num, NumCast, One, ToPrimitive, Zero};

use crate::traits::{DataType, FloatExt};
use crate::Dtype;

// Re-export the half crate types for convenience
pub use ::half::{bf16, f16};

/// IEEE 754 half-precision (16-bit) floating point type
///
/// Half provides more precision than BFloat16 but less dynamic range.
/// Best suited for:
/// - Inference workloads
/// - Storage of model weights
/// - Memory-constrained environments
#[derive(Clone, Copy, Default, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(transparent)]
pub struct Half(pub f16);

/// Brain floating point (16-bit) type
///
/// BFloat16 provides more dynamic range than Half but less precision.
/// Best suited for:
/// - Training with mixed precision
/// - Gradient accumulation
/// - Operations requiring wider dynamic range
#[derive(Clone, Copy, Default, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(transparent)]
pub struct BFloat16(pub bf16);

// ============================================================================
// Half Implementation
// ============================================================================

impl Half {
    /// Create a new Half from an f32 value
    #[inline]
    #[must_use]
    pub fn new(value: f32) -> Self {
        Self(f16::from_f32(value))
    }

    /// Get the inner value as f32
    #[inline]
    #[must_use]
    pub fn get(self) -> f32 {
        self.0.to_f32()
    }

    /// Get the raw f16 value
    #[inline]
    #[must_use]
    pub fn raw(self) -> f16 {
        self.0
    }
}

impl fmt::Debug for Half {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Half({})", self.get())
    }
}

impl fmt::Display for Half {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.get())
    }
}

impl From<f32> for Half {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl From<Half> for f32 {
    fn from(value: Half) -> Self {
        value.get()
    }
}

impl From<Half> for f64 {
    fn from(value: Half) -> Self {
        value.get() as f64
    }
}

// Arithmetic operations for Half
impl Add for Half {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Half {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Mul for Half {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl Div for Half {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl Rem for Half {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        Self(self.0 % rhs.0)
    }
}

impl Neg for Half {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl AddAssign for Half {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0 + rhs.0;
    }
}

impl SubAssign for Half {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0 - rhs.0;
    }
}

impl MulAssign for Half {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = self.0 * rhs.0;
    }
}

impl DivAssign for Half {
    fn div_assign(&mut self, rhs: Self) {
        self.0 = self.0 / rhs.0;
    }
}

impl Zero for Half {
    fn zero() -> Self {
        Self(f16::ZERO)
    }

    fn is_zero(&self) -> bool {
        self.0 == f16::ZERO
    }
}

impl One for Half {
    fn one() -> Self {
        Self(f16::ONE)
    }
}

impl Bounded for Half {
    fn min_value() -> Self {
        Self(f16::MIN)
    }

    fn max_value() -> Self {
        Self(f16::MAX)
    }
}

impl Num for Half {
    type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f32::from_str_radix(str, radix).map(Self::new)
    }
}

impl NumCast for Half {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        n.to_f32().map(Self::new)
    }
}

impl ToPrimitive for Half {
    fn to_i64(&self) -> Option<i64> {
        self.get().to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.get().to_u64()
    }

    fn to_f32(&self) -> Option<f32> {
        Some(self.get())
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.get() as f64)
    }
}

impl FromPrimitive for Half {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self::new(n as f32))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Self::new(n as f32))
    }

    fn from_f32(n: f32) -> Option<Self> {
        Some(Self::new(n))
    }

    fn from_f64(n: f64) -> Option<Self> {
        Some(Self::new(n as f32))
    }
}

impl Float for Half {
    fn nan() -> Self {
        Self(f16::NAN)
    }

    fn infinity() -> Self {
        Self(f16::INFINITY)
    }

    fn neg_infinity() -> Self {
        Self(f16::NEG_INFINITY)
    }

    fn neg_zero() -> Self {
        Self(f16::NEG_ZERO)
    }

    fn min_value() -> Self {
        Self(f16::MIN)
    }

    fn min_positive_value() -> Self {
        Self(f16::MIN_POSITIVE)
    }

    fn max_value() -> Self {
        Self(f16::MAX)
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
        Self::new(self.get().floor())
    }

    fn ceil(self) -> Self {
        Self::new(self.get().ceil())
    }

    fn round(self) -> Self {
        Self::new(self.get().round())
    }

    fn trunc(self) -> Self {
        Self::new(self.get().trunc())
    }

    fn fract(self) -> Self {
        Self::new(self.get().fract())
    }

    fn abs(self) -> Self {
        Self::new(self.get().abs())
    }

    fn signum(self) -> Self {
        Self::new(self.get().signum())
    }

    fn is_sign_positive(self) -> bool {
        self.0.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.0.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        Self::new(self.get().mul_add(a.get(), b.get()))
    }

    fn recip(self) -> Self {
        Self::new(self.get().recip())
    }

    fn powi(self, n: i32) -> Self {
        Self::new(self.get().powi(n))
    }

    fn powf(self, n: Self) -> Self {
        Self::new(self.get().powf(n.get()))
    }

    fn sqrt(self) -> Self {
        Self::new(self.get().sqrt())
    }

    fn exp(self) -> Self {
        Self::new(self.get().exp())
    }

    fn exp2(self) -> Self {
        Self::new(self.get().exp2())
    }

    fn ln(self) -> Self {
        Self::new(self.get().ln())
    }

    fn log(self, base: Self) -> Self {
        Self::new(self.get().log(base.get()))
    }

    fn log2(self) -> Self {
        Self::new(self.get().log2())
    }

    fn log10(self) -> Self {
        Self::new(self.get().log10())
    }

    fn max(self, other: Self) -> Self {
        Self::new(self.get().max(other.get()))
    }

    fn min(self, other: Self) -> Self {
        Self::new(self.get().min(other.get()))
    }

    fn abs_sub(self, other: Self) -> Self {
        Self::new((self.get() - other.get()).abs())
    }

    fn cbrt(self) -> Self {
        Self::new(self.get().cbrt())
    }

    fn hypot(self, other: Self) -> Self {
        Self::new(self.get().hypot(other.get()))
    }

    fn sin(self) -> Self {
        Self::new(self.get().sin())
    }

    fn cos(self) -> Self {
        Self::new(self.get().cos())
    }

    fn tan(self) -> Self {
        Self::new(self.get().tan())
    }

    fn asin(self) -> Self {
        Self::new(self.get().asin())
    }

    fn acos(self) -> Self {
        Self::new(self.get().acos())
    }

    fn atan(self) -> Self {
        Self::new(self.get().atan())
    }

    fn atan2(self, other: Self) -> Self {
        Self::new(self.get().atan2(other.get()))
    }

    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.get().sin_cos();
        (Self::new(s), Self::new(c))
    }

    fn exp_m1(self) -> Self {
        Self::new(self.get().exp_m1())
    }

    fn ln_1p(self) -> Self {
        Self::new(self.get().ln_1p())
    }

    fn sinh(self) -> Self {
        Self::new(self.get().sinh())
    }

    fn cosh(self) -> Self {
        Self::new(self.get().cosh())
    }

    fn tanh(self) -> Self {
        Self::new(self.get().tanh())
    }

    fn asinh(self) -> Self {
        Self::new(self.get().asinh())
    }

    fn acosh(self) -> Self {
        Self::new(self.get().acosh())
    }

    fn atanh(self) -> Self {
        Self::new(self.get().atanh())
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.get().integer_decode()
    }

    fn epsilon() -> Self {
        Self(f16::EPSILON)
    }

    fn to_degrees(self) -> Self {
        Self::new(self.get().to_degrees())
    }

    fn to_radians(self) -> Self {
        Self::new(self.get().to_radians())
    }
}

impl DataType for Half {
    fn dtype() -> Dtype {
        Dtype::Half
    }
}

impl FloatExt for Half {
    fn erf(self) -> Self {
        Self::new(libm::erff(self.get()))
    }

    fn erfc(self) -> Self {
        Self::new(libm::erfcf(self.get()))
    }
}

// ============================================================================
// BFloat16 Implementation
// ============================================================================

impl BFloat16 {
    /// Create a new BFloat16 from an f32 value
    #[inline]
    #[must_use]
    pub fn new(value: f32) -> Self {
        Self(bf16::from_f32(value))
    }

    /// Get the inner value as f32
    #[inline]
    #[must_use]
    pub fn get(self) -> f32 {
        self.0.to_f32()
    }

    /// Get the raw bf16 value
    #[inline]
    #[must_use]
    pub fn raw(self) -> bf16 {
        self.0
    }
}

impl fmt::Debug for BFloat16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BFloat16({})", self.get())
    }
}

impl fmt::Display for BFloat16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.get())
    }
}

impl From<f32> for BFloat16 {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl From<BFloat16> for f32 {
    fn from(value: BFloat16) -> Self {
        value.get()
    }
}

impl From<BFloat16> for f64 {
    fn from(value: BFloat16) -> Self {
        value.get() as f64
    }
}

// Arithmetic operations for BFloat16
impl Add for BFloat16 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub for BFloat16 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Mul for BFloat16 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl Div for BFloat16 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl Rem for BFloat16 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        Self(self.0 % rhs.0)
    }
}

impl Neg for BFloat16 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl AddAssign for BFloat16 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0 + rhs.0;
    }
}

impl SubAssign for BFloat16 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0 - rhs.0;
    }
}

impl MulAssign for BFloat16 {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = self.0 * rhs.0;
    }
}

impl DivAssign for BFloat16 {
    fn div_assign(&mut self, rhs: Self) {
        self.0 = self.0 / rhs.0;
    }
}

impl Zero for BFloat16 {
    fn zero() -> Self {
        Self(bf16::ZERO)
    }

    fn is_zero(&self) -> bool {
        self.0 == bf16::ZERO
    }
}

impl One for BFloat16 {
    fn one() -> Self {
        Self(bf16::ONE)
    }
}

impl Bounded for BFloat16 {
    fn min_value() -> Self {
        Self(bf16::MIN)
    }

    fn max_value() -> Self {
        Self(bf16::MAX)
    }
}

impl Num for BFloat16 {
    type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f32::from_str_radix(str, radix).map(Self::new)
    }
}

impl NumCast for BFloat16 {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        n.to_f32().map(Self::new)
    }
}

impl ToPrimitive for BFloat16 {
    fn to_i64(&self) -> Option<i64> {
        self.get().to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.get().to_u64()
    }

    fn to_f32(&self) -> Option<f32> {
        Some(self.get())
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.get() as f64)
    }
}

impl FromPrimitive for BFloat16 {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self::new(n as f32))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Self::new(n as f32))
    }

    fn from_f32(n: f32) -> Option<Self> {
        Some(Self::new(n))
    }

    fn from_f64(n: f64) -> Option<Self> {
        Some(Self::new(n as f32))
    }
}

impl Float for BFloat16 {
    fn nan() -> Self {
        Self(bf16::NAN)
    }

    fn infinity() -> Self {
        Self(bf16::INFINITY)
    }

    fn neg_infinity() -> Self {
        Self(bf16::NEG_INFINITY)
    }

    fn neg_zero() -> Self {
        Self(bf16::NEG_ZERO)
    }

    fn min_value() -> Self {
        Self(bf16::MIN)
    }

    fn min_positive_value() -> Self {
        Self(bf16::MIN_POSITIVE)
    }

    fn max_value() -> Self {
        Self(bf16::MAX)
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
        Self::new(self.get().floor())
    }

    fn ceil(self) -> Self {
        Self::new(self.get().ceil())
    }

    fn round(self) -> Self {
        Self::new(self.get().round())
    }

    fn trunc(self) -> Self {
        Self::new(self.get().trunc())
    }

    fn fract(self) -> Self {
        Self::new(self.get().fract())
    }

    fn abs(self) -> Self {
        Self::new(self.get().abs())
    }

    fn signum(self) -> Self {
        Self::new(self.get().signum())
    }

    fn is_sign_positive(self) -> bool {
        self.0.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.0.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        Self::new(self.get().mul_add(a.get(), b.get()))
    }

    fn recip(self) -> Self {
        Self::new(self.get().recip())
    }

    fn powi(self, n: i32) -> Self {
        Self::new(self.get().powi(n))
    }

    fn powf(self, n: Self) -> Self {
        Self::new(self.get().powf(n.get()))
    }

    fn sqrt(self) -> Self {
        Self::new(self.get().sqrt())
    }

    fn exp(self) -> Self {
        Self::new(self.get().exp())
    }

    fn exp2(self) -> Self {
        Self::new(self.get().exp2())
    }

    fn ln(self) -> Self {
        Self::new(self.get().ln())
    }

    fn log(self, base: Self) -> Self {
        Self::new(self.get().log(base.get()))
    }

    fn log2(self) -> Self {
        Self::new(self.get().log2())
    }

    fn log10(self) -> Self {
        Self::new(self.get().log10())
    }

    fn max(self, other: Self) -> Self {
        Self::new(self.get().max(other.get()))
    }

    fn min(self, other: Self) -> Self {
        Self::new(self.get().min(other.get()))
    }

    fn abs_sub(self, other: Self) -> Self {
        Self::new((self.get() - other.get()).abs())
    }

    fn cbrt(self) -> Self {
        Self::new(self.get().cbrt())
    }

    fn hypot(self, other: Self) -> Self {
        Self::new(self.get().hypot(other.get()))
    }

    fn sin(self) -> Self {
        Self::new(self.get().sin())
    }

    fn cos(self) -> Self {
        Self::new(self.get().cos())
    }

    fn tan(self) -> Self {
        Self::new(self.get().tan())
    }

    fn asin(self) -> Self {
        Self::new(self.get().asin())
    }

    fn acos(self) -> Self {
        Self::new(self.get().acos())
    }

    fn atan(self) -> Self {
        Self::new(self.get().atan())
    }

    fn atan2(self, other: Self) -> Self {
        Self::new(self.get().atan2(other.get()))
    }

    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.get().sin_cos();
        (Self::new(s), Self::new(c))
    }

    fn exp_m1(self) -> Self {
        Self::new(self.get().exp_m1())
    }

    fn ln_1p(self) -> Self {
        Self::new(self.get().ln_1p())
    }

    fn sinh(self) -> Self {
        Self::new(self.get().sinh())
    }

    fn cosh(self) -> Self {
        Self::new(self.get().cosh())
    }

    fn tanh(self) -> Self {
        Self::new(self.get().tanh())
    }

    fn asinh(self) -> Self {
        Self::new(self.get().asinh())
    }

    fn acosh(self) -> Self {
        Self::new(self.get().acosh())
    }

    fn atanh(self) -> Self {
        Self::new(self.get().atanh())
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.get().integer_decode()
    }

    fn epsilon() -> Self {
        Self(bf16::EPSILON)
    }

    fn to_degrees(self) -> Self {
        Self::new(self.get().to_degrees())
    }

    fn to_radians(self) -> Self {
        Self::new(self.get().to_radians())
    }
}

impl DataType for BFloat16 {
    fn dtype() -> Dtype {
        Dtype::BFloat16
    }
}

impl FloatExt for BFloat16 {
    fn erf(self) -> Self {
        Self::new(libm::erff(self.get()))
    }

    fn erfc(self) -> Self {
        Self::new(libm::erfcf(self.get()))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_half_creation() {
        let h = Half::new(1.5);
        assert!((h.get() - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_half_arithmetic() {
        let a = Half::new(2.0);
        let b = Half::new(3.0);

        let sum = a + b;
        assert!((sum.get() - 5.0).abs() < 0.01);

        let prod = a * b;
        assert!((prod.get() - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_half_datatype_trait() {
        assert_eq!(Half::dtype(), Dtype::Half);
        assert!(Half::is_floating_point());
        assert!(!Half::is_integer());
        assert!(!Half::is_complex());
    }

    #[test]
    fn test_bfloat16_creation() {
        let bf = BFloat16::new(1.5);
        assert!((bf.get() - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_bfloat16_arithmetic() {
        let a = BFloat16::new(2.0);
        let b = BFloat16::new(3.0);

        let sum = a + b;
        assert!((sum.get() - 5.0).abs() < 0.01);

        let prod = a * b;
        assert!((prod.get() - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_bfloat16_datatype_trait() {
        assert_eq!(BFloat16::dtype(), Dtype::BFloat16);
        assert!(BFloat16::is_floating_point());
        assert!(!BFloat16::is_integer());
        assert!(!BFloat16::is_complex());
    }

    #[test]
    fn test_half_special_values() {
        let nan = Half::nan();
        assert!(nan.is_nan());

        let inf = Half::infinity();
        assert!(inf.is_infinite());

        let zero = Half::zero();
        assert!(zero.is_zero());
    }

    #[test]
    fn test_bfloat16_special_values() {
        let nan = BFloat16::nan();
        assert!(nan.is_nan());

        let inf = BFloat16::infinity();
        assert!(inf.is_infinite());

        let zero = BFloat16::zero();
        assert!(zero.is_zero());
    }

    #[test]
    fn test_half_float_ext() {
        use crate::traits::FloatExt;

        let x = Half::new(0.5);
        let erf_val = x.erf();
        // erf(0.5) ≈ 0.5205
        assert!((erf_val.get() - 0.5205).abs() < 0.01);
    }

    #[test]
    fn test_bfloat16_float_ext() {
        use crate::traits::FloatExt;

        let x = BFloat16::new(0.5);
        let erf_val = x.erf();
        // erf(0.5) ≈ 0.5205
        assert!((erf_val.get() - 0.5205).abs() < 0.01);
    }
}
