//! Floating-point extension trait.

use super::{FloatOps, Scalar};

/// Floating-point extension trait.
///
/// Provides transcendental functions, rounding, and float-specific checks.
/// Implemented for f16, bf16, f32, f64. Extends `Scalar + FloatOps`, so
/// any bound `T: Float` automatically implies `T: Scalar` and `T: FloatOps`.
///
/// # Examples
///
/// ```
/// use coeus_core::Float;
///
/// let x: f32 = 2.0;
/// assert_eq!(x.sqrt(), 1.4142135_f32);
/// assert!(!x.is_nan());
/// assert!(x.is_finite());
/// ```
pub trait Float: Scalar + FloatOps {
    /// Largest finite value.
    const MAX: Self;
    /// Smallest positive normal value.
    const MIN_POSITIVE: Self;
    /// Not-a-Number.
    const NAN: Self;
    /// Negative infinity.
    const NEG_INFINITY: Self;
    /// Positive infinity.
    const INFINITY: Self;

    /// Floor: largest integer ≤ self.
    /// Floor: largest integer ≤ self.
    fn floor(self) -> Self;
    /// Ceiling: smallest integer ≥ self.
    fn ceil(self) -> Self;
    /// Round to nearest integer.
    fn round(self) -> Self;
    /// Truncate toward zero.
    fn trunc(self) -> Self;
    /// Fractional part.
    fn fract(self) -> Self;
    /// Absolute value.
    fn abs(self) -> Self;
    /// Sign function: -1, 0, or 1.
    fn signum(self) -> Self;
    /// Square root.
    fn sqrt(self) -> Self;
    /// Exponential: e^self.
    fn exp(self) -> Self;
    /// Base-2 exponential: 2^self.
    fn exp2(self) -> Self;
    /// Natural logarithm: ln(self).
    fn ln(self) -> Self;
    /// Base-2 logarithm.
    fn log2(self) -> Self;
    /// Base-10 logarithm.
    fn log10(self) -> Self;
    /// Sine.
    fn sin(self) -> Self;
    /// Cosine.
    fn cos(self) -> Self;
    /// Tangent.
    fn tan(self) -> Self;
    /// Arcsine.
    fn asin(self) -> Self;
    /// Arccosine.
    fn acos(self) -> Self;
    /// Arctangent.
    fn atan(self) -> Self;
    /// Hyperbolic sine.
    fn sinh(self) -> Self;
    /// Hyperbolic cosine.
    fn cosh(self) -> Self;
    /// Hyperbolic tangent.
    fn tanh(self) -> Self;
    /// Power: self^n.
    fn powf(self, n: Self) -> Self;
    /// Integer power: `self^exp` where `exp` is a signed integer exponent.
    ///
    /// Raises `self` to the integer power `exp` using repeated multiplication
    /// with sign preservation: `(-x)^exp = -(x^exp)` for odd `exp` and
    /// `(x^|exp|)` for even `exp`, matching
    /// `at::pow`/`Tensor.pow(scalar)` semantics when `scalar` is integer-valued.
    /// `exp = 0` returns `1`. Negative `exp` returns `1 / powi(|exp|)`.
    fn powi(self, exp: i32) -> Self;
    /// True if `self` rounds to an exact integer in `T` (i.e. truncates to itself).
    ///
    /// Used by `pow(x, scalar)` to dispatch between sign-preserving integer
    /// power and the fractional-power `exp(n·ln(x))` composition.
    fn is_integer(self) -> bool;
    /// True if self is NaN.
    fn is_nan(self) -> bool;
    /// True if self is positive or negative infinity.
    fn is_infinite(self) -> bool;
    /// True if self is a finite (non-infinite, non-NaN) value.
    fn is_finite(self) -> bool;
}
