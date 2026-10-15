/// Native-precision floating-point transcendental operations.
///
/// Sealed to `Float` implementors only. Integer types do **not** implement
/// this trait — calling `exp_op`, `log_op`, etc. on an integer type is a
/// compile error, not a runtime panic.
pub trait FloatOps: super::private::Sealed {
    /// Element-wise exponential: e^x.
    fn exp_op(self) -> Self;
    /// Element-wise base-2 exponential: 2^x.
    fn exp2_op(self) -> Self;
    /// Element-wise natural logarithm: ln(x).
    fn log_op(self) -> Self;
    /// Element-wise hyperbolic tangent: tanh(x).
    fn tanh_op(self) -> Self;
    /// Element-wise sine: sin(x).
    fn sin_op(self) -> Self;
    /// Element-wise cosine: cos(x).
    fn cos_op(self) -> Self;
    /// Gauss error function: erf(x).
    fn erf_op(self) -> Self;
    /// Complementary Gauss error function: erfc(x) = 1 - erf(x).
    fn erfc_op(self) -> Self;
    /// Natural logarithm of the absolute gamma function: ln|Γ(x)|.
    fn lgamma_op(self) -> Self;
    /// Element-wise tangent: tan(x).
    fn tan_op(self) -> Self;
    /// Element-wise arc-sine: asin(x).
    fn asin_op(self) -> Self;
    /// Element-wise arc-cosine: acos(x).
    fn acos_op(self) -> Self;
    /// Element-wise arc-tangent: atan(x).
    fn atan_op(self) -> Self;
    /// Element-wise hyperbolic sine: sinh(x).
    fn sinh_op(self) -> Self;
    /// Element-wise hyperbolic cosine: cosh(x).
    fn cosh_op(self) -> Self;
    /// Element-wise base-2 logarithm: log2(x).
    fn log2_op(self) -> Self;
    /// Element-wise base-10 logarithm: log10(x).
    fn log10_op(self) -> Self;
    /// Element-wise inverse hyperbolic tangent: atanh(x).
    fn atanh_op(self) -> Self;
    /// Element-wise inverse hyperbolic sine: asinh(x).
    fn asinh_op(self) -> Self;
    /// Element-wise inverse hyperbolic cosine: acosh(x).
    fn acosh_op(self) -> Self;
    /// Element-wise exp(x) - 1 with improved small-x accuracy.
    fn expm1_op(self) -> Self;
    /// Element-wise ln(1 + x) with improved small-x accuracy.
    fn log1p_op(self) -> Self;
    /// Gaussian Error Linear Unit: 0.5 * x * (1 + erf(x / sqrt(2))).
    fn gelu_op(self) -> Self;
    /// Logistic sigmoid: 1 / (1 + e^(-x)).
    fn sigmoid_op(self) -> Self;
}
