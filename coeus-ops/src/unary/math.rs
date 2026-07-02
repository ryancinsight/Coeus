// ── Unary math ops ──

use super::kernel::{elementwise_unary, elementwise_unary_assign};
use crate::backend_ops::{BackendOps, UnaryOp};
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;

/// Element-wise sine: sin(x).
///
/// Backward: `d/dx sin(x) = cos(x)`.
#[inline]
pub fn sin<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Sin)
}

/// Element-wise cosine: cos(x).
///
/// Backward: `d/dx cos(x) = -sin(x)`.
#[inline]
pub fn cos<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Cos)
}

/// Element-wise exponential.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::exp;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([2], &[0.0, 1.0]);
/// let b = exp(&a, &backend);
/// let s = b.as_slice();
/// assert!((s[0] - 1.0).abs() < 1e-5);
/// assert!((s[1] - std::f32::consts::E).abs() < 1e-5);
/// ```
#[inline]
pub fn exp<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Exp)
}

/// Element-wise natural log.
#[inline]
pub fn log<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Log)
}

/// Element-wise Gauss error function: erf(x) = (2/√π) ∫₀ˣ e^(−t²) dt.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::erf;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f64, SequentialBackend>::from_slice([2], &[0.0, 1.0]);
/// let b = erf(&a, &backend);
/// let s = b.as_slice();
/// assert!(s[0].abs() < 1e-15);
/// assert!((s[1] - 0.842_700_792_949_714_9).abs() < 1e-12);
/// ```
#[inline]
pub fn erf<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Erf)
}

/// Element-wise complementary error function: erfc(x) = 1 − erf(x).
///
/// Backward: `d/dx erfc(x) = −d/dx erf(x) = −(2/√π)·e^(−x²)`.
#[inline]
pub fn erfc<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Erfc)
}

/// Element-wise tangent: tan(x).
///
/// Backward: `d/dx tan(x) = 1 / cos²(x)` (sec²(x)).
#[inline]
pub fn tan<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Tan)
}

/// Element-wise arc-sine: asin(x), domain x ∈ [-1, 1].
///
/// Backward: `d/dx asin(x) = 1 / sqrt(1 - x²)`.
#[inline]
pub fn asin<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Asin)
}

/// Element-wise arc-cosine: acos(x), domain x ∈ [-1, 1].
///
/// Backward: `d/dx acos(x) = -1 / sqrt(1 - x²)`.
#[inline]
pub fn acos<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Acos)
}

/// Element-wise arc-tangent: atan(x).
///
/// Backward: `d/dx atan(x) = 1 / (1 + x²)`.
#[inline]
pub fn atan<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Atan)
}

/// Element-wise hyperbolic sine: sinh(x).
/// Backward: d/dx sinh(x) = cosh(x).
#[inline]
pub fn sinh<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Sinh)
}
/// Element-wise hyperbolic cosine: cosh(x).
/// Backward: d/dx cosh(x) = sinh(x).
#[inline]
pub fn cosh<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Cosh)
}
/// Element-wise base-2 logarithm: log2(x).
/// Backward: d/dx log2(x) = 1/(x * ln(2)).
#[inline]
pub fn log2<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Log2)
}
/// Element-wise base-10 logarithm: log10(x).
/// Backward: d/dx log10(x) = 1/(x * ln(10)).
#[inline]
pub fn log10<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Log10)
}
/// Element-wise base-2 exponential: 2^x.
/// Backward: d/dx 2^x = 2^x * ln(2).
#[inline]
pub fn exp2<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Exp2)
}
/// Element-wise inverse hyperbolic tangent: atanh(x), domain |x| < 1.
/// Backward: d/dx atanh(x) = 1/(1 - x²).
#[inline]
pub fn atanh<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Atanh)
}
/// Element-wise inverse hyperbolic sine: asinh(x).
/// Backward: d/dx asinh(x) = 1/sqrt(x² + 1).
#[inline]
pub fn asinh<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Asinh)
}
/// Element-wise inverse hyperbolic cosine: acosh(x), domain x > 1.
/// Backward: d/dx acosh(x) = 1/sqrt(x² - 1).
#[inline]
pub fn acosh<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Acosh)
}
/// Element-wise exp(x) - 1.
/// Backward: d/dx expm1(x) = exp(x).
#[inline]
pub fn expm1<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Expm1)
}
/// Element-wise ln(1 + x).
/// Backward: d/dx log1p(x) = 1/(1 + x).
#[inline]
pub fn log1p<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Log1p)
}

/// Element-wise negation (works for any Scalar).
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::neg;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([3], &[1.0, -2.0, 3.0]);
/// let b = neg(&a, &backend);
/// assert_eq!(b.as_slice(), &[-1.0, 2.0, -3.0]);
/// ```
#[inline]
pub fn neg<T: Scalar, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Neg)
}

/// Element-wise absolute value.
#[inline]
pub fn abs<T: Scalar, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Abs)
}

/// Element-wise square root.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::sqrt;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([3], &[1.0, 4.0, 9.0]);
/// let b = sqrt(&a, &backend);
/// let s = b.as_slice();
/// assert!((s[0] - 1.0).abs() < 1e-5);
/// assert!((s[1] - 2.0).abs() < 1e-5);
/// assert!((s[2] - 3.0).abs() < 1e-5);
/// ```
#[inline]
pub fn sqrt<T: Scalar, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Sqrt)
}

/// In-place element-wise sine.
#[inline]
pub fn sin_assign<T: Float, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Sin);
}

/// In-place element-wise cosine.
#[inline]
pub fn cos_assign<T: Float, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Cos);
}

/// In-place element-wise exponential.
#[inline]
pub fn exp_assign<T: Float, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Exp);
}

/// In-place element-wise natural log.
#[inline]
pub fn log_assign<T: Float, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Log);
}

/// In-place element-wise negation.
#[inline]
pub fn neg_assign<T: Scalar, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Neg);
}

/// In-place element-wise absolute value.
#[inline]
pub fn abs_assign<T: Scalar, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Abs);
}

/// In-place element-wise square root.
#[inline]
pub fn sqrt_assign<T: Scalar, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Sqrt);
}

/// Element-wise reciprocal: 1/x.
#[inline]
pub fn recip<T: Scalar, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Recip)
}

/// Element-wise signum: -1, 0, or 1.
#[inline]
pub fn sign<T: Scalar, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Sign)
}

/// Element-wise floor.
#[inline]
pub fn floor<T: Scalar, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Floor)
}

/// Element-wise ceil.
#[inline]
pub fn ceil<T: Scalar, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Ceil)
}

/// Element-wise round to nearest integer.
#[inline]
pub fn round<T: Scalar, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Round)
}

/// Element-wise truncation toward zero.
#[inline]
pub fn trunc<T: Scalar, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Trunc)
}

/// In-place element-wise reciprocal.
#[inline]
pub fn recip_assign<T: Scalar, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Recip);
}

/// In-place element-wise signum.
#[inline]
pub fn sign_assign<T: Scalar, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Sign);
}

/// In-place element-wise floor.
#[inline]
pub fn floor_assign<T: Scalar, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Floor);
}

/// In-place element-wise ceil.
#[inline]
pub fn ceil_assign<T: Scalar, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Ceil);
}

/// In-place element-wise round.
#[inline]
pub fn round_assign<T: Scalar, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Round);
}

/// In-place element-wise truncation.
#[inline]
pub fn trunc_assign<T: Scalar, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Trunc);
}
