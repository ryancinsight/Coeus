// ── Dtype traits ──
// Sealed trait hierarchy for numeric scalar types used in tensors.
//
// Design notes:
// - `Scalar` is the base: Copy + Pod + Send + Sync + 'static
// - `Float` extends Scalar with transcendental and rounding ops
// - `Int` extends Scalar with bitwise and modular ops
// - All traits are sealed (private Sealed supertrait) for monomorphization
// - bytemuck::Pod guarantees safe transmutation to/from [u8]

use bytemuck::Pod;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Rem, Sub};

// ── Sealed pattern ──
pub(crate) mod private {
    pub trait Sealed {}
}

/// Native-precision floating-point transcendental operations.
///
/// Sealed to `Float` implementors only. Integer types do **not** implement
/// this trait — calling `exp_op`, `log_op`, etc. on an integer type is a
/// compile error, not a runtime panic.
pub trait FloatOps: private::Sealed {
    /// Element-wise exponential: e^x.
    fn exp_op(self) -> Self;
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

/// Binary element-wise operation tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// Addition: a + b.
    Add,
    /// Subtraction: a - b.
    Sub,
    /// Multiplication: a * b.
    Mul,
    /// Division: a / b.
    Div,
}

/// Reduction operation tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    /// Sum of all elements.
    Sum,
    /// Arithmetic mean of all elements.
    Mean,
    /// Maximum element.
    Max,
    /// Minimum element.
    Min,
}

/// CPU unary operation dispatch tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuUnaryOp {
    /// Rectified Linear Unit: max(0, x).
    Relu,
    /// ReLU gradient: 1 if x > 0, else 0.
    ReluGrad,
    /// Logistic sigmoid: 1 / (1 + e^(-x)).
    Sigmoid,
    /// Sigmoid gradient: σ(x) * (1 - σ(x)).
    SigmoidGrad,
    /// Hyperbolic tangent: tanh(x).
    Tanh,
    /// Tanh gradient: 1 - tanh(x)^2.
    TanhGrad,
    /// Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
    Gelu,
    /// GELU gradient.
    GeluGrad,
    /// Element-wise sine: sin(x).
    Sin,
    /// Element-wise cosine: cos(x).
    Cos,
    /// Element-wise exponential: e^x.
    Exp,
    /// Element-wise natural logarithm: ln(x).
    Log,
    /// Gauss error function: erf(x).
    Erf,
    /// Complementary Gauss error function: erfc(x) = 1 - erf(x).
    Erfc,
    /// tan(x)
    Tan,
    /// arcsin(x)
    Asin,
    /// arccos(x)
    Acos,
    /// arctan(x)
    Atan,
    /// hyperbolic sine: sinh(x)
    Sinh,
    /// hyperbolic cosine: cosh(x)
    Cosh,
    /// base-2 logarithm: log2(x)
    Log2,
    /// base-10 logarithm: log10(x)
    Log10,
    /// inverse hyperbolic tangent: atanh(x)
    Atanh,
    /// inverse hyperbolic sine: asinh(x)
    Asinh,
    /// inverse hyperbolic cosine: acosh(x)
    Acosh,
    /// exp(x) - 1
    Expm1,
    /// ln(1 + x)
    Log1p,
    /// Element-wise negation: -x.
    Neg,
    /// Element-wise absolute value: |x|.
    Abs,
    /// Element-wise square root: sqrt(x).
    Sqrt,
    /// SiLU (Sigmoid Linear Unit): x * sigmoid(x).
    Silu,
    /// SiLU gradient.
    SiluGrad,
    /// Mish: x * tanh(softplus(x)).
    Mish,
    /// Mish gradient.
    MishGrad,
    /// ELU: x > 0 ? x : alpha * (e^x - 1).
    Elu,
    /// ELU gradient.
    EluGrad,
    /// Softplus: ln(1 + e^x).
    Softplus,
    /// Softplus gradient: sigmoid(x).
    SoftplusGrad,
    /// Tanh-approximated GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))).
    GeluTanh,
    /// Tanh-GELU gradient.
    GeluTanhGrad,
    /// Leaky ReLU: x > 0 ? x : slope * x. The packed `u64` is the negative-slope bit pattern.
    LeakyRelu(u64),
    /// Leaky ReLU gradient. The packed `u64` is the negative-slope bit pattern.
    LeakyReluGrad(u64),
    /// Hardtanh: clamp(x, min, max). The packed `u64` stores `(min, max)` as
    /// little-endian `f32` bit patterns.
    Hardtanh(u64),
    /// Hardtanh gradient: 1 inside (min, max), 0 outside. Same packed-min/max convention.
    HardtanhGrad(u64),
    /// Hardsigmoid: clamp(x/6 + 0.5, 0, 1). No parameters.
    Hardsigmoid,
    /// Hardsigmoid gradient: 1/6 inside (-3, 3), 0 outside.
    HardsigmoidGrad,
    /// Hardswish: x * ReLU6(x+3) / 6. No parameters.
    Hardswish,
    /// Hardswish gradient.
    HardswishGrad,
    /// Hardshrink: x if |x| > λ else 0. Packed `u64` is `λ.bits()`.
    Hardshrink(u64),
    /// Hardshrink gradient: 1 if |x| > λ else 0.
    HardshrinkGrad(u64),
    /// Softshrink: sign(x) * max(|x| - λ, 0). Packed `u64` is `λ.bits()`.
    Softshrink(u64),
    /// Softshrink gradient: 1 if |x| > λ else 0.
    SoftshrinkGrad(u64),
    /// Softsign: x / (1 + |x|). No parameters.
    Softsign,
    /// Softsign gradient: 1 / (1 + |x|)^2.
    SoftsignGrad,
    /// Threshold: x if x > threshold else value. The packed `u64` stores
    /// `(threshold, value)` as little-endian `f32` bit patterns.
    Threshold(u64),
    /// Threshold gradient: 1 if x > threshold else 0.
    ThresholdGrad(u64),
    /// CELU: max(0,x) + min(0, α·(exp(x/α) - 1)). Packed `u64` is `α.bits()` (default α = 1.0).
    Celu(u64),
    /// CELU gradient: 1 if x ≥ 0 else exp(x/α).
    CeluGrad(u64),
    /// Element-wise reciprocal: 1/x
    Recip,
    /// Element-wise signum: -1, 0, or 1
    Sign,
    /// Element-wise floor: largest integer ≤ x
    Floor,
    /// Element-wise ceil: smallest integer ≥ x
    Ceil,
    /// Element-wise round to nearest integer, ties to even (IEEE-754
    /// roundTiesToEven, matching torch.round)
    Round,
    /// Element-wise truncation toward zero
    Trunc,
}

/// CPU dispatch trait for unary operations.
///
/// Implemented for all `Scalar` types that support CPU-side unary kernels.
pub trait CpuUnaryDispatch: private::Sealed {
    /// Evaluate a unary operation on a single element.
    fn eval_unary(op: CpuUnaryOp, x: Self) -> Self;
}

/// Base numeric trait for all tensor element types.
///
/// # Safety / Design
/// - `Pod` enables zero-copy byte transmutation (bytemuck).
/// - `Num` gives arithmetic ops (Add, Sub, Mul, Div, Rem).
/// - Sealed prevents downstream impls, guaranteeing monomorphization.
///
/// # Examples
///
/// Scalar operations on contiguous slices (the SIMD seam):
///
/// ```
/// use coeus_core::Scalar;
///
/// let a = [1.0_f32, 2.0, 3.0];
/// let b = [4.0_f32, 5.0, 6.0];
/// let mut out = [0.0_f32; 3];
/// f32::add_slice(&a, &b, &mut out);
/// assert_eq!(out, [5.0, 7.0, 9.0]);
///
/// let dot = f32::dot_slice(&a, &b);
/// assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6
///
/// let mut acc = [10.0_f32, 10.0, 10.0];
/// f32::axpy_slice(2.0, &a, &mut acc);
/// assert_eq!(acc, [12.0, 14.0, 16.0]); // 10 + 2*[1,2,3]
/// ```
pub trait Scalar:
    private::Sealed
    + Copy
    + Clone
    + Send
    + Sync
    + Debug
    + Pod
    + PartialOrd
    + CpuUnaryDispatch
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + 'static
{
    /// Additive identity (0).
    fn zero() -> Self;

    /// Multiplicative identity (1).
    fn one() -> Self;
    /// Convert to f64 (for mixed-precision checkpoints).
    fn to_f64(self) -> f64;

    /// Convert from f64 (for initialization, etc.).
    fn from_f64(v: f64) -> Self;

    /// Convert from a structural index or dimension count.
    ///
    /// This is the native-precision path for index-derived tensor values such
    /// as `arange`; it avoids routing exact non-negative integer coordinates
    /// through `f64`.
    fn from_usize(v: usize) -> Self;

    /// Calculate square root of the value natively.
    fn sqrt_val(self) -> Self;

    /// Calculate absolute value of the value natively.
    fn abs_val(self) -> Self;

    /// Elementwise `a + b` into `out` over equal-length contiguous slices.
    ///
    /// One of the four per-type seams onto the SIMD-effect SSOT (`hermes-simd`):
    /// the default is a scalar loop; reduced/extended precision and integer types
    /// use it unchanged, while `f32`/`f64` override these to delegate to
    /// `hermes_simd::elementwise_{add,sub,mul,div}`. Each op is independent per
    /// lane, so the SIMD result is bitwise-identical to the scalar default; no
    /// reassociation occurs. `out`, `a`, and `b` must have equal length.
    #[inline]
    fn add_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
        for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *o = x + y;
        }
    }

    /// Elementwise `a - b` into `out`. See [`Scalar::add_slice`].
    #[inline]
    fn sub_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
        for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *o = x - y;
        }
    }

    /// Elementwise `a * b` into `out`. See [`Scalar::add_slice`].
    #[inline]
    fn mul_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
        for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *o = x * y;
        }
    }

    /// Elementwise `a / b` into `out`. See [`Scalar::add_slice`].
    #[inline]
    fn div_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
        for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *o = x / y;
        }
    }

    /// Dot product of two equal-length contiguous slices.
    ///
    /// This is the per-type seam onto the SIMD-effect SSOT for vector products.
    /// The default is a native-precision scalar fold; `f32`/`f64` override to
    /// `hermes_simd::dot`. Floating-point SIMD may reassociate the summation,
    /// so callers that compare against a sequential fold must use an
    /// analytically derived epsilon bound.
    #[inline]
    fn dot_slice(a: &[Self], b: &[Self]) -> Self {
        assert_eq!(a.len(), b.len(), "dot_slice: length mismatch");

        let mut acc = Self::zero();
        for (&x, &y) in a.iter().zip(b.iter()) {
            acc = acc + x * y;
        }
        acc
    }

    /// In-place multiplication of every contiguous slice element by `scalar`.
    ///
    /// This is the per-type seam onto the SIMD-effect SSOT for scalar scaling.
    /// The operation is lane-independent, so native-float SIMD overrides remain
    /// bitwise-identical to the scalar default for ordinary IEEE operands.
    #[inline]
    fn scale_slice(data: &mut [Self], scalar: Self) {
        for value in data {
            *value = *value * scalar;
        }
    }

    /// Fused scaled accumulate: `out[i] += alpha * x[i]` over a contiguous slice.
    ///
    /// The per-type seam onto the SIMD-effect SSOT for AXPY (BLAS level-1). The
    /// operation is lane-independent; native-float overrides route to
    /// `hermes_simd::axpy` and remain within the type's rounding error of this
    /// scalar default (differential tests use an epsilon bound, not bitwise
    /// equality). `x` and `out` must have equal length.
    #[inline]
    fn axpy_slice(alpha: Self, x: &[Self], out: &mut [Self]) {
        assert_eq!(x.len(), out.len(), "axpy_slice: length mismatch");
        for (o, &xi) in out.iter_mut().zip(x.iter()) {
            let next = *o + alpha * xi;
            *o = next;
        }
    }

    /// Sum of a contiguous slice — per-type seam onto the SIMD-effect SSOT.
    ///
    /// Default is a sequential left fold; `f32`/`f64` override to
    /// `hermes_simd::sum`. Summation is associative only approximately in
    /// floating point, so the SIMD result may differ from the sequential fold
    /// within the type's rounding error (differential tests use an epsilon
    /// bound, not bitwise equality). Empty slice sums to `Self::zero()`.
    #[inline]
    fn sum_slice(s: &[Self]) -> Self {
        match s.split_first() {
            Some((&first, rest)) => {
                let mut acc = first;
                for &v in rest {
                    acc = acc + v;
                }
                acc
            }
            None => Self::zero(),
        }
    }

    /// Minimum of a non-empty contiguous slice. `f32`/`f64` override to
    /// `hermes_simd::min`. min is exactly associative, so the SIMD result is
    /// value-identical to the sequential fold for non-NaN inputs.
    #[inline]
    fn min_slice(s: &[Self]) -> Self {
        let mut acc = s[0];
        for &v in &s[1..] {
            if v < acc {
                acc = v;
            }
        }
        acc
    }

    /// Maximum of a non-empty contiguous slice. `f32`/`f64` override to
    /// `hermes_simd::max`. See [`Scalar::min_slice`].
    #[inline]
    fn max_slice(s: &[Self]) -> Self {
        let mut acc = s[0];
        for &v in &s[1..] {
            if v > acc {
                acc = v;
            }
        }
        acc
    }
}

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
    /// True if self is NaN.
    fn is_nan(self) -> bool;
    /// True if self is positive or negative infinity.
    fn is_infinite(self) -> bool;
    /// True if self is a finite (non-infinite, non-NaN) value.
    fn is_finite(self) -> bool;
}

/// Integer extension trait.
///
/// Provides bitwise operations and integer-specific math.
/// Implemented for i8, i16, i32, i64, u8, u16, u32, u64.
pub trait Int: Scalar {
    /// Count of set bits (popcount).
    fn count_ones(self) -> u32;
    /// Count of unset bits.
    fn count_zeros(self) -> u32;
    /// Count of leading zero bits.
    fn leading_zeros(self) -> u32;
    /// Count of trailing zero bits.
    fn trailing_zeros(self) -> u32;
    /// Bitwise rotate left by `n` positions.
    fn rotate_left(self, n: u32) -> Self;
    /// Bitwise rotate right by `n` positions.
    fn rotate_right(self, n: u32) -> Self;
    /// Integer power: self^exp.
    fn pow(self, exp: u32) -> Self;
    /// Absolute value.
    fn abs(self) -> Self;
}
