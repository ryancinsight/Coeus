// ── Dtype traits ──
// Sealed trait hierarchy for numeric scalar types used in tensors.
//
// Design notes:
// - `Scalar` is the base: Copy + bytemuck::Pod + eunomia::Pod + Send + Sync +
//   'static
// - `Float` extends Scalar with transcendental and rounding ops
// - `Int` extends Scalar with bitwise and modular ops
// - All traits are sealed (private Sealed supertrait) for monomorphization
// - bytemuck::Pod and eunomia::Pod guarantee safe host/device byte layouts

use bytemuck::Pod;
use eunomia::{NumericElement, Pod as EunomiaPod};
use std::fmt::Debug;
use std::ops::Rem;

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
    /// Element-wise equality: 1 if a == b, else 0.
    Eq,
    /// Element-wise inequality: 1 if a != b, else 0.
    Ne,
    /// Element-wise less-than: 1 if a < b, else 0.
    Lt,
    /// Element-wise greater-than: 1 if a > b, else 0.
    Gt,
    /// Element-wise less-than-or-equal: 1 if a <= b, else 0.
    Le,
    /// Element-wise greater-than-or-equal: 1 if a >= b, else 0.
    Ge,
}

/// Reduction operation tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    /// Sum of all elements.
    Sum,
    /// Product of all elements.
    Prod,
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
    /// Natural logarithm of the absolute gamma function: ln|Γ(x)|.
    Lgamma,
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
    /// base-2 exponential: exp2(x) = 2^x
    Exp2,
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

impl CpuUnaryOp {
    /// Decode the low and high halves of a packed activation parameter pair.
    #[must_use]
    pub const fn decode_parameter_pair(bits: u64) -> [f32; 2] {
        [
            f32::from_bits(bits as u32),
            f32::from_bits((bits >> 32) as u32),
        ]
    }

    /// Decode the two runtime parameters carried by a parameterized activation.
    ///
    /// Hardtanh and threshold store two `f32` bit patterns. Single-parameter
    /// activations store one `f64` bit pattern, converted to the provider's
    /// `f32` parameter representation here; their unused second parameter is
    /// zero. Non-parameterized operations return `None`.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the provider parameter ABI is f32; single activation parameters enter as f64 bit patterns"
    )]
    pub const fn parameter_pair(self) -> Option<[f32; 2]> {
        let bits = match self {
            Self::Hardtanh(bits)
            | Self::HardtanhGrad(bits)
            | Self::Threshold(bits)
            | Self::ThresholdGrad(bits) => bits,
            Self::LeakyRelu(bits)
            | Self::LeakyReluGrad(bits)
            | Self::Hardshrink(bits)
            | Self::HardshrinkGrad(bits)
            | Self::Softshrink(bits)
            | Self::SoftshrinkGrad(bits)
            | Self::Celu(bits)
            | Self::CeluGrad(bits) => return Some([f64::from_bits(bits) as f32, 0.0]),
            _ => return None,
        };
        Some(Self::decode_parameter_pair(bits))
    }
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
/// - `EunomiaPod` is the canonical device-buffer layout contract consumed by
///   Hephaestus; keeping it on `Scalar` prevents backend seams from accepting
///   a host-only numeric type that cannot cross a device boundary.
/// - `NumericElement` is the eunomia SSOT element vocabulary (constants
///   `ZERO`/`ONE`/etc., `abs`/`sqrt`/`is_finite`/`is_nan`/`to_f64`/`from_f64`/
///   `try_from_count`, plus `Add`/`Sub`/`Mul`/`Div`/`Assigns`/`Copy`/`Send`/`Sync`/
///   `'static`/`Debug`/`PartialOrd`). Backend `Scalar` traits must extend—
///   not redeclare—`NumericElement`.
/// - `Rem<Output=Self>` is per-Scalar (not on `NumericElement`).
/// - Sealed via `eunomia::private::Sealed`.
///
/// # Examples
///
/// Slice kernels come from the `leto_ops::Scalar` supertrait — the stack's single
/// slice-kernel surface (SSOT) with its hermes-SIMD dispatch. They are resolved
/// through this trait's bound, never redeclared here:
///
/// ```
/// use coeus_core::Scalar;
/// use leto_ops::Scalar as LetoScalar;
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
///
/// fn requires_coeus_scalar<T: Scalar>() {}
/// requires_coeus_scalar::<f32>();
/// ```
pub trait Scalar:
    NumericElement + CpuUnaryDispatch + Pod + EunomiaPod + Rem<Output = Self> + Clone + leto_ops::Scalar
{
    // No `zero/one/to_f64/sqrt_val/abs_val`: use the eunomia SSOT directly
    // (`NumericElement::{ZERO, ONE, to_f64, sqrt, abs}`). This trait keeps
    // only what no provider owns.

    /// Return whether every byte in this value's representation is zero.
    ///
    /// Accelerator fills use this representation-level predicate to select a
    /// byte-clear operation without changing signed-zero or NaN payload bits.
    #[inline]
    fn has_zero_bit_pattern(&self) -> bool {
        bytemuck::bytes_of(self).iter().all(|&byte| byte == 0)
    }

    /// Construct a scalar from `f64`.
    ///
    /// Owned here: integers need it and eunomia has no int-inclusive
    /// `from_f64` (`TryFromCount` refuses out-of-range counts instead of
    /// saturating).
    fn from_f64(v: f64) -> Self;

    /// Addition defined for every input: integers wrap modulo 2^bits on
    /// overflow (two's-complement `wrapping_add`); floats follow IEEE 754
    /// (overflow saturates to ±infinity, matching native `+`). Named after
    /// `total_cmp`'s "the well-defined variant of the usual operator" sense,
    /// not because it changes float behavior — float `+` is already total.
    ///
    /// Every `Scalar` implementor states this explicitly (no default): the
    /// reduction operations (`coeus_dist::Sum`/`Product`) fold peer-supplied
    /// values through this method rather than `+`/`*` directly, so an
    /// adversarial or merely large peer value can never panic a debug or
    /// overflow-checked build. Integer wrapping matches the two's-complement
    /// behavior release builds already give unchecked `+`, and the reduction
    /// semantics of MPI/NCCL integer sums.
    fn total_add(self, rhs: Self) -> Self;

    /// Multiplication defined for every input. See [`Scalar::total_add`] for
    /// the wrap/IEEE split and the rationale.
    fn total_mul(self, rhs: Self) -> Self;

    // Slice kernels (`add/sub/mul/div/sum/dot/axpy/min/max_slice`) are NOT
    // redeclared here. They live once on the `leto_ops::Scalar` supertrait —
    // the stack's single slice-kernel surface with its hermes-SIMD dispatch —
    // and every `T: Scalar` resolves them through that bound (DIP: depend on
    // the provider abstraction, never redeclare it). `scale_slice` below stays
    // because no provider owns it yet; it is this trait's only kernel surface.

    /// In-place multiplication of every contiguous slice element by `scalar`.
    ///
    /// Coeus-only until a provider adopts it: the operation is
    /// lane-independent, so native-float SIMD overrides remain bitwise-identical
    /// to the scalar default for ordinary IEEE operands.
    #[inline]
    fn scale_slice(data: &mut [Self], scalar: Self) {
        for value in data {
            *value *= scalar;
        }
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
pub trait Float: Scalar + FloatOps + eunomia::FloatElement + leto_ops::RealScalar {
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
    ///
    /// Inherited from [`NumericElement::abs`] via the supertrait chain.
    /// Use `<T as NumericElement>::abs(x)` or `x.abs()` at call sites.
    // fn abs — provided by NumericElement supertrait, removed to avoid ambiguity
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
    // fn is_nan — provided by NumericElement supertrait, removed to avoid ambiguity
    // fn is_finite — provided by NumericElement supertrait, removed to avoid ambiguity
    /// True if self is positive or negative infinity.
    fn is_infinite(self) -> bool;
}

/// Integer extension trait.
///
/// Provides bitwise operations and integer-specific math.
/// Implemented for i8, i16, i32, i64, u8, u16, u32, u64.
pub trait Int: Scalar {
    // No `count_ones`/`abs`: use `NumericElement` directly. This trait keeps
    // only the bit intrinsics no provider owns.
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
}

#[cfg(test)]
mod cpu_unary_op_tests {
    use super::{CpuUnaryOp, Scalar};

    #[test]
    fn zero_bit_pattern_distinguishes_signed_zero() {
        assert!(0.0_f32.has_zero_bit_pattern());
        assert!(!(-0.0_f32).has_zero_bit_pattern());
        assert!(0_u32.has_zero_bit_pattern());
        assert!(!1_u32.has_zero_bit_pattern());
    }

    #[test]
    fn parameter_pairs_preserve_both_bit_patterns() {
        let first = -1.25_f32;
        let second = 2.5_f32;
        let bits = u64::from(first.to_bits()) | (u64::from(second.to_bits()) << 32);

        assert_eq!(
            CpuUnaryOp::Hardtanh(bits).parameter_pair(),
            Some([first, second])
        );
        assert_eq!(CpuUnaryOp::Relu.parameter_pair(), None);
    }
}
