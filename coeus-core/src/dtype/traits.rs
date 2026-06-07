// ── Dtype traits ──
// Sealed trait hierarchy for numeric scalar types used in tensors.
//
// Design notes:
// - `Scalar` is the base: Copy + Pod + Send + Sync + 'static
// - `Float` extends Scalar with transcendental and rounding ops
// - `Int` extends Scalar with bitwise and modular ops
// - All traits are sealed (private Sealed supertrait) for monomorphization
// - bytemuck::Pod guarantees safe transmutation to/from [u8]

use std::fmt::Debug;
use num_traits::{Num, Zero, One};
use bytemuck::Pod;

// ── Sealed pattern ──
pub(crate) mod private {
    pub trait Sealed {}
}

/// Helper trait for native-precision floating-point operations.
///
/// Implemented natively for floating point types.
/// For integer types, it is not mathematically supported and will panic.
pub trait FloatOps {
    fn exp_op(self) -> Self;
    fn log_op(self) -> Self;
    fn tanh_op(self) -> Self;
    fn sin_op(self) -> Self;
    fn cos_op(self) -> Self;
    fn gelu_op(self) -> Self;
    fn sigmoid_op(self) -> Self;
}

/// Base numeric trait for all tensor element types.
///
/// # Safety / Design
/// - `Pod` enables zero-copy byte transmutation (bytemuck).
/// - `Num` gives arithmetic ops (Add, Sub, Mul, Div, Rem).
/// - Sealed prevents downstream impls, guaranteeing monomorphization.
pub trait Scalar:
    private::Sealed
    + Num
    + Copy
    + Clone
    + Send
    + Sync
    + Debug
    + Pod
    + Zero
    + One
    + PartialOrd
    + FloatOps
    + 'static
{
    /// Convert to f64 (for mixed-precision checkpoints).
    fn to_f64(self) -> f64;

    /// Convert from f64 (for initialization, etc.).
    fn from_f64(v: f64) -> Self;

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
/// Implemented for f16, bf16, f32, f64.
pub trait Float: Scalar {
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

    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
    fn abs(self) -> Self;
    fn signum(self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn exp2(self) -> Self;
    fn ln(self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn is_nan(self) -> bool;
    fn is_infinite(self) -> bool;
    fn is_finite(self) -> bool;
}

/// Integer extension trait.
///
/// Provides bitwise operations and integer-specific math.
/// Implemented for i8, i16, i32, i64, u8, u16, u32, u64.
pub trait Int: Scalar + num_traits::NumCast {
    fn count_ones(self) -> u32;
    fn count_zeros(self) -> u32;
    fn leading_zeros(self) -> u32;
    fn trailing_zeros(self) -> u32;
    fn rotate_left(self, n: u32) -> Self;
    fn rotate_right(self, n: u32) -> Self;
    fn pow(self, exp: u32) -> Self;
    fn abs(self) -> Self;
}
