// ── Float implementations ──
// Scalar + Float for f32, f64, half::f16, half::bf16.

use super::traits::{private, Scalar, Float, FloatOps};
use half::{f16, bf16};

// ── Helper macros ──

macro_rules! impl_scalar_float_native {
    ($t:ty) => {
        impl private::Sealed for $t {}
        impl Scalar for $t {
            #[inline(always)]
            fn to_f64(self) -> f64 { self as f64 }
            #[inline(always)]
            fn from_f64(v: f64) -> Self { v as Self }
            #[inline(always)]
            fn sqrt_val(self) -> Self { self.sqrt() }
            #[inline(always)]
            fn abs_val(self) -> Self { self.abs() }
            #[inline]
            fn add_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
                // Delegate to the SIMD-effect SSOT. Lengths are equal by the
                // caller's contract; on a length mismatch fall back to scalar.
                if hermes_simd::elementwise_add::<$t>(a, b, out).is_err() {
                    for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) { *o = x + y; }
                }
            }
            #[inline]
            fn sub_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
                if hermes_simd::elementwise_sub::<$t>(a, b, out).is_err() {
                    for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) { *o = x - y; }
                }
            }
            #[inline]
            fn mul_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
                if hermes_simd::elementwise_mul::<$t>(a, b, out).is_err() {
                    for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) { *o = x * y; }
                }
            }
            #[inline]
            fn div_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
                if hermes_simd::elementwise_div::<$t>(a, b, out).is_err() {
                    for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) { *o = x / y; }
                }
            }
            #[inline]
            fn sum_slice(s: &[Self]) -> Self { hermes_simd::sum::<$t>(s) }
            #[inline]
            fn min_slice(s: &[Self]) -> Self { hermes_simd::min::<$t>(s) }
            #[inline]
            fn max_slice(s: &[Self]) -> Self { hermes_simd::max::<$t>(s) }
        }
        impl FloatOps for $t {
            #[inline(always)] fn exp_op(self) -> Self { self.exp() }
            #[inline(always)] fn log_op(self) -> Self { self.ln() }
            #[inline(always)] fn tanh_op(self) -> Self { self.tanh() }
            #[inline(always)] fn sin_op(self) -> Self { self.sin() }
            #[inline(always)] fn cos_op(self) -> Self { self.cos() }
            #[inline(always)] fn gelu_op(self) -> Self {
                let x_f = self;
                let c1 = Self::from_f64(0.5);
                let c2 = Self::from_f64(1.0);
                let c3 = Self::from_f64(0.797_884_560_8);
                let c4 = Self::from_f64(0.044_715);
                c1 * x_f * (c2 + (c3 * (x_f + c4 * x_f.powi(3))).tanh())
            }
            #[inline(always)] fn sigmoid_op(self) -> Self { 1.0 / (1.0 + (-self).exp()) }
        }
        impl Float for $t {
            const MAX: Self = <$t>::MAX;
            const MIN_POSITIVE: Self = <$t>::MIN_POSITIVE;
            const NAN: Self = <$t>::NAN;
            const NEG_INFINITY: Self = <$t>::NEG_INFINITY;
            const INFINITY: Self = <$t>::INFINITY;
            #[inline(always)] fn floor(self) -> Self { self.floor() }
            #[inline(always)] fn ceil(self) -> Self { self.ceil() }
            #[inline(always)] fn round(self) -> Self { self.round() }
            #[inline(always)] fn trunc(self) -> Self { self.trunc() }
            #[inline(always)] fn fract(self) -> Self { self.fract() }
            #[inline(always)] fn abs(self) -> Self { self.abs() }
            #[inline(always)] fn signum(self) -> Self { self.signum() }
            #[inline(always)] fn sqrt(self) -> Self { self.sqrt() }
            #[inline(always)] fn exp(self) -> Self { self.exp() }
            #[inline(always)] fn exp2(self) -> Self { self.exp2() }
            #[inline(always)] fn ln(self) -> Self { self.ln() }
            #[inline(always)] fn log2(self) -> Self { self.log2() }
            #[inline(always)] fn log10(self) -> Self { self.log10() }
            #[inline(always)] fn sin(self) -> Self { self.sin() }
            #[inline(always)] fn cos(self) -> Self { self.cos() }
            #[inline(always)] fn tan(self) -> Self { self.tan() }
            #[inline(always)] fn asin(self) -> Self { self.asin() }
            #[inline(always)] fn acos(self) -> Self { self.acos() }
            #[inline(always)] fn atan(self) -> Self { self.atan() }
            #[inline(always)] fn sinh(self) -> Self { self.sinh() }
            #[inline(always)] fn cosh(self) -> Self { self.cosh() }
            #[inline(always)] fn tanh(self) -> Self { self.tanh() }
            #[inline(always)] fn powf(self, n: Self) -> Self { self.powf(n) }
            #[inline(always)] fn is_nan(self) -> bool { self.is_nan() }
            #[inline(always)] fn is_infinite(self) -> bool { self.is_infinite() }
            #[inline(always)] fn is_finite(self) -> bool { self.is_finite() }
        }
    };
}

// Macro implementing Scalar and Float traits for half-precision floating-point types (f16, bf16).
//
// # Software Fallback Audit
// Basic arithmetic operators (+, -, *, /) on f16 and bf16 execute in native-precision using
// the core implementation in the half crate.
// However, transcendental functions (sin, cos, exp, log, tanh, etc.) are emulated using a
// widen-compute-narrow pattern (widen to f64, compute, narrow back to f16/bf16). This software fallback
// is documented because typical host CPUs lack hardware instruction support for 16-bit float transcendentals.
macro_rules! impl_scalar_float_half {
    ($t:ty) => {
        impl private::Sealed for $t {}
        impl Scalar for $t {
            #[inline(always)]
            fn to_f64(self) -> f64 { self.to_f64() }
            #[inline(always)]
            fn from_f64(v: f64) -> Self { Self::from_f64(v) }
            #[inline(always)]
            fn sqrt_val(self) -> Self { Self::from_f64(self.to_f64().sqrt()) }
            #[inline(always)]
            fn abs_val(self) -> Self { Self::from_f64(self.to_f64().abs()) }
        }
        impl FloatOps for $t {
            #[inline(always)] fn exp_op(self) -> Self { Self::from_f64(self.to_f64().exp()) }
            #[inline(always)] fn log_op(self) -> Self { Self::from_f64(self.to_f64().ln()) }
            #[inline(always)] fn tanh_op(self) -> Self { Self::from_f64(self.to_f64().tanh()) }
            #[inline(always)] fn sin_op(self) -> Self { Self::from_f64(self.to_f64().sin()) }
            #[inline(always)] fn cos_op(self) -> Self { Self::from_f64(self.to_f64().cos()) }
            #[inline(always)] fn gelu_op(self) -> Self {
                let x_f = self.to_f64();
                let res = 0.5 * x_f * (1.0 + (0.7978845608 * (x_f + 0.044715 * x_f.powi(3))).tanh());
                Self::from_f64(res)
            }
            #[inline(always)] fn sigmoid_op(self) -> Self {
                let x_f = self.to_f64();
                let res = 1.0 / (1.0 + (-x_f).exp());
                Self::from_f64(res)
            }
        }
        impl Float for $t {
            const MAX: Self = Self::MAX;
            const MIN_POSITIVE: Self = Self::MIN_POSITIVE;
            const NAN: Self = Self::NAN;
            const NEG_INFINITY: Self = Self::NEG_INFINITY;
            const INFINITY: Self = Self::INFINITY;
            #[inline(always)] fn floor(self) -> Self { Self::from_f64(self.to_f64().floor()) }
            #[inline(always)] fn ceil(self) -> Self { Self::from_f64(self.to_f64().ceil()) }
            #[inline(always)] fn round(self) -> Self { Self::from_f64(self.to_f64().round()) }
            #[inline(always)] fn trunc(self) -> Self { Self::from_f64(self.to_f64().trunc()) }
            #[inline(always)] fn fract(self) -> Self { Self::from_f64(self.to_f64().fract()) }
            #[inline(always)] fn abs(self) -> Self { Self::from_f64(self.to_f64().abs()) }
            #[inline(always)] fn signum(self) -> Self { Self::from_f64(self.to_f64().signum()) }
            #[inline(always)] fn sqrt(self) -> Self { Self::from_f64(self.to_f64().sqrt()) }
            #[inline(always)] fn exp(self) -> Self { Self::from_f64(self.to_f64().exp()) }
            #[inline(always)] fn exp2(self) -> Self { Self::from_f64(self.to_f64().exp2()) }
            #[inline(always)] fn ln(self) -> Self { Self::from_f64(self.to_f64().ln()) }
            #[inline(always)] fn log2(self) -> Self { Self::from_f64(self.to_f64().log2()) }
            #[inline(always)] fn log10(self) -> Self { Self::from_f64(self.to_f64().log10()) }
            #[inline(always)] fn sin(self) -> Self { Self::from_f64(self.to_f64().sin()) }
            #[inline(always)] fn cos(self) -> Self { Self::from_f64(self.to_f64().cos()) }
            #[inline(always)] fn tan(self) -> Self { Self::from_f64(self.to_f64().tan()) }
            #[inline(always)] fn asin(self) -> Self { Self::from_f64(self.to_f64().asin()) }
            #[inline(always)] fn acos(self) -> Self { Self::from_f64(self.to_f64().acos()) }
            #[inline(always)] fn atan(self) -> Self { Self::from_f64(self.to_f64().atan()) }
            #[inline(always)] fn sinh(self) -> Self { Self::from_f64(self.to_f64().sinh()) }
            #[inline(always)] fn cosh(self) -> Self { Self::from_f64(self.to_f64().cosh()) }
            #[inline(always)] fn tanh(self) -> Self { Self::from_f64(self.to_f64().tanh()) }
            #[inline(always)] fn powf(self, n: Self) -> Self { Self::from_f64(self.to_f64().powf(n.to_f64())) }
            #[inline(always)] fn is_nan(self) -> bool { self.to_f64().is_nan() }
            #[inline(always)] fn is_infinite(self) -> bool { self.to_f64().is_infinite() }
            #[inline(always)] fn is_finite(self) -> bool { self.to_f64().is_finite() }
        }
    };
}

impl_scalar_float_native!(f32);
impl_scalar_float_native!(f64);
impl_scalar_float_half!(f16);
impl_scalar_float_half!(bf16);
