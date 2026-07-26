use crate::dtype::traits::{private, Float, FloatOps, Scalar};
use eunomia::NumericElement;
use eunomia::{Bf16, F16};

macro_rules! impl_scalar_float_half {
    ($t:ty, $max:expr, $min_pos:expr) => {
        impl private::Sealed for $t {}
        impl Scalar for $t {
            #[inline(always)]
            fn zero() -> Self {
                Self::ZERO
            }
            #[inline(always)]
            fn one() -> Self {
                Self::ONE
            }
            #[inline(always)]
            fn to_f64(self) -> f64 {
                <Self as NumericElement>::to_f64(self)
            }
            #[inline(always)]
            fn from_f64(v: f64) -> Self {
                <Self as eunomia::FloatElement>::from_f64(v)
            }
            #[inline(always)]
            fn from_usize(v: usize) -> Self {
                <Self as eunomia::FloatElement>::from_f64(v as f64)
            }
            #[inline(always)]
            fn sqrt_val(self) -> Self {
                <Self as NumericElement>::sqrt(self)
            }
            #[inline(always)]
            fn abs_val(self) -> Self {
                <Self as NumericElement>::abs(self)
            }
        }
        impl FloatOps for $t {
            #[inline(always)]
            fn exp_op(self) -> Self {
                <Self as eunomia::FloatElement>::exp(self)
            }
            #[inline(always)]
            fn exp2_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.exp2())
            }
            #[inline(always)]
            fn log_op(self) -> Self {
                <Self as eunomia::FloatElement>::ln(self)
            }
            #[inline(always)]
            fn tanh_op(self) -> Self {
                <Self as eunomia::FloatElement>::tanh(self)
            }
            #[inline(always)]
            fn sin_op(self) -> Self {
                <Self as eunomia::FloatElement>::sin(self)
            }
            #[inline(always)]
            fn cos_op(self) -> Self {
                <Self as eunomia::FloatElement>::cos(self)
            }
            #[inline(always)]
            fn erf_op(self) -> Self {
                <Self as eunomia::FloatElement>::erf(self)
            }
            #[inline(always)]
            fn erfc_op(self) -> Self {
                <Self as eunomia::FloatElement>::erfc(self)
            }
            #[inline(always)]
            fn lgamma_op(self) -> Self {
                <Self as eunomia::FloatElement>::lgamma(self)
            }
            #[inline(always)]
            fn tan_op(self) -> Self {
                <Self as eunomia::FloatElement>::tan(self)
            }
            #[inline(always)]
            fn asin_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.asin())
            }
            #[inline(always)]
            fn acos_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.acos())
            }
            #[inline(always)]
            fn atan_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.atan())
            }
            #[inline(always)]
            fn sinh_op(self) -> Self {
                <Self as eunomia::FloatElement>::sinh(self)
            }
            #[inline(always)]
            fn cosh_op(self) -> Self {
                <Self as eunomia::FloatElement>::cosh(self)
            }
            #[inline(always)]
            fn log2_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.log2())
            }
            #[inline(always)]
            fn log10_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.log10())
            }
            #[inline(always)]
            fn atanh_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.atanh())
            }
            #[inline(always)]
            fn asinh_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.asinh())
            }
            #[inline(always)]
            fn acosh_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.acosh())
            }
            #[inline(always)]
            fn expm1_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.exp_m1())
            }
            #[inline(always)]
            fn log1p_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.ln_1p())
            }
            #[inline(always)]
            fn gelu_op(self) -> Self {
                let half = Self::from_f64(0.5);
                let one = Self::one();
                let inv_sqrt_two = Self::from_f64(core::f64::consts::FRAC_1_SQRT_2);
                half * self * (one + (self * inv_sqrt_two).erf_op())
            }
            #[inline(always)]
            fn sigmoid_op(self) -> Self {
                let x_f = <Self as NumericElement>::to_f64(self);
                let res = 1.0 / (1.0 + (-x_f).exp());
                Self::from_f64(res)
            }
        }
        impl Float for $t {
            const MAX: Self = $max;
            const MIN_POSITIVE: Self = $min_pos;
            const NAN: Self = Self::NAN;
            const NEG_INFINITY: Self = Self::NEG_INFINITY;
            const INFINITY: Self = Self::INFINITY;
            #[inline(always)]
            fn floor(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.floor())
            }
            #[inline(always)]
            fn ceil(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.ceil())
            }
            #[inline(always)]
            fn round(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.round())
            }
            #[inline(always)]
            fn trunc(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.trunc())
            }
            #[inline(always)]
            fn fract(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.fract())
            }
            #[inline(always)]
            fn abs(self) -> Self {
                <Self as NumericElement>::abs(self)
            }
            #[inline(always)]
            fn signum(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.signum())
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                <Self as NumericElement>::sqrt(self)
            }
            #[inline(always)]
            fn exp(self) -> Self {
                <Self as eunomia::FloatElement>::exp(self)
            }
            #[inline(always)]
            fn exp2(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.exp2())
            }
            #[inline(always)]
            fn ln(self) -> Self {
                <Self as eunomia::FloatElement>::ln(self)
            }
            #[inline(always)]
            fn log2(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.log2())
            }
            #[inline(always)]
            fn log10(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.log10())
            }
            #[inline(always)]
            fn sin(self) -> Self {
                <Self as eunomia::FloatElement>::sin(self)
            }
            #[inline(always)]
            fn cos(self) -> Self {
                <Self as eunomia::FloatElement>::cos(self)
            }
            #[inline(always)]
            fn tan(self) -> Self {
                <Self as eunomia::FloatElement>::tan(self)
            }
            #[inline(always)]
            fn asin(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.asin())
            }
            #[inline(always)]
            fn acos(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.acos())
            }
            #[inline(always)]
            fn atan(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.atan())
            }
            #[inline(always)]
            fn sinh(self) -> Self {
                <Self as eunomia::FloatElement>::sinh(self)
            }
            #[inline(always)]
            fn cosh(self) -> Self {
                <Self as eunomia::FloatElement>::cosh(self)
            }
            #[inline(always)]
            fn tanh(self) -> Self {
                <Self as eunomia::FloatElement>::tanh(self)
            }
            #[inline(always)]
            fn powf(self, n: Self) -> Self {
                <Self as eunomia::FloatElement>::powf(self, n)
            }
            #[inline(always)]
            fn powi(self, exp: i32) -> Self {
                <Self as eunomia::FloatElement>::powi(self, exp)
            }
            #[inline(always)]
            fn is_integer(self) -> bool {
                let f = <Self as NumericElement>::to_f64(self);
                f.is_finite() && f == f.trunc()
            }
            #[inline(always)]
            fn is_nan(self) -> bool {
                <Self as NumericElement>::is_nan(self)
            }
            #[inline(always)]
            fn is_infinite(self) -> bool {
                let f = <Self as NumericElement>::to_f64(self);
                f.is_infinite()
            }
            #[inline(always)]
            fn is_finite(self) -> bool {
                <Self as NumericElement>::is_finite(self)
            }
        }
    };
}

// F16: largest finite = 65504.0, smallest positive normal = 2^-14
impl_scalar_float_half!(F16, F16(0x7BFF), F16(0x0040));
// Bf16: largest finite ≈ 3.3895e38, smallest positive normal = 2^-126
impl_scalar_float_half!(Bf16, Bf16(0x7F7F), Bf16(0x0080));
