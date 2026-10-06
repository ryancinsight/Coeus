use crate::dtype::traits::{private, Float, FloatOps, Scalar};
use eunomia::{FloatElement, NumericElement};

macro_rules! impl_scalar_float_native {
    ($t:ty) => {
        impl private::Sealed for $t {}
        impl Scalar for $t {
            // `Scalar` keeps only what no provider owns; identities live on
            // `NumericElement` and are used directly at call sites.
            #[inline(always)]
            fn from_f64(v: f64) -> Self {
                v as Self
            }
            #[inline(always)]
            fn total_add(self, rhs: Self) -> Self {
                self + rhs
            }
            #[inline(always)]
            fn total_mul(self, rhs: Self) -> Self {
                self * rhs
            }
            // Slice kernels resolve through the `leto_ops::Scalar` supertrait
            // (single SSOT with hermes-SIMD dispatch); only the coeus-only
            // `scale_slice` is overridden here.
            #[inline]
            fn scale_slice(data: &mut [Self], scalar: Self) {
                hermes_simd::scale::<$t>(data, scalar);
            }
        }
        impl FloatOps for $t {
            #[inline(always)]
            fn exp_op(self) -> Self {
                self.exp()
            }
            #[inline(always)]
            fn exp2_op(self) -> Self {
                self.exp2()
            }
            #[inline(always)]
            fn log_op(self) -> Self {
                self.ln()
            }
            #[inline(always)]
            fn tanh_op(self) -> Self {
                self.tanh()
            }
            #[inline(always)]
            fn sin_op(self) -> Self {
                self.sin()
            }
            #[inline(always)]
            fn cos_op(self) -> Self {
                self.cos()
            }
            #[inline(always)]
            fn erf_op(self) -> Self {
                FloatElement::erf(self)
            }
            #[inline(always)]
            fn erfc_op(self) -> Self {
                FloatElement::erfc(self)
            }
            #[inline(always)]
            fn lgamma_op(self) -> Self {
                FloatElement::lgamma(self)
            }
            #[inline(always)]
            fn tan_op(self) -> Self {
                self.tan()
            }
            #[inline(always)]
            fn asin_op(self) -> Self {
                self.asin()
            }
            #[inline(always)]
            fn acos_op(self) -> Self {
                self.acos()
            }
            #[inline(always)]
            fn atan_op(self) -> Self {
                self.atan()
            }
            #[inline(always)]
            fn sinh_op(self) -> Self {
                self.sinh()
            }
            #[inline(always)]
            fn cosh_op(self) -> Self {
                self.cosh()
            }
            #[inline(always)]
            fn log2_op(self) -> Self {
                self.log2()
            }
            #[inline(always)]
            fn log10_op(self) -> Self {
                self.log10()
            }
            #[inline(always)]
            fn atanh_op(self) -> Self {
                self.atanh()
            }
            #[inline(always)]
            fn asinh_op(self) -> Self {
                self.asinh()
            }
            #[inline(always)]
            fn acosh_op(self) -> Self {
                self.acosh()
            }
            #[inline(always)]
            fn expm1_op(self) -> Self {
                self.exp_m1()
            }
            #[inline(always)]
            fn log1p_op(self) -> Self {
                self.ln_1p()
            }
            #[inline(always)]
            fn gelu_op(self) -> Self {
                let half = <$t as Scalar>::from_f64(0.5);
                let one = <$t as Scalar>::from_f64(1.0);
                let inv_sqrt_two = <$t as Scalar>::from_f64(core::f64::consts::FRAC_1_SQRT_2);
                half * self * (one + (self * inv_sqrt_two).erf_op())
            }
            #[inline(always)]
            fn sigmoid_op(self) -> Self {
                1.0 / (1.0 + (-self).exp())
            }
        }
        impl Float for $t {
            const MAX: Self = <$t>::MAX;
            const MIN_POSITIVE: Self = <$t>::MIN_POSITIVE;
            const NAN: Self = <$t>::NAN;
            const NEG_INFINITY: Self = <$t>::NEG_INFINITY;
            const INFINITY: Self = <$t>::INFINITY;
            #[inline(always)]
            fn floor(self) -> Self {
                self.floor()
            }
            #[inline(always)]
            fn ceil(self) -> Self {
                self.ceil()
            }
            #[inline(always)]
            fn round(self) -> Self {
                self.round()
            }
            #[inline(always)]
            fn trunc(self) -> Self {
                self.trunc()
            }
            #[inline(always)]
            fn fract(self) -> Self {
                self.fract()
            }
            #[inline(always)]
            fn signum(self) -> Self {
                self.signum()
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                self.sqrt()
            }
            #[inline(always)]
            fn exp(self) -> Self {
                self.exp()
            }
            #[inline(always)]
            fn exp2(self) -> Self {
                self.exp2()
            }
            #[inline(always)]
            fn ln(self) -> Self {
                self.ln()
            }
            #[inline(always)]
            fn log2(self) -> Self {
                self.log2()
            }
            #[inline(always)]
            fn log10(self) -> Self {
                self.log10()
            }
            #[inline(always)]
            fn sin(self) -> Self {
                self.sin()
            }
            #[inline(always)]
            fn cos(self) -> Self {
                self.cos()
            }
            #[inline(always)]
            fn tan(self) -> Self {
                self.tan()
            }
            #[inline(always)]
            fn asin(self) -> Self {
                self.asin()
            }
            #[inline(always)]
            fn acos(self) -> Self {
                self.acos()
            }
            #[inline(always)]
            fn atan(self) -> Self {
                self.atan()
            }
            #[inline(always)]
            fn sinh(self) -> Self {
                self.sinh()
            }
            #[inline(always)]
            fn cosh(self) -> Self {
                self.cosh()
            }
            #[inline(always)]
            fn tanh(self) -> Self {
                self.tanh()
            }
            #[inline(always)]
            fn powf(self, n: Self) -> Self {
                self.powf(n)
            }
            #[inline(always)]
            fn powi(self, exp: i32) -> Self {
                self.powi(exp)
            }
            #[inline(always)]
            fn is_integer(self) -> bool {
                self == self.trunc() && self.is_finite()
            }
            #[inline(always)]
            fn is_infinite(self) -> bool {
                self.is_infinite()
            }
        }
    };
}

impl_scalar_float_native!(f32);
impl_scalar_float_native!(f64);


