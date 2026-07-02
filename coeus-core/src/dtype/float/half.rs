use crate::dtype::traits::{private, Float, FloatOps, Scalar};
use half::{bf16, f16};

macro_rules! impl_scalar_float_half {
    ($t:ty) => {
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
                self.to_f64()
            }
            #[inline(always)]
            fn from_f64(v: f64) -> Self {
                Self::from_f64(v)
            }
            #[inline(always)]
            fn from_usize(v: usize) -> Self {
                Self::from_f64(v as f64)
            }
            #[inline(always)]
            fn sqrt_val(self) -> Self {
                Self::from_f64(self.to_f64().sqrt())
            }
            #[inline(always)]
            fn abs_val(self) -> Self {
                Self::from_f64(self.to_f64().abs())
            }
        }
        impl FloatOps for $t {
            #[inline(always)]
            fn exp_op(self) -> Self {
                Self::from_f64(self.to_f64().exp())
            }
            #[inline(always)]
            fn log_op(self) -> Self {
                Self::from_f64(self.to_f64().ln())
            }
            #[inline(always)]
            fn tanh_op(self) -> Self {
                Self::from_f64(self.to_f64().tanh())
            }
            #[inline(always)]
            fn sin_op(self) -> Self {
                Self::from_f64(self.to_f64().sin())
            }
            #[inline(always)]
            fn cos_op(self) -> Self {
                Self::from_f64(self.to_f64().cos())
            }
            #[inline(always)]
            fn erf_op(self) -> Self {
                Self::from_f64(crate::dtype::float::erf::erf_f64(self.to_f64()))
            }
            #[inline(always)]
            fn erfc_op(self) -> Self {
                Self::from_f64(1.0 - crate::dtype::float::erf::erf_f64(self.to_f64()))
            }
            #[inline(always)]
            fn tan_op(self) -> Self {
                Self::from_f64(self.to_f64().tan())
            }
            #[inline(always)]
            fn asin_op(self) -> Self {
                Self::from_f64(self.to_f64().asin())
            }
            #[inline(always)]
            fn acos_op(self) -> Self {
                Self::from_f64(self.to_f64().acos())
            }
            #[inline(always)]
            fn atan_op(self) -> Self {
                Self::from_f64(self.to_f64().atan())
            }
            #[inline(always)]
            fn sinh_op(self) -> Self {
                Self::from_f64(self.to_f64().sinh())
            }
            #[inline(always)]
            fn cosh_op(self) -> Self {
                Self::from_f64(self.to_f64().cosh())
            }
            #[inline(always)]
            fn log2_op(self) -> Self {
                Self::from_f64(self.to_f64().log2())
            }
            #[inline(always)]
            fn log10_op(self) -> Self {
                Self::from_f64(self.to_f64().log10())
            }
            #[inline(always)]
            fn gelu_op(self) -> Self {
                let x_f = self.to_f64();
                let res = 0.5
                    * x_f
                    * (1.0
                        + crate::dtype::float::erf::erf_f64(
                            x_f * core::f64::consts::FRAC_1_SQRT_2,
                        ));
                Self::from_f64(res)
            }
            #[inline(always)]
            fn sigmoid_op(self) -> Self {
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
            #[inline(always)]
            fn floor(self) -> Self {
                Self::from_f64(self.to_f64().floor())
            }
            #[inline(always)]
            fn ceil(self) -> Self {
                Self::from_f64(self.to_f64().ceil())
            }
            #[inline(always)]
            fn round(self) -> Self {
                Self::from_f64(self.to_f64().round())
            }
            #[inline(always)]
            fn trunc(self) -> Self {
                Self::from_f64(self.to_f64().trunc())
            }
            #[inline(always)]
            fn fract(self) -> Self {
                Self::from_f64(self.to_f64().fract())
            }
            #[inline(always)]
            fn abs(self) -> Self {
                Self::from_f64(self.to_f64().abs())
            }
            #[inline(always)]
            fn signum(self) -> Self {
                Self::from_f64(self.to_f64().signum())
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                Self::from_f64(self.to_f64().sqrt())
            }
            #[inline(always)]
            fn exp(self) -> Self {
                Self::from_f64(self.to_f64().exp())
            }
            #[inline(always)]
            fn exp2(self) -> Self {
                Self::from_f64(self.to_f64().exp2())
            }
            #[inline(always)]
            fn ln(self) -> Self {
                Self::from_f64(self.to_f64().ln())
            }
            #[inline(always)]
            fn log2(self) -> Self {
                Self::from_f64(self.to_f64().log2())
            }
            #[inline(always)]
            fn log10(self) -> Self {
                Self::from_f64(self.to_f64().log10())
            }
            #[inline(always)]
            fn sin(self) -> Self {
                Self::from_f64(self.to_f64().sin())
            }
            #[inline(always)]
            fn cos(self) -> Self {
                Self::from_f64(self.to_f64().cos())
            }
            #[inline(always)]
            fn tan(self) -> Self {
                Self::from_f64(self.to_f64().tan())
            }
            #[inline(always)]
            fn asin(self) -> Self {
                Self::from_f64(self.to_f64().asin())
            }
            #[inline(always)]
            fn acos(self) -> Self {
                Self::from_f64(self.to_f64().acos())
            }
            #[inline(always)]
            fn atan(self) -> Self {
                Self::from_f64(self.to_f64().atan())
            }
            #[inline(always)]
            fn sinh(self) -> Self {
                Self::from_f64(self.to_f64().sinh())
            }
            #[inline(always)]
            fn cosh(self) -> Self {
                Self::from_f64(self.to_f64().cosh())
            }
            #[inline(always)]
            fn tanh(self) -> Self {
                Self::from_f64(self.to_f64().tanh())
            }
            #[inline(always)]
            fn powf(self, n: Self) -> Self {
                Self::from_f64(self.to_f64().powf(n.to_f64()))
            }
            #[inline(always)]
            fn is_nan(self) -> bool {
                self.to_f64().is_nan()
            }
            #[inline(always)]
            fn is_infinite(self) -> bool {
                self.to_f64().is_infinite()
            }
            #[inline(always)]
            fn is_finite(self) -> bool {
                self.to_f64().is_finite()
            }
        }
    };
}

impl_scalar_float_half!(f16);
impl_scalar_float_half!(bf16);
