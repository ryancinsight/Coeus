use crate::dtype::traits::{private, Float, FloatOps, Scalar};

macro_rules! impl_scalar_float_native {
    ($t:ty, $erf_fn:path) => {
        impl private::Sealed for $t {}
        impl Scalar for $t {
            #[inline(always)]
            fn zero() -> Self {
                0.0 as $t
            }
            #[inline(always)]
            fn one() -> Self {
                1.0 as $t
            }
            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline(always)]
            fn from_f64(v: f64) -> Self {
                v as Self
            }
            #[inline(always)]
            fn from_usize(v: usize) -> Self {
                v as Self
            }
            #[inline(always)]
            fn sqrt_val(self) -> Self {
                self.sqrt()
            }
            #[inline(always)]
            fn abs_val(self) -> Self {
                self.abs()
            }
            #[inline]
            fn add_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
                if hermes_simd::elementwise_add::<$t>(a, b, out).is_err() {
                    for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                        *o = x + y;
                    }
                }
            }
            #[inline]
            fn sub_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
                if hermes_simd::elementwise_sub::<$t>(a, b, out).is_err() {
                    for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                        *o = x - y;
                    }
                }
            }
            #[inline]
            fn mul_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
                if hermes_simd::elementwise_mul::<$t>(a, b, out).is_err() {
                    for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                        *o = x * y;
                    }
                }
            }
            #[inline]
            fn div_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
                if hermes_simd::elementwise_div::<$t>(a, b, out).is_err() {
                    for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                        *o = x / y;
                    }
                }
            }
            #[inline]
            fn dot_slice(a: &[Self], b: &[Self]) -> Self {
                assert_eq!(a.len(), b.len(), "dot_slice: length mismatch");
                match hermes_simd::dot::<$t>(a, b) {
                    Ok(value) => value,
                    Err(_) => {
                        let mut acc = 0.0 as Self;
                        for (&x, &y) in a.iter().zip(b.iter()) {
                            acc += x * y;
                        }
                        acc
                    }
                }
            }
            #[inline]
            fn scale_slice(data: &mut [Self], scalar: Self) {
                hermes_simd::scale::<$t>(data, scalar);
            }
            #[inline]
            fn sum_slice(s: &[Self]) -> Self {
                hermes_simd::sum::<$t>(s)
            }
            #[inline]
            fn min_slice(s: &[Self]) -> Self {
                hermes_simd::min::<$t>(s)
            }
            #[inline]
            fn max_slice(s: &[Self]) -> Self {
                hermes_simd::max::<$t>(s)
            }
        }
        impl FloatOps for $t {
            #[inline(always)]
            fn exp_op(self) -> Self {
                self.exp()
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
                $erf_fn(self)
            }
            #[inline(always)]
            fn gelu_op(self) -> Self {
                let half = Self::from_f64(0.5);
                let one = Self::from_f64(1.0);
                let inv_sqrt_two = Self::from_f64(core::f64::consts::FRAC_1_SQRT_2);
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
            fn abs(self) -> Self {
                self.abs()
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
            fn is_nan(self) -> bool {
                self.is_nan()
            }
            #[inline(always)]
            fn is_infinite(self) -> bool {
                self.is_infinite()
            }
            #[inline(always)]
            fn is_finite(self) -> bool {
                self.is_finite()
            }
        }
    };
}

impl_scalar_float_native!(f32, crate::dtype::float::erf::erf_f32);
impl_scalar_float_native!(f64, crate::dtype::float::erf::erf_f64);
