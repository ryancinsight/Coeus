//! Coeus numeric-trait impls (`Scalar`/`FloatOps`/`CpuUnaryDispatch`) for the
//! SSOT [`eunomia::Complex`]. The complex *type* and its arithmetic live in
//! eunomia (the datatype vocabulary); coeus owns only the coeus-specific trait
//! surface over it. Consolidates the former coeus-local `Complex` definition.

use crate::dtype::traits::{private, Float, FloatOps, Scalar};
use eunomia::Complex;

impl<T: Float> private::Sealed for Complex<T> {}

impl<T: Float> FloatOps for Complex<T> {
    #[inline(always)]
    fn exp_op(self) -> Self {
        let r = self.re.exp();
        Self {
            re: r * self.im.cos(),
            im: r * self.im.sin(),
        }
    }

    #[inline(always)]
    fn log_op(self) -> Self {
        let r = (self.re * self.re + self.im * self.im).sqrt();
        let theta = T::from_f64(self.im.to_f64().atan2(self.re.to_f64()));
        Self {
            re: r.ln(),
            im: theta,
        }
    }

    #[inline(always)]
    fn tanh_op(self) -> Self {
        let two = T::from_f64(2.0);
        let x2 = self.re * two;
        let y2 = self.im * two;
        let denom = x2.cosh() + y2.cos();
        Self {
            re: x2.sinh() / denom,
            im: y2.sin() / denom,
        }
    }

    #[inline(always)]
    fn sin_op(self) -> Self {
        Self {
            re: self.re.sin() * self.im.cosh(),
            im: self.re.cos() * self.im.sinh(),
        }
    }

    #[inline(always)]
    fn cos_op(self) -> Self {
        Self {
            re: self.re.cos() * self.im.cosh(),
            im: T::zero() - (self.re.sin() * self.im.sinh()),
        }
    }

    #[inline(always)]
    fn erf_op(self) -> Self {
        panic!("erf not supported on complex types")
    }

    #[inline(always)]
    fn tan_op(self) -> Self {
        Self::zero()
    }

    #[inline(always)]
    fn asin_op(self) -> Self {
        Self::zero()
    }

    #[inline(always)]
    fn acos_op(self) -> Self {
        Self::zero()
    }

    #[inline(always)]
    fn atan_op(self) -> Self {
        Self::zero()
    }

    #[inline(always)]
    fn gelu_op(self) -> Self {
        panic!("gelu not supported on complex types")
    }

    #[inline(always)]
    fn sigmoid_op(self) -> Self {
        let one = Self::one();
        let exp_neg_z = Self {
            re: T::zero() - self.re,
            im: T::zero() - self.im,
        }
        .exp_op();
        one / (one + exp_neg_z)
    }
}

impl<T: Float> Scalar for Complex<T> {
    #[inline(always)]
    fn zero() -> Self {
        Self {
            re: <T as Scalar>::zero(),
            im: <T as Scalar>::zero(),
        }
    }

    #[inline(always)]
    fn one() -> Self {
        Self {
            re: <T as Scalar>::one(),
            im: <T as Scalar>::zero(),
        }
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self.re.to_f64()
    }

    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        Self {
            re: T::from_f64(v),
            im: T::zero(),
        }
    }

    #[inline(always)]
    fn from_usize(v: usize) -> Self {
        Self {
            re: T::from_usize(v),
            im: T::zero(),
        }
    }

    #[inline(always)]
    fn sqrt_val(self) -> Self {
        let r = (self.re * self.re + self.im * self.im).sqrt();
        let u = ((r + self.re) / T::from_f64(2.0)).sqrt();
        let v = ((r - self.re) / T::from_f64(2.0)).sqrt();
        let v = if self.im.to_f64() < 0.0 {
            T::zero() - v
        } else {
            v
        };
        Self { re: u, im: v }
    }

    #[inline(always)]
    fn abs_val(self) -> Self {
        let mag = (self.re * self.re + self.im * self.im).sqrt();
        Self {
            re: mag,
            im: T::zero(),
        }
    }
}

impl<T: Float> crate::dtype::CpuUnaryDispatch for Complex<T> {
    #[inline]
    fn eval_unary(op: crate::dtype::CpuUnaryOp, x: Self) -> Self {
        use crate::dtype::{CpuUnaryOp, FloatOps, Scalar};
        match op {
            CpuUnaryOp::Relu => panic!("Relu not supported on complex types"),
            CpuUnaryOp::ReluGrad => panic!("ReluGrad not supported on complex types"),
            CpuUnaryOp::Sigmoid => x.sigmoid_op(),
            CpuUnaryOp::SigmoidGrad => x * (Self::one() - x),
            CpuUnaryOp::Tanh => x.tanh_op(),
            CpuUnaryOp::TanhGrad => Self::one() - x * x,
            CpuUnaryOp::Gelu => panic!("Gelu not supported on complex types"),
            CpuUnaryOp::GeluGrad => panic!("GeluGrad not supported on complex types"),
            CpuUnaryOp::Sin => x.sin_op(),
            CpuUnaryOp::Cos => x.cos_op(),
            CpuUnaryOp::Tan => x.tan_op(),
            CpuUnaryOp::Asin => x.asin_op(),
            CpuUnaryOp::Acos => x.acos_op(),
            CpuUnaryOp::Atan => x.atan_op(),
            CpuUnaryOp::Exp => x.exp_op(),
            CpuUnaryOp::Log => x.log_op(),
            CpuUnaryOp::Neg => Self::zero() - x,
            CpuUnaryOp::Abs => x.abs_val(),
            CpuUnaryOp::Sqrt => x.sqrt_val(),
            CpuUnaryOp::Recip => {
                // 1 / (a + bi) = (a - bi) / (a² + b²)
                let mag_sq = x.re * x.re + x.im * x.im;
                Self {
                    re: x.re / mag_sq,
                    im: T::zero() - x.im / mag_sq,
                }
            }
            CpuUnaryOp::Sign => {
                let mag = (x.re * x.re + x.im * x.im).sqrt();
                if mag == T::zero() {
                    Self::zero()
                } else {
                    Self {
                        re: x.re / mag,
                        im: x.im / mag,
                    }
                }
            }
            _ => panic!("Unary operation not supported for Complex"),
        }
    }
}
