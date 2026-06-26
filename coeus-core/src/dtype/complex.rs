use crate::dtype::traits::{private, Float, FloatOps, Scalar};
use bytemuck::{Pod, Zeroable};
use num_traits::{Num, One, Zero};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

/// Complex number representation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Complex<T> {
    pub re: T,
    pub im: T,
}

impl<T> Complex<T> {
    /// Create a new complex number.
    #[inline(always)]
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

unsafe impl<T: Zeroable> Zeroable for Complex<T> {}
unsafe impl<T: Pod> Pod for Complex<T> {}

impl<T: Add<Output = T>> Add for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl<T: Sub<Output = T>> Sub for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

impl<T: Num + Clone> Mul for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re.clone() * other.re.clone() - self.im.clone() * other.im.clone(),
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl<T: Num + Clone> Div for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn div(self, other: Self) -> Self {
        let denom = other.re.clone() * other.re.clone() + other.im.clone() * other.im.clone();
        Self {
            re: (self.re.clone() * other.re.clone() + self.im.clone() * other.im.clone())
                / denom.clone(),
            im: (self.im * other.re.clone() - self.re * other.im) / denom,
        }
    }
}

impl<T: Rem<Output = T>> Rem for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn rem(self, other: Self) -> Self {
        Self {
            re: self.re % other.re,
            im: self.im % other.im,
        }
    }
}

impl<T: Neg<Output = T>> Neg for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl<T: Zero> Zero for Complex<T> {
    #[inline(always)]
    fn zero() -> Self {
        Self {
            re: T::zero(),
            im: T::zero(),
        }
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }
}

impl<T: One + Zero + Num + Clone> One for Complex<T> {
    #[inline(always)]
    fn one() -> Self {
        Self {
            re: T::one(),
            im: T::zero(),
        }
    }
}

impl<T: Num + Clone> Num for Complex<T> {
    type FromStrRadixErr = ();
    #[inline(always)]
    fn from_str_radix(_str: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Err(())
    }
}

impl<T: PartialOrd> PartialOrd for Complex<T> {
    #[inline(always)]
    fn partial_cmp(&self, _other: &Self) -> Option<std::cmp::Ordering> {
        None
    }
}

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
        use num_traits::{One, Zero};
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
