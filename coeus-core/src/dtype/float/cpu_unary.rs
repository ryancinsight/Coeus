use half::{bf16, f16};

macro_rules! impl_cpu_unary_dispatch_float {
    ($t:ty) => {
        impl $crate::dtype::CpuUnaryDispatch for $t {
            #[inline(always)]
            fn eval_unary(op: $crate::dtype::CpuUnaryOp, x: Self) -> Self {
                use $crate::dtype::{CpuUnaryOp, FloatOps, Scalar};
                match op {
                    CpuUnaryOp::Relu => {
                        if x > Self::zero() {
                            x
                        } else {
                            Self::zero()
                        }
                    }
                    CpuUnaryOp::ReluGrad => {
                        if x > Self::zero() {
                            Self::one()
                        } else {
                            Self::zero()
                        }
                    }
                    CpuUnaryOp::Sigmoid => x.sigmoid_op(),
                    CpuUnaryOp::SigmoidGrad => x * (Self::one() - x),
                    CpuUnaryOp::Tanh => x.tanh_op(),
                    CpuUnaryOp::TanhGrad => Self::one() - x * x,
                    CpuUnaryOp::Gelu => x.gelu_op(),
                    CpuUnaryOp::GeluGrad => {
                        let half = Self::from_f64(0.5);
                        let one = Self::one();
                        let inv_sqrt_two = Self::from_f64(core::f64::consts::FRAC_1_SQRT_2);
                        let inv_sqrt_two_pi = Self::from_f64(0.3989422804014327);
                        let x2 = x * x;
                        half * (one + (x * inv_sqrt_two).erf_op())
                            + x * ((Self::zero() - half * x2).exp_op()) * inv_sqrt_two_pi
                    }
                    CpuUnaryOp::Sin => x.sin_op(),
                    CpuUnaryOp::Cos => x.cos_op(),
                    CpuUnaryOp::Exp => x.exp_op(),
                    CpuUnaryOp::Log => x.log_op(),
                    CpuUnaryOp::Neg => Self::zero() - x,
                    CpuUnaryOp::Abs => x.abs_val(),
                    CpuUnaryOp::Sqrt => x.sqrt_val(),
                    CpuUnaryOp::Silu => x * x.sigmoid_op(),
                    CpuUnaryOp::SiluGrad => {
                        let s = x.sigmoid_op();
                        s * (Self::one() + x * (Self::one() - s))
                    }
                    CpuUnaryOp::Mish => {
                        let sp = (Self::one() + x.exp_op()).log_op();
                        x * sp.tanh_op()
                    }
                    CpuUnaryOp::MishGrad => {
                        let sp = (Self::one() + x.exp_op()).log_op();
                        let w = sp.tanh_op();
                        let sig = x.sigmoid_op();
                        w + x * (Self::one() - w * w) * sig
                    }
                    CpuUnaryOp::Elu => {
                        if x >= Self::zero() {
                            x
                        } else {
                            x.exp_op() - Self::one()
                        }
                    }
                    CpuUnaryOp::EluGrad => {
                        if x >= Self::zero() {
                            Self::one()
                        } else {
                            x.exp_op()
                        }
                    }
                    CpuUnaryOp::Softplus => (Self::one() + x.exp_op()).log_op(),
                    CpuUnaryOp::SoftplusGrad => x.sigmoid_op(),
                    CpuUnaryOp::GeluTanh => {
                        let c1 = Self::from_f64(0.7978845608);
                        let c2 = Self::from_f64(0.044715);
                        let half = Self::from_f64(0.5);
                        let one = Self::one();
                        let v = c1 * (x + c2 * x * x * x);
                        half * x * (one + v.tanh_op())
                    }
                    CpuUnaryOp::GeluTanhGrad => {
                        let c1 = Self::from_f64(0.7978845608);
                        let c2 = Self::from_f64(0.044715);
                        let c3 = Self::from_f64(0.134145);
                        let half = Self::from_f64(0.5);
                        let one = Self::one();
                        let v = c1 * (x + c2 * x * x * x);
                        let t = v.tanh_op();
                        let dt = c1 * (one + c3 * x * x);
                        half * (one + t) + half * x * (one - t * t) * dt
                    }
                    CpuUnaryOp::LeakyRelu(slope_bits) => {
                        let slope = Self::from_f64(f64::from_bits(slope_bits));
                        if x >= Self::zero() {
                            x
                        } else {
                            slope * x
                        }
                    }
                    CpuUnaryOp::LeakyReluGrad(slope_bits) => {
                        let slope = Self::from_f64(f64::from_bits(slope_bits));
                        if x >= Self::zero() {
                            Self::one()
                        } else {
                            slope
                        }
                    }
                    CpuUnaryOp::Hardtanh(bits) => {
                        let min_v = Self::from_f64(f32::from_bits(bits as u32) as f64);
                        let max_v = Self::from_f64(f32::from_bits((bits >> 32) as u32) as f64);
                        if x < min_v {
                            min_v
                        } else if x > max_v {
                            max_v
                        } else {
                            x
                        }
                    }
                    CpuUnaryOp::HardtanhGrad(bits) => {
                        let min_v = Self::from_f64(f32::from_bits(bits as u32) as f64);
                        let max_v = Self::from_f64(f32::from_bits((bits >> 32) as u32) as f64);
                        if x > min_v && x < max_v {
                            Self::one()
                        } else {
                            Self::zero()
                        }
                    }
                    CpuUnaryOp::Hardsigmoid => {
                        let six = Self::from_f64(6.0);
                        let half = Self::from_f64(0.5);
                        let one = Self::one();
                        let v = x / six + half;
                        if v < Self::zero() {
                            Self::zero()
                        } else if v > one {
                            one
                        } else {
                            v
                        }
                    }
                    CpuUnaryOp::HardsigmoidGrad => {
                        let three = Self::from_f64(3.0);
                        let six = Self::from_f64(6.0);
                        if x > -three && x < three {
                            Self::one() / six
                        } else {
                            Self::zero()
                        }
                    }
                    CpuUnaryOp::Hardswish => {
                        let three = Self::from_f64(3.0);
                        let six = Self::from_f64(6.0);
                        let v = x + three;
                        let relu6 = if v < Self::zero() {
                            Self::zero()
                        } else if v > six {
                            six
                        } else {
                            v
                        };
                        x * relu6 / six
                    }
                    CpuUnaryOp::HardswishGrad => {
                        let three = Self::from_f64(3.0);
                        let six = Self::from_f64(6.0);
                        let two = Self::from_f64(2.0);
                        let one = Self::one();
                        if x < -three {
                            Self::zero()
                        } else if x <= three {
                            (two * x + three) / six
                        } else {
                            one
                        }
                    }
                    CpuUnaryOp::Hardshrink(lam_bits) => {
                        let lam = Self::from_f64(f64::from_bits(lam_bits));
                        let ax = if x < Self::zero() {
                            Self::zero() - x
                        } else {
                            x
                        };
                        if ax > lam {
                            x
                        } else {
                            Self::zero()
                        }
                    }
                    CpuUnaryOp::HardshrinkGrad(lam_bits) => {
                        let lam = Self::from_f64(f64::from_bits(lam_bits));
                        let ax = if x < Self::zero() {
                            Self::zero() - x
                        } else {
                            x
                        };
                        if ax > lam {
                            Self::one()
                        } else {
                            Self::zero()
                        }
                    }
                    CpuUnaryOp::Softshrink(lam_bits) => {
                        let lam = Self::from_f64(f64::from_bits(lam_bits));
                        let ax = if x < Self::zero() {
                            Self::zero() - x
                        } else {
                            x
                        };
                        if ax > lam {
                            let s = if x < Self::zero() {
                                Self::zero() - Self::one()
                            } else {
                                Self::one()
                            };
                            s * (ax - lam)
                        } else {
                            Self::zero()
                        }
                    }
                    CpuUnaryOp::SoftshrinkGrad(lam_bits) => {
                        let lam = Self::from_f64(f64::from_bits(lam_bits));
                        let ax = if x < Self::zero() {
                            Self::zero() - x
                        } else {
                            x
                        };
                        if ax > lam {
                            Self::one()
                        } else {
                            Self::zero()
                        }
                    }
                    CpuUnaryOp::Softsign => {
                        let one = Self::one();
                        let ax = if x < Self::zero() {
                            Self::zero() - x
                        } else {
                            x
                        };
                        x / (one + ax)
                    }
                    CpuUnaryOp::SoftsignGrad => {
                        let one = Self::one();
                        let ax = if x < Self::zero() {
                            Self::zero() - x
                        } else {
                            x
                        };
                        let denom = (one + ax) * (one + ax);
                        one / denom
                    }
                    CpuUnaryOp::Threshold(bits) => {
                        let thr = Self::from_f64(f32::from_bits(bits as u32) as f64);
                        let val = Self::from_f64(f32::from_bits((bits >> 32) as u32) as f64);
                        if x > thr {
                            x
                        } else {
                            val
                        }
                    }
                    CpuUnaryOp::ThresholdGrad(bits) => {
                        let thr = Self::from_f64(f32::from_bits(bits as u32) as f64);
                        if x > thr {
                            Self::one()
                        } else {
                            Self::zero()
                        }
                    }
                    CpuUnaryOp::Celu(alpha_bits) => {
                        let alpha = Self::from_f64(f64::from_bits(alpha_bits));
                        let one = Self::one();
                        if x >= Self::zero() {
                            x
                        } else {
                            alpha * ((x / alpha).exp_op() - one)
                        }
                    }
                    CpuUnaryOp::CeluGrad(alpha_bits) => {
                        let alpha = Self::from_f64(f64::from_bits(alpha_bits));
                        if x >= Self::zero() {
                            Self::one()
                        } else {
                            (x / alpha).exp_op()
                        }
                    }
                    CpuUnaryOp::Recip => Self::one() / x,
                    CpuUnaryOp::Sign => {
                        if x > Self::zero() {
                            Self::one()
                        } else if x < Self::zero() {
                            Self::zero() - Self::one()
                        } else {
                            Self::zero()
                        }
                    }
                    CpuUnaryOp::Floor => Self::from_f64(Self::to_f64(x).floor()),
                    CpuUnaryOp::Ceil => Self::from_f64(Self::to_f64(x).ceil()),
                    CpuUnaryOp::Round => Self::from_f64(Self::to_f64(x).round()),
                    CpuUnaryOp::Trunc => Self::from_f64(Self::to_f64(x).trunc()),
                }
            }
        }
    };
}

impl_cpu_unary_dispatch_float!(f32);
impl_cpu_unary_dispatch_float!(f64);
impl_cpu_unary_dispatch_float!(f16);
impl_cpu_unary_dispatch_float!(bf16);
