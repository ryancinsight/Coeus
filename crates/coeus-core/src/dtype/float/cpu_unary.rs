use eunomia::{Bf16, F16};

macro_rules! impl_cpu_unary_dispatch_float {
    ($t:ty) => {
        impl $crate::dtype::CpuUnaryDispatch for $t {
            #[inline(always)]
            fn eval_unary(op: $crate::dtype::CpuUnaryOp, x: Self) -> Self {
                use ::eunomia::NumericElement;
                use $crate::dtype::{CpuUnaryOp, FloatOps, Scalar};
                match op {
                    CpuUnaryOp::Relu => {
                        if x > Self::ZERO {
                            x
                        } else {
                            Self::ZERO
                        }
                    }
                    CpuUnaryOp::ReluGrad => {
                        if x > Self::ZERO {
                            Self::ONE
                        } else {
                            Self::ZERO
                        }
                    }
                    CpuUnaryOp::Sigmoid => x.sigmoid_op(),
                    CpuUnaryOp::SigmoidGrad => x * (Self::ONE - x),
                    CpuUnaryOp::Tanh => x.tanh_op(),
                    CpuUnaryOp::TanhGrad => Self::ONE - x * x,
                    CpuUnaryOp::Gelu => x.gelu_op(),
                    CpuUnaryOp::GeluGrad => {
                        let half = Self::from_f64(0.5);
                        let one = Self::ONE;
                        let inv_sqrt_two = Self::from_f64(core::f64::consts::FRAC_1_SQRT_2);
                        let inv_sqrt_two_pi = Self::from_f64(0.3989422804014327);
                        let x2 = x * x;
                        half * (one + (x * inv_sqrt_two).erf_op())
                            + x * ((Self::ZERO - half * x2).exp_op()) * inv_sqrt_two_pi
                    }
                    CpuUnaryOp::Sin => x.sin_op(),
                    CpuUnaryOp::Cos => x.cos_op(),
                    CpuUnaryOp::Exp => x.exp_op(),
                    CpuUnaryOp::Log => x.log_op(),
                    CpuUnaryOp::Erf => x.erf_op(),
                    CpuUnaryOp::Erfc => x.erfc_op(),
                    CpuUnaryOp::Lgamma => x.lgamma_op(),
                    CpuUnaryOp::Tan => x.tan_op(),
                    CpuUnaryOp::Asin => x.asin_op(),
                    CpuUnaryOp::Acos => x.acos_op(),
                    CpuUnaryOp::Atan => x.atan_op(),
                    CpuUnaryOp::Sinh => x.sinh_op(),
                    CpuUnaryOp::Cosh => x.cosh_op(),
                    CpuUnaryOp::Log2 => x.log2_op(),
                    CpuUnaryOp::Log10 => x.log10_op(),
                    CpuUnaryOp::Exp2 => x.exp2_op(),
                    CpuUnaryOp::Atanh => x.atanh_op(),
                    CpuUnaryOp::Asinh => x.asinh_op(),
                    CpuUnaryOp::Acosh => x.acosh_op(),
                    CpuUnaryOp::Expm1 => x.expm1_op(),
                    CpuUnaryOp::Log1p => x.log1p_op(),
                    CpuUnaryOp::Neg => Self::ZERO - x,
                    CpuUnaryOp::Abs => x.abs_val(),
                    CpuUnaryOp::Sqrt => x.sqrt_val(),
                    CpuUnaryOp::Silu => x * x.sigmoid_op(),
                    CpuUnaryOp::SiluGrad => {
                        let s = x.sigmoid_op();
                        s * (Self::ONE + x * (Self::ONE - s))
                    }
                    CpuUnaryOp::Mish => {
                        let sp = (Self::ONE + x.exp_op()).log_op();
                        x * sp.tanh_op()
                    }
                    CpuUnaryOp::MishGrad => {
                        let sp = (Self::ONE + x.exp_op()).log_op();
                        let w = sp.tanh_op();
                        let sig = x.sigmoid_op();
                        w + x * (Self::ONE - w * w) * sig
                    }
                    CpuUnaryOp::Elu => {
                        if x >= Self::ZERO {
                            x
                        } else {
                            x.exp_op() - Self::ONE
                        }
                    }
                    CpuUnaryOp::EluGrad => {
                        if x >= Self::ZERO {
                            Self::ONE
                        } else {
                            x.exp_op()
                        }
                    }
                    CpuUnaryOp::Softplus => (Self::ONE + x.exp_op()).log_op(),
                    CpuUnaryOp::SoftplusGrad => x.sigmoid_op(),
                    CpuUnaryOp::GeluTanh => {
                        let c1 = Self::from_f64(0.7978845608);
                        let c2 = Self::from_f64(0.044715);
                        let half = Self::from_f64(0.5);
                        let one = Self::ONE;
                        let v = c1 * (x + c2 * x * x * x);
                        half * x * (one + v.tanh_op())
                    }
                    CpuUnaryOp::GeluTanhGrad => {
                        let c1 = Self::from_f64(0.7978845608);
                        let c2 = Self::from_f64(0.044715);
                        let c3 = Self::from_f64(0.134145);
                        let half = Self::from_f64(0.5);
                        let one = Self::ONE;
                        let v = c1 * (x + c2 * x * x * x);
                        let t = v.tanh_op();
                        let dt = c1 * (one + c3 * x * x);
                        half * (one + t) + half * x * (one - t * t) * dt
                    }
                    CpuUnaryOp::LeakyRelu(slope_bits) => {
                        let slope = Self::from_f64(f64::from_bits(slope_bits));
                        if x >= Self::ZERO {
                            x
                        } else {
                            slope * x
                        }
                    }
                    // PReLU / LeakyReLU gradient oracle: dx = 1 if x > 0 else slope.
                    // Matches PyTorch's `F.prelu` / `F.leaky_relu` contract which
                    // returns `slope` (not 1) at the kink position x = 0. The
                    // forward's `x >= 0 ? x : slope*x` is unaffected (both predicates
                    // yield 0 at x = 0); only the gradient predicate is tightened.
                    CpuUnaryOp::LeakyReluGrad(slope_bits) => {
                        let slope = Self::from_f64(f64::from_bits(slope_bits));
                        if x > Self::ZERO {
                            Self::ONE
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
                            Self::ONE
                        } else {
                            Self::ZERO
                        }
                    }
                    CpuUnaryOp::Hardsigmoid => {
                        let six = Self::from_f64(6.0);
                        let half = Self::from_f64(0.5);
                        let one = Self::ONE;
                        let v = x / six + half;
                        if v < Self::ZERO {
                            Self::ZERO
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
                            Self::ONE / six
                        } else {
                            Self::ZERO
                        }
                    }
                    CpuUnaryOp::Hardswish => {
                        let three = Self::from_f64(3.0);
                        let six = Self::from_f64(6.0);
                        let v = x + three;
                        let relu6 = if v < Self::ZERO {
                            Self::ZERO
                        } else if v > six {
                            six
                        } else {
                            v
                        };
                        x * relu6 / six
                    }
                    CpuUnaryOp::HardswishGrad => {
                        // Piecewise: 0 if x ≤ -3, (2x+3)/6 if -3 < x < 3,
                        // 1 if x ≥ 3. Matches `torch.nn.functional.hardswish`'s
                        // CPU kernel, which takes the zero branch at `x ≤ -3`
                        // (inclusive). The previous exclusive bound
                        // (`x < -three`) leaked the middle-branch evaluation
                        // at the kink x = -3, producing -0.5 instead of 0.
                        let three = Self::from_f64(3.0);
                        let six = Self::from_f64(6.0);
                        let two = Self::from_f64(2.0);
                        let one = Self::ONE;
                        if x <= -three {
                            Self::ZERO
                        } else if x < three {
                            (two * x + three) / six
                        } else {
                            one
                        }
                    }
                    CpuUnaryOp::Hardshrink(lam_bits) => {
                        let lam = Self::from_f64(f64::from_bits(lam_bits));
                        let ax = if x < Self::ZERO { Self::ZERO - x } else { x };
                        if ax > lam {
                            x
                        } else {
                            Self::ZERO
                        }
                    }
                    CpuUnaryOp::HardshrinkGrad(lam_bits) => {
                        let lam = Self::from_f64(f64::from_bits(lam_bits));
                        let ax = if x < Self::ZERO { Self::ZERO - x } else { x };
                        if ax > lam {
                            Self::ONE
                        } else {
                            Self::ZERO
                        }
                    }
                    CpuUnaryOp::Softshrink(lam_bits) => {
                        let lam = Self::from_f64(f64::from_bits(lam_bits));
                        let ax = if x < Self::ZERO { Self::ZERO - x } else { x };
                        if ax > lam {
                            let s = if x < Self::ZERO {
                                Self::ZERO - Self::ONE
                            } else {
                                Self::ONE
                            };
                            s * (ax - lam)
                        } else {
                            Self::ZERO
                        }
                    }
                    CpuUnaryOp::SoftshrinkGrad(lam_bits) => {
                        let lam = Self::from_f64(f64::from_bits(lam_bits));
                        let ax = if x < Self::ZERO { Self::ZERO - x } else { x };
                        if ax > lam {
                            Self::ONE
                        } else {
                            Self::ZERO
                        }
                    }
                    CpuUnaryOp::Softsign => {
                        let one = Self::ONE;
                        let ax = if x < Self::ZERO { Self::ZERO - x } else { x };
                        x / (one + ax)
                    }
                    CpuUnaryOp::SoftsignGrad => {
                        let one = Self::ONE;
                        let ax = if x < Self::ZERO { Self::ZERO - x } else { x };
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
                            Self::ONE
                        } else {
                            Self::ZERO
                        }
                    }
                    CpuUnaryOp::Celu(alpha_bits) => {
                        let alpha = Self::from_f64(f64::from_bits(alpha_bits));
                        let one = Self::ONE;
                        if x >= Self::ZERO {
                            x
                        } else {
                            alpha * ((x / alpha).exp_op() - one)
                        }
                    }
                    CpuUnaryOp::CeluGrad(alpha_bits) => {
                        let alpha = Self::from_f64(f64::from_bits(alpha_bits));
                        if x >= Self::ZERO {
                            Self::ONE
                        } else {
                            (x / alpha).exp_op()
                        }
                    }
                    CpuUnaryOp::Recip => Self::ONE / x,
                    CpuUnaryOp::Sign => {
                        if x > Self::ZERO {
                            Self::ONE
                        } else if x < Self::ZERO {
                            Self::ZERO - Self::ONE
                        } else {
                            Self::ZERO
                        }
                    }
                    CpuUnaryOp::Floor => {
                        Self::from_f64(<Self as NumericElement>::to_f64(x).floor())
                    }
                    CpuUnaryOp::Ceil => Self::from_f64(<Self as NumericElement>::to_f64(x).ceil()),
                    // Ties-to-even (banker's rounding) per IEEE-754 roundTiesToEven,
                    // matching torch.round, WGSL round(), and CUDA rintf.
                    CpuUnaryOp::Round => {
                        Self::from_f64(<Self as NumericElement>::to_f64(x).round_ties_even())
                    }
                    CpuUnaryOp::Trunc => {
                        Self::from_f64(<Self as NumericElement>::to_f64(x).trunc())
                    }
                }
            }
        }
    };
}

impl_cpu_unary_dispatch_float!(f32);
impl_cpu_unary_dispatch_float!(f64);
impl_cpu_unary_dispatch_float!(F16);
impl_cpu_unary_dispatch_float!(Bf16);
