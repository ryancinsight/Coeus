use half::{bf16, f16};

macro_rules! impl_cpu_unary_dispatch_float {
    ($t:ty) => {
        impl $crate::dtype::CpuUnaryDispatch for $t {
            #[inline]
            fn eval_unary(op: $crate::dtype::CpuUnaryOp, x: Self) -> Self {
                use num_traits::{One, Zero};
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
                        let c1 = Self::from_f64(0.7978845608);
                        let c2 = Self::from_f64(0.044715);
                        let c3 = Self::from_f64(0.134145);

                        let x2 = x * x;
                        let v = c1 * (x + c2 * x * x2);
                        let t = v.tanh_op();
                        let dy = c1 * (one + c3 * x2);
                        half * (one + t) + half * x * (one - t * t) * dy
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
                }
            }
        }
    };
}

impl_cpu_unary_dispatch_float!(f32);
impl_cpu_unary_dispatch_float!(f64);
impl_cpu_unary_dispatch_float!(f16);
impl_cpu_unary_dispatch_float!(bf16);
