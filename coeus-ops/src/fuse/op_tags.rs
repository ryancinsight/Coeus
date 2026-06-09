use coeus_core::FloatOps;
use coeus_core::Scalar;

// ── Binary Operations ZST Tags ──

pub trait BinaryOpTag: 'static + Send + Sync + Copy + Clone {
    const WGSL_SYMBOL: &'static str;
    fn apply<T: Scalar>(x: T, y: T) -> T;
}

#[derive(Clone, Copy)]
pub struct Add;
impl BinaryOpTag for Add {
    const WGSL_SYMBOL: &'static str = "+";
    #[inline(always)]
    fn apply<T: Scalar>(x: T, y: T) -> T {
        x + y
    }
}

#[derive(Clone, Copy)]
pub struct Sub;
impl BinaryOpTag for Sub {
    const WGSL_SYMBOL: &'static str = "-";
    #[inline(always)]
    fn apply<T: Scalar>(x: T, y: T) -> T {
        x - y
    }
}

#[derive(Clone, Copy)]
pub struct Mul;
impl BinaryOpTag for Mul {
    const WGSL_SYMBOL: &'static str = "*";
    #[inline(always)]
    fn apply<T: Scalar>(x: T, y: T) -> T {
        x * y
    }
}

#[derive(Clone, Copy)]
pub struct Div;
impl BinaryOpTag for Div {
    const WGSL_SYMBOL: &'static str = "/";
    #[inline(always)]
    fn apply<T: Scalar>(x: T, y: T) -> T {
        x / y
    }
}

// ── Unary Operations ZST Tags ──

pub trait UnaryOpTag<T: Scalar>: 'static + Send + Sync + Copy + Clone {
    const WGSL_TEMPLATE: &'static str;
    fn apply(x: T) -> T;
}

#[derive(Clone, Copy)]
pub struct Relu;
impl<T: Scalar> UnaryOpTag<T> for Relu {
    const WGSL_TEMPLATE: &'static str = "max(({}), 0.0)";
    #[inline(always)]
    fn apply(x: T) -> T {
        if x > T::zero() {
            x
        } else {
            T::zero()
        }
    }
}

#[derive(Clone, Copy)]
pub struct Sigmoid;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Sigmoid {
    const WGSL_TEMPLATE: &'static str = "1.0 / (1.0 + exp(-({})))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.sigmoid_op()
    }
}

#[derive(Clone, Copy)]
pub struct Tanh;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Tanh {
    const WGSL_TEMPLATE: &'static str = "tanh(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.tanh_op()
    }
}

#[derive(Clone, Copy)]
pub struct Gelu;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Gelu {
    const WGSL_TEMPLATE: &'static str =
        "0.5 * ({}) * (1.0 + tanh(0.7978845608 * (({}) + 0.044715 * ({}) * ({}) * ({}))))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.gelu_op()
    }
}

#[derive(Clone, Copy)]
pub struct GeluGrad;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for GeluGrad {
    const WGSL_TEMPLATE: &'static str = "0.5 * (1.0 + tanh(0.7978845608 * (({}) + 0.044715 * ({}) * ({}) * ({})))) + 0.5 * ({}) * (1.0 - tanh(0.7978845608 * (({}) + 0.044715 * ({}) * ({}) * ({}))) * tanh(0.7978845608 * (({}) + 0.044715 * ({}) * ({}) * ({})))) * 0.7978845608 * (1.0 + 0.134145 * ({}) * ({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        let half = T::from_f64(0.5);
        let one = T::one();
        let c1 = T::from_f64(0.7978845608);
        let c2 = T::from_f64(0.044715);
        let c3 = T::from_f64(0.134145);

        let x2 = x * x;
        let v = c1 * (x + c2 * x * x2);
        let t = v.tanh_op();
        let dy = c1 * (one + c3 * x2);
        half * (one + t) + half * x * (one - t * t) * dy
    }
}

#[derive(Clone, Copy)]
pub struct Sin;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Sin {
    const WGSL_TEMPLATE: &'static str = "sin(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.sin_op()
    }
}

#[derive(Clone, Copy)]
pub struct Cos;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Cos {
    const WGSL_TEMPLATE: &'static str = "cos(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.cos_op()
    }
}

#[derive(Clone, Copy)]
pub struct Exp;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Exp {
    const WGSL_TEMPLATE: &'static str = "exp(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.exp_op()
    }
}

#[derive(Clone, Copy)]
pub struct Log;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Log {
    const WGSL_TEMPLATE: &'static str = "log(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.log_op()
    }
}

#[derive(Clone, Copy)]
pub struct Neg;
impl<T: Scalar> UnaryOpTag<T> for Neg {
    const WGSL_TEMPLATE: &'static str = "-(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        T::zero() - x
    }
}

#[derive(Clone, Copy)]
pub struct Abs;
impl<T: Scalar> UnaryOpTag<T> for Abs {
    const WGSL_TEMPLATE: &'static str = "abs(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.abs_val()
    }
}

#[derive(Clone, Copy)]
pub struct Sqrt;
impl<T: Scalar> UnaryOpTag<T> for Sqrt {
    const WGSL_TEMPLATE: &'static str = "sqrt(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.sqrt_val()
    }
}

#[derive(Clone, Copy)]
pub struct Silu;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Silu {
    const WGSL_TEMPLATE: &'static str = "({}) / (1.0 + exp(-({})))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x * x.sigmoid_op()
    }
}

#[derive(Clone, Copy)]
pub struct SiluGrad;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for SiluGrad {
    const WGSL_TEMPLATE: &'static str =
        "(1.0 / (1.0 + exp(-({})))) * (1.0 + ({}) * (1.0 - (1.0 / (1.0 + exp(-({}))))))";
    #[inline(always)]
    fn apply(x: T) -> T {
        let sig = x.sigmoid_op();
        sig * (T::one() + x * (T::one() - sig))
    }
}

#[derive(Clone, Copy)]
pub struct Mish;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Mish {
    const WGSL_TEMPLATE: &'static str = "({}) * tanh(log(1.0 + exp(({}))))";
    #[inline(always)]
    fn apply(x: T) -> T {
        let sp = (T::one() + x.exp_op()).log_op();
        x * sp.tanh_op()
    }
}

#[derive(Clone, Copy)]
pub struct MishGrad;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for MishGrad {
    const WGSL_TEMPLATE: &'static str = "tanh(log(1.0 + exp(({})))) + ({}) * (1.0 - tanh(log(1.0 + exp(({})))) * tanh(log(1.0 + exp(({}))))) * (1.0 / (1.0 + exp(-({}))))";
    #[inline(always)]
    fn apply(x: T) -> T {
        let sp = (T::one() + x.exp_op()).log_op();
        let w = sp.tanh_op();
        let sig = x.sigmoid_op();
        w + x * (T::one() - w * w) * sig
    }
}
// ── Phase 7 Activation Tags ──

#[derive(Clone, Copy)]
pub struct Elu;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Elu {
    const WGSL_TEMPLATE: &'static str = "select(exp({}) - 1.0, {}, {} >= 0.0)";
    #[inline(always)]
    fn apply(x: T) -> T {
        if x >= T::zero() {
            x
        } else {
            x.exp_op() - T::one()
        }
    }
}

#[derive(Clone, Copy)]
pub struct EluGrad;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for EluGrad {
    const WGSL_TEMPLATE: &'static str = "select(exp({}), 1.0, {} >= 0.0)";
    #[inline(always)]
    fn apply(x: T) -> T {
        if x >= T::zero() {
            T::one()
        } else {
            x.exp_op()
        }
    }
}

#[derive(Clone, Copy)]
pub struct Softplus;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Softplus {
    const WGSL_TEMPLATE: &'static str = "log(1.0 + exp({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        (T::one() + x.exp_op()).log_op()
    }
}

#[derive(Clone, Copy)]
pub struct SoftplusGrad;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for SoftplusGrad {
    const WGSL_TEMPLATE: &'static str = "1.0 / (1.0 + exp(-{}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.sigmoid_op()
    }
}

#[derive(Clone, Copy)]
pub struct GeluTanh;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for GeluTanh {
    const WGSL_TEMPLATE: &'static str =
        "0.5 * {} * (1.0 + tanh(0.7978845608 * ({} + 0.044715 * {} * {} * {})))";
    #[inline(always)]
    fn apply(x: T) -> T {
        let c1 = T::from_f64(0.7978845608);
        let c2 = T::from_f64(0.044715);
        let half = T::from_f64(0.5);
        let one = T::one();
        let v = c1 * (x + c2 * x * x * x);
        half * x * (one + v.tanh_op())
    }
}

#[derive(Clone, Copy)]
pub struct GeluTanhGrad;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for GeluTanhGrad {
    const WGSL_TEMPLATE: &'static str =
        "0.5 * (1.0 + tanh(0.7978845608 * ({} + 0.044715 * {} * {} * {}))) + \
         0.5 * {} * (1.0 - tanh(0.7978845608 * ({} + 0.044715 * {} * {} * {})) * \
         tanh(0.7978845608 * ({} + 0.044715 * {} * {} * {}))) * \
         0.7978845608 * (1.0 + 0.134145 * {} * {})";
    #[inline(always)]
    fn apply(x: T) -> T {
        let c1 = T::from_f64(0.7978845608);
        let c2 = T::from_f64(0.044715);
        let c3 = T::from_f64(0.134145);
        let half = T::from_f64(0.5);
        let one = T::one();
        let v = c1 * (x + c2 * x * x * x);
        let t = v.tanh_op();
        let dt = c1 * (one + c3 * x * x);
        half * (one + t) + half * x * (one - t * t) * dt
    }
}

/// LeakyRelu tag — NOT a ZST; carries slope encoded as `f64::to_bits()`.
///
/// Cannot implement `UnaryOpTag` because `WGSL_TEMPLATE` must be `&'static str`
/// and the slope is a runtime value. Handled explicitly in fuse evaluator.
#[derive(Clone, Copy)]
pub struct LeakyReluTag {
    /// `f64::to_bits(slope)` — negative-region slope.
    pub slope_bits: u64,
}

impl LeakyReluTag {
    #[inline]
    pub fn new(slope: f64) -> Self {
        Self {
            slope_bits: f64::to_bits(slope),
        }
    }

    #[inline]
    pub fn slope(&self) -> f64 {
        f64::from_bits(self.slope_bits)
    }

    #[inline(always)]
    pub fn apply<T: Scalar>(&self, x: T) -> T {
        let slope = T::from_f64(self.slope());
        if x >= T::zero() {
            x
        } else {
            slope * x
        }
    }
}

/// LeakyRelu gradient tag — NOT a ZST; carries slope encoded as `f64::to_bits()`.
///
/// Same static-string constraint prevents `UnaryOpTag` implementation.
/// Handled explicitly in fuse evaluator.
#[derive(Clone, Copy)]
pub struct LeakyReluGradTag {
    /// `f64::to_bits(slope)` — negative-region slope.
    pub slope_bits: u64,
}

impl LeakyReluGradTag {
    #[inline]
    pub fn new(slope: f64) -> Self {
        Self {
            slope_bits: f64::to_bits(slope),
        }
    }

    #[inline]
    pub fn slope(&self) -> f64 {
        f64::from_bits(self.slope_bits)
    }

    #[inline(always)]
    pub fn apply<T: Scalar>(&self, x: T) -> T {
        let slope = T::from_f64(self.slope());
        if x >= T::zero() {
            T::one()
        } else {
            slope
        }
    }
}
