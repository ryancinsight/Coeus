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
    fn wgsl_expr(child: &str) -> String {
        Self::WGSL_TEMPLATE.replace("{}", child)
    }
    fn apply(x: T) -> T;
}

/// WGSL expression for an Abramowitz-Stegun erf approximation.
///
/// The maximum absolute error is below the existing backend parity tolerance
/// used by WGPU tests, and WGSL has no native `erf` intrinsic.
#[must_use]
pub fn wgsl_erf_approx_expr(arg: &str) -> String {
    let z = format!("({arg})");
    let t = format!("(1.0 / (1.0 + 0.3275911 * abs({z})))");
    format!(
        "(sign({z}) * (1.0 - (((((1.061405429 * {t} - 1.453152027) * {t} + 1.421413741) * {t} - 0.284496736) * {t} + 0.254829592) * {t}) * exp(-({z}) * ({z}))))"
    )
}

/// WGSL expression for exact-contract GELU using `erf(x / sqrt(2))`.
#[must_use]
pub fn wgsl_gelu_expr(arg: &str) -> String {
    let x = format!("({arg})");
    let erf = wgsl_erf_approx_expr(&format!("{x} * 0.7071067811865476"));
    format!("(0.5 * {x} * (1.0 + {erf}))")
}

/// WGSL expression for the exact-contract GELU derivative.
#[must_use]
pub fn wgsl_gelu_grad_expr(arg: &str) -> String {
    let x = format!("({arg})");
    let erf = wgsl_erf_approx_expr(&format!("{x} * 0.7071067811865476"));
    format!("(0.5 * (1.0 + {erf}) + {x} * exp(-0.5 * {x} * {x}) * 0.3989422804014327)")
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
    fn wgsl_expr(child: &str) -> String {
        wgsl_gelu_expr(child)
    }
    #[inline(always)]
    fn apply(x: T) -> T {
        x.gelu_op()
    }
}

#[derive(Clone, Copy)]
pub struct GeluGrad;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for GeluGrad {
    const WGSL_TEMPLATE: &'static str = "0.5 * (1.0 + tanh(0.7978845608 * (({}) + 0.044715 * ({}) * ({}) * ({})))) + 0.5 * ({}) * (1.0 - tanh(0.7978845608 * (({}) + 0.044715 * ({}) * ({}) * ({}))) * tanh(0.7978845608 * (({}) + 0.044715 * ({}) * ({}) * ({})))) * 0.7978845608 * (1.0 + 0.134145 * ({}) * ({}))";
    fn wgsl_expr(child: &str) -> String {
        wgsl_gelu_grad_expr(child)
    }
    #[inline(always)]
    fn apply(x: T) -> T {
        let half = T::from_f64(0.5);
        let one = T::one();
        let inv_sqrt_two = T::from_f64(core::f64::consts::FRAC_1_SQRT_2);
        let inv_sqrt_two_pi = T::from_f64(0.3989422804014327);
        let x2 = x * x;
        half * (one + (x * inv_sqrt_two).erf_op())
            + x * ((T::zero() - half * x2).exp_op()) * inv_sqrt_two_pi
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

// ── Recip ───────────────────────────────────────────────────────────────────

/// Element-wise reciprocal: 1/x.
#[derive(Clone, Copy)]
pub struct Recip;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Recip {
    const WGSL_TEMPLATE: &'static str = "(1.0 / ({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        T::one() / x
    }
}

// ── Sign ────────────────────────────────────────────────────────────────────

/// Element-wise signum: -1, 0, or 1.
#[derive(Clone, Copy)]
pub struct Sign;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Sign {
    const WGSL_TEMPLATE: &'static str = "sign(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        if x > T::zero() {
            T::one()
        } else if x < T::zero() {
            T::zero() - T::one()
        } else {
            T::zero()
        }
    }
}

// ── Floor ───────────────────────────────────────────────────────────────────

/// Element-wise floor.
#[derive(Clone, Copy)]
pub struct Floor;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Floor {
    const WGSL_TEMPLATE: &'static str = "floor(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        T::from_f64(T::to_f64(x).floor())
    }
}

// ── Ceil ────────────────────────────────────────────────────────────────────

/// Element-wise ceil.
#[derive(Clone, Copy)]
pub struct Ceil;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Ceil {
    const WGSL_TEMPLATE: &'static str = "ceil(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        T::from_f64(T::to_f64(x).ceil())
    }
}

// ── Round ───────────────────────────────────────────────────────────────────

/// Element-wise round to nearest integer.
#[derive(Clone, Copy)]
pub struct Round;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Round {
    const WGSL_TEMPLATE: &'static str = "round(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        T::from_f64(T::to_f64(x).round())
    }
}

// ── Trunc ───────────────────────────────────────────────────────────────────

/// Element-wise truncation toward zero.
#[derive(Clone, Copy)]
pub struct Trunc;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Trunc {
    const WGSL_TEMPLATE: &'static str = "trunc(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        T::from_f64(T::to_f64(x).trunc())
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
