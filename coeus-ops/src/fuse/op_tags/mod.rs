mod binary;
mod leaky_relu;
mod wgsl;

pub use binary::{Add, BinaryOpTag, Div, Mul, Sub};
pub use leaky_relu::{LeakyReluGradTag, LeakyReluTag};
pub use wgsl::{wgsl_erf_approx_expr, wgsl_gelu_expr, wgsl_gelu_grad_expr};

use coeus_core::FloatOps;
use coeus_core::Scalar;

// ── Unary Operations ZST Tags ──

/// Tag trait for unary operations in the fused expression DAG.
pub trait UnaryOpTag<T: Scalar>: 'static + Send + Sync + Copy + Clone {
    /// WGSL template string with `{}` as the child expression placeholder.
    const WGSL_TEMPLATE: &'static str;
    /// Render the WGSL expression for this operation applied to `child`.
    fn wgsl_expr(child: &str) -> String {
        Self::WGSL_TEMPLATE.replace("{}", child)
    }
    /// Apply the unary operation to a scalar value.
    fn apply(x: T) -> T;
}

#[derive(Clone, Copy)]
/// ReLU operation tag.
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
/// Sigmoid operation tag.
pub struct Sigmoid;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Sigmoid {
    const WGSL_TEMPLATE: &'static str = "1.0 / (1.0 + exp(-({})))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.sigmoid_op()
    }
}

#[derive(Clone, Copy)]
/// Tanh operation tag.
pub struct Tanh;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Tanh {
    const WGSL_TEMPLATE: &'static str = "tanh(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.tanh_op()
    }
}

#[derive(Clone, Copy)]
/// Exact GELU operation tag.
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
/// Exact GELU gradient operation tag.
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
/// Sine operation tag.
pub struct Sin;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Sin {
    const WGSL_TEMPLATE: &'static str = "sin(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.sin_op()
    }
}

#[derive(Clone, Copy)]
/// Cosine operation tag.
pub struct Cos;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Cos {
    const WGSL_TEMPLATE: &'static str = "cos(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.cos_op()
    }
}

#[derive(Clone, Copy)]
/// Exponential operation tag.
pub struct Exp;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Exp {
    const WGSL_TEMPLATE: &'static str = "exp(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.exp_op()
    }
}

#[derive(Clone, Copy)]
/// Gauss error function operation tag.
pub struct Erf;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Erf {
    const WGSL_TEMPLATE: &'static str = "erf(({}))";
    fn wgsl_expr(child: &str) -> String {
        wgsl_erf_approx_expr(child)
    }
    #[inline(always)]
    fn apply(x: T) -> T {
        x.erf_op()
    }
}

#[derive(Clone, Copy)]
/// Complementary error function tag.
pub struct Erfc;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Erfc {
    const WGSL_TEMPLATE: &'static str = "erfc({})";
    fn wgsl_expr(child: &str) -> String {
        // erfc = 1 - erf(x); approximate via the existing erf polynomial
        format!("(1.0 - ({}))", wgsl_erf_approx_expr(child))
    }
    #[inline(always)]
    fn apply(x: T) -> T {
        x.erfc_op()
    }
}

#[derive(Clone, Copy)]
/// Tangent operation tag.
pub struct Tan;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Tan {
    const WGSL_TEMPLATE: &'static str = "tan({})";
    fn wgsl_expr(child: &str) -> String {
        format!("tan({})", child)
    }
    #[inline(always)]
    fn apply(x: T) -> T {
        x.tan_op()
    }
}

#[derive(Clone, Copy)]
/// Arc-sine operation tag.
pub struct Asin;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Asin {
    const WGSL_TEMPLATE: &'static str = "asin({})";
    fn wgsl_expr(child: &str) -> String {
        format!("asin({})", child)
    }
    #[inline(always)]
    fn apply(x: T) -> T {
        x.asin_op()
    }
}

#[derive(Clone, Copy)]
/// Arc-cosine operation tag.
pub struct Acos;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Acos {
    const WGSL_TEMPLATE: &'static str = "acos({})";
    fn wgsl_expr(child: &str) -> String {
        format!("acos({})", child)
    }
    #[inline(always)]
    fn apply(x: T) -> T {
        x.acos_op()
    }
}

#[derive(Clone, Copy)]
/// Arc-tangent operation tag.
pub struct Atan;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Atan {
    const WGSL_TEMPLATE: &'static str = "atan({})";
    fn wgsl_expr(child: &str) -> String {
        format!("atan({})", child)
    }
    #[inline(always)]
    fn apply(x: T) -> T {
        x.atan_op()
    }
}

#[derive(Clone, Copy)]
/// Hyperbolic sine operation tag.
pub struct Sinh;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Sinh {
    const WGSL_TEMPLATE: &'static str = "sinh({})";
    fn wgsl_expr(child: &str) -> String {
        format!("sinh({})", child)
    }
    #[inline(always)]
    fn apply(x: T) -> T {
        x.sinh_op()
    }
}
#[derive(Clone, Copy)]
/// Hyperbolic cosine operation tag.
pub struct Cosh;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Cosh {
    const WGSL_TEMPLATE: &'static str = "cosh({})";
    fn wgsl_expr(child: &str) -> String {
        format!("cosh({})", child)
    }
    #[inline(always)]
    fn apply(x: T) -> T {
        x.cosh_op()
    }
}
#[derive(Clone, Copy)]
/// Base-2 logarithm operation tag.
pub struct Log2;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Log2 {
    const WGSL_TEMPLATE: &'static str = "log2({})";
    fn wgsl_expr(child: &str) -> String {
        format!("log2({})", child)
    }
    #[inline(always)]
    fn apply(x: T) -> T {
        x.log2_op()
    }
}
#[derive(Clone, Copy)]
/// Base-10 logarithm operation tag.
pub struct Log10;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Log10 {
    const WGSL_TEMPLATE: &'static str = "log({})*0.43429448190325182";
    fn wgsl_expr(child: &str) -> String {
        format!("(log({}) * 0.43429448190325182)", child)
    }
    #[inline(always)]
    fn apply(x: T) -> T {
        x.log10_op()
    }
}

#[derive(Clone, Copy)]
/// Base-2 exponential operation tag.
pub struct Exp2;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Exp2 {
    const WGSL_TEMPLATE: &'static str = "exp2({})";
    fn wgsl_expr(child: &str) -> String {
        format!("exp2({})", child)
    }
    #[inline(always)]
    fn apply(x: T) -> T {
        x.exp2_op()
    }
}

#[derive(Clone, Copy)]
/// Inverse hyperbolic tangent operation tag.
pub struct Atanh;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Atanh {
    const WGSL_TEMPLATE: &'static str = "atanh({})";
    fn wgsl_expr(child: &str) -> String {
        format!("atanh({})", child)
    }
    #[inline(always)]
    fn apply(x: T) -> T {
        x.atanh_op()
    }
}

#[derive(Clone, Copy)]
/// Inverse hyperbolic sine operation tag.
pub struct Asinh;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Asinh {
    const WGSL_TEMPLATE: &'static str = "asinh({})";
    fn wgsl_expr(child: &str) -> String {
        format!("asinh({})", child)
    }
    #[inline(always)]
    fn apply(x: T) -> T {
        x.asinh_op()
    }
}

#[derive(Clone, Copy)]
/// Inverse hyperbolic cosine operation tag.
pub struct Acosh;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Acosh {
    const WGSL_TEMPLATE: &'static str = "acosh({})";
    fn wgsl_expr(child: &str) -> String {
        format!("acosh({})", child)
    }
    #[inline(always)]
    fn apply(x: T) -> T {
        x.acosh_op()
    }
}

#[derive(Clone, Copy)]
/// exp(x) - 1 operation tag.
pub struct Expm1;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Expm1 {
    const WGSL_TEMPLATE: &'static str = "(exp({}) - 1.0)";
    fn wgsl_expr(child: &str) -> String {
        format!("(exp({}) - 1.0)", child)
    }
    #[inline(always)]
    fn apply(x: T) -> T {
        x.expm1_op()
    }
}

#[derive(Clone, Copy)]
/// ln(1 + x) operation tag.
pub struct Log1p;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Log1p {
    const WGSL_TEMPLATE: &'static str = "log(1.0 + ({}))";
    fn wgsl_expr(child: &str) -> String {
        format!("log(1.0 + ({}))", child)
    }
    #[inline(always)]
    fn apply(x: T) -> T {
        x.log1p_op()
    }
}

#[derive(Clone, Copy)]
/// Natural log operation tag.
pub struct Log;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Log {
    const WGSL_TEMPLATE: &'static str = "log(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.log_op()
    }
}

#[derive(Clone, Copy)]
/// Negation operation tag.
pub struct Neg;
impl<T: Scalar> UnaryOpTag<T> for Neg {
    const WGSL_TEMPLATE: &'static str = "-(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        T::zero() - x
    }
}

#[derive(Clone, Copy)]
/// Absolute value operation tag.
pub struct Abs;
impl<T: Scalar> UnaryOpTag<T> for Abs {
    const WGSL_TEMPLATE: &'static str = "abs(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.abs_val()
    }
}

#[derive(Clone, Copy)]
/// Square root operation tag.
pub struct Sqrt;
impl<T: Scalar> UnaryOpTag<T> for Sqrt {
    const WGSL_TEMPLATE: &'static str = "sqrt(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.sqrt_val()
    }
}

#[derive(Clone, Copy)]
/// SiLU operation tag.
pub struct Silu;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Silu {
    const WGSL_TEMPLATE: &'static str = "({}) / (1.0 + exp(-({})))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x * x.sigmoid_op()
    }
}

#[derive(Clone, Copy)]
/// SiLU gradient operation tag.
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
/// Mish operation tag.
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
/// Mish gradient operation tag.
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
/// ELU operation tag.
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
/// ELU gradient operation tag.
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
/// Softplus operation tag.
pub struct Softplus;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Softplus {
    const WGSL_TEMPLATE: &'static str = "log(1.0 + exp({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        (T::one() + x.exp_op()).log_op()
    }
}

#[derive(Clone, Copy)]
/// Softplus gradient operation tag.
pub struct SoftplusGrad;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for SoftplusGrad {
    const WGSL_TEMPLATE: &'static str = "1.0 / (1.0 + exp(-{}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.sigmoid_op()
    }
}

#[derive(Clone, Copy)]
/// Tanh-approximation GELU operation tag.
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
/// Tanh-approximation GELU gradient operation tag.
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
        <T as Scalar>::from_f64(<T as Scalar>::to_f64(x).floor())
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
        <T as Scalar>::from_f64(<T as Scalar>::to_f64(x).ceil())
    }
}

// ── Round ───────────────────────────────────────────────────────────────────

/// Element-wise round to nearest integer, ties to even (banker's rounding).
///
/// Matches `torch.round` / IEEE-754 roundTiesToEven; WGSL's `round()` builtin
/// has the same ties-to-even contract.
#[derive(Clone, Copy)]
pub struct Round;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Round {
    const WGSL_TEMPLATE: &'static str = "round(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        <T as Scalar>::from_f64(<T as Scalar>::to_f64(x).round_ties_even())
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
        <T as Scalar>::from_f64(<T as Scalar>::to_f64(x).trunc())
    }
}
