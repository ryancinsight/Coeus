use super::UnaryOpTag;
use crate::fuse::op_tags::wgsl_erf_approx_expr;
use coeus_core::{FloatOps, Scalar};

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
/// `exp(x) - 1` operation tag.
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
/// `ln(1 + x)` operation tag.
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
