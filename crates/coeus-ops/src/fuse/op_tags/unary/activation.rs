use super::UnaryOpTag;
use crate::fuse::op_tags::{wgsl_gelu_expr, wgsl_gelu_grad_expr};
use coeus_core::{FloatOps, NumericElement, Scalar};

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
        let half = <T as coeus_core::Scalar>::from_f64(0.5);
        let one = <T as NumericElement>::ONE;
        let inv_sqrt_two = <T as coeus_core::Scalar>::from_f64(core::f64::consts::FRAC_1_SQRT_2);
        let inv_sqrt_two_pi = <T as coeus_core::Scalar>::from_f64(0.3989422804014327);
        let x2 = x * x;
        half * (one + (x * inv_sqrt_two).erf_op())
            + x * ((<T as NumericElement>::ZERO - half * x2).exp_op()) * inv_sqrt_two_pi
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
        sig * (<T as NumericElement>::ONE + x * (<T as NumericElement>::ONE - sig))
    }
}

#[derive(Clone, Copy)]
/// Mish operation tag.
pub struct Mish;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Mish {
    const WGSL_TEMPLATE: &'static str = "({}) * tanh(log(1.0 + exp(({}))))";
    #[inline(always)]
    fn apply(x: T) -> T {
        let sp = (<T as NumericElement>::ONE + x.exp_op()).log_op();
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
        let sp = (<T as NumericElement>::ONE + x.exp_op()).log_op();
        let w = sp.tanh_op();
        let sig = x.sigmoid_op();
        w + x * (<T as NumericElement>::ONE - w * w) * sig
    }
}

#[derive(Clone, Copy)]
/// ELU operation tag.
pub struct Elu;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Elu {
    const WGSL_TEMPLATE: &'static str = "select(exp({}) - 1.0, {}, {} >= 0.0)";
    #[inline(always)]
    fn apply(x: T) -> T {
        if x >= <T as NumericElement>::ZERO {
            x
        } else {
            x.exp_op() - <T as NumericElement>::ONE
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
        if x >= <T as NumericElement>::ZERO {
            <T as NumericElement>::ONE
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
        (<T as NumericElement>::ONE + x.exp_op()).log_op()
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
        let c1 = <T as coeus_core::Scalar>::from_f64(0.7978845608);
        let c2 = <T as coeus_core::Scalar>::from_f64(0.044715);
        let half = <T as coeus_core::Scalar>::from_f64(0.5);
        let one = <T as NumericElement>::ONE;
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
        let c1 = <T as coeus_core::Scalar>::from_f64(0.7978845608);
        let c2 = <T as coeus_core::Scalar>::from_f64(0.044715);
        let c3 = <T as coeus_core::Scalar>::from_f64(0.134145);
        let half = <T as coeus_core::Scalar>::from_f64(0.5);
        let one = <T as NumericElement>::ONE;
        let v = c1 * (x + c2 * x * x * x);
        let t = v.tanh_op();
        let dt = c1 * (one + c3 * x * x);
        half * (one + t) + half * x * (one - t * t) * dt
    }
}
