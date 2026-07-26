mod activation;
mod elementary;
mod transcendental;

use coeus_core::Scalar;

pub use activation::{
    Elu, EluGrad, Gelu, GeluGrad, GeluTanh, GeluTanhGrad, Mish, MishGrad, Silu, SiluGrad, Softplus,
    SoftplusGrad,
};
pub use elementary::{Abs, Ceil, Floor, Neg, Recip, Relu, Round, Sign, Sqrt, Trunc};
pub use transcendental::{
    Acos, Acosh, Asin, Asinh, Atan, Atanh, Cos, Cosh, Erf, Erfc, Exp, Exp2, Expm1, Log, Log10,
    Log1p, Log2, Sigmoid, Sin, Sinh, Tan, Tanh,
};

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
