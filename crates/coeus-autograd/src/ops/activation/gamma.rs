//! Gamma-log reverse mode, using the provider-owned real digamma capability.
use super::{unary_op, UnaryAutogradOp};
use crate::Var;
use coeus_core::Float;
use coeus_ops::{BackendOps, DigammaOps};
use coeus_tensor::Tensor;

/// Zero-sized tag for the differentiable natural log of absolute gamma.
pub struct LgammaOp;

impl<T: Float, B: BackendOps<T> + DigammaOps<T> + Default> UnaryAutogradOp<T, B> for LgammaOp {
    const OP_NAME: &'static str = "lgamma";

    #[inline]
    fn forward(input: &Tensor<T, B>, backend: &B) -> Result<Tensor<T, B>, B::Error> {
        coeus_ops::lgamma(input, backend)
    }

    #[inline]
    fn backward(
        grad_out: &Tensor<T, B>,
        input: &Tensor<T, B>,
        _output: &Tensor<T, B>,
        backend: &B,
    ) -> Result<Tensor<T, B>, B::Error> {
        let derivative = coeus_ops::digamma(input, backend)?;
        coeus_ops::mul(grad_out, &derivative, backend)
    }
}

/// Tracked natural logarithm of absolute gamma, with `grad_out * digamma(x)`.
///
/// The provider capability is explicit: CPU is supported; device backends must
/// implement `DigammaOps` before they can instantiate this operation.
#[inline]
pub fn lgamma<T: Float, B: BackendOps<T> + DigammaOps<T> + Default>(
    input: &Var<T, B>,
) -> Result<Var<T, B>, B::Error> {
    unary_op::<T, B, LgammaOp>(input)
}
