use super::unary_op;
use super::UnaryAutogradOp;
use crate::var::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;

/// ZST tag for Tanh autograd.
pub struct TanhOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for TanhOp {
    const OP_NAME: &'static str = "tanh";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::tanh(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        _x: &Tensor<T, B>,
        y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let deriv = coeus_ops::elementwise_unary(y, backend, coeus_ops::UnaryOp::TanhGrad).expect("elementwise_unary");
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

/// Tracked Tanh activation.
#[must_use]
#[inline]
pub fn tanh<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, TanhOp>(a)
}
