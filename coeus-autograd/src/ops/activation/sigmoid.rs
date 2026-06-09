use super::unary_op;
use super::UnaryAutogradOp;
use crate::var::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;

/// ZST tag for Sigmoid autograd.
pub struct SigmoidOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for SigmoidOp {
    const OP_NAME: &'static str = "sigmoid";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::sigmoid(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        _x: &Tensor<T, B>,
        y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let deriv = coeus_ops::elementwise_unary(y, backend, coeus_ops::UnaryOp::SigmoidGrad);
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

/// Tracked Sigmoid activation.
#[must_use]
#[inline]
pub fn sigmoid<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, SigmoidOp>(a)
}
