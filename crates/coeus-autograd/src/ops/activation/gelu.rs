use super::UnaryAutogradOp;
use super::unary_op;
use crate::var::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;

/// ZST tag for GELU autograd.
pub struct GeluOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for GeluOp {
    const OP_NAME: &'static str = "gelu";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Result<Tensor<T, B>, B::Error> {
        coeus_ops::gelu(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Result<Tensor<T, B>, B::Error> {
        let deriv = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::GeluGrad)?;
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

/// ZST tag for GELU tanh approximation autograd.
pub struct GeluTanhOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for GeluTanhOp {
    const OP_NAME: &'static str = "gelu_tanh";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Result<Tensor<T, B>, B::Error> {
        coeus_ops::gelu_tanh(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Result<Tensor<T, B>, B::Error> {
        let deriv = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::GeluTanhGrad)?;
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

/// Tracked GELU activation.
#[must_use]
#[inline]
pub fn gelu<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
) -> Result<Var<T, B>, B::Error> {
    unary_op::<T, B, GeluOp>(a)
}

/// Tracked GELU tanh approximation.
#[must_use]
#[inline]
pub fn gelu_tanh<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
) -> Result<Var<T, B>, B::Error> {
    unary_op::<T, B, GeluTanhOp>(a)
}
