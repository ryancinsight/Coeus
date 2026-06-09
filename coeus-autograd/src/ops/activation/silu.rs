use super::unary_op;
use super::UnaryAutogradOp;
use crate::var::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;

/// ZST tag for SiLU autograd.
pub struct SiluOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for SiluOp {
    const OP_NAME: &'static str = "silu";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::silu(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let mask = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::SiluGrad);
        coeus_ops::mul(grad_out, &mask, backend)
    }
}

/// ZST tag for Mish autograd.
pub struct MishOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for MishOp {
    const OP_NAME: &'static str = "mish";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::mish(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let mask = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::MishGrad);
        coeus_ops::mul(grad_out, &mask, backend)
    }
}

/// ZST tag for Softplus autograd.
pub struct SoftplusOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for SoftplusOp {
    const OP_NAME: &'static str = "softplus";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::softplus(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        // SoftplusGrad = sigmoid(x)
        let deriv = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::SoftplusGrad);
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

/// Tracked SiLU activation.
#[must_use]
#[inline]
pub fn silu<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, SiluOp>(a)
}

/// Tracked Mish activation.
#[must_use]
#[inline]
pub fn mish<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, MishOp>(a)
}

/// Tracked Softplus activation.
#[must_use]
#[inline]
pub fn softplus<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, SoftplusOp>(a)
}
