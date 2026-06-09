use super::unary_op;
use super::UnaryAutogradOp;
use crate::var::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;

/// ZST tag for Exponential autograd.
pub struct ExpOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for ExpOp {
    const OP_NAME: &'static str = "exp";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::exp(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        _x: &Tensor<T, B>,
        y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        coeus_ops::mul(grad_out, y, backend)
    }
}

/// ZST tag for Natural Logarithm autograd.
pub struct LogOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for LogOp {
    const OP_NAME: &'static str = "log";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::log(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        coeus_ops::div(grad_out, x, backend)
    }
}

/// Tracked Exponential function.
#[must_use]
#[inline]
pub fn exp<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, ExpOp>(a)
}

/// Tracked Natural Logarithm.
#[must_use]
#[inline]
pub fn log<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, LogOp>(a)
}
