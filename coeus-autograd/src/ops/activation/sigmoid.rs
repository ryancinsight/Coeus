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
        let deriv = coeus_ops::elementwise_unary(y, backend, coeus_ops::UnaryOp::SigmoidGrad).expect("elementwise_unary");
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

/// Tracked Sigmoid activation.
///
/// # Examples
///
/// `σ(x) = 1 / (1 + e^{-x})` with `σ'(x) = σ(x)(1 - σ(x))`. At `x = 0`,
/// `σ(0) = 0.5` and `σ'(0) = 0.25`, so the scalar-sum gradient is `0.25`.
///
/// ```
/// use coeus_autograd::Var;
/// use coeus_core::MoiraiBackend;
/// use coeus_tensor::Tensor;
///
/// let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([2], &[0.0, 0.0]), true);
/// let y = coeus_autograd::sigmoid(&x);
/// assert!((y.tensor.as_slice()[0] - 0.5).abs() < 1e-5);
/// let loss = coeus_autograd::sum(&y);
/// loss.backward();
/// let grad = x.grad().unwrap();
/// assert!((grad.as_slice()[0] - 0.25).abs() < 1e-5); // 0.5 * (1 - 0.5)
/// assert!((grad.as_slice()[1] - 0.25).abs() < 1e-5);
/// ```
#[must_use]
#[inline]
pub fn sigmoid<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, SigmoidOp>(a)
}
