use super::unary_op;
use super::UnaryAutogradOp;
use crate::var::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;

unary_autograd!(
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
    /// let x = Var::<f32, MoiraiBackend>::new(
    ///     Tensor::from_slice([2], &[0.0, 0.0]).expect("invariant: example shape matches data"),
    ///     true,
    /// ).expect("invariant: example gradient buffer allocation succeeds");
    /// let y = coeus_autograd::sigmoid(&x).expect("invariant: example activation succeeds");
    /// assert!((y.tensor.as_slice()[0] - 0.5).abs() < 1e-5);
    /// let loss = coeus_autograd::sum(&y).expect("invariant: example reduction succeeds");
    /// loss.backward().expect("invariant: valid autograd fixture completes backward");
    /// let grad = x.grad().expect("invariant: backward populates the tracked leaf gradient");
    /// assert!((grad.as_slice()[0] - 0.25).abs() < 1e-5); // 0.5 * (1 - 0.5)
    /// assert!((grad.as_slice()[1] - 0.25).abs() < 1e-5);
    /// ```
    SigmoidOp, "sigmoid", sigmoid, |g, _x, y, b| {
        super::backward_via_unary_derivative(g, y, b, coeus_ops::UnaryOp::SigmoidGrad)
    }
);
