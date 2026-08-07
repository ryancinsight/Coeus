use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{BackendError, Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for Huber loss (PyTorch `F.huber_loss(reduction='mean')`).
///
/// The forward uses the **classical** Huber definition — which differs from
/// `smooth_l1_loss` by omitting the `1/δ` factor in the quadratic region
/// and rescaling the linear region by `δ`:
///
///   forward quadratic (`|z| < δ`): `0.5 * z²`
///   forward linear    (`|z| ≥ δ`): `δ * |z| - 0.5 * δ²`
///   backward quadratic: `z`
///   backward linear:   `sign(z) * δ`
pub struct HuberLossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Element-wise differences `pred[i] - target[i]`, stored on the selected
    /// provider for backward.
    pub diffs: Tensor<T, B>,
    /// Delta threshold separating quadratic from linear regions.
    pub delta: T,
    /// Number of elements in the loss reduction.
    pub n: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for HuberLossNode<T, B> {
    fn op_name(&self) -> &'static str {
        "huber_loss"
    }
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            let abs_diff =
                coeus_ops::elementwise_unary(&self.diffs, &backend, coeus_ops::UnaryOp::Abs)?;
            let delta = Tensor::full_on([1], self.delta, &backend);
            let boundary = coeus_ops::elementwise_binary(
                &delta,
                &abs_diff,
                &backend,
                coeus_ops::BinaryOp::Sub,
            )?;
            let quadratic_region =
                coeus_ops::elementwise_unary(&boundary, &backend, coeus_ops::UnaryOp::ReluGrad)?;
            let sign =
                coeus_ops::elementwise_unary(&self.diffs, &backend, coeus_ops::UnaryOp::Sign)?;
            let linear =
                coeus_ops::elementwise_binary(&sign, &delta, &backend, coeus_ops::BinaryOp::Mul)?;
            let element_gradient =
                coeus_ops::where_cond(&quadratic_region, &self.diffs, &linear, &backend)?;
            let g_out = coeus_ops::mean(grad_out, &backend)?;
            let scale = Tensor::full_on([1], g_out / T::from_usize(self.n), &backend);
            let grad_tensor = coeus_ops::elementwise_binary(
                &element_gradient,
                &scale,
                &backend,
                coeus_ops::BinaryOp::Mul,
            )?;
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend)?;
        }

        Ok(())
    }
}

/// Huber loss (PyTorch `F.huber_loss(reduction='mean', delta=...)`).
///
/// Forward per element `z = pred - target`:
///   - quadratic branch (`|z| < delta`): `0.5 * z * z`
///   - linear branch   (`|z| >= delta`): `delta * |z| - 0.5 * delta * delta`
///
/// Backward per element:
///   - quadratic: `z`
///   - linear:   `sign(z) * delta`
///
/// This matches the classical Huber definition; PyTorch's
/// `smooth_l1_loss` is the `0.5·z²/β`-form alternative.
///
/// # Errors
///
/// Returns the backend error type when the input shapes differ, the reduction
/// is empty, or `delta` is non-finite or non-positive.
pub fn huber_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    pred: &Var<T, B>,
    target: &Var<T, B>,
    delta: T,
) -> Result<Var<T, B>, B::Error> {
    let backend = B::default();
    if pred.tensor.shape() != target.tensor.shape() {
        return Err(B::Error::from(BackendError::ShapeMismatch {
            operation: "huber_loss",
            lhs: pred.tensor.shape().to_vec(),
            rhs: target.tensor.shape().to_vec(),
        }));
    }
    let n = pred.tensor.numel();
    if n == 0 {
        return Err(B::Error::from(BackendError::Storage {
            operation: "huber_loss",
            reason: "mean reduction requires at least one element".to_owned(),
        }));
    }
    if !Float::is_finite(delta) || delta <= T::zero() {
        return Err(B::Error::from(BackendError::Storage {
            operation: "huber_loss",
            reason: "delta must be finite and greater than zero".to_owned(),
        }));
    }
    let diffs = coeus_ops::elementwise_binary(
        &pred.tensor,
        &target.tensor,
        &backend,
        coeus_ops::BinaryOp::Sub,
    )?;
    let abs_diff = coeus_ops::elementwise_unary(&diffs, &backend, coeus_ops::UnaryOp::Abs)?;
    let delta_scalar = Tensor::full_on([1], delta, &backend);
    let boundary = coeus_ops::elementwise_binary(
        &delta_scalar,
        &abs_diff,
        &backend,
        coeus_ops::BinaryOp::Sub,
    )?;
    let quadratic_region =
        coeus_ops::elementwise_unary(&boundary, &backend, coeus_ops::UnaryOp::ReluGrad)?;
    let half = Tensor::full_on([1], T::from_f64(0.5), &backend);
    let squared =
        coeus_ops::elementwise_binary(&diffs, &diffs, &backend, coeus_ops::BinaryOp::Mul)?;
    let quadratic =
        coeus_ops::elementwise_binary(&squared, &half, &backend, coeus_ops::BinaryOp::Mul)?;
    let linear_scaled = coeus_ops::elementwise_binary(
        &abs_diff,
        &delta_scalar,
        &backend,
        coeus_ops::BinaryOp::Mul,
    )?;
    let delta_squared = coeus_ops::elementwise_binary(
        &delta_scalar,
        &delta_scalar,
        &backend,
        coeus_ops::BinaryOp::Mul,
    )?;
    let linear_offset =
        coeus_ops::elementwise_binary(&delta_squared, &half, &backend, coeus_ops::BinaryOp::Mul)?;
    let linear = coeus_ops::elementwise_binary(
        &linear_scaled,
        &linear_offset,
        &backend,
        coeus_ops::BinaryOp::Sub,
    )?;
    let elements = coeus_ops::where_cond(&quadratic_region, &quadratic, &linear, &backend)?;
    let out_tensor = coeus_ops::mean_axis(&elements.reshape([n]), 0, &backend)?;
    let requires_grad = crate::grad_mode::should_track_var(pred);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = HuberLossNode {
            output_grad,
            inputs: vec![pred.clone()],
            diffs,
            delta,
            n,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };
    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}
