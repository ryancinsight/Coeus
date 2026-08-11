//! Smooth L1 (Huber-β) loss autograd node.
//!
//! PyTorch contract (`SmoothL1Loss(reduction="mean", beta=1.0)`):
//!   loss(n) = 0.5 * (z_n)^2 / beta           if |z_n| < beta
//!           = |z_n| - 0.5 * beta              otherwise
//!
//! where `z = pred - target`. The gradient is the standard subgradient
//! matching PyTorch's `F.smooth_l1_loss`:
//!   d/dz = z / beta                          if |z| < beta
//!        = sign(z)                           otherwise
//!
//! At the kink (`|z| == beta`) Coeus takes the **right limit** to match
//! PyTorch's numerically stable reduction (the L1 piece, gradient `sign(z)`).
//! This avoids an implementation-defined subgradient at the boundary and
//! is empirically verified via parity tests against `torch.nn.functional.smooth_l1_loss`.
//!
//! Forward and backward stay on the selected provider; no input-sized host
//! staging occurs.

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for Smooth L1 loss (mean reduction).
pub struct SmoothL1LossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the scalar output.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Provider-resident `z = pred - target`.
    pub diffs: Tensor<T, B>,
    /// Provider-resident quadratic-region mask `|z| < beta`.
    pub quad_mask: Tensor<T, B>,
    /// Threshold β.
    pub beta: T,
    /// Number of elements in the mean reduction.
    pub n: usize,
    /// Original tensor shape for gradient reconstruction.
    pub shape: coeus_core::Shape,
    /// Provider-resident mean scale `1 / element_count`.
    pub mean_scale: Tensor<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for SmoothL1LossNode<T, B>
{
    fn op_name(&self) -> &'static str {
        "smooth_l1_loss"
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
        // d/d_pred: scale * (z / beta) if |z| < beta, else scale * sign(z).
        // At |z| == beta the mask is false → L1 piece with sign(z) (right
        // limit); at z == 0 sign(0) = 0 matches PyTorch's reduce-at-zero.
        let scale = coeus_ops::mul(grad_out, &self.mean_scale, &backend);
        let inv_beta_tensor = Tensor::full_on([1], T::one() / self.beta, &backend);
        let quad = coeus_ops::mul(
            &coeus_ops::mul(&self.diffs, &scale, &backend),
            &inv_beta_tensor,
            &backend,
        );
        let linear = coeus_ops::mul(&coeus_ops::sign(&self.diffs, &backend), &scale, &backend);
        let d_pred = coeus_ops::where_cond(&self.quad_mask, &quad, &linear, &backend)?;

        if let Some(Some(ref g)) = input_grads.first() {
            coeus_ops::add_assign(g.write(), &d_pred, &backend)?;
        }
        if let Some(Some(ref g)) = input_grads.get(1) {
            let d_target = coeus_ops::neg(&d_pred, &backend);
            coeus_ops::add_assign(g.write(), &d_target, &backend)?;
        }
        Ok(())
    }
}

/// Tracked Smooth L1 loss: `mean_i loss_smooth(pred[i] - target[i], beta)`.
/// `pred` and `target` must share shape. Mirrors PyTorch
/// `SmoothL1Loss(reduction="mean", beta=float)`.
///
/// The complete forward and backward computation stays on the selected
/// provider; no input-sized host staging occurs.
pub fn smooth_l1_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    pred: &Var<T, B>,
    target: &Var<T, B>,
    beta: T,
) -> Var<T, B> {
    let backend = B::default();
    assert_eq!(
        pred.tensor.shape(),
        target.tensor.shape(),
        "smooth_l1_loss requires pred and target to have identical shapes"
    );
    assert!(
        beta > T::zero(),
        "smooth_l1_loss requires beta > 0; got beta = 0 (would divide by zero)"
    );
    let n = pred.tensor.numel();
    assert!(n > 0, "smooth_l1_loss requires at least one element");
    let shape = pred.tensor.shape_cloned();

    // z = pred - target, |z|, and the quadratic-region mask |z| < beta.
    let diffs = coeus_ops::sub(&pred.tensor, &target.tensor, &backend);
    let abs_z = coeus_ops::abs(&diffs, &backend);
    let beta_tensor = Tensor::full_on([1], beta, &backend);
    let quad_mask = coeus_ops::lt(&abs_z, &beta_tensor.broadcast(shape.clone()), &backend);
    //   quadratic: 0.5 * z² / beta
    //   linear:    |z| - 0.5 * beta
    let half = T::from_f64(0.5);
    let inv_beta_tensor = Tensor::full_on([1], T::one() / beta, &backend);
    let quadratic = coeus_ops::mul(
        &coeus_ops::mul(&diffs, &diffs, &backend),
        &coeus_ops::mul(
            &Tensor::full_on(shape.clone(), half, &backend),
            &inv_beta_tensor,
            &backend,
        ),
        &backend,
    );
    let linear = coeus_ops::sub(
        &abs_z,
        &Tensor::full_on(shape.clone(), half * beta, &backend),
        &backend,
    );
    let per_elem = coeus_ops::where_cond(&quad_mask, &quadratic, &linear, &backend)
        .expect("smooth_l1_loss: provider where_cond dispatch");
    let loss = coeus_ops::mean_axis(&per_elem.reshape([n]), 0, &backend)
        .expect("invariant: validated non-empty Smooth L1 reduction has axis zero");

    let requires_grad =
        crate::grad_mode::should_track_var(pred) || crate::grad_mode::should_track_var(target);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = grad.as_ref().cloned().map(|output_grad| {
        let node = SmoothL1LossNode {
            output_grad,
            inputs: vec![pred.clone(), target.clone()],
            diffs,
            quad_mask,
            beta,
            n,
            shape,
            mean_scale: Tensor::full_on([1], T::one() / T::from_f64(n as f64), &backend),
        };
        Arc::new(node) as Arc<dyn BackwardNode<T, B>>
    });
    Var {
        tensor: loss,
        grad,
        creator,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::MoiraiBackend;

    fn var_from(data: &[f64]) -> Var<f64, MoiraiBackend> {
        Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([data.len()], data),
            true,
        )
    }

    #[test]
    fn smooth_l1_forward_matches_reference() {
        // z = [1, -2, 3], beta = 1.5:
        //   |1| < 1.5 → 0.5·1/1.5 = 1/3
        //   |−2| ≥ 1.5 → 2 − 0.75 = 1.25
        //   |3| ≥ 1.5 → 3 − 0.75 = 2.25
        //   mean = (1/3 + 1.25 + 2.25) / 3
        let pred = var_from(&[1.0, 2.0, 5.0]);
        let target = var_from(&[0.0, 4.0, 2.0]);
        let loss = smooth_l1_loss(&pred, &target, 1.5);
        let expected = (1.0 / 3.0 + 1.25 + 2.25) / 3.0;
        assert_eq!(loss.tensor.shape(), &[1]);
        assert!((loss.tensor.as_slice()[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn smooth_l1_backward_matches_analytic_gradient() {
        // z = [1, -2, 3], beta = 1.5:
        //   quad grad = z/beta = [1/1.5, -, -]; linear grad = sign(z).
        //   d_pred = [1/1.5, -1, 1] / 3
        let pred = var_from(&[1.0, 2.0, 5.0]);
        let target = var_from(&[0.0, 4.0, 2.0]);
        let loss = smooth_l1_loss(&pred, &target, 1.5);
        loss.backward().expect("invariant: backward completes");
        let pred_grad = pred.grad().expect("pred must receive a gradient");
        let target_grad = target.grad().expect("target must receive a gradient");
        let expected = [1.0 / 1.5 / 3.0, -1.0 / 3.0, 1.0 / 3.0];
        for (i, (&g, &e)) in pred_grad.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-12,
                "smooth_l1 pred grad[{i}]: got {g}, expected {e}"
            );
        }
        for (i, (&g, &e)) in target_grad
            .as_slice()
            .iter()
            .zip(expected.iter())
            .enumerate()
        {
            assert!(
                (g + e).abs() < 1e-12,
                "smooth_l1 target grad[{i}]: got {g}, expected {}",
                -e
            );
        }
    }

    #[test]
    fn smooth_l1_kink_takes_right_limit() {
        // z = [1.5, -1.5] with beta = 1.5: at the kink take the L1 piece
        // (gradient sign(z)), matching PyTorch's stable reduction.
        let pred = var_from(&[1.5, 0.0]);
        let target = var_from(&[0.0, 1.5]);
        let loss = smooth_l1_loss(&pred, &target, 1.5);
        loss.backward().expect("invariant: backward completes");
        let grad = pred.grad().expect("pred must receive a gradient");
        let expected = [1.0 / 2.0, -1.0 / 2.0];
        for (i, (&g, &e)) in grad.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-12,
                "smooth_l1 kink grad[{i}]: got {g}, expected {e}"
            );
        }
    }
}
