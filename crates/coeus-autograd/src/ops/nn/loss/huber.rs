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
///
/// Forward and backward stay on the selected provider: the node retains the
/// provider-resident difference tensor and the `|z| <= δ` mask rather than a
/// host `Vec<T>` payload.
pub struct HuberLossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Provider-resident element-wise differences `pred - target`.
    pub diffs: Tensor<T, B>,
    /// Provider-resident quadratic-region mask `|z| <= delta`.
    pub quad_mask: Tensor<T, B>,
    /// Delta threshold separating quadratic from linear regions.
    pub delta: T,
    /// Number of elements in the loss reduction.
    pub n: usize,
    /// Original tensor shape for gradient reconstruction.
    pub shape: coeus_core::Shape,
    /// Provider-resident mean scale `1 / element_count`.
    pub mean_scale: Tensor<T, B>,
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
        // d/dz: `z` in the quadratic region, `sign(z) * delta` in the linear
        // region, scaled by `grad_out / n`. Composed from provider ops only.
        let scale = coeus_ops::mul(grad_out, &self.mean_scale, &backend);
        let quad = coeus_ops::mul(&self.diffs, &scale, &backend);
        let delta_scale = coeus_ops::mul(
            &scale,
            &Tensor::full_on(scale.shape_cloned(), self.delta, &backend),
            &backend,
        );
        let linear = coeus_ops::mul(
            &coeus_ops::sign(&self.diffs, &backend),
            &delta_scale,
            &backend,
        );
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
/// The complete forward and backward computation stays on the selected
/// provider; no input-sized host staging occurs.
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
    let shape = pred.tensor.shape_cloned();

    // z = pred - target, |z|, and the quadratic-region mask, all on-provider.
    let diffs = coeus_ops::sub(&pred.tensor, &target.tensor, &backend);
    let abs_z = coeus_ops::abs(&diffs, &backend);
    let delta_tensor = Tensor::full_on([1], delta, &backend);
    let quad_mask = coeus_ops::le(&abs_z, &delta_tensor.broadcast(shape.clone()), &backend);
    // Classical Huber branch selection:
    //   quadratic: 0.5 * z²
    //   linear:    delta * |z| - 0.5 * delta²
    let half = T::from_f64(0.5);
    let half_sq = half * delta * delta;
    let quadratic = coeus_ops::mul(
        &coeus_ops::mul(&diffs, &diffs, &backend),
        &Tensor::full_on(shape.clone(), half, &backend),
        &backend,
    );
    let linear = coeus_ops::sub(
        &coeus_ops::mul(&abs_z, &delta_tensor.broadcast(shape.clone()), &backend),
        &Tensor::full_on(shape.clone(), half_sq, &backend),
        &backend,
    );
    let per_elem = coeus_ops::where_cond(&quad_mask, &quadratic, &linear, &backend)?;
    let loss = coeus_ops::mean_axis(&per_elem.reshape([n]), 0, &backend)
        .expect("invariant: validated non-empty Huber reduction has axis zero");

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
            inputs: vec![pred.clone(), target.clone()],
            diffs,
            quad_mask,
            delta,
            n,
            shape,
            mean_scale: Tensor::full_on([1], T::one() / T::from_f64(n as f64), &backend),
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };
    Ok(Var {
        tensor: loss,
        grad,
        creator,
    })
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
    fn huber_forward_matches_classical_reference() {
        // z = [1, -2, 3], delta = 1.5:
        //   |1| <= 1.5 → 0.5·1 = 0.5
        //   |−2| > 1.5 → 1.5·2 − 0.5·2.25 = 3 − 1.125 = 1.875
        //   |3| > 1.5  → 1.5·3 − 0.5·2.25 = 4.5 − 1.125 = 3.375
        //   mean = (0.5 + 1.875 + 3.375) / 3 = 5.75 / 3
        let pred = var_from(&[1.0, 2.0, 5.0]);
        let target = var_from(&[0.0, 4.0, 2.0]);
        let loss = huber_loss(&pred, &target, 1.5).expect("valid huber loss");
        assert_eq!(loss.tensor.shape(), &[1]);
        assert!((loss.tensor.as_slice()[0] - 5.75 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn huber_backward_matches_analytic_gradient() {
        // z = [1, -2, 3], delta = 1.5:
        //   grad = [1, sign(−2)·1.5 = −1.5, 1.5] / 3
        let pred = var_from(&[1.0, 2.0, 5.0]);
        let target = var_from(&[0.0, 4.0, 2.0]);
        let loss = huber_loss(&pred, &target, 1.5).expect("valid huber loss");
        loss.backward().expect("invariant: backward completes");
        let pred_grad = pred.grad().expect("pred must receive a gradient");
        let target_grad = target.grad().expect("target must receive a gradient");
        let expected = [1.0 / 3.0, -1.5 / 3.0, 1.5 / 3.0];
        for (i, (&g, &e)) in pred_grad.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-12,
                "huber pred grad[{i}]: got {g}, expected {e}"
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
                "huber target grad[{i}]: got {g}, expected {}",
                -e
            );
        }
    }

    #[test]
    fn huber_zero_difference_uses_quadratic_branch() {
        // z = 0 is in the quadratic region: gradient contribution 0.
        let pred = var_from(&[2.0, 0.0]);
        let target = var_from(&[0.0, 0.0]);
        let loss = huber_loss(&pred, &target, 1.0).expect("valid huber loss");
        loss.backward().expect("invariant: backward completes");
        let grad = pred.grad().expect("pred must receive a gradient");
        let expected = [1.0 / 2.0, 0.0];
        for (i, (&g, &e)) in grad.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-12,
                "huber zero-diff grad[{i}]: got {g}, expected {e}"
            );
        }
    }

    #[test]
    fn huber_rejects_shape_mismatch() {
        let pred = var_from(&[1.0, 2.0]);
        let target = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([3], &[1.0, 2.0, 3.0]),
            true,
        );
        assert!(huber_loss(&pred, &target, 1.0).is_err());
    }

    #[test]
    #[should_panic(expected = "mean reduction requires at least one element")]
    fn huber_rejects_empty_input() {
        let pred = var_from(&[]);
        let target = var_from(&[]);
        let _ = huber_loss(&pred, &target, 1.0).expect("valid huber loss");
    }

    #[test]
    fn huber_rejects_non_positive_delta() {
        let pred = var_from(&[1.0, 2.0]);
        let target = var_from(&[0.0, 1.0]);
        assert!(huber_loss(&pred, &target, 0.0).is_err());
        assert!(huber_loss(&pred, &target, f64::NAN).is_err());
    }
}
