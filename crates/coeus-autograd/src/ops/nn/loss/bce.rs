use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for binary cross-entropy loss.
///
/// Forward and backward stay on the selected provider: the node retains the
/// provider-resident clamped probabilities and target tensor rather than host
/// `Vec<T>` payloads.
pub struct BinaryCrossEntropyNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Provider-resident clamped prediction values in `[eps, 1-eps]`.
    pub probs: Tensor<T, B>,
    /// Provider-resident target values (0.0 or 1.0).
    pub targets: Tensor<T, B>,
    /// Number of elements in the loss reduction.
    pub n: usize,
    /// Provider-resident mean scale `1 / element_count`.
    pub mean_scale: Tensor<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for BinaryCrossEntropyNode<T, B>
{
    fn op_name(&self) -> &'static str {
        "binary_cross_entropy"
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
            // d/dp = -(t/p - (1-t)/(1-p)) / n, all on-provider.
            let scale = coeus_ops::mul(grad_out, &self.mean_scale, &backend);
            let ones = Tensor::full_on([1], T::one(), &backend);
            let inv_p = coeus_ops::div(&self.targets, &self.probs, &backend);
            let one_minus_t = coeus_ops::sub(&ones, &self.targets, &backend);
            let one_minus_p = coeus_ops::sub(&ones, &self.probs, &backend);
            let inv_one_minus_p = coeus_ops::div(&one_minus_t, &one_minus_p, &backend);
            let diff = coeus_ops::sub(&inv_p, &inv_one_minus_p, &backend);
            let d_pred = coeus_ops::mul(&coeus_ops::neg(&diff, &backend), &scale, &backend);
            coeus_ops::add_assign(g.write(), &d_pred, &backend)?;
        }
        Ok(())
    }
}

/// Tracked Binary Cross-Entropy Loss.
/// `pred`: `[N]` probabilities (clamped to `[eps, 1-eps]`), `target`: `[N]`
/// float targets (0.0 or 1.0), `eps`: numerical stability clamp (e.g.
/// `T::from_f64(1e-7)`). The complete forward and backward computation stays
/// on the selected provider; no input-sized host staging occurs.
pub fn binary_cross_entropy<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    pred: &Var<T, B>,
    target: &Var<T, B>,
    eps: T,
) -> Var<T, B> {
    let backend = B::default();
    let shape = pred.tensor.shape_cloned();
    let n = pred.tensor.numel();
    assert!(n > 0, "binary_cross_entropy requires at least one element");

    // Clamp pred to [eps, 1-eps] with two where_cond selects.
    let eps_tensor = Tensor::full_on(shape.clone(), eps, &backend);
    let one_minus_eps_tensor = Tensor::full_on(shape.clone(), T::one() - eps, &backend);
    let below = coeus_ops::lt(&pred.tensor, &eps_tensor, &backend);
    let clamped_low = coeus_ops::where_cond(&below, &eps_tensor, &pred.tensor, &backend)
        .expect("binary_cross_entropy: low clamp");
    let above = coeus_ops::gt(&clamped_low, &one_minus_eps_tensor, &backend);
    let probs = coeus_ops::where_cond(&above, &one_minus_eps_tensor, &clamped_low, &backend)
        .expect("binary_cross_entropy: high clamp");

    // loss = -(t * log(p) + (1-t) * log(1-p)), all on-provider.
    let ones = Tensor::full_on(shape.clone(), T::one(), &backend);
    let log_p = coeus_ops::log(&probs, &backend);
    let log_1mp = coeus_ops::log(&coeus_ops::sub(&ones, &probs, &backend), &backend);
    let term = coeus_ops::add(
        &coeus_ops::mul(&target.tensor, &log_p, &backend),
        &coeus_ops::mul(
            &coeus_ops::sub(&ones, &target.tensor, &backend),
            &log_1mp,
            &backend,
        ),
        &backend,
    );
    let neg_term = coeus_ops::neg(&term, &backend);
    let loss = coeus_ops::mean_axis(&neg_term.reshape([n]), 0, &backend)
        .expect("invariant: validated non-empty BCE reduction has axis zero");

    let requires_grad = crate::grad_mode::should_track_var(pred);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = grad.as_ref().cloned().map(|output_grad| {
        let node = BinaryCrossEntropyNode {
            output_grad,
            inputs: vec![pred.clone()],
            probs,
            targets: target.tensor.clone(),
            n,
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
    fn bce_forward_matches_reference() {
        // pred = [0.8, 0.2], target = [1, 0], eps = 1e-7:
        //   loss = mean(-(1·log(0.8) + 0), -(0 + 1·log(0.8))) = -log(0.8).
        let pred = var_from(&[0.8, 0.2]);
        let target = var_from(&[1.0, 0.0]);
        let loss = binary_cross_entropy(&pred, &target, 1e-7);
        assert_eq!(loss.tensor.shape(), &[1]);
        assert!((loss.tensor.as_slice()[0] - (-0.8_f64.ln())).abs() < 1e-12);
    }

    #[test]
    fn bce_forward_clamps_predictions() {
        // pred = [1.5, -0.5] clamp to [1-eps, eps]; loss finite.
        let pred = var_from(&[1.5, -0.5]);
        let target = var_from(&[1.0, 0.0]);
        let loss = binary_cross_entropy(&pred, &target, 1e-7);
        let v = loss.tensor.as_slice()[0];
        assert!(v.is_finite(), "clamped BCE must be finite, got {v}");
    }

    #[test]
    fn bce_backward_matches_analytic_gradient() {
        // d/dp = -(t/p - (1-t)/(1-p)) / n.
        let pred = var_from(&[0.8, 0.2]);
        let target = var_from(&[1.0, 0.0]);
        let loss = binary_cross_entropy(&pred, &target, 1e-7);
        loss.backward().expect("invariant: backward completes");
        let grad = pred.grad().expect("pred must receive a gradient");
        let expected = [-(1.0 / 0.8) / 2.0, (1.0 / 0.8) / 2.0];
        for (i, (&g, &e)) in grad.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-9, "bce grad[{i}]: got {g}, expected {e}");
        }
    }

    #[test]
    #[should_panic(expected = "at least one element")]
    fn bce_rejects_empty_input() {
        let pred = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([0], &[]), true);
        let target = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([0], &[]), true);
        let _ = binary_cross_entropy(&pred, &target, 1e-7);
    }
}
