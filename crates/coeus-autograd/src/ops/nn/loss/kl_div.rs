use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for KL divergence loss.
///
/// Computes `mean(target * (log(target) - input))` where `input` holds
/// log-probabilities and `target` holds probabilities. The gradient w.r.t.
/// `input` is `-target / N * grad_out` per element. Forward and backward stay
/// on the selected provider; the node retains the provider-resident target
/// tensor rather than a host `Vec<T>` payload.
pub struct KlDivLossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Provider-resident target probabilities, saved for backward.
    pub target_saved: Tensor<T, B>,
    /// Original input shape for backward gradient materialization.
    pub input_shape: Vec<usize>,
    /// Number of elements in the loss reduction.
    pub n: usize,
    /// Provider-resident mean scale `1 / element_count`.
    pub mean_scale: Tensor<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for KlDivLossNode<T, B> {
    fn op_name(&self) -> &'static str {
        "kl_divergence"
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
            // d(loss)/d(input_i) = -target_i / N * grad_out
            let scale = coeus_ops::mul(grad_out, &self.mean_scale, &backend);
            let d_input = coeus_ops::mul(
                &coeus_ops::neg(&self.target_saved, &backend),
                &scale,
                &backend,
            );
            coeus_ops::add_assign(g.write(), &d_input, &backend)?;
        }
        Ok(())
    }
}

/// Tracked KL Divergence loss (PyTorch `F.kl_div` with `reduction='mean'`).
///
/// `input`: log-probabilities (log Q), `target`: probabilities (P).
/// Computes `mean(target * (log(target) - input))` with the `target == 0`
/// term taken as `0` by convention (`0 * log(0) = 0`). The complete forward
/// and backward computation stays on the selected provider; no input-sized
/// host staging occurs.
///
/// Returns a scalar Var (shape `[1]`).
pub fn kl_divergence<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    let n = input.tensor.numel();
    let input_shape = input.tensor.shape().to_vec();
    assert_eq!(
        target.tensor.numel(),
        n,
        "input and target must have the same number of elements"
    );
    assert_eq!(
        target.tensor.shape(),
        input.tensor.shape(),
        "input and target must have the same shape"
    );

    // loss = mean(target * (log(target) - input)), with the target == 0 term
    // taken as 0 by convention. All on-provider. `log(target)` is evaluated
    // on a safe copy (0 → 1) so no -inf lane exists; the original target
    // (0 at those positions) zeroes the term, avoiding `0 * -inf = NaN`.
    let ones = Tensor::full_on(target.tensor.shape_cloned(), T::one(), &backend);
    let zeros = Tensor::zeros_on(target.tensor.shape_cloned(), &backend);
    let zero_mask = coeus_ops::eq(&target.tensor, &zeros, &backend);
    let safe_target = coeus_ops::where_cond(&zero_mask, &ones, &target.tensor, &backend)
        .expect("kl_divergence: provider safe-target mask");
    let log_target = coeus_ops::log(&safe_target, &backend);
    let diff = coeus_ops::sub(&log_target, &input.tensor, &backend);
    let term = coeus_ops::mul(&target.tensor, &diff, &backend);
    let loss = coeus_ops::mean_axis(&term.reshape([n]), 0, &backend)
        .expect("invariant: validated non-empty KL divergence reduction has axis zero");

    let requires_grad = crate::grad_mode::should_track_var(input);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = grad.as_ref().cloned().map(|output_grad| {
        let node = KlDivLossNode {
            output_grad,
            inputs: vec![input.clone()],
            target_saved: target.tensor.clone(),
            input_shape,
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
    fn kl_div_forward_matches_reference() {
        // target = [0.5, 0.0, 0.25], input (log-probabilities) = [-1, 0, -2]:
        //   term0 = 0.5 * (log(0.5) - (-1))
        //   term1 = 0 (target == 0 convention)
        //   term2 = 0.25 * (log(0.25) - (-2))
        let target = var_from(&[0.5, 0.0, 0.25]);
        let input = var_from(&[-1.0, 0.0, -2.0]);
        let loss = kl_divergence(&input, &target);
        let expected = (0.5 * (0.5_f64.ln() + 1.0) + 0.25 * (0.25_f64.ln() + 2.0)) / 3.0;
        assert_eq!(loss.tensor.shape(), &[1]);
        assert!((loss.tensor.as_slice()[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn kl_div_backward_matches_analytic_gradient() {
        // d/dinput = -target / n.
        let target = var_from(&[0.5, 0.0, 0.25]);
        let input = var_from(&[-1.0, 0.0, -2.0]);
        let loss = kl_divergence(&input, &target);
        loss.backward().expect("invariant: backward completes");
        let grad = input.grad().expect("input must receive a gradient");
        let expected = [-0.5 / 3.0, 0.0 / 3.0, -0.25 / 3.0];
        for (i, (&g, &e)) in grad.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-12,
                "kl_div grad[{i}]: got {g}, expected {e}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "same number of elements")]
    fn kl_div_rejects_element_mismatch() {
        let target = var_from(&[0.5, 0.5]);
        let input = var_from(&[-1.0]);
        let _ = kl_divergence(&input, &target);
    }
}
