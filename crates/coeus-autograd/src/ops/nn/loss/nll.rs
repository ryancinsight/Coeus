use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for negative log-likelihood loss.
///
/// Forward and backward stay on the selected provider: the node retains the
/// provider-resident one-hot target mask and mean scale rather than host
/// `Vec<T>` payloads. The `targets: &[usize]` host slice is a boundary upload.
pub struct NllLossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Provider-resident one-hot target mask (shape `[N, C]`).
    pub target_mask: Tensor<T, B>,
    /// Batch size.
    pub n: usize,
    /// Number of classes.
    pub c: usize,
    /// Provider-resident mean scale `1 / batch_size`.
    pub mean_scale: Tensor<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for NllLossNode<T, B> {
    fn op_name(&self) -> &'static str {
        "nll_loss"
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
        if let Some(Some(ref g)) = input_grads.get(0) {
            // d/dlog_probs = -mask * grad_out / n, all on-provider.
            let scale = coeus_ops::mul(grad_out, &self.mean_scale, &backend);
            let d_log = coeus_ops::mul(
                &coeus_ops::neg(&self.target_mask, &backend),
                &scale,
                &backend,
            );
            coeus_ops::add_assign(g.write(), &d_log, &backend)?;
        }
        Ok(())
    }
}

/// Tracked Negative Log-Likelihood Loss.
/// `log_probs`: `[N, C]` (already log-probabilities), `targets`: `[N]` class
/// indices. The complete forward and backward computation stays on the
/// selected provider; no input-sized host staging occurs beyond the one-hot
/// target-mask boundary upload.
pub fn nll_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    log_probs: &Var<T, B>,
    targets: &[usize],
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let shape = log_probs.tensor.shape();
    let n = shape[0];
    let c = shape[1];
    assert_eq!(targets.len(), n, "targets length must match batch size");

    // One-hot target mask on-provider; selected = mask * log_probs.
    let target_f: Vec<T> = targets.iter().map(|&i| T::from_usize(i)).collect();
    let target_tensor = Tensor::from_slice_on([n], &target_f, &backend);
    let target_mask = coeus_ops::one_hot(&target_tensor, c, &backend);
    let selected = coeus_ops::mul(&log_probs.tensor, &target_mask, &backend);
    // loss = -mean_i selected[i, target[i]] = -sum over batch / n.
    let row_sum = coeus_ops::sum_axis(&selected, 1, &backend)
        .expect("invariant: validated [N, C] NLL axis-one reduction");
    let neg_sum = coeus_ops::neg(&row_sum, &backend);
    let loss = coeus_ops::mean_axis(&neg_sum.reshape([n]), 0, &backend)
        .expect("invariant: validated non-empty NLL reduction has axis zero");

    let requires_grad = crate::grad_mode::should_track_var(log_probs);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = grad.as_ref().cloned().map(|output_grad| {
        let node = NllLossNode {
            output_grad,
            inputs: vec![log_probs.clone()],
            target_mask,
            n,
            c,
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

    #[test]
    fn nll_forward_matches_reference() {
        // log_probs = [[-1, -2], [-3, -4]], targets = [0, 1]:
        //   loss = mean(-(-1), -(-4)) = mean(1, 4) = 2.5.
        let log_probs = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[-1.0, -2.0, -3.0, -4.0]),
            true,
        );
        let targets = [0usize, 1];
        let loss = nll_loss(&log_probs, &targets);
        assert_eq!(loss.tensor.shape(), &[1]);
        assert!((loss.tensor.as_slice()[0] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn nll_backward_matches_analytic_gradient() {
        // d/dlog_probs = -one_hot(targets) / n.
        let log_probs = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[-1.0, -2.0, -3.0, -4.0]),
            true,
        );
        let targets = [0usize, 1];
        let loss = nll_loss(&log_probs, &targets);
        loss.backward().expect("invariant: backward completes");
        let grad = log_probs.grad().expect("log_probs must receive a gradient");
        let expected = [[-0.5, 0.0], [0.0, -0.5]];
        for (i, &g) in grad.as_slice().iter().enumerate() {
            let e = expected[i / 2][i % 2];
            assert!(
                (g - e).abs() < 1e-12,
                "nll grad[{i}]: got {g}, expected {e}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "targets")]
    fn nll_rejects_target_length_mismatch() {
        let log_probs = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[-1.0, -2.0, -3.0, -4.0]),
            true,
        );
        let targets = [0usize];
        let _ = nll_loss(&log_probs, &targets);
    }
}
