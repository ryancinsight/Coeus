use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for the Poisson negative-log-likelihood loss (log-input form).
///
/// Forward and backward stay on the selected provider: the node retains the
/// provider-resident `exp(input)` tensor and mean scale rather than host
/// `Vec<T>` payloads.
pub struct PoissonNllNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Provider-resident `exp(input)`, saved for backward.
    pub exp_input: Tensor<T, B>,
    /// Number of elements in the mean reduction.
    pub n: usize,
    /// Original tensor shape for gradient reconstruction.
    pub shape: coeus_core::Shape,
    /// Provider-resident mean scale `1 / element_count`.
    pub mean_scale: Tensor<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for PoissonNllNode<T, B> {
    fn op_name(&self) -> &'static str {
        "poisson_nll"
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
        let scale = coeus_ops::mul(grad_out, &self.mean_scale, &backend);

        // d/d_input = (exp(input) - target) / n.
        if let Some(Some(ref g)) = input_grads.first() {
            let d_input = coeus_ops::sub(&self.exp_input, &self.inputs[1].tensor, &backend);
            let d_input = coeus_ops::mul(&d_input, &scale, &backend);
            coeus_ops::add_assign(g.write(), &d_input, &backend)?;
        }

        // d/d_target = -input / n.
        if let Some(Some(ref g)) = input_grads.get(1) {
            let d_target = coeus_ops::neg(&self.inputs[0].tensor, &backend);
            let d_target = coeus_ops::mul(&d_target, &scale, &backend);
            coeus_ops::add_assign(g.write(), &d_target, &backend)?;
        }
        Ok(())
    }
}

/// Tracked Poisson negative-log-likelihood loss in the log-input regime
/// (PyTorch `PoissonNLLLoss(log_input=True, full=False, reduction="mean")`):
/// `loss = mean_i(exp(input_i) - target_i * input_i)`.
///
/// `input` holds the log-rate `log(λ)`, `target` the observed counts; both
/// share shape. The Stirling `full` correction term is not included (matching
/// the PyTorch default). The complete forward and backward computation stays
/// on the selected provider; no input-sized host staging occurs.
pub fn poisson_nll<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    assert_eq!(
        input.tensor.shape(),
        target.tensor.shape(),
        "poisson_nll requires input and target to have identical shapes"
    );
    let n = input.tensor.numel();
    assert!(n > 0, "poisson_nll requires at least one element");
    let shape = input.tensor.shape_cloned();

    // loss_i = exp(input) - target * input, all on-provider.
    let exp_input = coeus_ops::exp(&input.tensor, &backend);
    let product = coeus_ops::mul(&target.tensor, &input.tensor, &backend);
    let per_elem = coeus_ops::sub(&exp_input, &product, &backend);
    let loss = coeus_ops::mean_axis(&per_elem.reshape([n]), 0, &backend)
        .expect("invariant: validated non-empty Poisson NLL reduction has axis zero");

    let requires_grad =
        crate::grad_mode::should_track_var(input) || crate::grad_mode::should_track_var(target);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = grad.as_ref().cloned().map(|output_grad| {
        let node = PoissonNllNode {
            output_grad,
            inputs: vec![input.clone(), target.clone()],
            exp_input,
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
    fn poisson_nll_forward_matches_reference() {
        // input = [0, 1], target = [1, 2]:
        //   loss = mean(exp(0) - 1·0, exp(1) - 2·1) = mean(1, e - 2).
        let input = var_from(&[0.0, 1.0]);
        let target = var_from(&[1.0, 2.0]);
        let loss = poisson_nll(&input, &target);
        let expected = f64::midpoint(1.0, std::f64::consts::E - 2.0);
        assert_eq!(loss.tensor.shape(), &[1]);
        assert!((loss.tensor.as_slice()[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn poisson_nll_backward_matches_analytic_gradient() {
        // d/dinput = (exp(input) - target) / n; d/dtarget = -input / n.
        let input = var_from(&[0.0, 1.0]);
        let target = var_from(&[1.0, 2.0]);
        let loss = poisson_nll(&input, &target);
        loss.backward().expect("invariant: backward completes");
        let input_grad = input.grad().expect("input must receive a gradient");
        let target_grad = target.grad().expect("target must receive a gradient");
        let expected_input = [(1.0 - 1.0) / 2.0, (std::f64::consts::E - 2.0) / 2.0];
        let expected_target = [0.0 / 2.0, -1.0 / 2.0];
        for (i, (&g, &e)) in input_grad
            .as_slice()
            .iter()
            .zip(expected_input.iter())
            .enumerate()
        {
            assert!(
                (g - e).abs() < 1e-12,
                "poisson_nll input grad[{i}]: got {g}, expected {e}"
            );
        }
        for (i, (&g, &e)) in target_grad
            .as_slice()
            .iter()
            .zip(expected_target.iter())
            .enumerate()
        {
            assert!(
                (g - e).abs() < 1e-12,
                "poisson_nll target grad[{i}]: got {g}, expected {e}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "identical shapes")]
    fn poisson_nll_rejects_shape_mismatch() {
        let input = var_from(&[0.0, 1.0]);
        let target = var_from(&[1.0]);
        let _ = poisson_nll(&input, &target);
    }
}
