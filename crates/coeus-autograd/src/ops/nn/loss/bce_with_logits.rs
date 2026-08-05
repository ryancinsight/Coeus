use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for binary cross-entropy computed from logits.
pub struct BceWithLogitsNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Provider-resident `sigmoid(logit) - target` derivative factor.
    pub sig_minus_target: Tensor<T, B>,
    /// Provider-resident logits used for the target derivative.
    pub logits: Tensor<T, B>,
    /// Provider-resident mean scale multiplied by the upstream output gradient
    /// during backward.
    pub scale: Tensor<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for BceWithLogitsNode<T, B>
{
    fn op_name(&self) -> &'static str {
        "bce_with_logits"
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
        let scale = coeus_ops::mul(grad_out, &self.scale, &backend);

        // d/d_logit = (sigmoid(z) - y) / n, with all arithmetic on the
        // selected provider.
        if let Some(Some(gradient)) = input_grads.first() {
            let grad_logits = coeus_ops::mul(&self.sig_minus_target, &scale, &backend);
            coeus_ops::add_assign(gradient.write(), &grad_logits, &backend)?;
        }

        // d/d_target = -z / n, with the logits retained as a provider tensor.
        if let Some(Some(gradient)) = input_grads.get(1) {
            let neg_logits = coeus_ops::neg(&self.logits, &backend);
            let grad_target = coeus_ops::mul(&neg_logits, &scale, &backend);
            coeus_ops::add_assign(gradient.write(), &grad_target, &backend)?;
        }

        Ok(())
    }
}

/// Tracked binary cross-entropy with logits: the numerically stable composition
/// of a sigmoid and binary cross-entropy. `logits` and `target` must share shape.
///
/// Per element with `z = logit`, `y = target`:
/// `loss = max(z, 0) - z*y + log(1 + exp(-|z|))`, averaged over all elements.
/// This is the `reduction="mean"` form of PyTorch `BCEWithLogitsLoss`.
pub fn bce_with_logits<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    logits: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    assert_eq!(
        logits.tensor.shape(),
        target.tensor.shape(),
        "bce_with_logits requires logits and target to have identical shapes"
    );
    let n = logits.tensor.numel();
    assert!(n > 0, "bce_with_logits requires at least one element");

    // Stable BCE-with-logits expression. Each operation remains in the
    // backend's provider implementation: Leto for CPU and Hephaestus for
    // accelerator backends.
    let positive_logits = coeus_ops::relu(&logits.tensor, &backend);
    let absolute_logits = coeus_ops::abs(&logits.tensor, &backend);
    let negative_absolute_logits = coeus_ops::neg(&absolute_logits, &backend);
    let exponential_tail = coeus_ops::exp(&negative_absolute_logits, &backend);
    let log_tail = coeus_ops::log1p(&exponential_tail, &backend);
    let weighted_target = coeus_ops::mul(&logits.tensor, &target.tensor, &backend);
    let signed_terms = coeus_ops::sub(&positive_logits, &weighted_target, &backend);
    let loss_terms = coeus_ops::add(&signed_terms, &log_tail, &backend);
    let loss = coeus_ops::mean_axis(&loss_terms.reshape([n]), 0, &backend)
        .expect("invariant: validated non-empty BCE reduction has axis zero");

    let sig_minus_target = coeus_ops::sub(
        &coeus_ops::sigmoid(&logits.tensor, &backend),
        &target.tensor,
        &backend,
    );
    let scale = Tensor::full_on([1], T::from_f64(1.0 / n as f64), &backend);
    let requires_grad =
        crate::grad_mode::should_track_var(logits) || crate::grad_mode::should_track_var(target);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad
            .as_ref()
            .expect("invariant: tracked output has a gradient buffer")
            .clone();
        let node = BceWithLogitsNode {
            output_grad,
            inputs: vec![logits.clone(), target.clone()],
            sig_minus_target,
            logits: logits.tensor.clone(),
            scale,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };
    Var {
        tensor: loss,
        grad,
        creator,
    }
}
