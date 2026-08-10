use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for L1 (mean absolute error) loss.
pub struct L1LossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Provider-resident element-wise differences `pred - target`.
    pub diffs: Tensor<T, B>,
    /// Number of elements in the mean reduction.
    pub n: usize,
    /// Original logical tensor shape.
    pub shape: coeus_core::Shape,
    /// Provider-resident mean scale `1 / element_count`.
    pub mean_scale: Tensor<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for L1LossNode<T, B> {
    fn op_name(&self) -> &'static str {
        "l1_loss"
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
        // d/d_pred mean|pred - target| = sign(pred - target) / n.
        // The kink subgradient is zero, matching torch.sign(0) == 0.
        let scale = coeus_ops::mul(grad_out, &self.mean_scale, &backend);
        let sign = coeus_ops::sign(&self.diffs, &backend);
        let d_pred = coeus_ops::mul(&sign, &scale, &backend);

        if let Some(Some(ref gradient)) = input_grads.first() {
            coeus_ops::add_assign(gradient.write(), &d_pred, &backend)?;
        }

        if let Some(Some(ref gradient)) = input_grads.get(1) {
            let d_target = coeus_ops::neg(&d_pred, &backend);
            coeus_ops::add_assign(gradient.write(), &d_target, &backend)?;
        }

        Ok(())
    }
}

/// Tracked L1 (mean absolute error) loss: `mean_i |pred[i] - target[i]|`.
///
/// `pred` and `target` must have identical shape. The mean reduction covers
/// every element, not only the leading dimension. All arithmetic remains on
/// the selected backend: Leto for CPU and Hephaestus for accelerator backends.
pub fn l1_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    pred: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    assert_eq!(
        pred.tensor.shape(),
        target.tensor.shape(),
        "l1_loss requires pred and target to have identical shapes"
    );
    let n = pred.tensor.numel();
    assert!(n > 0, "l1_loss requires at least one element");

    let diffs = coeus_ops::sub(&pred.tensor, &target.tensor, &backend);
    let absolute_diffs = coeus_ops::abs(&diffs, &backend);
    let loss = coeus_ops::mean_axis(&absolute_diffs.reshape([n]), 0, &backend)
        .expect("invariant: validated non-empty L1 reduction has axis zero");
    let requires_grad =
        crate::grad_mode::should_track_var(pred) || crate::grad_mode::should_track_var(target);
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
        let mean_scale = Tensor::full_on([1], T::one() / T::from_f64(n as f64), &backend);
        let node = L1LossNode {
            output_grad,
            inputs: vec![pred.clone(), target.clone()],
            n,
            shape: pred.tensor.shape_cloned(),
            diffs,
            mean_scale,
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
