use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_ops::CrossEntropyOps;
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node retaining provider-native cross-entropy backward state.
pub struct CrossEntropyLossNode<T, B>
where
    T: Scalar,
    B: coeus_ops::BackendOps<T> + CrossEntropyOps<T> + Default,
{
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Provider-native targets retained from forward.
    pub targets: <B as CrossEntropyOps<T>>::Targets,
    /// Provider-resident probabilities retained from forward.
    pub probabilities: Tensor<T, B>,
}

impl<T, B> BackwardNode<T, B> for CrossEntropyLossNode<T, B>
where
    T: Float,
    B: coeus_ops::BackendOps<T> + CrossEntropyOps<T> + Default,
{
    fn op_name(&self) -> &'static str {
        "cross_entropy_loss"
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
        if let Some(Some(gradient)) = input_grads.first() {
            let backend = B::default();
            let destination = gradient.write();
            let (destination_storage, destination_layout) = destination.storage_mut_and_layout();
            backend.cross_entropy_backward_accumulate(
                grad_out.storage(),
                grad_out.layout(),
                self.probabilities.storage(),
                self.probabilities.layout(),
                &self.targets,
                destination_storage,
                destination_layout,
            )?;
        }
        Ok(())
    }
}

/// Attach provider-resident mean cross-entropy state to the autograd graph.
pub fn cross_entropy_loss<T, B>(
    logits: &Var<T, B>,
    targets: <B as CrossEntropyOps<T>>::Targets,
    output: Tensor<T, B>,
    probabilities: Tensor<T, B>,
) -> Var<T, B>
where
    T: Float,
    B: coeus_ops::BackendOps<T> + CrossEntropyOps<T> + Default,
{
    let backend = B::default();
    let requires_grad = crate::grad_mode::should_track_var(logits);
    let grad = requires_grad.then(|| Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))));
    let creator = grad.as_ref().map(|output_grad| {
        Arc::new(CrossEntropyLossNode {
            output_grad: Arc::clone(output_grad),
            inputs: vec![logits.clone()],
            targets,
            probabilities,
        }) as Arc<dyn BackwardNode<T, B>>
    });

    Var {
        tensor: output,
        grad,
        creator,
    }
}
