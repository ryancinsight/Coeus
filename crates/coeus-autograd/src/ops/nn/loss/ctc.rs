use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_ops::{BackendOps, CtcBatch, CtcOps};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd graph connections and provider-owned CTC recurrence state.
pub struct CtcLossNode<T, B>
where
    T: Scalar,
    B: BackendOps<T> + CtcOps<T> + Default,
{
    output_grad: Arc<GradBuffer<T, B>>,
    inputs: [Var<T, B>; 1],
    state: B::CtcState,
}

impl<T, B> BackwardNode<T, B> for CtcLossNode<T, B>
where
    T: Float,
    B: BackendOps<T> + CtcOps<T> + Default,
{
    fn op_name(&self) -> &'static str {
        "ctc_loss"
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
            let destination = gradient.write();
            let (storage, layout) = destination.storage_mut_and_layout();
            B::default().ctc_backward_accumulate(
                &self.state,
                grad_out.storage(),
                grad_out.layout(),
                storage,
                layout,
            )?;
        }
        Ok(())
    }
}

/// Compute mean CTC loss and retain provider state for differentiation.
///
/// `log_probs` has shape `[frames, batch, classes]`. Targets are concatenated
/// in batch order; both length slices contain one entry per sample. The loss
/// divides each sample by `max(target_length, 1)` before taking the batch mean.
/// Empty targets retain the all-blank path; impossible alignments have infinite
/// loss and their backward pass returns an error without changing the input
/// gradient. Backward differentiates independent log probabilities, yielding
/// negative posterior occupancy. Compose with [`crate::log_softmax()`] to obtain
/// derivatives with respect to logits.
///
/// # Errors
///
/// Returns the provider's typed error for invalid layouts, lengths, labels,
/// active numeric inputs, allocation failure, or nonrepresentable arithmetic.
///
/// # Examples
///
/// ```
/// use coeus_autograd::{ctc_loss, Var};
/// use coeus_core::SequentialBackend;
/// use coeus_tensor::Tensor;
///
/// let input = Var::<f32, SequentialBackend>::new(
///     Tensor::from_slice([1, 1, 2], &[-std::f32::consts::LN_2; 2]), true);
/// let loss = ctc_loss(&input, &[], &[1], &[0], 0)?;
/// assert_eq!(loss.tensor.as_slice(), &[std::f32::consts::LN_2]);
/// loss.backward()?;
/// assert_eq!(input.grad().expect("invariant: tracked input has a gradient").as_slice(),
///     &[-1.0, 0.0]);
/// # Ok::<(), coeus_core::BackendError>(())
/// ```
pub fn ctc_loss<T, B>(
    log_probs: &Var<T, B>,
    targets: &[usize],
    input_lengths: &[usize],
    target_lengths: &[usize],
    blank: usize,
) -> Result<Var<T, B>, B::Error>
where
    T: Float,
    B: BackendOps<T> + CtcOps<T> + Default,
{
    let backend = B::default();
    let mut output = Tensor::zeros_on([1], &backend);
    let (storage, layout) = output.storage_mut_and_layout();
    let state = backend.ctc_forward(
        log_probs.tensor.storage(),
        log_probs.tensor.layout(),
        CtcBatch {
            targets,
            input_lengths,
            target_lengths,
            blank,
        },
        storage,
        layout,
    )?;
    let requires_grad = crate::grad_mode::should_track_var(log_probs);
    let grad = requires_grad.then(|| Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))));
    let creator = grad.as_ref().map(|output_grad| {
        // The existing graph erases heterogeneous operation nodes at its graph boundary.
        Arc::new(CtcLossNode {
            output_grad: Arc::clone(output_grad),
            inputs: [log_probs.clone()],
            state,
        }) as Arc<dyn BackwardNode<T, B>>
    });
    Ok(Var {
        tensor: output,
        grad,
        creator,
    })
}
