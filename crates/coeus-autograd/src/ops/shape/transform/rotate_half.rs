use crate::{grad_buffer::GradBuffer, node::BackwardNode, var::Var};
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Backward node for provider-selected half-vector rotation.
struct RotateHalfNode<
    T: Scalar,
    B: coeus_ops::BackendOps<T> + coeus_ops::RotateHalfOps<T> + Default,
> {
    output_grad: Arc<GradBuffer<T, B>>,
    inputs: Vec<Var<T, B>>,
}

impl<T, B> BackwardNode<T, B> for RotateHalfNode<T, B>
where
    T: Scalar,
    B: coeus_ops::BackendOps<T> + coeus_ops::RotateHalfOps<T> + Default,
{
    fn op_name(&self) -> &'static str {
        "rotate_half"
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
        let Some(Some(accumulator)) = input_grads.first() else {
            return Ok(());
        };
        let backend = B::default();
        let rotated = coeus_ops::rotate_half(grad_out, &backend)?;
        let gradient = coeus_ops::elementwise_unary(&rotated, &backend, coeus_ops::UnaryOp::Neg)?;
        coeus_ops::add_assign(accumulator.write(), &gradient, &backend)
    }
}

/// Rotate the equal halves of the final axis as `[-x₂, x₁]` and track its
/// exact transpose for backward propagation.
///
/// # Errors
///
/// Returns the selected backend's typed failure for invalid shape, allocation,
/// layout conversion, or provider dispatch.
pub fn rotate_half<T, B>(input: &Var<T, B>) -> Result<Var<T, B>, B::Error>
where
    T: Scalar,
    B: coeus_ops::BackendOps<T> + coeus_ops::RotateHalfOps<T> + Default,
{
    let backend = B::default();
    let tensor = coeus_ops::rotate_half(&input.tensor, &backend)?;
    if !crate::grad_mode::should_track_var(input) {
        return Ok(Var::new(tensor, false));
    }

    let output_grad = Arc::new(GradBuffer::new(Tensor::zeros_on(
        tensor.shape_cloned(),
        &backend,
    )));
    let node = RotateHalfNode {
        output_grad: Arc::clone(&output_grad),
        inputs: vec![input.clone()],
    };
    Ok(Var {
        tensor,
        grad: Some(output_grad),
        creator: Some(Arc::new(node)),
    })
}
