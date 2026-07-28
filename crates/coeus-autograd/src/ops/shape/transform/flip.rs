// ── Tracked flip ──
//
// flip has a straight-through gradient: the gradient is just flipped back
// along the same axis.

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct FlipNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub axis: usize,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for FlipNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn op_name(&self) -> &'static str {
        "flip"
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
            // Gradient of flip is flip (self-inverse).
            let flipped_grad = coeus_ops::flip(grad_out, self.axis, &backend)?;
            let lock = g.write();
            coeus_ops::add_assign(lock, &flipped_grad, &backend)?;
        }

        Ok(())
    }
}

/// Tracked flip: reverse `input` along `axis`.
///
/// Backward: flip the output gradient along the same axis (flip is self-inverse).
///
/// # Panics
/// Panics if `axis >= input.ndim()`.
#[must_use]
#[inline]
pub fn flip<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    axis: usize,
) -> Result<Var<T, B>, B::Error>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let out_tensor = coeus_ops::flip(&input.tensor, axis, &backend)?;

    let requires_grad = crate::grad_mode::should_track_var(input);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )?)))
    } else {
        None
    };
    let creator = if let Some(ref output_grad) = grad {
        let node = FlipNode {
            output_grad: output_grad.clone(),
            inputs: vec![input.clone()],
            axis,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };
    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}
