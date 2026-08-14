// ── Tracked stack op ──
//
// `stack(inputs, dim)` inserts a new dimension at `dim` and concatenates all
// input tensors along it.  The backward is an `unstuck` via `split` + squeeze
// on each resulting chunk.
#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct StackNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub dim: usize,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for StackNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "stack"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        // Split the stacked output-gradient back into `n` slices along `dim`,
        // each of size 1; then squeeze `dim` to recover the original rank.
        let chunks = coeus_ops::split(grad_out, 1, self.dim);
        for (chunk, acc) in chunks.into_iter().zip(input_grads.iter()) {
            let Some(ref g) = *acc else {
                continue;
            };
            // Remove the stacked dimension to match the input rank
            let squeezed = chunk.squeeze(self.dim);
            let lock = g.write();
            coeus_ops::add_assign(lock, &squeezed, &backend)?;
        }
        Ok(())
    }
}

/// Tracked stack: inserts a new axis at `dim` and concatenates `inputs` along it.
///
/// All inputs must have the same shape.  Output shape has one extra dimension
/// of size `n` at position `dim`.  Backward propagates gradients via an unstack
/// (split-along-dim + squeeze).
///
/// # Panics
/// Panics if `inputs` is empty or if shapes do not match.
#[must_use]
#[inline]
pub fn stack<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    inputs: &[&Var<T, B>],
    dim: usize,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    assert!(!inputs.is_empty(), "stack: inputs must be non-empty");
    let backend = B::default();
    let tensors: Vec<&Tensor<T, B>> = inputs.iter().map(|v| &v.tensor).collect();
    let out_tensor = coeus_ops::stack(&tensors, dim);

    let requires_grad = inputs.iter().any(|v| crate::grad_mode::should_track_var(v));
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };
    let creator = if requires_grad {
        let node = StackNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: inputs.iter().map(|v| (*v).clone()).collect(),
            dim,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };
    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}
