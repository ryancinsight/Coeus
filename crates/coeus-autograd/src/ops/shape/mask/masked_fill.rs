// ── Tracked masked_fill ──
//
// Backward passes gradient through unmasked positions and zeros it where
// the mask selected the fill value. The mask itself is non-differentiable.
#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct MaskedFillNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub mask_tensor: Tensor<T, B>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for MaskedFillNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn op_name(&self) -> &'static str {
        "masked_fill"
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
            let grad_input =
                coeus_ops::masked_fill(grad_out, &self.mask_tensor, T::zero(), &backend);
            let lock = g.write();
            coeus_ops::add_assign(lock, &grad_input, &backend)?;
        }
        Ok(())
    }
}

/// Tracked masked fill.
///
/// Gradient flows only to `input`; `mask` receives zero gradient.
///
/// # Panics
/// Panics if `input.shape() != mask.shape()`.
#[must_use]
#[inline]
pub fn masked_fill<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    mask: &Var<T, B>,
    value: T,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let out_tensor = coeus_ops::masked_fill(&input.tensor, &mask.tensor, value, &backend);

    let requires_grad = crate::grad_mode::should_track_var(input);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };
    let creator = if requires_grad {
        let node = MaskedFillNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![input.clone()],
            mask_tensor: mask.tensor.clone(),
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
