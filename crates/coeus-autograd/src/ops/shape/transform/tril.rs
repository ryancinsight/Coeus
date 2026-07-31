// ── Tracked tril / triu ──
//
// tril and triu have pass-through gradients: the backward simply applies
// the same mask to the incoming gradient (grad of a zeroed element stays 0).

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

// ── tril ────────────────────────────────────────────────────────────────────

pub struct TrilNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub k: isize,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for TrilNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn op_name(&self) -> &'static str {
        "tril"
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
            // Gradient of tril(x, k) is tril(grad_out, k): zeroed positions
            // have zero gradient since they contributed nothing to the output.
            let masked = coeus_ops::tril(grad_out, self.k, &backend);
            let lock = g.write();
            coeus_ops::add_assign(lock, &masked, &backend)?;
        }
        Ok(())
    }
}

/// Tracked lower-triangular mask on the last two dimensions.
///
/// Elements above diagonal `k` are set to zero in the output; their gradient
/// is also zero (pass-through mask backward).
///
/// # Panics
/// Panics if `input.ndim() < 2`.
#[must_use]
#[inline]
pub fn tril<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    k: isize,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let out_tensor = coeus_ops::tril(&input.tensor, k, &backend);

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
        let node = TrilNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![input.clone()],
            k,
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

// ── triu ────────────────────────────────────────────────────────────────────

pub struct TriuNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub k: isize,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for TriuNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn op_name(&self) -> &'static str {
        "triu"
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
            let masked = coeus_ops::triu(grad_out, self.k, &backend);
            let lock = g.write();
            coeus_ops::add_assign(lock, &masked, &backend)?;
        }
        Ok(())
    }
}

/// Tracked upper-triangular mask on the last two dimensions.
///
/// # Panics
/// Panics if `input.ndim() < 2`.
#[must_use]
#[inline]
pub fn triu<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    k: isize,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let out_tensor = coeus_ops::triu(&input.tensor, k, &backend);

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
        let node = TriuNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![input.clone()],
            k,
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
