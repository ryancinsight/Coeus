// ── Tracked gather op ──
//
// gather(input, dim, index):
//   out[…, k, …] = input[…, index[…, k, …], …]
//
// Backward:
//   d_input = scatter_add(zeros_like(input), dim, index, grad_out)
//   d_index = 0 (integer index, non-differentiable)

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct GatherNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    /// The gather index tensor, stored for backward.
    pub index: Tensor<T, B>,
    pub dim: usize,
    /// Shape of the input tensor (needed to create zeros for scatter_add).
    pub input_shape: Vec<usize>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for GatherNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn op_name(&self) -> &'static str {
        "gather"
    }
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        // Gradient flows through `input` only (index is non-differentiable).
        if let Some(Some(ref g)) = input_grads.first() {
            // d_input = scatter_add(zeros_like(input), dim, index, grad_out)
            let zeros = Tensor::zeros_on(self.input_shape.clone(), &backend);
            let d_input = coeus_ops::scatter_add(&zeros, self.dim, &self.index, grad_out, &backend);
            coeus_ops::add_assign(g.write(), &d_input, &backend);
        }
    }
}

/// Tracked index-based element selection along `dim`.
///
/// `index` must be a tensor of the same scalar type as `input` but containing
/// integer values (cast to `usize` when indexing).  Integer index tensors are
/// not yet a first-class type in coeus, so callers create index tensors of
/// type `T` with integer-valued elements.
///
/// # Backward
/// Only `input` receives a gradient via `scatter_add`; `index` is treated as
/// a constant (zero gradient).
///
/// # Panics
/// Same as `coeus_ops::gather`.
#[must_use]
#[inline]
pub fn gather<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    dim: usize,
    index: &Var<T, B>,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let out_tensor = coeus_ops::gather(&input.tensor, dim, &index.tensor, &backend);

    let requires_grad = input.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };
    let creator = if requires_grad {
        let node = GatherNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![input.clone()],
            index: index.tensor.clone(),
            dim,
            input_shape: input.tensor.shape().to_vec(),
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
