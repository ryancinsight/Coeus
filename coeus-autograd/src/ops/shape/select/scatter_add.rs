// ── Tracked scatter_add ──
//
// Forward: `out = scatter_add(input, dim, index, src)` — `out` starts as a copy
// of `input` and, for each `k` along `dim`, `src[…, k, …]` is added into
// `out[…, index[…, k, …], …]`.
//
// Backward (two tracked inputs, `input` and `src`; `index` non-differentiable):
//   grad_input = grad_out                       (input is copied through; the
//                                                add does not scale it)
//   grad_src   = gather(grad_out, dim, index)   (each scattered source element
//                                                receives the output gradient
//                                                at the position it landed in)
//
// This is the exact transpose of `gather` — gather's backward scatter_adds,
// scatter_add's backward gathers. Matches PyTorch `Tensor.scatter_add`.

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct ScatterAddNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// `[input, src]` — the destination tensor and the scattered source.
    pub inputs: Vec<Var<T, B>>,
    /// The index tensor (T-encoded integer indices), stored for backward.
    pub index: Tensor<T, B>,
    pub dim: usize,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for ScatterAddNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn op_name(&self) -> &'static str {
        "scatter_add"
    }
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();

        // ∂/∂input: input is copied into the output unchanged, so its gradient
        // is the output gradient verbatim.
        if let Some(Some(ref g)) = input_grads.first() {
            let grad_input = grad_out.to_contiguous_on(&backend);
            coeus_ops::add_assign(g.write(), &grad_input, &backend);
        }

        // ∂/∂src: each source element lands at `index`, so its gradient gathers
        // the output gradient from that position.
        if let Some(Some(ref g)) = input_grads.get(1) {
            let grad_src = coeus_ops::gather(grad_out, self.dim, &self.index, &backend);
            coeus_ops::add_assign(g.write(), &grad_src, &backend);
        }
    }
}

/// Tracked `scatter_add` (`torch.Tensor.scatter_add`): returns `input` with
/// `src` added into the positions named by `index` along `dim`.
///
/// `index` must be a tensor of the same scalar type as `input` holding
/// integer-valued elements (index tensors are not yet a first-class type in
/// coeus).
///
/// # Backward
/// Gradient flows to both `input` (identity) and `src` (gathered at `index`);
/// `index` is treated as a constant.
///
/// # Panics
/// Same as `coeus_ops::scatter_add`.
#[must_use]
pub fn scatter_add<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    dim: usize,
    index: &Var<T, B>,
    src: &Var<T, B>,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let out_tensor = coeus_ops::scatter_add(
        &input.tensor,
        dim,
        &index.tensor,
        &src.tensor,
        &backend,
    );

    let requires_grad =
        crate::grad_mode::should_track_var(input) || crate::grad_mode::should_track_var(src);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };
    let creator = if requires_grad {
        let node = ScatterAddNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![input.clone(), src.clone()],
            index: index.tensor.clone(),
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
