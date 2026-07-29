// ── Tracked index_put ──
//
// Forward: `out = index_put(x, idx, v, accumulate)` — positions `idx` (a 1-D
// index vector) of `x` are replaced by `v` (`accumulate = false`) or have `v`
// added to them (`accumulate = true`); all other positions keep `x`.
//
// Backward (two tracked inputs, `x` and `v`):
//   grad_x = grad_out with the `idx` positions zeroed  (accumulate = false)
//          = grad_out                                   (accumulate = true; x
//                                                        is fully preserved)
//   grad_v = grad_out gathered at `idx` (`index_select`) — each replaced/added
//            output position feeds its gradient back to the source value.
//
// Matches PyTorch `Tensor.index_put((idx,), v, accumulate)` autograd.

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct IndexPutNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// `[x, v]` — the destination tensor and the inserted values.
    pub inputs: Vec<Var<T, B>>,
    /// The 1-D index tensor (f64-encoded integer indices).
    pub index_tensor: Tensor<T, B>,
    /// Whether the forward accumulated (`+=`) rather than overwrote.
    pub accumulate: bool,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for IndexPutNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn op_name(&self) -> &'static str {
        "index_put"
    }
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();

        // ∂/∂x: overwrite zeroes the replaced positions (they no longer depend
        // on x); accumulation preserves x everywhere, so the gradient passes
        // through unchanged.
        if let Some(Some(ref g)) = input_grads.first() {
            let grad_x = if self.accumulate {
                grad_out.to_contiguous_on(&backend)
            } else {
                let k = self.index_tensor.numel();
                let zeros = Tensor::zeros_on([k], &backend);
                coeus_ops::index_put(grad_out, &self.index_tensor, &zeros, false, &backend)
            };
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_x, &backend).expect("autograd gradient accumulation");
        }

        // ∂/∂v: each value lands at its `idx` position, so its gradient is the
        // output gradient gathered there.
        if let Some(Some(ref g)) = input_grads.get(1) {
            let grad_v = coeus_ops::index_select(grad_out, 0, &self.index_tensor, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_v, &backend).expect("autograd gradient accumulation");
        }
    }
}

/// Tracked `index_put` (`torch.Tensor.index_put`): returns `x` with the 1-D
/// index positions `indices` set to (`accumulate = false`) or increased by
/// (`accumulate = true`) `values`.
///
/// Gradient flows to both `x` (identity, minus the overwritten positions when
/// not accumulating) and `values` (gathered at `indices`).
#[must_use]
pub fn index_put<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    indices: &Var<T, B>,
    values: &Var<T, B>,
    accumulate: bool,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let out_tensor = coeus_ops::index_put(
        &input.tensor,
        &indices.tensor,
        &values.tensor,
        accumulate,
        &backend,
    );

    let requires_grad =
        crate::grad_mode::should_track_var(input) || crate::grad_mode::should_track_var(values);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };
    let creator = if requires_grad {
        let node = IndexPutNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![input.clone(), values.clone()],
            index_tensor: indices.tensor.clone(),
            accumulate,
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
