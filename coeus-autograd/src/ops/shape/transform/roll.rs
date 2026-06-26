// ── Tracked roll ──
//
// roll has a straight-through gradient: the backward is roll with negated shifts
// (unroll = reverse roll, which is roll by -shift).

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct RollNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub shifts: Vec<isize>,
    pub dims: Vec<usize>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for RollNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn op_name(&self) -> &'static str {
        "roll"
    }
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            // Backward of roll(x, shifts, dims) is roll(grad, -shifts, dims).
            let neg_shifts: Vec<isize> = self.shifts.iter().map(|&s| -s).collect();
            let unrolled = coeus_ops::roll(grad_out, &neg_shifts, &self.dims, &backend);
            let lock = g.write();
            coeus_ops::add_assign(lock, &unrolled, &backend);
        }
    }
}

/// Tracked circular shift along `dims` by `shifts`.
///
/// Backward: roll by negated shifts (unroll).
#[must_use]
#[inline]
pub fn roll<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    shifts: &[isize],
    dims: &[usize],
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let out_tensor = coeus_ops::roll(&input.tensor, shifts, dims, &backend);

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
        let node = RollNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![input.clone()],
            shifts: shifts.to_vec(),
            dims: dims.to_vec(),
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
