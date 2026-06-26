// ── Tracked broadcast_to ──
//
// Backward reduces the output gradient over every dimension that was
// broadcast from size 1 in the input.

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct BroadcastNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub broadcast_dims: Vec<usize>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for BroadcastNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn op_name(&self) -> &'static str {
        "broadcast_to"
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
            let mut grad_input = grad_out.clone();
            for &axis in &self.broadcast_dims {
                grad_input = coeus_ops::sum_axis(&grad_input, axis, &backend);
            }
            let lock = g.write();
            coeus_ops::add_assign(lock, &grad_input, &backend);
        }
    }
}

/// Tracked broadcast materialization with rank-preserving broadcasting.
///
/// # Panics
/// Panics if `target_shape.len() != input.ndim()` or if any dimension is
/// incompatible for broadcasting.
#[must_use]
#[inline]
pub fn broadcast_to<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    target_shape: impl Into<Vec<usize>>,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let target_shape = target_shape.into();

    assert_eq!(
        target_shape.len(),
        input.tensor.ndim(),
        "broadcast_to: target rank {} must match input rank {}",
        target_shape.len(),
        input.tensor.ndim(),
    );

    let broadcast_dims: Vec<usize> = input
        .tensor
        .shape()
        .iter()
        .zip(target_shape.iter())
        .enumerate()
        .filter_map(|(dim, (&src, &dst))| {
            if src == dst {
                None
            } else if src == 1 {
                Some(dim)
            } else {
                panic!(
                    "broadcast_to: incompatible dimension at axis {dim}: input={src}, target={dst}"
                );
            }
        })
        .collect();

    let out_tensor = coeus_ops::broadcast_to(&input.tensor, &target_shape, &backend);

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
        let node = BroadcastNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![input.clone()],
            broadcast_dims,
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
