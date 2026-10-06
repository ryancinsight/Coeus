use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Scalar, Shape};
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct CatNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub split_sizes: Vec<usize>,
    pub dim: usize,
    pub out_shape: Shape,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for CatNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "cat"
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
        let ndim = self.out_shape.len();
        let dim = self.dim;
        let mut offset = 0usize;

        for (&sz, acc) in self.split_sizes.iter().zip(input_grads.iter()) {
            let Some(ref g) = *acc else {
                offset += sz;
                continue;
            };

            let ranges: Vec<(usize, usize)> = (0..ndim)
                .map(|d| {
                    if d == dim {
                        (offset, offset + sz)
                    } else {
                        (0, self.out_shape[d])
                    }
                })
                .collect();

            let lock = g.write();
            let (g_storage, g_layout) = lock.storage_and_layout_mut();
            let sliced_out_layout = grad_out.layout().slice(&ranges);

            backend.elementwise_binary_update(
                coeus_ops::BinaryOp::Add,
                g_storage,
                g_layout,
                grad_out.storage(),
                &sliced_out_layout,
            )?;
            offset += sz;
        }
        Ok(())
    }
}

/// Concatenates tracked variables along `dim`, propagating gradients to each input.
///
/// Backward propagation slices the output gradient by each input's extent along
/// `dim` and accumulates the matching slice into that input's gradient buffer.
///
/// # Panics
///
/// Panics when `inputs` is empty.
pub fn cat<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    inputs: &[&Var<T, B>],
    dim: usize,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    assert!(!inputs.is_empty(), "cat: inputs must be non-empty");
    let backend = B::default();

    let split_sizes: Vec<usize> = inputs.iter().map(|v| v.tensor.shape()[dim]).collect();
    let tensors: Vec<&Tensor<T, B>> = inputs.iter().map(|v| &v.tensor).collect();
    let out_tensor = coeus_ops::cat(&tensors, dim);

    let requires_grad = inputs.iter().any(|v| crate::grad_mode::should_track_var(v));
    if !requires_grad {
        return Var::new(out_tensor, false);
    }

    let out_shape = out_tensor.shape_cloned();
    let output_grad = Arc::new(GradBuffer::new(Tensor::zeros_on(
        out_shape.clone(),
        &backend,
    )));
    let grad = Some(output_grad.clone());

    let node = CatNode {
        output_grad,
        inputs: inputs.iter().map(|v| (*v).clone()).collect(),
        split_sizes,
        dim,
        out_shape,
    };
    let creator = Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>);

    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}
