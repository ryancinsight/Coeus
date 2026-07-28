use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Scalar, Shape};
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct SplitNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub offset: usize,
    pub size: usize,
    pub dim: usize,
    pub input_shape: Shape,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for SplitNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "split"
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
        let Some(Some(ref acc)) = input_grads.first() else {
            return Ok(());
        };

        let ndim = self.input_shape.len();
        let dim = self.dim;
        let ranges: Vec<(usize, usize)> = (0..ndim)
            .map(|d| {
                if d == dim {
                    (self.offset, self.offset + self.size)
                } else {
                    (0, self.input_shape[d])
                }
            })
            .collect();

        let lock = acc.write();
        let (parent_storage, parent_layout) = lock.storage_mut_and_layout()?;
        let sliced_layout = parent_layout.slice(&ranges);

        let parent_storage_imm: &B::DeviceBuffer<T> =
            unsafe { &*(parent_storage as *const B::DeviceBuffer<T>) };
        backend.elementwise_binary(
            coeus_ops::BinaryOp::Add,
            parent_storage_imm,
            &sliced_layout,
            grad_out.storage(),
            grad_out.layout(),
            parent_storage,
            &sliced_layout,
        )?;

        Ok(())
    }
}

/// Splits a tracked variable into chunks of size `chunk_size` along `dim`.
///
/// Backward propagation scatters each chunk gradient back into the parent input
/// gradient at the chunk's recorded offset.
///
/// # Panics
///
/// Panics when `chunk_size` is zero.
pub fn split<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    chunk_size: usize,
    dim: usize,
) -> Result<Vec<Var<T, B>>, B::Error>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    assert!(chunk_size > 0, "split: chunk_size must be > 0");
    let backend = B::default();
    let input_shape = x.tensor.shape_cloned();

    let chunks = coeus_ops::split(&x.tensor, chunk_size, dim)?;
    let requires_grad = crate::grad_mode::should_track_var(x);

    let mut results = Vec::with_capacity(chunks.len());
    let mut offset = 0usize;

    for chunk_tensor in chunks {
        let this_size = chunk_tensor.shape()[dim];

        if !requires_grad {
            results.push(Var::new(chunk_tensor, false)?);
            offset += this_size;
            continue;
        }

        let output_grad = Arc::new(GradBuffer::new(Tensor::zeros_on(
            chunk_tensor.shape_cloned(),
            &backend,
        )?));
        let grad = Some(output_grad.clone());

        let node = SplitNode {
            output_grad,
            inputs: vec![x.clone()],
            offset,
            size: this_size,
            dim,
            input_shape: input_shape.clone(),
        };
        let creator = Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>);

        results.push(Var {
            tensor: chunk_tensor,
            grad,
            creator,
        });
        offset += this_size;
    }
    Ok(results)
}
