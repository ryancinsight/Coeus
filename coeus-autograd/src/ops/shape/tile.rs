// ── Tracked tile ──
//
// Backward of tile(input, reps): sum the output gradient over each repeated
// copy. For each output element, its gradient contributes to exactly one
// input element; when reps > 1, multiple output elements share the same
// source so we sum their gradients.

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct TileNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub reps: Vec<usize>,
    pub in_shape: Vec<usize>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for TileNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn op_name(&self) -> &'static str {
        "tile"
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
            // Reconstruct the effective input shape after padding.
            let in_ndim = self.in_shape.len();
            let n = in_ndim.max(self.reps.len());
            let pad_in = n - in_ndim;
            let pad_reps = n - self.reps.len();
            let eff_in: Vec<usize> = (0..n)
                .map(|d| if d < pad_in { 1 } else { self.in_shape[d - pad_in] })
                .collect();
            let eff_reps: Vec<usize> = (0..n)
                .map(|d| if d < pad_reps { 1 } else { self.reps[d - pad_reps] })
                .collect();
            let out_shape: Vec<usize> = (0..n).map(|d| eff_in[d] * eff_reps[d]).collect();

            let go_cont = grad_out.to_contiguous();
            let go_s = go_cont.as_slice();

            let in_numel: usize = eff_in.iter().product();
            let out_numel: usize = out_shape.iter().product();

            // Row-major strides.
            let mut out_strides = vec![1usize; n];
            for d in (0..n - 1).rev() {
                out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
            }
            let mut in_strides = vec![1usize; n];
            for d in (0..n - 1).rev() {
                in_strides[d] = in_strides[d + 1] * eff_in[d + 1];
            }

            // Sum output gradients over repeated copies into the input gradient.
            let mut gi_data = vec![T::zero(); in_numel];
            for out_flat in 0..out_numel {
                let mut in_flat = 0usize;
                let mut rem = out_flat;
                for d in 0..n {
                    let out_coord = rem / out_strides[d];
                    rem %= out_strides[d];
                    let in_coord = out_coord % eff_in[d];
                    in_flat += in_coord * in_strides[d];
                }
                gi_data[in_flat] = gi_data[in_flat] + go_s[out_flat];
            }

            // Reshape from eff_in back to original in_shape.
            let gi_inc = if pad_in > 0 {
                Tensor::from_slice(self.in_shape.clone(), &gi_data[..self.in_shape.iter().product::<usize>()])
            } else {
                Tensor::from_slice(eff_in, &gi_data)
            };

            let gl = g.write();
            coeus_ops::add_assign(gl, &gi_inc, &backend);
        }
    }
}

/// Tracked tile: replicate `input` by `reps[d]` times along each dimension `d`.
#[must_use]
#[inline]
pub fn tile<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    reps: &[usize],
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let out_tensor = coeus_ops::tile(&input.tensor, reps, &backend);

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
        let node = TileNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![input.clone()],
            reps: reps.to_vec(),
            in_shape: input.tensor.shape().to_vec(),
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
