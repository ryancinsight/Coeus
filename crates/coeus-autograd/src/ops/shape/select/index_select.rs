// ── Tracked index_select ──
//
// Backward: scatter gradient back to source positions (each output gradient
// element is added to the input gradient at the selected index position).
// Multiple selections of the same index accumulate (scatter-add semantics).

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct IndexSelectNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    /// The index tensor (f64-encoded integer indices).
    pub index_tensor: Tensor<T, B>,
    pub dim: usize,
    /// Input shape before selection (for scatter-back).
    pub input_shape: Vec<usize>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for IndexSelectNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn op_name(&self) -> &'static str {
        "index_select"
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
            // Scatter-add the output gradient back to input positions.
            // For each output flat index, find which input index it came from
            // and accumulate the gradient there.
            let in_shape = &self.input_shape;
            let ndim = in_shape.len();
            let dim = self.dim;

            let idx_cont = self.index_tensor.to_contiguous();
            let idx_s = idx_cont.as_slice();
            let go_cont = grad_out.to_contiguous();
            let go_s = go_cont.as_slice();

            // Build output shape from input_shape with dim replaced by k.
            let k = idx_s.len();
            let mut out_shape = in_shape.clone();
            out_shape[dim] = k;

            // Compute row-major strides for input and output.
            let mut in_strides = vec![1usize; ndim];
            for d in (0..ndim - 1).rev() {
                in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
            }
            let mut out_strides = vec![1usize; ndim];
            for d in (0..ndim - 1).rev() {
                out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
            }

            let gl = g.write();

            // Scatter output gradient back into input positions.
            let in_numel: usize = in_shape.iter().product();
            let mut gi_data = vec![T::zero(); in_numel];
            for (out_flat, &grad_out_element) in go_s.iter().enumerate() {
                let mut coords = vec![0usize; ndim];
                let mut rem = out_flat;
                for d in 0..ndim {
                    coords[d] = rem / out_strides[d];
                    rem %= out_strides[d];
                }
                let sel = <T as Scalar>::to_f64(idx_s[coords[dim]]) as usize;
                let mut in_flat = 0usize;
                for d in 0..ndim {
                    let c = if d == dim { sel } else { coords[d] };
                    in_flat += c * in_strides[d];
                }
                gi_data[in_flat] += grad_out_element;
            }

            // Accumulate increment into the gradient buffer.
            let gi_increment = Tensor::from_slice(in_shape.clone(), &gi_data);
            coeus_ops::add_assign(gl, &gi_increment, &backend)?;
        }
        Ok(())
    }
}

/// Tracked index selection along `dim`.
///
/// `index` must be 1-D. Backward scatters (accumulates) output gradients
/// back to the selected input positions.
#[must_use]
#[inline]
pub fn index_select<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    dim: usize,
    index: &Var<T, B>,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let out_tensor = coeus_ops::index_select(&input.tensor, dim, &index.tensor, &backend);

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
        let node = IndexSelectNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![input.clone()],
            index_tensor: index.tensor.clone(),
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
