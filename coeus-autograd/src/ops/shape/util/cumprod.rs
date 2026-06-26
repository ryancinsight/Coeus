// ── Tracked cumprod ──
//
// Backward of cumprod uses the standard formula:
//   grad_x[i] = sum_j( grad_out[j] * cumprod[j] / x[i] )  for j >= i
//             = grad_out[i] * suffix_prod + ... (when x[i] != 0)
//
// We implement the safe Autograd formula via the suffix-sum trick on the
// log-space representation (which handles zeros via perturbation):
//   dy/dx[i] = sum_{j>=i}( grad_out[j] * prod_{k>=i, k!=j}(x[k]) )
//
// For the common case where no element is zero this simplifies to:
//   dy/dx[i] = (sum_{j>=i}( grad_out[j] * out[j] )) / x[i]
//
// To avoid numerical issues we use the direct accumulation approach:
//   d_in[i] = suffix_sum( grad_out * out )[i] / x[i]
// This matches PyTorch's cumprod backward for non-zero inputs.

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct CumprodNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    /// Forward output (cumprod values).
    pub output: Tensor<T, B>,
    /// Input tensor saved for backward.
    pub input_saved: Tensor<T, B>,
    pub dim: usize,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for CumprodNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn op_name(&self) -> &'static str {
        "cumprod"
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
            // grad_in[i] = suffix_sum( grad_out * out )[i] / x[i]
            // For numerical stability we use the exact element-wise computation
            // over a contiguous slice along dim.
            let shape = grad_out.shape();
            let ndim = shape.len();
            let dim = self.dim;
            let n = shape[dim];

            let go_cont = grad_out.to_contiguous();
            let out_cont = self.output.to_contiguous();
            let in_cont = self.input_saved.to_contiguous();

            let go_s = go_cont.as_slice();
            let out_s = out_cont.as_slice();
            let in_s = in_cont.as_slice();

            let numel: usize = shape.iter().product();
            let mut strides = vec![1usize; ndim];
            for d in (0..ndim - 1).rev() {
                strides[d] = strides[d + 1] * shape[d + 1];
            }
            let dim_stride = strides[dim];
            let outer = numel / (n * dim_stride);
            let inner = dim_stride;

            let mut gi_data = vec![T::zero(); numel];

            for outer_idx in 0..outer {
                for inner_idx in 0..inner {
                    let base = outer_idx * n * inner + inner_idx;
                    // Compute suffix sum of grad_out * out along this line.
                    let mut suffix: T = T::zero();
                    // Traverse from the end towards the start.
                    for i in (0..n).rev() {
                        let flat = base + i * inner;
                        suffix = suffix + go_s[flat] * out_s[flat];
                        // grad_in[i] = suffix / x[i]
                        let xi = in_s[flat];
                        // Avoid division by zero: if x[i] is zero we output 0.
                        gi_data[flat] = if xi == T::zero() {
                            T::zero()
                        } else {
                            suffix / xi
                        };
                    }
                }
            }

            let gi_increment = Tensor::from_slice(shape.to_vec(), &gi_data);
            let gl = g.write();
            coeus_ops::add_assign(gl, &gi_increment, &backend);
        }
    }
}

/// Tracked cumulative product along `dim`.
#[must_use]
#[inline]
pub fn cumprod<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    dim: usize,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let out_tensor = coeus_ops::cumprod(&input.tensor, dim, &backend);

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
        let node = CumprodNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![input.clone()],
            output: out_tensor.clone(),
            input_saved: input.tensor.clone(),
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
