//! Differentiable sort along an axis.
//!
//! # Sort backward
//! Scatters the output gradient back to original unsorted positions using the
//! argsort indices returned during the forward pass. Equivalent to PyTorch's
//! `torch.sort` backward.

// ── Autograd nodes: sort ──
//
// # sort
//   Forward: stable sort values and return argsort indices along `dim`.
//   Backward: scatter output gradient back to original positions via the argsort
//     indices (inverse permutation scatter).
//
//   grad_in[..orig_pos..] = grad_out[..sorted_pos..]
//
//   Implemented as `scatter_add(zeros_like(input), dim, sort_indices, grad_out)`
//   which maps sorted gradient positions back to original positions.

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for `sort`.
///
/// Stores the argsort indices so backward can use `scatter_add` to move sorted
/// output gradients back to their original unsorted positions.
pub struct SortNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    /// Gradient buffer for the sorted output.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// The single differentiable input (values to be sorted).
    pub inputs: Vec<Var<T, B>>,
    /// Argsort indices returned by forward, reused by `scatter_add` in backward.
    pub sort_indices: Tensor<T, B>,
    /// Axis along which the sort was performed.
    pub dim: usize,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for SortNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "sort"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        let Some(Some(ref g)) = input_grads.first() else {
            return;
        };

        // Scatter grad_out[sorted_pos] -> original_pos using sort_indices.
        // scatter_add(zeros, dim, sort_indices, grad_out) performs:
        //   result[..sort_indices[i].., ..] += grad_out[..i.., ..]
        // This is the inverse of gather, routing sorted grads back to input.
        let input_shape = self.sort_indices.shape_cloned();
        let zeros = Tensor::zeros_on(input_shape, &backend);
        let grad_in =
            coeus_ops::scatter_add(&zeros, self.dim, &self.sort_indices, grad_out, &backend);

        let gl = g.write();
        coeus_ops::add_assign(gl, &grad_in, &backend).expect("autograd gradient accumulation");
    }
}

/// Differentiable sort along `dim`, similar to PyTorch `torch.sort`.
///
/// Returns `(sorted_values, sort_indices)`.  Only `sorted_values` tracks
/// gradients; `sort_indices` is not differentiable and is returned as a
/// detached tensor.
///
/// # Backward
/// Scatters `grad_out` back to original positions using `sort_indices` via
/// `scatter_add`, exactly matching PyTorch's sort backward semantics.
///
/// # Panics
/// Panics if `dim >= input.tensor.ndim()`.
#[must_use]
pub fn sort<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    dim: usize,
    descending: bool,
) -> (Var<T, B>, Var<T, B>)
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let (sorted_vals, sort_indices) = coeus_ops::sort(&input.tensor, dim, descending, &backend);

    let requires_grad = crate::grad_mode::should_track_var(input);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            sorted_vals.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };
    let creator = if requires_grad {
        let node = SortNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![input.clone()],
            sort_indices: sort_indices.clone(),
            dim,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    let sorted_var = Var {
        tensor: sorted_vals,
        grad,
        creator,
    };
    let indices_var = Var::new(sort_indices, false);
    (sorted_var, indices_var)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::MoiraiBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn sort_forward_and_backward_1d() {
        let data = vec![3.0f64, 1.0, 4.0, 1.0, 5.0];
        let x = Var::<f64, MoiraiBackend>::new(Tensor::from_slice([5], &data), true);
        let (sorted, indices) = sort(&x, 0, false);
        // sorted ascending: [1, 1, 3, 4, 5]
        let s = sorted.tensor.as_slice().to_vec();
        assert!(
            s[0] <= s[1] && s[1] <= s[2] && s[2] <= s[3] && s[3] <= s[4],
            "not ascending: {s:?}"
        );
        // indices are not tracked
        assert!(indices.grad.is_none());

        // grad_out = [1, 1, 1, 1, 1]; scatter back via sort_indices
        sorted.backward();
        let dx = x.grad().unwrap();
        let dx_sum: f64 = dx.as_slice().iter().sum();
        assert!(
            (dx_sum - 5.0).abs() < 1e-12,
            "sum of grad should be 5.0, got {dx_sum}"
        );
    }

    #[test]
    fn sort_backward_dim1() {
        let data = vec![3.0f64, 1.0, 4.0, 2.0, 5.0, 0.0];
        let x = Var::<f64, MoiraiBackend>::new(Tensor::from_slice([2, 3], &data), true);
        let (sorted, _) = sort(&x, 1, false);
        sorted.backward();
        let dx = x.grad().unwrap();
        // Each element should receive exactly 1 gradient
        for v in dx.as_slice() {
            assert!((*v - 1.0).abs() < 1e-12, "expected grad 1.0, got {v}");
        }
    }
}
