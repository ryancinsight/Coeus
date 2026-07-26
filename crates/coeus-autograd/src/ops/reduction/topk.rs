//! Differentiable top-k selection along an axis.
//!
//! # Topk backward
//! Scatters the k output gradients back to their original positions using the
//! topk indices. Positions not in the top-k receive zero gradient.
//! Equivalent to PyTorch's `torch.topk` backward.

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for `topk`.
///
/// Stores the selected indices so backward can use `scatter_add` to route the
/// top-k output gradients back to their original input positions.
pub struct TopkNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    /// Gradient buffer for the top-k output values.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// The single differentiable input.
    pub inputs: Vec<Var<T, B>>,
    /// Indices of the selected top-k elements, reused by `scatter_add` in backward.
    pub topk_indices: Tensor<T, B>,
    /// Axis along which topk was performed.
    pub dim: usize,
    /// Shape of the original input tensor.
    pub input_shape: Vec<usize>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for TopkNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "topk"
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

        let zeros = Tensor::zeros_on(self.input_shape.clone(), &backend);
        let grad_in =
            coeus_ops::scatter_add(&zeros, self.dim, &self.topk_indices, grad_out, &backend);

        let gl = g.write();
        coeus_ops::add_assign(gl, &grad_in, &backend);
    }
}

/// Differentiable top-k along `dim`, matching PyTorch `torch.topk`.
///
/// Returns `(top_values, top_indices)`. Only `top_values` tracks gradients;
/// `top_indices` is always detached (indices are not differentiable).
///
/// # Backward
/// Scatters `grad_out` (shape: input with `dim` → k) back to the original input
/// positions using `top_indices` via `scatter_add`. Positions outside the top-k
/// receive zero gradient, matching PyTorch's `torch.topk` backward.
///
/// # Panics
/// Panics if `k == 0`, `k > input.tensor.shape()[dim]`, or `dim >= input.tensor.ndim()`.
#[must_use]
pub fn topk<T: Scalar + leto_ops::Scalar, B>(
    input: &Var<T, B>,
    k: usize,
    dim: usize,
    largest: bool,
) -> (Var<T, B>, Var<T, B>)
where
    B: coeus_ops::BackendOps<T> + coeus_ops::BackendOps<i64> + Default,
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>:
        coeus_core::CpuAddressableStorage<i64> + coeus_core::CpuAddressableStorageMut<i64>,
{
    let backend = B::default();
    let (top_vals, idx_i64) = coeus_ops::topk(&input.tensor, k, dim, largest);

    let idx_data: Vec<T> = idx_i64
        .to_contiguous_on(&backend)
        .as_slice()
        .iter()
        .map(|&x| T::from_f64(x as f64))
        .collect();
    let top_indices = Tensor::from_slice_on(idx_i64.shape().to_vec(), &idx_data, &backend);

    let input_shape = input.tensor.shape_cloned().to_vec();
    let requires_grad = crate::grad_mode::should_track_var(input);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            top_vals.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };
    let creator = if requires_grad {
        let node = TopkNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![input.clone()],
            topk_indices: top_indices.clone(),
            dim,
            input_shape,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    let vals_var = Var {
        tensor: top_vals,
        grad,
        creator,
    };
    let idx_var = Var::new(top_indices, false);
    (vals_var, idx_var)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::MoiraiBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn topk_forward_and_backward_1d() {
        let data = vec![3.0f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let x = Var::<f64, MoiraiBackend>::new(Tensor::from_slice([7], &data), true);
        let (vals, _) = topk(&x, 3, 0, true);
        let vs = vals.tensor.as_slice().to_vec();
        assert_eq!(vs.len(), 3);
        let mut sorted = vs.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(sorted, vs, "topk should be descending");

        vals.backward();
        let dx = x.grad().unwrap();
        let ones: usize = dx
            .as_slice()
            .iter()
            .filter(|&&v| (v - 1.0).abs() < 1e-12)
            .count();
        let zeros: usize = dx.as_slice().iter().filter(|&&v| v.abs() < 1e-12).count();
        assert_eq!(ones, 3, "topk backward: expected 3 gradient-1 positions");
        assert_eq!(zeros, 4, "topk backward: expected 4 gradient-0 positions");
    }

    #[test]
    fn topk_backward_smallest() {
        let data = vec![3.0f64, 1.0, 4.0, 1.0, 5.0];
        let x = Var::<f64, MoiraiBackend>::new(Tensor::from_slice([5], &data), true);
        let (vals, _) = topk(&x, 2, 0, false);
        vals.backward();
        let dx = x.grad().unwrap();
        let grad_sum: f64 = dx.as_slice().iter().sum();
        assert!(
            (grad_sum - 2.0).abs() < 1e-12,
            "grad sum should be 2, got {grad_sum}"
        );
    }
}
