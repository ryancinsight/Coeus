use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for softmax, storing the output for backward.
pub struct SoftmaxNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Saved softmax output `y = softmax(x)` for backward.
    pub y_clone: Tensor<T, B>,
    /// Axis along which softmax was computed.
    pub dim_u: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for SoftmaxNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "softmax"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        if let Some(Some(ref g_in)) = input_grads.get(0) {
            accumulate_softmax_grad(grad_out, &self.y_clone, self.dim_u, g_in);
        }
    }
}

/// Softmax reverse pass `dx = y * (grad_out - sum_dim(y * grad_out))`, accumulated
/// into `g_in`.
///
/// Shared by [`SoftmaxNode`] and the masked/causal softmax nodes: their gradient is
/// the same jacobian applied to their (masked) output. Because masked positions hold
/// `y = 0`, they neither receive gradient (`dx = 0`) nor contribute to the per-row
/// sum, and an all-masked row (`y` all zero) propagates zero — matching the forward,
/// with no `-inf`/`NaN` ever formed.
pub(crate) fn accumulate_softmax_grad<T, B>(
    grad_out: &Tensor<T, B>,
    y: &Tensor<T, B>,
    dim_u: usize,
    g_in: &Arc<GradBuffer<T, B>>,
) where
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
{
    let backend = B::default();
    let gy = coeus_ops::mul(grad_out, y, &backend);
    let sum_gy = coeus_ops::sum_axis(&gy, dim_u, &backend)
        .expect("invariant: softmax backward axis matches the input rank");
    let mut dx = coeus_ops::sub(grad_out, &sum_gy, &backend);
    coeus_ops::mul_assign(&mut dx, y, &backend);
    let gl = g_in.write();
    coeus_ops::add_assign(gl, &dx, &backend);
}

/// Tracked Softmax.
pub fn softmax<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    dim: isize,
) -> Var<T, B> {
    let ndim = input.tensor.ndim();
    let dim_u = if dim < 0 {
        (ndim as isize + dim) as usize
    } else {
        dim as usize
    };
    assert!(
        dim_u < ndim,
        "softmax dim {dim} out of bounds for ndim={ndim}"
    );
    let backend = B::default();

    let max_t = coeus_ops::max_axis(&input.tensor, dim_u, &backend)
        .expect("invariant: softmax axis is validated");
    let shift_x = coeus_ops::sub(&input.tensor, &max_t, &backend);
    let exp_x_t = coeus_ops::exp(&shift_x, &backend);
    let sum_t = coeus_ops::sum_axis(&exp_x_t, dim_u, &backend)
        .expect("invariant: softmax axis is validated");
    let y_t = coeus_ops::div(&exp_x_t, &sum_t, &backend);

    let requires_grad = crate::grad_mode::should_track_var(input);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            y_t.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = vec![input.clone()];
        let y_clone = y_t.clone();

        let node = SoftmaxNode {
            output_grad,
            inputs,
            y_clone,
            dim_u,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Var {
        tensor: y_t,
        grad,
        creator,
    }
}

/// Tracked Softmin over `dim` — `softmax(-input)` (`torch.nn.functional.softmin`).
///
/// Differentiable via composition of the tracked `neg` and [`softmax`].
#[must_use]
pub fn softmin<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    dim: isize,
) -> Var<T, B> {
    softmax(&crate::ops::neg(input), dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::MoiraiBackend;

    #[test]
    fn softmin_equals_softmax_of_negation_and_is_differentiable() {
        let x = Var::<f64, MoiraiBackend>::new(Tensor::from_slice([3], &[1.0, 2.0, 3.0]), true);
        let out = softmin(&x, 0);

        let neg =
            Var::<f64, MoiraiBackend>::new(Tensor::from_slice([3], &[-1.0, -2.0, -3.0]), false);
        let reference = softmax(&neg, 0);
        for (i, (&a, &b)) in out
            .tensor
            .as_slice()
            .iter()
            .zip(reference.tensor.as_slice())
            .enumerate()
        {
            assert!((a - b).abs() < 1e-12, "softmin[{i}]: {a} vs {b}");
        }

        // Smallest input receives the largest weight; distribution sums to 1.
        let y = out.tensor.as_slice();
        assert!(
            y[0] > y[1] && y[1] > y[2],
            "softmin must rank inversely to input"
        );
        assert!((y.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        out.backward();
        assert!(x.grad().is_some(), "softmin must be differentiable");
    }
}
