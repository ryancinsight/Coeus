// ── Autograd node: log_softmax ──
//
// Forward:
//   log_softmax(x)_i = x_i − max(x) − log(Σ_j exp(x_j − max(x)))
//
// Backward:
//   ∂L/∂x_i = g_i − softmax_i · Σ_j g_j
//
// Stores softmax probabilities (exp of log-probs) as Tensor<T,B> so that the
// backward pass can use coeus_ops tensor operations without raw slice access.

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for numerically-stable log-softmax.
///
/// Stores softmax probabilities (`exp(log_softmax(x))`) for the backward pass.
pub struct LogSoftmaxNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    /// softmax(x) = exp(log_softmax(x)), same shape as input.
    pub probs: Tensor<T, B>,
    pub axis: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for LogSoftmaxNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "log_softmax"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    // ∂L/∂x_i = g_i − p_i · Σ_j g_j
    //
    // Implemented as:
    //   sum_g  = sum_axis(g, axis)          — shape matches probs after broadcasting
    //   scaled = mul(probs, sum_g)           — element-wise, broadcasts sum_g along axis
    //   dx     = sub(g, scaled)              — element-wise
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let Some(Some(ref acc)) = input_grads.first() else {
            return;
        };
        let backend = B::default();
        // Σ_j g_j along axis; result shape has axis dimension reduced (broadcast-compatible)
        let sum_g = coeus_ops::sum_axis(grad_out, self.axis, &backend);
        // p_i · Σ_j g_j
        let scaled = coeus_ops::mul(&self.probs, &sum_g, &backend);
        // g_i − p_i · Σ_j g_j
        let dx = coeus_ops::sub(grad_out, &scaled, &backend);

        let lock = acc.write();
        coeus_ops::add_assign(lock, &dx, &backend);
    }
}

/// Tracked numerically-stable log-softmax.
///
/// # Arguments
/// - `input`: input variable, any shape
/// - `axis`: axis along which log-softmax is computed (negative indexing not supported here;
///   use `dim` in the public API wrapper in `ops::mod` if needed)
///
/// # Backward
/// `∂L/∂x_i = g_i − softmax_i · Σ_j g_j` — computed in `T` precision via tensor ops.
pub fn log_softmax<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    axis: usize,
) -> Var<T, B> {
    let backend = B::default();
    let ndim = input.tensor.ndim();
    assert!(
        axis < ndim,
        "log_softmax: axis {axis} out of bounds for ndim {ndim}"
    );

    // Forward: log-softmax values
    let log_prob_tensor = coeus_ops::log_softmax_axis(&input.tensor, axis, &backend);

    // softmax probs = exp(log_probs), stored for backward
    let probs = coeus_ops::exp(&log_prob_tensor, &backend);

    let requires_grad = crate::grad_mode::should_track_var(input);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            log_prob_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = LogSoftmaxNode {
            output_grad,
            inputs: vec![input.clone()],
            probs,
            axis,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Var {
        tensor: log_prob_tensor,
        grad,
        creator,
    }
}
