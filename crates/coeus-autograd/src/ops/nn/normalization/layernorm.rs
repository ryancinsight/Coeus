use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for layer normalization.
pub struct LayerNormNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Normalized feature dimension.
    pub d: usize,
    /// Captured weight tensor reshaped for broadcasting.
    pub w_reshaped_captured: Tensor<T, B>,
    /// Saved normalized input `x_hat = (x - mean) / std` for backward.
    pub x_hat_clone: Tensor<T, B>,
    /// Saved inverse standard deviation for backward.
    pub istdev_clone: Tensor<T, B>,
    /// Constant tensor holding the feature dimension `d`.
    pub d_const: Tensor<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for LayerNormNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "layernorm"
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
        let backend = B::default();
        let dy = grad_out; // [N, D]
        let mut dy_w = coeus_ops::mul(dy, &self.w_reshaped_captured, &backend);
        if let Some(Some(ref gw)) = input_grads.get(1) {
            let dg_t = coeus_ops::sum_axis(
                &coeus_ops::mul(dy, &self.x_hat_clone, &backend),
                0,
                &backend,
            );
            let dg = dg_t.reshape([self.d]);
            let gl = gw.write();
            coeus_ops::add_assign(gl, &dg, &backend);
        }

        // ── dL/dbeta = sum(dy, dim=0) [D] ──
        if let Some(Some(ref gb)) = input_grads.get(2) {
            let db_t = coeus_ops::sum_axis(dy, 0, &backend);
            let db = db_t.reshape([self.d]);
            let gl = gb.write();
            coeus_ops::add_assign(gl, &db, &backend);
        }

        // ── dL/dx ──
        if let Some(Some(ref gx)) = input_grads.get(0) {
            let sum_dy_w = coeus_ops::sum_axis(&dy_w, 1, &backend); // [N, 1]
            let dy_w_xhat = coeus_ops::mul(&dy_w, &self.x_hat_clone, &backend); // [N, D]
            let sum_dy_w_xhat = coeus_ops::sum_axis(&dy_w_xhat, 1, &backend); // [N, 1]

            // term2 = x_hat * sum_dy_w_xhat + sum_dy_w
            let mut term2 = coeus_ops::mul(&self.x_hat_clone, &sum_dy_w_xhat, &backend); // [N, D]
            coeus_ops::add_assign(&mut term2, &sum_dy_w, &backend);

            // dy_w = dy_w * d_const
            coeus_ops::mul_assign(&mut dy_w, &self.d_const, &backend);

            // term = dy_w - term2
            coeus_ops::sub_assign(&mut dy_w, &term2, &backend);
            let mut term = dy_w;

            // dx = term * istdev_clone / d_const
            coeus_ops::mul_assign(&mut term, &self.istdev_clone, &backend);
            coeus_ops::div_assign(&mut term, &self.d_const, &backend);
            let dx = term;

            let gl = gx.write();
            coeus_ops::add_assign(gl, &dx, &backend);
        }
    }
}

/// Tracked Layer Normalization.
pub fn layernorm<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Var<T, B>,
    out_tensor: Tensor<T, B>,
    x_hat: Tensor<T, B>,
    istdev: Tensor<T, B>,
    d_const: Tensor<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    let requires_grad = crate::grad_mode::should_track_var(input)
        || crate::grad_mode::should_track_var(weight)
        || crate::grad_mode::should_track_var(bias);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = vec![input.clone(), weight.clone(), bias.clone()];
        let d = weight.tensor.shape()[0];
        let w_reshaped_captured = weight.tensor.reshape([1, d]);
        let x_hat_clone = x_hat.clone();
        let istdev_clone = istdev.clone();
        let d_const = d_const.clone();

        let node = LayerNormNode {
            output_grad,
            inputs,
            d,
            w_reshaped_captured,
            x_hat_clone,
            istdev_clone,
            d_const,
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
