use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for RMS normalization.
pub struct RMSNormNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Normalized feature dimension.
    pub d: usize,
    /// Captured weight tensor reshaped for broadcasting.
    pub w_reshaped_captured: Tensor<T, B>,
    /// Saved normalized input `x_hat = x / rms(x)` for backward.
    pub x_hat_clone: Tensor<T, B>,
    /// Saved RMS value for backward.
    pub rms_clone: Tensor<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for RMSNormNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "rmsnorm"
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

        // ── dL/dgamma = sum(dy * x_hat, dim=0) [D] ──
        if let Some(Some(ref gw)) = input_grads.get(1) {
            let dg_t = coeus_ops::sum_axis(
                &coeus_ops::mul(dy, &self.x_hat_clone, &backend),
                0,
                &backend,
            )
            .expect("invariant: rmsnorm gamma gradient axis is valid");
            let dg = dg_t.reshape([self.d]);
            let gl = gw.write();
            coeus_ops::add_assign(gl, &dg, &backend).expect("autograd gradient accumulation");
        }

        // ── dL/dx ──
        if let Some(Some(ref gx)) = input_grads.get(0) {
            let mut dy_w = coeus_ops::mul(dy, &self.w_reshaped_captured, &backend); // [N, D]
            let dy_w_xhat = coeus_ops::mul(&dy_w, &self.x_hat_clone, &backend); // [N, D]
            let scaled_sum = coeus_ops::mean_axis(&dy_w_xhat, 1, &backend)
                .expect("invariant: rmsnorm backward axis is valid"); // [N, 1]

            let term_prod = coeus_ops::mul(&self.x_hat_clone, &scaled_sum, &backend); // [N, D]
            coeus_ops::sub_assign(&mut dy_w, &term_prod, &backend)
                .expect("autograd gradient accumulation"); // [N, D]

            let mut dx = dy_w;
            coeus_ops::div_assign(&mut dx, &self.rms_clone, &backend)
                .expect("autograd gradient accumulation"); // [N, D]

            let gl = gx.write();
            coeus_ops::add_assign(gl, &dx, &backend).expect("autograd gradient accumulation");
        }
    }
}

/// Tracked RMS Normalization.
pub fn rmsnorm<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    out_tensor: Tensor<T, B>,
    x_hat: Tensor<T, B>,
    rms: Tensor<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    let requires_grad =
        crate::grad_mode::should_track_var(input) || crate::grad_mode::should_track_var(weight);
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
        let inputs = vec![input.clone(), weight.clone()];
        let d = weight.tensor.shape()[0];
        let w_reshaped_captured = weight.tensor.reshape([1, d]);
        let x_hat_clone = x_hat.clone();
        let rms_clone = rms.clone();

        let node = RMSNormNode {
            output_grad,
            inputs,
            d,
            w_reshaped_captured,
            x_hat_clone,
            rms_clone,
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
