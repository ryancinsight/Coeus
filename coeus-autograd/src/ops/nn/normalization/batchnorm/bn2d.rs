use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct BatchNorm2dNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub w_reshaped_captured: Tensor<T, B>,
    pub x_hat_clone: Tensor<T, B>,
    pub xmu_clone: Tensor<T, B>,
    pub istdev_clone: Tensor<T, B>,
    pub minus_half: Tensor<T, B>,
    pub m_const_captured: Tensor<T, B>,
    pub two_const: Tensor<T, B>,
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub m: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for BatchNorm2dNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "batchnorm2d"
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

        let go_nhwc = grad_out.permute(&[0, 2, 3, 1]).to_contiguous_on(&backend); // [N, H, W, C]
        let go_flat = go_nhwc.reshape([self.m, self.c]); // [M, C]

        // ── dL/dbeta = sum(dy, dim=0) [C] ──
        if let Some(Some(ref gb)) = input_grads.get(2) {
            let db_t = coeus_ops::sum_axis(&go_flat, 0, &backend); // [1, C]
            let db = db_t.reshape([self.c]);
            let gl = gb.write();
            coeus_ops::add_assign(gl, &db, &backend);
        }

        // ── dL/dgamma = sum(dy * x_hat, dim=0) [C] ──
        if let Some(Some(ref gw_var)) = input_grads.get(1) {
            let dy_xhat = coeus_ops::mul(&go_flat, &self.x_hat_clone, &backend);
            let dg_t = coeus_ops::sum_axis(&dy_xhat, 0, &backend); // [1, C]
            let dg = dg_t.reshape([self.c]);
            let gl = gw_var.write();
            coeus_ops::add_assign(gl, &dg, &backend);
        }

        // ── dL/dx ──
        if let Some(Some(ref gx)) = input_grads.get(0) {
            let dxhat = coeus_ops::mul(&go_flat, &self.w_reshaped_captured, &backend); // [M, C]
            let sum_dxhat = coeus_ops::sum_axis(&dxhat, 0, &backend); // [1, C]
            let dxhat_xmu = coeus_ops::mul(&dxhat, &self.xmu_clone, &backend);
            let sum_dxhat_xmu = coeus_ops::sum_axis(&dxhat_xmu, 0, &backend); // [1, C]

            let mut istdev_cube = coeus_ops::mul(&self.istdev_clone, &self.istdev_clone, &backend);
            coeus_ops::mul_assign(&mut istdev_cube, &self.istdev_clone, &backend);

            coeus_ops::mul_assign(&mut istdev_cube, &self.minus_half, &backend);
            let dvar_scale = istdev_cube; // [1, C]

            let mut term3 = coeus_ops::mul(&self.istdev_clone, &sum_dxhat, &backend); // [1, C]
            coeus_ops::div_assign(&mut term3, &self.m_const_captured, &backend); // [1, C]

            let mut dvar_part = coeus_ops::mul(&dvar_scale, &sum_dxhat_xmu, &backend); // [1, C]
            coeus_ops::mul_assign(&mut dvar_part, &self.two_const, &backend);
            coeus_ops::div_assign(&mut dvar_part, &self.m_const_captured, &backend); // [1, C]

            let term2 = coeus_ops::mul(&self.xmu_clone, &dvar_part, &backend); // [M, C]

            let mut term1 = coeus_ops::mul(&dxhat, &self.istdev_clone, &backend); // [M, C]
            coeus_ops::add_assign(&mut term1, &term2, &backend);
            coeus_ops::sub_assign(&mut term1, &term3, &backend);
            let dx_flat = term1; // [M, C]

            let dx_nhwc = dx_flat.reshape([self.n, self.h, self.w, self.c]);
            let dx_nchw = dx_nhwc.permute(&[0, 3, 1, 2]).to_contiguous_on(&backend);

            let gl = gx.write();
            coeus_ops::add_assign(gl, &dx_nchw, &backend);
        }
    }
}

/// Pre-computed intermediates and spatial dimensions for [`batchnorm2d`].
///
/// Groups the tensor results produced during the forward pass (needed for backward)
/// together with the shape scalars `n`, `c`, `h`, `w`, `m` so that `batchnorm2d`
/// accepts a single typed descriptor instead of fifteen positional arguments.
pub struct BatchNorm2dArgs<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Forward-pass output tensor `[N, C, H, W]`.
    pub out_tensor: Tensor<T, B>,
    /// Normalised input `x_hat = (x - mu) / sqrt(var + eps)`, shape `[M, C]`.
    pub x_hat: Tensor<T, B>,
    /// Centred input `x - mu`, shape `[M, C]`.
    pub xmu: Tensor<T, B>,
    /// Inverse standard deviation `1 / sqrt(var + eps)`, shape `[1, C]`.
    pub istdev: Tensor<T, B>,
    /// Scalar tensor holding the batch count `M = N * H * W`, shape `[1]`.
    pub m_const: Tensor<T, B>,
    /// Constant tensor holding `-0.5`, shape `[1]`.
    pub minus_half: Tensor<T, B>,
    /// Constant tensor holding `2.0`, shape `[1]`.
    pub two_const: Tensor<T, B>,
    /// Batch size.
    pub n: usize,
    /// Channel count.
    pub c: usize,
    /// Spatial height.
    pub h: usize,
    /// Spatial width.
    pub w: usize,
    /// Spatial batch size `N * H * W`.
    pub m: usize,
}

/// Tracked 2D Batch Normalization.
pub fn batchnorm2d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Var<T, B>,
    args: BatchNorm2dArgs<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    let requires_grad = input.grad.is_some() || weight.grad.is_some() || bias.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            args.out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = vec![input.clone(), weight.clone(), bias.clone()];
        let w_reshaped_captured = weight.tensor.reshape([1, args.c]);

        let node = BatchNorm2dNode {
            output_grad,
            inputs,
            w_reshaped_captured,
            x_hat_clone: args.x_hat,
            xmu_clone: args.xmu,
            istdev_clone: args.istdev,
            minus_half: args.minus_half,
            m_const_captured: args.m_const,
            two_const: args.two_const,
            n: args.n,
            c: args.c,
            h: args.h,
            w: args.w,
            m: args.m,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Var {
        tensor: args.out_tensor,
        grad,
        creator,
    }
}
