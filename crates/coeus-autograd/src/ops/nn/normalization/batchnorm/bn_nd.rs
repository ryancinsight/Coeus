use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

// ── Permute/reshape dispatch helpers ──

fn bn_permute_to_nhwc<T, B, const DIM: usize>(tensor: &Tensor<T, B>, backend: &B) -> Tensor<T, B>
where
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default,
{
    match DIM {
        1 => tensor.permute(&[0, 2, 1]).to_contiguous_on(backend),
        2 => tensor.permute(&[0, 2, 3, 1]).to_contiguous_on(backend),
        3 => tensor.permute(&[0, 2, 3, 4, 1]).to_contiguous_on(backend),
        _ => panic!("bn_permute_to_nhwc: unsupported DIM {DIM}"),
    }
}

fn bn_permute_from_nhwc<T, B, const DIM: usize>(tensor: &Tensor<T, B>, backend: &B) -> Tensor<T, B>
where
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default,
{
    match DIM {
        1 => tensor.permute(&[0, 2, 1]).to_contiguous_on(backend),
        2 => tensor.permute(&[0, 3, 1, 2]).to_contiguous_on(backend),
        3 => tensor.permute(&[0, 4, 1, 2, 3]).to_contiguous_on(backend),
        _ => panic!("bn_permute_from_nhwc: unsupported DIM {DIM}"),
    }
}

fn bn_reshape_to_flat<T, B, const DIM: usize>(
    tensor: Tensor<T, B>,
    m: usize,
    c: usize,
) -> Tensor<T, B>
where
    T: Scalar,
    B: coeus_ops::BackendOps<T>,
{
    tensor.reshape([m, c])
}

fn bn_reshape_from_flat<T, B, const DIM: usize>(
    tensor: Tensor<T, B>,
    n: usize,
    spatial: &[usize],
    c: usize,
) -> Tensor<T, B>
where
    T: Scalar,
    B: coeus_ops::BackendOps<T>,
{
    match DIM {
        1 => tensor.reshape([n, spatial[0], c]),
        2 => tensor.reshape([n, spatial[0], spatial[1], c]),
        3 => tensor.reshape([n, spatial[0], spatial[1], spatial[2], c]),
        _ => panic!("bn_reshape_from_flat: unsupported DIM {DIM}"),
    }
}

// ── Backward node ──

/// Autograd node for N-dimensional batch normalization.
pub struct BatchNormNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default, const DIM: usize> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Captured weight tensor reshaped for broadcasting.
    pub w_reshaped_captured: Tensor<T, B>,
    /// Saved normalized input `x_hat` for backward.
    pub x_hat_clone: Tensor<T, B>,
    /// Saved `(x - mean)` for backward.
    pub xmu_clone: Tensor<T, B>,
    /// Saved inverse standard deviation for backward.
    pub istdev_clone: Tensor<T, B>,
    /// Constant tensor holding `-0.5` for backward.
    pub minus_half: Tensor<T, B>,
    /// Constant tensor holding the normalization count `m`.
    pub m_const_captured: Tensor<T, B>,
    /// Constant tensor holding `2.0` for backward.
    pub two_const: Tensor<T, B>,
    /// Batch size.
    pub n: usize,
    /// Number of channels.
    pub c: usize,
    /// Spatial dimensions (padded with 0 for lower-D cases).
    pub spatial: [usize; 3],
    /// Normalization count `n * product(spatial)`.
    pub m: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default, const DIM: usize> BackwardNode<T, B>
    for BatchNormNode<T, B, DIM>
{
    #[inline]
    fn op_name(&self) -> &'static str {
        match DIM {
            1 => "batchnorm1d",
            2 => "batchnorm2d",
            3 => "batchnorm3d",
            _ => "batchnormNd",
        }
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

        let go_nhwc = bn_permute_to_nhwc::<T, B, DIM>(grad_out, &backend); // [N, ..., C]
        let go_flat = bn_reshape_to_flat::<T, B, DIM>(go_nhwc, self.m, self.c); // [M, C]

        // ── dL/dbeta = sum(dy, dim=0) [C] ──
        if let Some(Some(ref gb)) = input_grads.get(2) {
            let db_t = coeus_ops::sum_axis(&go_flat, 0, &backend)
                .expect("invariant: batchnorm beta gradient axis is valid"); // [1, C]
            let db = db_t.reshape([self.c]);
            let gl = gb.write();
            coeus_ops::add_assign(gl, &db, &backend).expect("autograd gradient accumulation");
        }

        // ── dL/dgamma = sum(dy * x_hat, dim=0) [C] ──
        if let Some(Some(ref gw_var)) = input_grads.get(1) {
            let dy_xhat = coeus_ops::mul(&go_flat, &self.x_hat_clone, &backend);
            let dg_t = coeus_ops::sum_axis(&dy_xhat, 0, &backend)
                .expect("invariant: batchnorm gamma gradient axis is valid"); // [1, C]
            let dg = dg_t.reshape([self.c]);
            let gl = gw_var.write();
            coeus_ops::add_assign(gl, &dg, &backend).expect("autograd gradient accumulation");
        }

        // ── dL/dx ──
        if let Some(Some(ref gx)) = input_grads.get(0) {
            let dxhat = coeus_ops::mul(&go_flat, &self.w_reshaped_captured, &backend); // [M, C]
            let sum_dxhat = coeus_ops::sum_axis(&dxhat, 0, &backend)
                .expect("invariant: batchnorm backward axis is valid"); // [1, C]
            let dxhat_xmu = coeus_ops::mul(&dxhat, &self.xmu_clone, &backend);
            let sum_dxhat_xmu = coeus_ops::sum_axis(&dxhat_xmu, 0, &backend)
                .expect("invariant: batchnorm backward axis is valid"); // [1, C]

            let mut istdev_cube = coeus_ops::mul(&self.istdev_clone, &self.istdev_clone, &backend);
            coeus_ops::mul_assign(&mut istdev_cube, &self.istdev_clone, &backend)
                .expect("autograd gradient accumulation");

            coeus_ops::mul_assign(&mut istdev_cube, &self.minus_half, &backend)
                .expect("autograd gradient accumulation");
            let dvar_scale = istdev_cube; // [1, C]

            let mut term3 = coeus_ops::mul(&self.istdev_clone, &sum_dxhat, &backend); // [1, C]
            coeus_ops::div_assign(&mut term3, &self.m_const_captured, &backend)
                .expect("autograd gradient accumulation"); // [1, C]

            let mut dvar_part = coeus_ops::mul(&dvar_scale, &sum_dxhat_xmu, &backend); // [1, C]
            coeus_ops::mul_assign(&mut dvar_part, &self.two_const, &backend)
                .expect("autograd gradient accumulation");
            coeus_ops::div_assign(&mut dvar_part, &self.m_const_captured, &backend)
                .expect("autograd gradient accumulation"); // [1, C]

            let term2 = coeus_ops::mul(&self.xmu_clone, &dvar_part, &backend); // [M, C]

            let mut term1 = coeus_ops::mul(&dxhat, &self.istdev_clone, &backend); // [M, C]
            coeus_ops::add_assign(&mut term1, &term2, &backend)
                .expect("autograd gradient accumulation");
            coeus_ops::sub_assign(&mut term1, &term3, &backend)
                .expect("autograd gradient accumulation");
            let dx_flat = term1; // [M, C]

            let dx_nhwc = bn_reshape_from_flat::<T, B, DIM>(dx_flat, self.n, &self.spatial, self.c);
            let dx_nchw = bn_permute_from_nhwc::<T, B, DIM>(&dx_nhwc, &backend);

            let gl = gx.write();
            coeus_ops::add_assign(gl, &dx_nchw, &backend).expect("autograd gradient accumulation");
        }
    }
}

// ── Args ──

/// Pre-computed intermediates and spatial dimensions for tracked batch normalization.
pub struct BatchNormArgs<T: Scalar, B: coeus_ops::BackendOps<T> + Default, const DIM: usize> {
    /// Normalized output tensor from the forward pass.
    pub out_tensor: Tensor<T, B>,
    /// Standardized input `(x - mean) / std` saved for backward.
    pub x_hat: Tensor<T, B>,
    /// Mean-centered input `x - mean` saved for backward.
    pub xmu: Tensor<T, B>,
    /// Reciprocal standard deviation `1 / sqrt(var + eps)` saved for backward.
    pub istdev: Tensor<T, B>,
    /// Scalar constant tensor holding `1 / m` for mean gradient computation.
    pub m_const: Tensor<T, B>,
    /// Scalar constant tensor holding `-0.5` for variance gradient computation.
    pub minus_half: Tensor<T, B>,
    /// Scalar constant tensor holding `2.0` for variance gradient computation.
    pub two_const: Tensor<T, B>,
    /// Batch size (number of samples).
    pub n: usize,
    /// Number of channels.
    pub c: usize,
    /// Spatial dimensions packed as `[d, h, w]` (unused dims set to 1).
    pub spatial: [usize; 3],
    /// Total number of elements per channel: `n * d * h * w`.
    pub m: usize,
}

// ── Tracked forward ──

fn batchnorm_nd_inner<T: Float, B: coeus_ops::BackendOps<T> + Default, const DIM: usize>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Var<T, B>,
    args: BatchNormArgs<T, B, DIM>,
) -> Var<T, B> {
    let backend = B::default();
    let requires_grad = crate::grad_mode::should_track_var(input)
        || crate::grad_mode::should_track_var(weight)
        || crate::grad_mode::should_track_var(bias);
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

        let node = BatchNormNode::<T, B, DIM> {
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
            spatial: args.spatial,
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

/// Tracked 1D Batch Normalization.
pub fn batchnorm1d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Var<T, B>,
    args: BatchNormArgs<T, B, 1>,
) -> Var<T, B> {
    batchnorm_nd_inner::<T, B, 1>(input, weight, bias, args)
}

/// Tracked 2D Batch Normalization.
pub fn batchnorm2d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Var<T, B>,
    args: BatchNormArgs<T, B, 2>,
) -> Var<T, B> {
    batchnorm_nd_inner::<T, B, 2>(input, weight, bias, args)
}

/// Tracked 3D Batch Normalization.
pub fn batchnorm3d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Var<T, B>,
    args: BatchNormArgs<T, B, 3>,
) -> Var<T, B> {
    batchnorm_nd_inner::<T, B, 3>(input, weight, bias, args)
}
