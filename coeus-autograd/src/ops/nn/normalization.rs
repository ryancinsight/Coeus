use std::sync::{Arc, Mutex};
use coeus_core::{Scalar, Float};
use coeus_tensor::Tensor;
use crate::node::BackwardNode;
use crate::var::Var;

pub struct LayerNormNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    pub d: usize,
    pub w_reshaped_captured: Tensor<T, B>,
    pub x_hat_clone: Tensor<T, B>,
    pub istdev_clone: Tensor<T, B>,
    pub d_const: Tensor<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for LayerNormNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "layernorm"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();
        let dy = grad_out; // [N, D]
        let mut dy_w = coeus_ops::mul(dy, &self.w_reshaped_captured, &backend);
        if let Some(Some(ref gw)) = input_grads.get(1) {
            let dg_t = coeus_ops::sum_axis(&coeus_ops::mul(dy, &self.x_hat_clone, &backend), 0, &backend);
            let dg = dg_t.reshape([self.d]);
            let mut gl = gw.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &dg, &backend);
        }

        // ── dL/dbeta = sum(dy, dim=0) [D] ──
        if let Some(Some(ref gb)) = input_grads.get(2) {
            let db_t = coeus_ops::sum_axis(dy, 0, &backend);
            let db = db_t.reshape([self.d]);
            let mut gl = gb.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &db, &backend);
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

            let mut gl = gx.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &dx, &backend);
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
    let requires_grad = input.grad.is_some() || weight.grad.is_some() || bias.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(out_tensor.shape_cloned(), &backend))))
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

    Var { tensor: out_tensor, grad, creator }
}

pub struct RMSNormNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    pub d: usize,
    pub w_reshaped_captured: Tensor<T, B>,
    pub x_hat_clone: Tensor<T, B>,
    pub rms_clone: Tensor<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for RMSNormNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "rmsnorm"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();
        let dy = grad_out; // [N, D]

        // ── dL/dgamma = sum(dy * x_hat, dim=0) [D] ──
        if let Some(Some(ref gw)) = input_grads.get(1) {
            let dg_t = coeus_ops::sum_axis(&coeus_ops::mul(dy, &self.x_hat_clone, &backend), 0, &backend);
            let dg = dg_t.reshape([self.d]);
            let mut gl = gw.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &dg, &backend);
        }

        // ── dL/dx ──
        if let Some(Some(ref gx)) = input_grads.get(0) {
            let mut dy_w = coeus_ops::mul(dy, &self.w_reshaped_captured, &backend); // [N, D]
            let dy_w_xhat = coeus_ops::mul(&dy_w, &self.x_hat_clone, &backend); // [N, D]
            let scaled_sum = coeus_ops::mean_axis(&dy_w_xhat, 1, &backend); // [N, 1]

            let term_prod = coeus_ops::mul(&self.x_hat_clone, &scaled_sum, &backend); // [N, D]
            coeus_ops::sub_assign(&mut dy_w, &term_prod, &backend); // [N, D]

            let mut dx = dy_w;
            coeus_ops::div_assign(&mut dx, &self.rms_clone, &backend); // [N, D]

            let mut gl = gx.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &dx, &backend);
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
    let requires_grad = input.grad.is_some() || weight.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(out_tensor.shape_cloned(), &backend))))
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

    Var { tensor: out_tensor, grad, creator }
}

pub struct BatchNorm1dNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
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
    pub l: usize,
    pub m: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for BatchNorm1dNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "batchnorm1d"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();

        let go_nlc = grad_out.permute(&[0, 2, 1]).to_contiguous_on(&backend); // [N, L, C]
        let go_flat = go_nlc.reshape([self.m, self.c]); // [M, C]

        // ── dL/dbeta = sum(dy, dim=0) [C] ──
        if let Some(Some(ref gb)) = input_grads.get(2) {
            let db_t = coeus_ops::sum_axis(&go_flat, 0, &backend); // [1, C]
            let db = db_t.reshape([self.c]);
            let mut gl = gb.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &db, &backend);
        }

        // ── dL/dgamma = sum(dy * x_hat, dim=0) [C] ──
        if let Some(Some(ref gw_var)) = input_grads.get(1) {
            let dy_xhat = coeus_ops::mul(&go_flat, &self.x_hat_clone, &backend);
            let dg_t = coeus_ops::sum_axis(&dy_xhat, 0, &backend); // [1, C]
            let dg = dg_t.reshape([self.c]);
            let mut gl = gw_var.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &dg, &backend);
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

            let dx_nlc = dx_flat.reshape([self.n, self.l, self.c]);
            let dx_ncl = dx_nlc.permute(&[0, 2, 1]).to_contiguous_on(&backend);

            let mut gl = gx.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &dx_ncl, &backend);
        }
    }
}

/// Tracked 1D Batch Normalization.
#[allow(clippy::too_many_arguments)]
pub fn batchnorm1d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Var<T, B>,
    out_tensor: Tensor<T, B>,
    x_hat: Tensor<T, B>,
    xmu: Tensor<T, B>,
    istdev: Tensor<T, B>,
    m_const: Tensor<T, B>,
    minus_half: Tensor<T, B>,
    two_const: Tensor<T, B>,
    n: usize,
    c: usize,
    l: usize,
    m: usize,
) -> Var<T, B> {
    let backend = B::default();
    let requires_grad = input.grad.is_some() || weight.grad.is_some() || bias.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(out_tensor.shape_cloned(), &backend))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = vec![input.clone(), weight.clone(), bias.clone()];
        let w_reshaped_captured = weight.tensor.reshape([1, c]);
        let x_hat_clone = x_hat.clone();
        let xmu_clone = xmu.clone();
        let istdev_clone = istdev.clone();
        let minus_half = minus_half.clone();
        let m_const_captured = m_const.clone();
        let two_const = two_const.clone();

        let node = BatchNorm1dNode {
            output_grad,
            inputs,
            w_reshaped_captured,
            x_hat_clone,
            xmu_clone,
            istdev_clone,
            minus_half,
            m_const_captured,
            two_const,
            n,
            c,
            l,
            m,
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

pub struct BatchNorm2dNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
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
    fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();

        let go_nhwc = grad_out.permute(&[0, 2, 3, 1]).to_contiguous_on(&backend); // [N, H, W, C]
        let go_flat = go_nhwc.reshape([self.m, self.c]); // [M, C]

        // ── dL/dbeta = sum(dy, dim=0) [C] ──
        if let Some(Some(ref gb)) = input_grads.get(2) {
            let db_t = coeus_ops::sum_axis(&go_flat, 0, &backend); // [1, C]
            let db = db_t.reshape([self.c]);
            let mut gl = gb.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &db, &backend);
        }

        // ── dL/dgamma = sum(dy * x_hat, dim=0) [C] ──
        if let Some(Some(ref gw_var)) = input_grads.get(1) {
            let dy_xhat = coeus_ops::mul(&go_flat, &self.x_hat_clone, &backend);
            let dg_t = coeus_ops::sum_axis(&dy_xhat, 0, &backend); // [1, C]
            let dg = dg_t.reshape([self.c]);
            let mut gl = gw_var.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &dg, &backend);
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

            let mut gl = gx.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &dx_nchw, &backend);
        }
    }
}

/// Tracked 2D Batch Normalization.
#[allow(clippy::too_many_arguments)]
pub fn batchnorm2d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Var<T, B>,
    out_tensor: Tensor<T, B>,
    x_hat: Tensor<T, B>,
    xmu: Tensor<T, B>,
    istdev: Tensor<T, B>,
    m_const: Tensor<T, B>,
    minus_half: Tensor<T, B>,
    two_const: Tensor<T, B>,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    m: usize,
) -> Var<T, B> {
    let backend = B::default();
    let requires_grad = input.grad.is_some() || weight.grad.is_some() || bias.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(out_tensor.shape_cloned(), &backend))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = vec![input.clone(), weight.clone(), bias.clone()];
        let w_reshaped_captured = weight.tensor.reshape([1, c]);
        let x_hat_clone = x_hat.clone();
        let xmu_clone = xmu.clone();
        let istdev_clone = istdev.clone();
        let minus_half = minus_half.clone();
        let m_const_captured = m_const.clone();
        let two_const = two_const.clone();

        let node = BatchNorm2dNode {
            output_grad,
            inputs,
            w_reshaped_captured,
            x_hat_clone,
            xmu_clone,
            istdev_clone,
            minus_half,
            m_const_captured,
            two_const,
            n,
            c,
            h,
            w,
            m,
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

pub struct BatchNorm3dNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
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
    pub d: usize,
    pub h: usize,
    pub w: usize,
    pub m: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for BatchNorm3dNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "batchnorm3d"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();

        let go_ndhwc = grad_out.permute(&[0, 2, 3, 4, 1]).to_contiguous_on(&backend); // [N, D, H, W, C]
        let go_flat = go_ndhwc.reshape([self.m, self.c]); // [M, C]

        // ── dL/dbeta = sum(dy, dim=0) [C] ──
        if let Some(Some(ref gb)) = input_grads.get(2) {
            let db_t = coeus_ops::sum_axis(&go_flat, 0, &backend); // [1, C]
            let db = db_t.reshape([self.c]);
            let mut gl = gb.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &db, &backend);
        }

        // ── dL/dgamma = sum(dy * x_hat, dim=0) [C] ──
        if let Some(Some(ref gw_var)) = input_grads.get(1) {
            let dy_xhat = coeus_ops::mul(&go_flat, &self.x_hat_clone, &backend);
            let dg_t = coeus_ops::sum_axis(&dy_xhat, 0, &backend); // [1, C]
            let dg = dg_t.reshape([self.c]);
            let mut gl = gw_var.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &dg, &backend);
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

            let dx_ndhwc = dx_flat.reshape([self.n, self.d, self.h, self.w, self.c]);
            let dx_ncdhw = dx_ndhwc.permute(&[0, 4, 1, 2, 3]).to_contiguous_on(&backend);

            let mut gl = gx.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &dx_ncdhw, &backend);
        }
    }
}

/// Tracked 3D Batch Normalization.
#[allow(clippy::too_many_arguments)]
pub fn batchnorm3d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Var<T, B>,
    out_tensor: Tensor<T, B>,
    x_hat: Tensor<T, B>,
    xmu: Tensor<T, B>,
    istdev: Tensor<T, B>,
    m_const: Tensor<T, B>,
    minus_half: Tensor<T, B>,
    two_const: Tensor<T, B>,
    n: usize,
    c: usize,
    d: usize,
    h: usize,
    w: usize,
    m: usize,
) -> Var<T, B> {
    let backend = B::default();
    let requires_grad = input.grad.is_some() || weight.grad.is_some() || bias.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(out_tensor.shape_cloned(), &backend))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = vec![input.clone(), weight.clone(), bias.clone()];
        let w_reshaped_captured = weight.tensor.reshape([1, c]);
        let x_hat_clone = x_hat.clone();
        let xmu_clone = xmu.clone();
        let istdev_clone = istdev.clone();
        let minus_half = minus_half.clone();
        let m_const_captured = m_const.clone();
        let two_const = two_const.clone();

        let node = BatchNorm3dNode {
            output_grad,
            inputs,
            w_reshaped_captured,
            x_hat_clone,
            xmu_clone,
            istdev_clone,
            minus_half,
            m_const_captured,
            two_const,
            n,
            c,
            d,
            h,
            w,
            m,
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
