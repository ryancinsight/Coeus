use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;
use std::cell::RefCell;

/// 1D Batch Normalization for `[N, C, L]` inputs.
///
/// Normalizes over the N, L dimensions (per-channel mean/variance).
/// Running stats are updated during each forward call in training mode.
/// In eval mode (`is_training = false`) uses running_mean/running_var.
#[derive(Clone)]
pub struct BatchNorm1d<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Number of channels (C dimension).
    pub num_features: usize,
    /// Learnable scale (gamma): `[C]`.
    pub weight: Var<T, B>,
    /// Learnable shift (beta): `[C]`.
    pub bias: Var<T, B>,
    /// Numerical stability constant added to variance.
    pub eps: f64,
    /// Exponential moving average factor for running stats.
    pub momentum: f64,
    /// Whether the layer is in training mode.
    pub is_training: bool,
    /// Running mean `[C]`.
    pub running_mean: RefCell<Tensor<T, B>>,
    /// Running variance `[C]`.
    pub running_var: RefCell<Tensor<T, B>>,
    /// Cached epsilon tensor: `[1]`.
    eps_t: Tensor<T, B>,
    /// Cached momentum tensor: `[1]`.
    mom_t: Tensor<T, B>,
    /// Cached 1 - momentum tensor: `[1]`.
    one_minus_mom_t: Tensor<T, B>,
    /// Cached -0.5 constant tensor: `[1]`.
    minus_half: Tensor<T, B>,
    /// Cached 2.0 constant tensor: `[1]`.
    two_const: Tensor<T, B>,
    /// Cached ones tensor of shape `[1, C]`.
    ones_c: Tensor<T, B>,
    /// Cached spatial batch size m constants: (m, m_const, corr_t)
    m_cache: RefCell<Option<(usize, Tensor<T, B>, Tensor<T, B>)>>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BatchNorm1d<T, B> {
    /// Create with ones weight, zeros bias, and initialized running stats.
    pub fn new(num_features: usize, eps: f64, momentum: f64) -> Self {
        let backend = B::default();
        let eps_t = Tensor::full_on([1], T::from_f64(eps), &backend);
        let mom_t = Tensor::full_on([1], T::from_f64(momentum), &backend);
        let one_minus_mom_t = Tensor::full_on([1], T::from_f64(1.0 - momentum), &backend);
        let minus_half = Tensor::full_on([1], T::from_f64(-0.5), &backend);
        let two_const = Tensor::full_on([1], T::from_f64(2.0), &backend);
        let ones_c = Tensor::ones_on([1, num_features], &backend);
        Self {
            num_features,
            weight: Var::new(Tensor::ones_on([num_features], &backend), true),
            bias: Var::new(Tensor::zeros_on([num_features], &backend), true),
            eps,
            momentum,
            is_training: true,
            running_mean: RefCell::new(Tensor::zeros_on([num_features], &backend)),
            running_var: RefCell::new(Tensor::ones_on([num_features], &backend)),
            eps_t,
            mom_t,
            one_minus_mom_t,
            minus_half,
            two_const,
            ones_c,
            m_cache: RefCell::new(None),
        }
    }

    /// Construct from pre-existing weight, bias, and running-stat tensors (e.g., after checkpoint load).
    pub fn from_parts(
        num_features: usize,
        weight: Var<T, B>,
        bias: Var<T, B>,
        eps: f64,
        momentum: f64,
        running_mean: Tensor<T, B>,
        running_var: Tensor<T, B>,
    ) -> Self {
        let backend = B::default();
        let eps_t = Tensor::full_on([1], T::from_f64(eps), &backend);
        let mom_t = Tensor::full_on([1], T::from_f64(momentum), &backend);
        let one_minus_mom_t = Tensor::full_on([1], T::from_f64(1.0 - momentum), &backend);
        let minus_half = Tensor::full_on([1], T::from_f64(-0.5), &backend);
        let two_const = Tensor::full_on([1], T::from_f64(2.0), &backend);
        let ones_c = Tensor::ones_on([1, num_features], &backend);
        Self {
            num_features,
            weight,
            bias,
            eps,
            momentum,
            is_training: true,
            running_mean: RefCell::new(running_mean),
            running_var: RefCell::new(running_var),
            eps_t,
            mom_t,
            one_minus_mom_t,
            minus_half,
            two_const,
            ones_c,
            m_cache: RefCell::new(None),
        }
    }

    /// Set training/eval mode.
    pub fn set_training(&mut self, mode: bool) {
        self.is_training = mode;
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for BatchNorm1d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn train(&mut self, mode: bool) {
        self.is_training = mode;
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        // PyTorch `nn.BatchNorm1d` accepts both `[N, C]` and `[N, C, L]` inputs;
        // the `[N, C]` form is the degenerate case `L = 1`.  Squeeze-unsqueeze via
        // autograd-tracked `reshape` so the 2D path stays differentiable and the
        // existing 3D kernel runs unchanged on the reshaped tensor.
        let input_is_2d = input.tensor.ndim() == 2;
        let upstream: Var<T, B> = if input_is_2d {
            let n = input.tensor.shape()[0];
            let c = input.tensor.shape()[1];
            // [N, C] -> [N, C, 1]; preserve grad-creator by going through `reshape`.
            coeus_autograd::reshape(input, vec![n, c, 1])
        } else {
            input.clone()
        };
        let out_3d = self.forward_3d(&upstream);
        if input_is_2d {
            let n = input.tensor.shape()[0];
            let c = input.tensor.shape()[1];
            // [N, C, 1] -> [N, C]; preserve grad-creator.
            coeus_autograd::reshape(&out_3d, vec![n, c])
        } else {
            out_3d
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BatchNorm1d<T, B> {
    /// 3D forward path: `[N, C, L] -> [N, C, L]`.  Separated from the `Module`
    /// trait surface so the 2D-input adapter above can call it without going
    /// through the trait vtable.
    fn forward_3d(&self, input: &Var<T, B>) -> Var<T, B> {
        let n = input.tensor.shape()[0];
        let c = input.tensor.shape()[1];
        let l = input.tensor.shape()[2];
        let backend = B::default();

        // ── Eval mode: use running stats without updating them ──
        if !self.is_training {
            let rm = self.running_mean.borrow();
            let rv = self.running_var.borrow();
            // Normalize using running stats: (x - running_mean) / sqrt(running_var + eps)
            let nlc = input.tensor.permute(&[0, 2, 1]).to_contiguous_on(&backend);
            let flat = nlc.reshape([n * l, c]);
            let rm_row = rm.reshape([1, c]);
            let rv_row = rv.reshape([1, c]);
            let mut istdev = rv_row.clone();
            coeus_ops::add_assign(&mut istdev, &self.eps_t, &backend);
            coeus_ops::sqrt_assign(&mut istdev, &backend);
            let ones = Tensor::ones_on([1, c], &backend);
            let mut istdev_inv = ones;
            coeus_ops::div_assign(&mut istdev_inv, &istdev, &backend);
            let xmu = coeus_ops::sub(&flat, &rm_row, &backend);
            let x_hat = coeus_ops::mul(&xmu, &istdev_inv, &backend);
            let w_r = self.weight.tensor.reshape([1, c]);
            let b_r = self.bias.tensor.reshape([1, c]);
            let mut y = coeus_ops::mul(&x_hat, &w_r, &backend);
            coeus_ops::add_assign(&mut y, &b_r, &backend);
            let y_nlc = y.reshape([n, l, c]);
            let out_tensor = y_nlc.permute(&[0, 2, 1]).to_contiguous_on(&backend);
            return Var::new(out_tensor, false);
        }

        let m = n * l; // spatial batch size

        // Retrieve or update cached m constants
        let (m_const, corr_t) = {
            let mut cache = self.m_cache.borrow_mut();
            if let Some((cached_m, ref cached_m_const, ref cached_corr_t)) = *cache {
                if cached_m == m {
                    (cached_m_const.clone(), cached_corr_t.clone())
                } else {
                    let m_const = Tensor::full_on([1], T::from_f64(m as f64), &backend);
                    let correction = if m > 1 {
                        m as f64 / (m - 1) as f64
                    } else {
                        1.0
                    };
                    let corr_t = Tensor::full_on([1], T::from_f64(correction), &backend);
                    *cache = Some((m, m_const.clone(), corr_t.clone()));
                    (m_const, corr_t)
                }
            } else {
                let m_const = Tensor::full_on([1], T::from_f64(m as f64), &backend);
                let correction = if m > 1 {
                    m as f64 / (m - 1) as f64
                } else {
                    1.0
                };
                let corr_t = Tensor::full_on([1], T::from_f64(correction), &backend);
                *cache = Some((m, m_const.clone(), corr_t.clone()));
                (m_const, corr_t)
            }
        };

        // ── View as [M, C] via NCL → NLC → [M, C] ──
        let nlc = input.tensor.permute(&[0, 2, 1]).to_contiguous_on(&backend); // [N, L, C]
        let flat = nlc.reshape([m, c]); // [M, C]

        // ── Per-channel mean [1, C] ──
        let mean_t = coeus_ops::mean_axis(&flat, 0, &backend)
            .expect("invariant: batchnorm1d channel axis is valid"); // [1, C]

        // ── Centered: x - mu [M, C] ──
        let xmu = coeus_ops::sub(&flat, &mean_t, &backend);

        // ── Per-channel variance [1, C] ──
        let xmu_sq = coeus_ops::mul(&xmu, &xmu, &backend);
        let var_t = coeus_ops::mean_axis(&xmu_sq, 0, &backend)
            .expect("invariant: batchnorm1d channel axis is valid"); // [1, C]

        // ── 1/sqrt(var + eps) [1, C] ──
        let mut stdev = var_t.clone();
        coeus_ops::add_assign(&mut stdev, &self.eps_t, &backend);
        coeus_ops::sqrt_assign(&mut stdev, &backend);

        let mut istdev = self.ones_c.clone();
        coeus_ops::div_assign(&mut istdev, &stdev, &backend); // [1, C]

        // ── x_hat = xmu * istdev [M, C] ──
        let x_hat = coeus_ops::mul(&xmu, &istdev, &backend);

        // ── y = gamma * x_hat + beta ──
        let w_reshaped = self.weight.tensor.reshape([1, c]);
        let b_reshaped = self.bias.tensor.reshape([1, c]);
        let mut y_flat = coeus_ops::mul(&x_hat, &w_reshaped, &backend);
        coeus_ops::add_assign(&mut y_flat, &b_reshaped, &backend);

        // ── Output: [M, C] → [N, L, C] → permute → [N, C, L] ──
        let y_nlc = y_flat.reshape([n, l, c]);
        let out_tensor = y_nlc.permute(&[0, 2, 1]).to_contiguous_on(&backend);

        // ── Update running stats (exponential moving average) ──
        if let (Ok(mut rm), Ok(mut rv)) = (
            self.running_mean.try_borrow_mut(),
            self.running_var.try_borrow_mut(),
        ) {
            let mean_c = mean_t.reshape([c]);
            let var_c = var_t.reshape([c]);

            coeus_ops::mul_assign(&mut *rm, &self.one_minus_mom_t, &backend);
            let term_mean = coeus_ops::mul(&mean_c, &self.mom_t, &backend);
            coeus_ops::add_assign(&mut *rm, &term_mean, &backend);

            coeus_ops::mul_assign(&mut *rv, &self.one_minus_mom_t, &backend);
            let var_corrected = coeus_ops::mul(&var_c, &corr_t, &backend);
            let term_var = coeus_ops::mul(&var_corrected, &self.mom_t, &backend);
            coeus_ops::add_assign(&mut *rv, &term_var, &backend);
        }

        coeus_autograd::batchnorm1d(
            input,
            &self.weight,
            &self.bias,
            coeus_autograd::BatchNormArgs {
                out_tensor,
                x_hat,
                xmu,
                istdev,
                m_const,
                minus_half: self.minus_half.clone(),
                two_const: self.two_const.clone(),
                n,
                c,
                spatial: [l, 1, 1],
                m,
            },
        )
    }
}
