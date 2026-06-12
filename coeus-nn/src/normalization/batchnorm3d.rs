use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;
use std::cell::RefCell;

/// 3D Batch Normalization for `[N, C, D, H, W]` inputs.
///
/// Normalizes over the N, D, H, W dimensions (per-channel mean/variance).
/// Running stats are updated during each forward call.
#[derive(Clone)]
pub struct BatchNorm3d<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    pub num_features: usize,
    pub weight: Var<T, B>,
    pub bias: Var<T, B>,
    pub eps: f64,
    pub momentum: f64,
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
    /// Cached ones tensor of shape [1, C]
    ones_c: Tensor<T, B>,
    /// Cached spatial batch size m constants: (m, m_const, corr_t)
    m_cache: RefCell<Option<(usize, Tensor<T, B>, Tensor<T, B>)>>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BatchNorm3d<T, B> {
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
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for BatchNorm3d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let n = input.tensor.shape()[0];
        let c = input.tensor.shape()[1];
        let d = input.tensor.shape()[2];
        let h = input.tensor.shape()[3];
        let w = input.tensor.shape()[4];
        let m = n * d * h * w; // spatial batch size
        let backend = B::default();

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

        // ── View as [M, C] via NCDHW → NDHWC → [M, C] ──
        let ndhwc = input
            .tensor
            .permute(&[0, 2, 3, 4, 1])
            .to_contiguous_on(&backend); // [N, D, H, W, C]
        let flat = ndhwc.reshape([m, c]); // [M, C]

        // ── Per-channel mean [1, C] ──
        let mean_t = coeus_ops::mean_axis(&flat, 0, &backend); // [1, C]

        // ── Centered: x - mu [M, C] ──
        let xmu = coeus_ops::sub(&flat, &mean_t, &backend);

        // ── Per-channel variance [1, C] ──
        let xmu_sq = coeus_ops::mul(&xmu, &xmu, &backend);
        let var_t = coeus_ops::mean_axis(&xmu_sq, 0, &backend); // [1, C]

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

        // ── Output: [M, C] → [N, D, H, W, C] → permute → [N, C, D, H, W] ──
        let y_ndhwc = y_flat.reshape([n, d, h, w, c]);
        let out_tensor = y_ndhwc.permute(&[0, 4, 1, 2, 3]).to_contiguous_on(&backend);

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

        coeus_autograd::batchnorm3d(
            input,
            &self.weight,
            &self.bias,
            coeus_autograd::BatchNorm3dArgs {
                out_tensor,
                x_hat,
                xmu,
                istdev,
                m_const,
                minus_half: self.minus_half.clone(),
                two_const: self.two_const.clone(),
                n,
                c,
                d,
                h,
                w,
                m,
            },
        )
    }
}
