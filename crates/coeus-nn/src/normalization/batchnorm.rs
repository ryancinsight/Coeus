//! Generic N-dimensional batch normalization.
//!
//! [`BatchNorm`](crate::normalization::batchnorm::BatchNorm) is the one implementation behind the
//! [`BatchNorm1d`](crate::normalization::BatchNorm1d),
//! [`BatchNorm2d`](crate::normalization::BatchNorm2d) and
//! [`BatchNorm3d`](crate::normalization::BatchNorm3d) aliases: the forward
//! pass is const-generic over the spatial rank `DIM`.
//!
//! The permutation/reshape tables below deliberately mirror the `match DIM`
//! helpers in `coeus-autograd`'s `ops/nn/normalization/batchnorm/bn_nd.rs`.
//! Those helpers are `pub(crate)` to `coeus-autograd` and so cannot be named
//! across the crate boundary, so the *table* (not the orchestration) is
//! duplicated here on purpose.

use super::validation;
use crate::module::{Module, ModuleError};
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend, Scalar};
use coeus_tensor::Tensor;
use std::cell::RefCell;

/// Diagnostic module name for a given spatial rank.
const fn module_name<const DIM: usize>() -> &'static str {
    match DIM {
        1 => "BatchNorm1d",
        2 => "BatchNorm2d",
        3 => "BatchNorm3d",
        _ => "BatchNormNd",
    }
}

/// Expected-rank description for the rank check (`1-D` also accepts `[N, C]`).
const fn rank_str<const DIM: usize>() -> &'static str {
    match DIM {
        1 => "2 or 3",
        2 => "4",
        3 => "5",
        _ => "configured batch-normalization rank",
    }
}

// ── Permute/reshape dispatch (mirrors `coeus-autograd`'s `bn_nd.rs`) ──

/// `[N, C, spatial...] -> [N, spatial..., C]`.
fn permute_to_nhwc<T: Scalar, B: coeus_ops::BackendOps<T> + Default, const DIM: usize>(
    tensor: &Tensor<T, B>,
    backend: &B,
) -> Tensor<T, B> {
    match DIM {
        1 => tensor.permute(&[0, 2, 1]).to_contiguous_on(backend),
        2 => tensor.permute(&[0, 2, 3, 1]).to_contiguous_on(backend),
        3 => tensor.permute(&[0, 2, 3, 4, 1]).to_contiguous_on(backend),
        _ => panic!("BatchNorm permute_to_nhwc: unsupported DIM {DIM}"),
    }
}

/// `[N, spatial..., C] -> [N, C, spatial...]`.
fn permute_from_nhwc<T: Scalar, B: coeus_ops::BackendOps<T> + Default, const DIM: usize>(
    tensor: &Tensor<T, B>,
    backend: &B,
) -> Tensor<T, B> {
    match DIM {
        1 => tensor.permute(&[0, 2, 1]).to_contiguous_on(backend),
        2 => tensor.permute(&[0, 3, 1, 2]).to_contiguous_on(backend),
        3 => tensor.permute(&[0, 4, 1, 2, 3]).to_contiguous_on(backend),
        _ => panic!("BatchNorm permute_from_nhwc: unsupported DIM {DIM}"),
    }
}

/// `[M, C] -> [N, spatial..., C]`.
fn reshape_from_flat<T: Scalar, B: coeus_ops::BackendOps<T>, const DIM: usize>(
    tensor: Tensor<T, B>,
    n: usize,
    spatial: &[usize],
    c: usize,
) -> Tensor<T, B> {
    match DIM {
        1 => tensor.reshape([n, spatial[0], c]),
        2 => tensor.reshape([n, spatial[0], spatial[1], c]),
        3 => tensor.reshape([n, spatial[0], spatial[1], spatial[2], c]),
        _ => panic!("BatchNorm reshape_from_flat: unsupported DIM {DIM}"),
    }
}

/// N-dimensional batch normalization for `DIM`-spatial-rank inputs.
///
/// `BatchNorm<T, B, 1>` normalizes `[N, C]` or `[N, C, L]` inputs (the
/// `[N, C]` form is the degenerate `L = 1` case, handled by the squeeze
/// adapter in [`Module::forward`]); `BatchNorm<T, B, 2>` handles
/// `[N, C, H, W]` and `BatchNorm<T, B, 3>` handles `[N, C, D, H, W]`.
/// Prefer the [`BatchNorm1d`](crate::normalization::BatchNorm1d),
/// [`BatchNorm2d`](crate::normalization::BatchNorm2d) and
/// [`BatchNorm3d`](crate::normalization::BatchNorm3d) aliases.
///
/// Running stats are updated during each forward call in training mode; in
/// eval mode (`is_training = false`) the frozen `running_mean`/`running_var`
/// are used instead.
#[derive(Clone)]
pub struct BatchNorm<
    T: Float,
    B: coeus_ops::BackendOps<T> + Default = MoiraiBackend,
    const DIM: usize = 1,
> {
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
    /// Whether the layer is in training mode (updates running stats when true).
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
    /// Cached spatial batch size m constants: `(m, m_const, corr_t)`.
    m_cache: RefCell<Option<(usize, Tensor<T, B>, Tensor<T, B>)>>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default, const DIM: usize> BatchNorm<T, B, DIM> {
    /// Create with ones weight, zeros bias, and initialized running stats.
    pub fn new(num_features: usize, eps: f64, momentum: f64) -> Self {
        let backend = B::default();
        Self::from_parts(
            num_features,
            Var::new(Tensor::ones_on([num_features], &backend), true),
            Var::new(Tensor::zeros_on([num_features], &backend), true),
            eps,
            momentum,
            Tensor::zeros_on([num_features], &backend),
            Tensor::ones_on([num_features], &backend),
        )
    }

    /// Construct from pre-existing weight, bias, and running-stat tensors (e.g. after checkpoint load).
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
        let eps_t = Tensor::full_on([1], coeus_core::FloatElement::from_f64(eps), &backend);
        let mom_t = Tensor::full_on([1], coeus_core::FloatElement::from_f64(momentum), &backend);
        let one_minus_mom_t = Tensor::full_on(
            [1],
            coeus_core::FloatElement::from_f64(1.0 - momentum),
            &backend,
        );
        let minus_half = Tensor::full_on([1], coeus_core::FloatElement::from_f64(-0.5), &backend);
        let two_const = Tensor::full_on([1], coeus_core::FloatElement::from_f64(2.0), &backend);
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

    /// Retrieve (or build and cache) the per-batch `m` constants `(m_const, corr_t)`.
    fn m_constants(
        &self,
        m: usize,
        backend: &B,
    ) -> Result<(Tensor<T, B>, Tensor<T, B>), ModuleError<B::Error>> {
        let module: &str = module_name::<DIM>();
        let mut cache = self
            .m_cache
            .try_borrow_mut()
            .map_err(|_| validation::state_borrow(module, "m_cache"))?;
        if let Some((cached_m, ref cached_m_const, ref cached_corr_t)) = *cache {
            if cached_m == m {
                return Ok((cached_m_const.clone(), cached_corr_t.clone()));
            }
        }
        let m_const = Tensor::full_on([1], T::from_count(m), backend);
        let correction = if m > 1 {
            T::from_count(m) / T::from_count(m - 1)
        } else {
            <T as coeus_core::NumericElement>::ONE
        };
        let corr_t = Tensor::full_on([1], correction, backend);
        *cache = Some((m, m_const.clone(), corr_t.clone()));
        Ok((m_const, corr_t))
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default, const DIM: usize> Module<T, B>
    for BatchNorm<T, B, DIM>
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn train(&mut self, mode: bool) {
        self.is_training = mode;
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let module: &str = module_name::<DIM>();
        let shape = input.tensor.shape();
        // 1-D additionally accepts the rank-2 `[N, C]` form (the `L = 1` case).
        let rank_ok = shape.len() == DIM + 2 || (DIM == 1 && shape.len() == 2);
        if !rank_ok {
            return Err(validation::invalid_rank(
                module,
                rank_str::<DIM>(),
                shape.len(),
            ));
        }
        let c = shape[1];
        if c != self.num_features {
            return Err(validation::channel_mismatch(module, self.num_features, c));
        }
        if !self.eps.is_finite() || self.eps < 0.0 {
            return Err(ModuleError::InvalidEpsilon { module });
        }
        for (parameter, actual) in [
            ("weight", self.weight.tensor.shape()),
            ("bias", self.bias.tensor.shape()),
        ] {
            if actual != [self.num_features] {
                return Err(validation::shape_mismatch(
                    module,
                    parameter,
                    &[self.num_features],
                    actual,
                ));
            }
        }
        // PyTorch `nn.BatchNorm1d` accepts both `[N, C]` and `[N, C, L]` inputs;
        // the `[N, C]` form is the degenerate case `L = 1`.  Squeeze-unsqueeze via
        // autograd-tracked `reshape` so the 2D path stays differentiable and the
        // existing 3D kernel runs unchanged on the reshaped tensor.
        let input_is_2d = DIM == 1 && shape.len() == 2;
        let upstream: Var<T, B> = if input_is_2d {
            let n = shape[0];
            let c = shape[1];
            // [N, C] -> [N, C, 1]; preserve grad-creator by going through `reshape`.
            coeus_autograd::reshape(input, vec![n, c, 1])
        } else {
            input.clone()
        };
        let out = self.forward_nd(&upstream)?;
        if input_is_2d {
            let n = shape[0];
            let c = shape[1];
            // [N, C, 1] -> [N, C]; preserve grad-creator.
            Ok(coeus_autograd::reshape(&out, vec![n, c]))
        } else {
            Ok(out)
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default, const DIM: usize> BatchNorm<T, B, DIM> {
    /// Rank-`DIM` forward path: `[N, C, spatial...] -> [N, C, spatial...]`.
    /// Separated from the `Module` trait surface so the 2D-input adapter above
    /// can call it without going through the trait vtable.
    fn forward_nd(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let module: &str = module_name::<DIM>();
        let shape = input.tensor.shape();
        let n = shape[0];
        let c = shape[1];
        let mut spatial = [1usize; 3];
        for (slot, &extent) in spatial.iter_mut().zip(shape[2..].iter()) {
            *slot = extent;
        }
        let backend = B::default();
        let m_spatial: usize = spatial.iter().product();
        let m = n * m_spatial; // spatial batch size

        // ── Eval mode: use running stats without updating them ──
        if !self.is_training {
            let rm = self
                .running_mean
                .try_borrow()
                .map_err(|_| validation::state_borrow(module, "running_mean"))?;
            let rv = self
                .running_var
                .try_borrow()
                .map_err(|_| validation::state_borrow(module, "running_var"))?;
            for (parameter, actual) in [("running_mean", rm.shape()), ("running_var", rv.shape())] {
                if actual != [self.num_features] {
                    return Err(validation::shape_mismatch(
                        module,
                        parameter,
                        &[self.num_features],
                        actual,
                    ));
                }
            }
            // Normalize using running stats: (x - running_mean) / coeus_core::FloatElement::sqrt(running_var + eps)
            let nhwc = permute_to_nhwc::<T, B, DIM>(&input.tensor, &backend);
            let flat = nhwc.reshape([m, c]);
            let rm_row = rm.reshape([1, c]);
            let rv_row = rv.reshape([1, c]);
            let mut istdev = rv_row.clone();
            coeus_ops::add_assign(&mut istdev, &self.eps_t, &backend)
                .map_err(|source| validation::backend(module, source))?;
            coeus_ops::sqrt_assign(&mut istdev, &backend)
                .map_err(|source| validation::backend(module, source))?;
            let ones = Tensor::ones_on([1, c], &backend);
            let mut istdev_inv = ones;
            coeus_ops::div_assign(&mut istdev_inv, &istdev, &backend)
                .map_err(|source| validation::backend(module, source))?;
            let xmu = coeus_ops::sub(&flat, &rm_row, &backend);
            let x_hat = coeus_ops::mul(&xmu, &istdev_inv, &backend);
            let w_r = self.weight.tensor.reshape([1, c]);
            let b_r = self.bias.tensor.reshape([1, c]);
            let mut y = coeus_ops::mul(&x_hat, &w_r, &backend);
            coeus_ops::add_assign(&mut y, &b_r, &backend)
                .map_err(|source| validation::backend(module, source))?;
            let y_nhwc = reshape_from_flat::<T, B, DIM>(y, n, &spatial, c);
            let out_tensor = permute_from_nhwc::<T, B, DIM>(&y_nhwc, &backend);
            return Ok(Var::new(out_tensor, false));
        }

        if m < 2 {
            return Err(validation::insufficient_elements(module, 2, m));
        }

        let (m_const, corr_t) = self.m_constants(m, &backend)?;

        // ── View as [M, C] via NCHW... → NHWC... → [M, C] ──
        let nhwc = permute_to_nhwc::<T, B, DIM>(&input.tensor, &backend);
        let flat = nhwc.reshape([m, c]); // [M, C]

        // ── Per-channel mean [1, C] ──
        let mean_t = coeus_ops::mean_axis(&flat, 0, &backend)
            .map_err(|source| validation::backend(module, source))?; // [1, C]

        // ── Centered: x - mu [M, C] ──
        let xmu = coeus_ops::sub(&flat, &mean_t, &backend);

        // ── Per-channel variance [1, C] ──
        let xmu_sq = coeus_ops::mul(&xmu, &xmu, &backend);
        let var_t = coeus_ops::mean_axis(&xmu_sq, 0, &backend)
            .map_err(|source| validation::backend(module, source))?; // [1, C]

        // ── 1/coeus_core::FloatElement::sqrt(var + eps) [1, C] ──
        let mut stdev = var_t.clone();
        coeus_ops::add_assign(&mut stdev, &self.eps_t, &backend)
            .map_err(|source| validation::backend(module, source))?;
        coeus_ops::sqrt_assign(&mut stdev, &backend)
            .map_err(|source| validation::backend(module, source))?;

        let mut istdev = self.ones_c.clone();
        coeus_ops::div_assign(&mut istdev, &stdev, &backend)
            .map_err(|source| validation::backend(module, source))?; // [1, C]

        // ── x_hat = xmu * istdev [M, C] ──
        let x_hat = coeus_ops::mul(&xmu, &istdev, &backend);

        // ── y = gamma * x_hat + beta ──
        let w_reshaped = self.weight.tensor.reshape([1, c]);
        let b_reshaped = self.bias.tensor.reshape([1, c]);
        let mut y_flat = coeus_ops::mul(&x_hat, &w_reshaped, &backend);
        coeus_ops::add_assign(&mut y_flat, &b_reshaped, &backend)
            .map_err(|source| validation::backend(module, source))?;

        // ── Output: [M, C] → [N, ..., C] → permute → [N, C, ...] ──
        let y_nhwc = reshape_from_flat::<T, B, DIM>(y_flat, n, &spatial, c);
        let out_tensor = permute_from_nhwc::<T, B, DIM>(&y_nhwc, &backend);

        // ── Update running stats (exponential moving average) ──
        let mut next_mean = self
            .running_mean
            .try_borrow()
            .map_err(|_| validation::state_borrow(module, "running_mean"))?
            .clone();
        let mut next_var = self
            .running_var
            .try_borrow()
            .map_err(|_| validation::state_borrow(module, "running_var"))?
            .clone();
        for (parameter, actual) in [
            ("running_mean", next_mean.shape()),
            ("running_var", next_var.shape()),
        ] {
            if actual != [self.num_features] {
                return Err(validation::shape_mismatch(
                    module,
                    parameter,
                    &[self.num_features],
                    actual,
                ));
            }
        }
        let mean_c = mean_t.reshape([c]);
        let var_c = var_t.reshape([c]);
        coeus_ops::mul_assign(&mut next_mean, &self.one_minus_mom_t, &backend)
            .map_err(|source| validation::backend(module, source))?;
        let term_mean = coeus_ops::mul(&mean_c, &self.mom_t, &backend);
        coeus_ops::add_assign(&mut next_mean, &term_mean, &backend)
            .map_err(|source| validation::backend(module, source))?;
        coeus_ops::mul_assign(&mut next_var, &self.one_minus_mom_t, &backend)
            .map_err(|source| validation::backend(module, source))?;
        let var_corrected = coeus_ops::mul(&var_c, &corr_t, &backend);
        let term_var = coeus_ops::mul(&var_corrected, &self.mom_t, &backend);
        coeus_ops::add_assign(&mut next_var, &term_var, &backend)
            .map_err(|source| validation::backend(module, source))?;
        let mut running_mean = self
            .running_mean
            .try_borrow_mut()
            .map_err(|_| validation::state_borrow(module, "running_mean"))?;
        let mut running_var = self
            .running_var
            .try_borrow_mut()
            .map_err(|_| validation::state_borrow(module, "running_var"))?;
        *running_mean = next_mean;
        *running_var = next_var;

        Ok(coeus_autograd::batchnorm_nd::<T, B, DIM>(
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
                spatial,
                m,
            },
        ))
    }
}
