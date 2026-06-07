// ── Instance Normalization ──
//
// InstanceNorm1d and InstanceNorm2d are thin wrappers over the channel-normalization
// pattern: each sample, each channel is normalized independently.
// This is equivalent to GroupNorm where G = C (each group has exactly 1 channel).
//
// Since G must be a const generic and C is runtime, we cannot directly instantiate
// GroupNorm<T, B, C>. Instead, InstanceNorm uses the same approach manually:
// reshape to [N*C, spatial], apply LayerNorm, reshape back, apply affine.

use std::cell::RefCell;
use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;
use coeus_autograd::Var;
use crate::module::Module;

#[derive(Clone)]
struct InstanceNormCache<T: Float, B: coeus_ops::BackendOps<T> + Default> {
    spatial: usize,
    ln_weight: Var<T, B>,
    ln_bias: Var<T, B>,
    eps_t: Tensor<T, B>,
    d_const: Tensor<T, B>,
    ones_cache: RefCell<Option<(usize, Tensor<T, B>)>>,
}

/// Instance Normalization for 1D inputs `[N, C, L]` or `[N, C]`.
///
/// Normalizes over the L (spatial) dimension independently per sample per channel.
#[derive(Clone)]
pub struct InstanceNorm1d<
    T: Float,
    B: coeus_ops::BackendOps<T> + Default = MoiraiBackend,
> {
    /// Trainable scale (gamma): shape `[num_features]`.
    pub weight: Var<T, B>,
    /// Trainable shift (beta): shape `[num_features]`.
    pub bias: Var<T, B>,
    pub num_features: usize,
    pub eps: f64,
    cache: RefCell<Option<InstanceNormCache<T, B>>>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> InstanceNorm1d<T, B> {
    /// Create an InstanceNorm1d layer.
    pub fn new(num_features: usize, eps: f64) -> Self {
        let backend = B::default();
        let weight = Var::new(Tensor::ones_on([num_features], &backend), true);
        let bias   = Var::new(Tensor::zeros_on([num_features], &backend), true);
        Self { weight, bias, num_features, eps, cache: RefCell::new(None) }
    }

    fn get_cache(&self, spatial: usize) -> std::cell::RefMut<'_, Option<InstanceNormCache<T, B>>> {
        let mut cache = self.cache.borrow_mut();
        let need_recreate = match &*cache {
            Some(c) => c.spatial != spatial,
            None => true,
        };
        if need_recreate {
            let backend = B::default();
            let ln_weight = Var::new(Tensor::ones_on([spatial], &backend), false);
            let ln_bias = Var::new(Tensor::zeros_on([spatial], &backend), false);
            let eps_t = Tensor::full_on([1], T::from_f64(self.eps), &backend);
            let d_const = Tensor::full_on([1], T::from_f64(spatial as f64), &backend);
            *cache = Some(InstanceNormCache {
                spatial,
                ln_weight,
                ln_bias,
                eps_t,
                d_const,
                ones_cache: RefCell::new(None),
            });
        }
        cache
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for InstanceNorm1d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    /// Forward: input `[N, C]` or `[N, C, L]`.
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let shape = input.tensor.shape_cloned();
        let n = shape[0];
        let c = shape[1];
        let spatial: usize = shape.get(2).copied().unwrap_or(1);

        // Reshape to [N*C, spatial] → normalize over spatial per (sample, channel) pair
        let flat = coeus_autograd::reshape(input, [n * c, spatial]);

        // Get cache
        let cache_borrow = self.get_cache(spatial);
        let cache = cache_borrow.as_ref().unwrap();

        let backend = B::default();

        // Mean over last dimension of flat: [N*C, spatial] -> [N*C, 1]
        let mean_t = coeus_ops::mean_axis(&flat.tensor, 1, &backend);

        // Centered: x - mu
        let xmu = coeus_ops::sub(&flat.tensor, &mean_t, &backend);

        // Variance
        let xmu_sq = coeus_ops::mul(&xmu, &xmu, &backend);
        let mut stdev = coeus_ops::mean_axis(&xmu_sq, 1, &backend);

        // 1/sqrt(var + eps)
        coeus_ops::add_assign(&mut stdev, &cache.eps_t, &backend);
        coeus_ops::sqrt_assign(&mut stdev, &backend);

        let ones = {
            let mut o_cache = cache.ones_cache.borrow_mut();
            let n_channels = n * c;
            if let Some((cached_n, ref cached_ones)) = *o_cache {
                if cached_n == n_channels {
                    cached_ones.clone()
                } else {
                    let ones = Tensor::ones_on([n_channels, 1], &backend);
                    *o_cache = Some((n_channels, ones.clone()));
                    ones
                }
            } else {
                let ones = Tensor::ones_on([n_channels, 1], &backend);
                *o_cache = Some((n_channels, ones.clone()));
                ones
            }
        };
        let mut istdev = ones;
        coeus_ops::div_assign(&mut istdev, &stdev, &backend);

        // Normalize
        let x_hat = coeus_ops::mul(&xmu, &istdev, &backend);

        // Scale and bias
        let w_reshaped = cache.ln_weight.tensor.reshape([1, spatial]);
        let b_reshaped = cache.ln_bias.tensor.reshape([1, spatial]);
        let mut out_tensor = coeus_ops::mul(&x_hat, &w_reshaped, &backend);
        coeus_ops::add_assign(&mut out_tensor, &b_reshaped, &backend);

        let normed_flat = coeus_autograd::layernorm(
            &flat,
            &cache.ln_weight,
            &cache.ln_bias,
            out_tensor,
            x_hat,
            istdev,
            cache.d_const.clone(),
        );

        let normed = coeus_autograd::reshape(&normed_flat, shape.clone());

        // Broadcast affine: weight/bias [C] → [1, C, 1, ...]
        let mut bshape = vec![1usize; shape.len()];
        bshape[1] = c;
        let w = coeus_autograd::reshape(&self.weight, bshape.as_slice());
        let b = coeus_autograd::reshape(&self.bias,   bshape.as_slice());
        let scaled = coeus_autograd::mul(&normed, &w);
        coeus_autograd::add(&scaled, &b)
    }
}

/// Instance Normalization for 2D inputs `[N, C, H, W]`.
///
/// Normalizes over the H*W spatial dimensions independently per sample per channel.
#[derive(Clone)]
pub struct InstanceNorm2d<
    T: Float,
    B: coeus_ops::BackendOps<T> + Default = MoiraiBackend,
> {
    pub weight: Var<T, B>,
    pub bias: Var<T, B>,
    pub num_features: usize,
    pub eps: f64,
    cache: RefCell<Option<InstanceNormCache<T, B>>>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> InstanceNorm2d<T, B> {
    /// Create an InstanceNorm2d layer.
    pub fn new(num_features: usize, eps: f64) -> Self {
        let backend = B::default();
        let weight = Var::new(Tensor::ones_on([num_features], &backend), true);
        let bias   = Var::new(Tensor::zeros_on([num_features], &backend), true);
        Self { weight, bias, num_features, eps, cache: RefCell::new(None) }
    }

    fn get_cache(&self, spatial: usize) -> std::cell::RefMut<'_, Option<InstanceNormCache<T, B>>> {
        let mut cache = self.cache.borrow_mut();
        let need_recreate = match &*cache {
            Some(c) => c.spatial != spatial,
            None => true,
        };
        if need_recreate {
            let backend = B::default();
            let ln_weight = Var::new(Tensor::ones_on([spatial], &backend), false);
            let ln_bias = Var::new(Tensor::zeros_on([spatial], &backend), false);
            let eps_t = Tensor::full_on([1], T::from_f64(self.eps), &backend);
            let d_const = Tensor::full_on([1], T::from_f64(spatial as f64), &backend);
            *cache = Some(InstanceNormCache {
                spatial,
                ln_weight,
                ln_bias,
                eps_t,
                d_const,
                ones_cache: RefCell::new(None),
            });
        }
        cache
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for InstanceNorm2d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    /// Forward: input `[N, C, H, W]`.
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let shape = input.tensor.shape_cloned();
        let n = shape[0];
        let c = shape[1];
        let h = shape.get(2).copied().unwrap_or(1);
        let w_dim = shape.get(3).copied().unwrap_or(1);
        let spatial = h * w_dim;

        let flat = coeus_autograd::reshape(input, [n * c, spatial]);

        // Get cache
        let cache_borrow = self.get_cache(spatial);
        let cache = cache_borrow.as_ref().unwrap();

        let backend = B::default();

        // Mean over last dimension of flat: [N*C, spatial] -> [N*C, 1]
        let mean_t = coeus_ops::mean_axis(&flat.tensor, 1, &backend);

        // Centered: x - mu
        let xmu = coeus_ops::sub(&flat.tensor, &mean_t, &backend);

        // Variance
        let xmu_sq = coeus_ops::mul(&xmu, &xmu, &backend);
        let mut stdev = coeus_ops::mean_axis(&xmu_sq, 1, &backend);

        // 1/sqrt(var + eps)
        coeus_ops::add_assign(&mut stdev, &cache.eps_t, &backend);
        coeus_ops::sqrt_assign(&mut stdev, &backend);

        let ones = {
            let mut o_cache = cache.ones_cache.borrow_mut();
            let n_channels = n * c;
            if let Some((cached_n, ref cached_ones)) = *o_cache {
                if cached_n == n_channels {
                    cached_ones.clone()
                } else {
                    let ones = Tensor::ones_on([n_channels, 1], &backend);
                    *o_cache = Some((n_channels, ones.clone()));
                    ones
                }
            } else {
                let ones = Tensor::ones_on([n_channels, 1], &backend);
                *o_cache = Some((n_channels, ones.clone()));
                ones
            }
        };
        let mut istdev = ones;
        coeus_ops::div_assign(&mut istdev, &stdev, &backend);

        // Normalize
        let x_hat = coeus_ops::mul(&xmu, &istdev, &backend);

        // Scale and bias
        let w_reshaped = cache.ln_weight.tensor.reshape([1, spatial]);
        let b_reshaped = cache.ln_bias.tensor.reshape([1, spatial]);
        let mut out_tensor = coeus_ops::mul(&x_hat, &w_reshaped, &backend);
        coeus_ops::add_assign(&mut out_tensor, &b_reshaped, &backend);

        let normed_flat = coeus_autograd::layernorm(
            &flat,
            &cache.ln_weight,
            &cache.ln_bias,
            out_tensor,
            x_hat,
            istdev,
            cache.d_const.clone(),
        );

        let normed = coeus_autograd::reshape(&normed_flat, shape.clone());

        let mut bshape = vec![1usize; shape.len()];
        bshape[1] = c;
        let wv = coeus_autograd::reshape(&self.weight, bshape.as_slice());
        let bv = coeus_autograd::reshape(&self.bias,   bshape.as_slice());
        let scaled = coeus_autograd::mul(&normed, &wv);
        coeus_autograd::add(&scaled, &bv)
    }
}
