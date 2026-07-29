//! Instance normalization layers.
//!
//! [`InstanceNorm1d`], [`InstanceNorm2d`], and [`InstanceNorm3d`] normalize
//! each sample/channel slice across its spatial dimensions independently. This
//! is equivalent to group normalization with one channel per group.

use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;
use std::cell::RefCell;

// ── Shared cache ──────────────────────────────────────────────────────────────

#[derive(Clone)]
struct InstanceNormCache<T: Float, B: coeus_ops::BackendOps<T> + Default> {
    spatial: usize,
    ln_weight: Var<T, B>,
    ln_bias: Var<T, B>,
    eps_t: Tensor<T, B>,
    d_const: Tensor<T, B>,
    ones_cache: RefCell<Option<(usize, Tensor<T, B>)>>,
}

/// Rebuild `cache` if the spatial size changed.
fn ensure_cache<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    cache: &mut Option<InstanceNormCache<T, B>>,
    spatial: usize,
    eps: f64,
) {
    let needs_rebuild = cache.as_ref().is_none_or(|c| c.spatial != spatial);
    if needs_rebuild {
        let backend = B::default();
        let ln_weight = Var::new(Tensor::ones_on([spatial], &backend), false);
        let ln_bias = Var::new(Tensor::zeros_on([spatial], &backend), false);
        let eps_t = Tensor::full_on([1], T::from_f64(eps), &backend);
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
}

// ── Shared normalization body ─────────────────────────────────────────────────
//
// Input is already reshaped to `[N*C, spatial]` by the caller.
// `weight`/`bias` are the per-channel affine parameters ([C]).

fn instance_norm_forward<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    flat: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Var<T, B>,
    cache: &InstanceNormCache<T, B>,
    n: usize,
    c: usize,
    orig_shape: coeus_core::Shape,
) -> Var<T, B> {
    let spatial = cache.spatial;
    let backend = B::default();
    let n_channels = n * c;

    let mean_t = coeus_ops::mean_axis(&flat.tensor, 1, &backend)
        .expect("invariant: instancenorm feature axis is valid"); // [N*C, 1]
    let xmu = coeus_ops::sub(&flat.tensor, &mean_t, &backend);
    let xmu_sq = coeus_ops::mul(&xmu, &xmu, &backend);
    let mut stdev = coeus_ops::mean_axis(&xmu_sq, 1, &backend)
        .expect("invariant: instancenorm feature axis is valid"); // population var
    coeus_ops::add_assign(&mut stdev, &cache.eps_t, &backend)
        .expect("normalization backend operation");
    coeus_ops::sqrt_assign(&mut stdev, &backend).expect("normalization backend operation");

    let ones = {
        let mut o_cache = cache.ones_cache.borrow_mut();
        match &*o_cache {
            Some((cached_n, ref cached_ones)) if *cached_n == n_channels => cached_ones.clone(),
            _ => {
                let ones = Tensor::ones_on([n_channels, 1], &backend);
                *o_cache = Some((n_channels, ones.clone()));
                ones
            }
        }
    };
    let mut istdev = ones;
    coeus_ops::div_assign(&mut istdev, &stdev, &backend).expect("normalization backend operation");

    let x_hat = coeus_ops::mul(&xmu, &istdev, &backend);

    let w_reshaped = cache.ln_weight.tensor.reshape([1, spatial]);
    let b_reshaped = cache.ln_bias.tensor.reshape([1, spatial]);
    let mut out_tensor = coeus_ops::mul(&x_hat, &w_reshaped, &backend);
    coeus_ops::add_assign(&mut out_tensor, &b_reshaped, &backend)
        .expect("normalization backend operation");

    let normed_flat = coeus_autograd::layernorm(
        flat,
        &cache.ln_weight,
        &cache.ln_bias,
        out_tensor,
        x_hat,
        istdev,
        cache.d_const.clone(),
    );

    let mut bshape = vec![1usize; orig_shape.len()];
    bshape[1] = c;
    let normed = coeus_autograd::reshape(&normed_flat, orig_shape);
    let wv = coeus_autograd::reshape(weight, bshape.as_slice());
    let bv = coeus_autograd::reshape(bias, bshape.as_slice());
    let scaled = coeus_autograd::mul(&normed, &wv);
    coeus_autograd::add(&scaled, &bv)
}

// ── InstanceNorm1d ────────────────────────────────────────────────────────────

/// Instance Normalization for 1D inputs `[N, C, L]` or `[N, C]`.
///
/// Normalizes over the `L` (spatial) dimension independently per sample per channel.
///
/// # Examples
///
/// ```
/// use coeus_nn::{InstanceNorm1d, Module};
/// use coeus_autograd::Var;
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
///
/// let in1 = InstanceNorm1d::<f32, SequentialBackend>::new(4, 1e-5);
/// let x = Var::new(Tensor::ones_on([2, 4, 8], &SequentialBackend::new()), false);
/// let y = in1.forward(&x);
/// assert_eq!(y.tensor.shape(), &[2, 4, 8]);
/// ```
#[derive(Clone)]
pub struct InstanceNorm1d<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Trainable scale (gamma): shape `[num_features]`.
    pub weight: Var<T, B>,
    /// Trainable shift (beta): shape `[num_features]`.
    pub bias: Var<T, B>,
    /// Number of channels.
    pub num_features: usize,
    /// Numerical stability constant added to variance.
    pub eps: f64,
    cache: RefCell<Option<InstanceNormCache<T, B>>>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> InstanceNorm1d<T, B> {
    /// Create an InstanceNorm1d layer.
    pub fn new(num_features: usize, eps: f64) -> Self {
        let backend = B::default();
        Self {
            weight: Var::new(Tensor::ones_on([num_features], &backend), true),
            bias: Var::new(Tensor::zeros_on([num_features], &backend), true),
            num_features,
            eps,
            cache: RefCell::new(None),
        }
    }
}

/// Implements the [`crate::module::Module`] interface for [`InstanceNorm1d`].
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for InstanceNorm1d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    /// Forward: input `[N, C]` or `[N, C, L]`.
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let shape = input.tensor.shape_cloned();
        let n = shape[0];
        let c = shape[1];
        let spatial = shape.get(2).copied().unwrap_or(1);
        let flat = coeus_autograd::reshape(input, [n * c, spatial]);

        let mut cache = self.cache.borrow_mut();
        ensure_cache::<T, B>(&mut *cache, spatial, self.eps);
        let cache = cache.as_ref().unwrap();

        instance_norm_forward(&flat, &self.weight, &self.bias, cache, n, c, shape)
    }
}

// ── InstanceNorm2d ────────────────────────────────────────────────────────────

/// Instance Normalization for 2D inputs `[N, C, H, W]`.
///
/// Normalizes over the `H × W` spatial dimensions independently per sample per channel.
///
/// # Examples
///
/// ```
/// use coeus_nn::{InstanceNorm2d, Module};
/// use coeus_autograd::Var;
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
///
/// let in2 = InstanceNorm2d::<f32, SequentialBackend>::new(4, 1e-5);
/// let x = Var::new(Tensor::ones_on([2, 4, 8, 8], &SequentialBackend::new()), false);
/// let y = in2.forward(&x);
/// assert_eq!(y.tensor.shape(), &[2, 4, 8, 8]);
/// ```
#[derive(Clone)]
pub struct InstanceNorm2d<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Trainable scale (gamma): shape `[num_features]`.
    pub weight: Var<T, B>,
    /// Trainable shift (beta): shape `[num_features]`.
    pub bias: Var<T, B>,
    /// Number of channels.
    pub num_features: usize,
    /// Numerical stability constant added to variance.
    pub eps: f64,
    cache: RefCell<Option<InstanceNormCache<T, B>>>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> InstanceNorm2d<T, B> {
    /// Create an InstanceNorm2d layer.
    pub fn new(num_features: usize, eps: f64) -> Self {
        let backend = B::default();
        Self {
            weight: Var::new(Tensor::ones_on([num_features], &backend), true),
            bias: Var::new(Tensor::zeros_on([num_features], &backend), true),
            num_features,
            eps,
            cache: RefCell::new(None),
        }
    }
}

/// Implements the [`crate::module::Module`] interface for [`InstanceNorm2d`].
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for InstanceNorm2d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    /// Forward: input `[N, C, H, W]`.
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let shape = input.tensor.shape_cloned();
        let n = shape[0];
        let c = shape[1];
        let spatial = shape.get(2).copied().unwrap_or(1) * shape.get(3).copied().unwrap_or(1);
        let flat = coeus_autograd::reshape(input, [n * c, spatial]);

        let mut cache = self.cache.borrow_mut();
        ensure_cache::<T, B>(&mut *cache, spatial, self.eps);
        let cache = cache.as_ref().unwrap();

        instance_norm_forward(&flat, &self.weight, &self.bias, cache, n, c, shape)
    }
}

// ── InstanceNorm3d ────────────────────────────────────────────────────────────

/// Instance Normalization for 3D inputs `[N, C, D, H, W]`.
///
/// Normalizes over the `D × H × W` spatial volume independently per sample per channel.
///
/// # Examples
///
/// ```
/// use coeus_nn::{InstanceNorm3d, Module};
/// use coeus_autograd::Var;
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
///
/// let in3 = InstanceNorm3d::<f32, SequentialBackend>::new(4, 1e-5);
/// let x = Var::new(Tensor::ones_on([1, 4, 4, 4, 4], &SequentialBackend::new()), false);
/// let y = in3.forward(&x);
/// assert_eq!(y.tensor.shape(), &[1, 4, 4, 4, 4]);
/// ```
#[derive(Clone)]
pub struct InstanceNorm3d<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Trainable scale (gamma): shape `[num_features]`.
    pub weight: Var<T, B>,
    /// Trainable shift (beta): shape `[num_features]`.
    pub bias: Var<T, B>,
    /// Number of channels.
    pub num_features: usize,
    /// Numerical stability constant added to variance.
    pub eps: f64,
    cache: RefCell<Option<InstanceNormCache<T, B>>>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> InstanceNorm3d<T, B> {
    /// Create an InstanceNorm3d layer.
    pub fn new(num_features: usize, eps: f64) -> Self {
        let backend = B::default();
        Self {
            weight: Var::new(Tensor::ones_on([num_features], &backend), true),
            bias: Var::new(Tensor::zeros_on([num_features], &backend), true),
            num_features,
            eps,
            cache: RefCell::new(None),
        }
    }
}

/// Implements the [`crate::module::Module`] interface for [`InstanceNorm3d`].
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for InstanceNorm3d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    /// Forward: input `[N, C, D, H, W]`.
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let shape = input.tensor.shape_cloned();
        let n = shape[0];
        let c = shape[1];
        let spatial = shape.get(2).copied().unwrap_or(1)
            * shape.get(3).copied().unwrap_or(1)
            * shape.get(4).copied().unwrap_or(1);
        let flat = coeus_autograd::reshape(input, [n * c, spatial]);

        let mut cache = self.cache.borrow_mut();
        ensure_cache::<T, B>(&mut *cache, spatial, self.eps);
        let cache = cache.as_ref().unwrap();

        instance_norm_forward(&flat, &self.weight, &self.bias, cache, n, c, shape)
    }
}
