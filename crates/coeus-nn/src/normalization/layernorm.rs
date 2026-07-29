//! Layer normalization layers and helpers.
//!
//! [`LayerNorm`] normalizes activations over their last dimension and applies
//! optional trainable affine parameters.

use super::validation;
use crate::module::{Module, ModuleError};
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;
use std::cell::RefCell;

/// Functional (stateless) layer normalization.
///
/// For rank-2 inputs this matches [`LayerNorm::forward`]; for rank≥3 it matches
/// [`LayerNorm::forward_nd`] by normalizing the last dimension.
///
/// `weight` and `bias` default to ones/zeros of shape `[normalized_shape]`.
///
/// # Errors
///
/// Returns a typed module or backend failure when the input rank, trailing
/// dimension, affine parameter shapes, or epsilon violate the LayerNorm
/// contract, or when a backend operation fails.
pub fn layer_norm<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    normalized_shape: usize,
    weight: Option<&Var<T, B>>,
    bias: Option<&Var<T, B>>,
    eps: f64,
) -> Result<Var<T, B>, ModuleError<B::Error>>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let w = weight
        .cloned()
        .unwrap_or_else(|| Var::new(Tensor::ones_on([normalized_shape], &backend), false));
    let b = bias
        .cloned()
        .unwrap_or_else(|| Var::new(Tensor::zeros_on([normalized_shape], &backend), false));
    let ln = LayerNorm::from_parts(w, b, eps);
    if input.tensor.ndim() == 2 {
        ln.forward(input)
    } else {
        ln.forward_nd(input)
    }
}

/// Layer Normalization module.
///
/// Applies Layer Normalization over the last dimension of tensors with rank
/// two or greater.
#[derive(Clone)]
pub struct LayerNorm<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Trainable scale parameter gamma: `[D]`.
    pub weight: Var<T, B>,
    /// Trainable shift parameter beta: `[D]`.
    pub bias: Var<T, B>,
    /// Small value for numerical stability.
    pub eps: f64,
    /// Cached epsilon tensor: `[1]`.
    eps_t: Tensor<T, B>,
    /// Cached dimension constant: `[1]`.
    d_const: Tensor<T, B>,
    /// Cached ones tensor of shape `[N, 1]`: (N, ones_tensor).
    ones_cache: RefCell<Option<(usize, Tensor<T, B>)>>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> LayerNorm<T, B> {
    /// Create a new LayerNorm layer for a given feature dimension.
    pub fn new(normalized_shape: usize, eps: f64) -> Self {
        let backend = B::default();
        let weight = Var::new(Tensor::ones_on([normalized_shape], &backend), true);
        let bias = Var::new(Tensor::zeros_on([normalized_shape], &backend), true);
        let eps_t = Tensor::full_on([1], T::from_f64(eps), &backend);
        let d_const = Tensor::full_on([1], T::from_f64(normalized_shape as f64), &backend);
        Self {
            weight,
            bias,
            eps,
            eps_t,
            d_const,
            ones_cache: RefCell::new(None),
        }
    }

    /// Create a LayerNorm layer from existing parameters.
    pub fn from_parts(weight: Var<T, B>, bias: Var<T, B>, eps: f64) -> Self {
        let backend = B::default();
        let normalized_shape = weight.tensor.shape()[0];
        let eps_t = Tensor::full_on([1], T::from_f64(eps), &backend);
        let d_const = Tensor::full_on([1], T::from_f64(normalized_shape as f64), &backend);
        Self {
            weight,
            bias,
            eps,
            eps_t,
            d_const,
            ones_cache: RefCell::new(None),
        }
    }
}

/// Implements the [`crate::module::Module`] interface for [`LayerNorm`].
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for LayerNorm<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    /// Write updated `(weight, bias)` values back. Overrides the
    /// default no-op: `coeus_tensor::Tensor`'s copy-on-write storage
    /// means a clone taken via `parameters()` detaches from this
    /// module on first mutation, so an optimizer that mutates its own
    /// owned copy needs this round-trip to propagate updates back.
    fn load_parameters(&mut self, params: &[Var<T, B>]) {
        self.weight = params[0].clone();
        self.bias = params[1].clone();
    }

    /// Forward pass over the final dimension of any input with rank ≥ 2.
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        const MODULE: &str = "LayerNorm";
        let shape = input.tensor.shape_cloned();
        if shape.len() != 2 {
            return self.forward_nd(input);
        }
        let _n = shape[0];
        let d = shape[1];
        for (parameter, actual) in [
            ("weight", self.weight.tensor.shape()),
            ("bias", self.bias.tensor.shape()),
        ] {
            if actual != [d] {
                return Err(validation::shape_mismatch(MODULE, parameter, &[d], actual));
            }
        }
        if !self.eps.is_finite() || self.eps < 0.0 {
            return Err(ModuleError::InvalidEpsilon { module: MODULE });
        }
        let backend = B::default();

        // ── Mean over last dimension ──
        let mean_t = coeus_ops::mean_axis(&input.tensor, 1, &backend)
            .map_err(|source| validation::backend(MODULE, source))?; // [N, 1]

        // ── Centered: x - mu ──
        let xmu = coeus_ops::sub(&input.tensor, &mean_t, &backend); // [N, D]

        // ── Variance ──
        let xmu_sq = coeus_ops::mul(&xmu, &xmu, &backend);
        let mut stdev = coeus_ops::mean_axis(&xmu_sq, 1, &backend)
            .map_err(|source| validation::backend(MODULE, source))?; // [N, 1]

        // ── 1/sqrt(var + eps) ──
        coeus_ops::add_assign(&mut stdev, &self.eps_t, &backend)
            .map_err(|source| validation::backend(MODULE, source))?;
        coeus_ops::sqrt_assign(&mut stdev, &backend)
            .map_err(|source| validation::backend(MODULE, source))?;

        let ones = {
            let mut cache = self
                .ones_cache
                .try_borrow_mut()
                .map_err(|_| validation::state_borrow(MODULE, "ones_cache"))?;
            if let Some((cached_n, ref cached_ones)) = *cache {
                if cached_n == shape[0] {
                    cached_ones.clone()
                } else {
                    let ones = Tensor::ones_on([shape[0], 1], &backend);
                    *cache = Some((shape[0], ones.clone()));
                    ones
                }
            } else {
                let ones = Tensor::ones_on([shape[0], 1], &backend);
                *cache = Some((shape[0], ones.clone()));
                ones
            }
        };
        let mut istdev = ones;
        coeus_ops::div_assign(&mut istdev, &stdev, &backend)
            .map_err(|source| validation::backend(MODULE, source))?; // [N, 1]

        // ── Normalize ──
        let x_hat = coeus_ops::mul(&xmu, &istdev, &backend); // [N, D]

        // ── Scale and bias ──
        let w_reshaped = self.weight.tensor.reshape([1, d]);
        let b_reshaped = self.bias.tensor.reshape([1, d]);
        let mut out_tensor = coeus_ops::mul(&x_hat, &w_reshaped, &backend);
        coeus_ops::add_assign(&mut out_tensor, &b_reshaped, &backend)
            .map_err(|source| validation::backend(MODULE, source))?;

        Ok(coeus_autograd::layernorm(
            input,
            &self.weight,
            &self.bias,
            out_tensor,
            x_hat,
            istdev,
            self.d_const.clone(),
        ))
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> LayerNorm<T, B> {
    /// Forward pass for any rank ≥ 2 input.
    ///
    /// Collapses all leading dimensions into a single batch dimension, applies
    /// the standard 2-D LayerNorm over the last (`D`) dimension, then restores
    /// the original shape.  All three operations use tracked `coeus_autograd::reshape`
    /// so gradients flow through the entire reshape→normalize→reshape chain.
    ///
    /// # Errors
    ///
    /// Returns a typed module or backend failure when the input has fewer than
    /// two dimensions, its trailing dimension differs from the configured
    /// normalized shape, or normalization execution fails.
    ///
    /// # Examples
    /// ```text
    /// // 3-D Transformer hidden states [batch, seq, d_model]:
    /// let output = layer_norm.forward_nd(&x);
    ///
    /// // 4-D activation map [batch, channels, h, w]:
    /// let output = layer_norm.forward_nd(&x);
    /// ```
    pub fn forward_nd(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let shape = input.tensor.shape_cloned();
        let ndim = shape.len();
        if ndim < 2 {
            return Err(validation::invalid_rank("LayerNorm", "at least 2", ndim));
        }
        if ndim == 2 {
            // Fast path: avoid a no-op reshape pair.
            return self.forward(input);
        }
        let d = shape[ndim - 1];
        let leading: usize = shape[..ndim - 1].iter().product();
        // Tracked flatten [... , D] → [leading, D]
        let flat = coeus_autograd::reshape(input, [leading, d]);
        // Apply 2-D LayerNorm
        let normed = self.forward(&flat)?;
        // Tracked unflatten back to original shape
        Ok(coeus_autograd::reshape(&normed, shape))
    }
}
