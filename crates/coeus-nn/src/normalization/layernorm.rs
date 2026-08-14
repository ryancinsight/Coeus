//! Layer normalization over a configured trailing shape.

use super::validation;
use crate::module::{Module, ModuleError};
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;
use std::cell::RefCell;

/// Shape of the trailing dimensions normalized by [`LayerNorm`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormalizedShape(Vec<usize>);

impl NormalizedShape {
    /// Borrow the configured trailing dimensions.
    #[must_use]
    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }

    fn is_valid(&self) -> bool {
        !self.0.is_empty() && self.0.iter().all(|&dimension| dimension > 0)
    }
}

impl From<usize> for NormalizedShape {
    fn from(dimension: usize) -> Self {
        Self(vec![dimension])
    }
}

impl From<Vec<usize>> for NormalizedShape {
    fn from(dimensions: Vec<usize>) -> Self {
        Self(dimensions)
    }
}

impl From<&[usize]> for NormalizedShape {
    fn from(dimensions: &[usize]) -> Self {
        Self(dimensions.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for NormalizedShape {
    fn from(dimensions: [usize; N]) -> Self {
        Self(dimensions.into())
    }
}

/// Functional (stateless) layer normalization.
///
/// Normalizes the configured trailing dimensions of every input with rank at
/// least two. `weight` and `bias` default to ones and zeros with the requested
/// `normalized_shape`.
///
/// # Errors
///
/// Returns a typed module or backend failure when the input rank, trailing
/// dimensions, affine parameter shapes, or epsilon violate the LayerNorm
/// contract, or when a backend operation fails.
pub fn layer_norm<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    normalized_shape: impl Into<NormalizedShape>,
    weight: Option<&Var<T, B>>,
    bias: Option<&Var<T, B>>,
    eps: f64,
) -> Result<Var<T, B>, ModuleError<B::Error>>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let normalized_shape = normalized_shape.into();
    if !normalized_shape.is_valid() {
        return Err(validation::shape_mismatch(
            "LayerNorm",
            "normalized_shape",
            &[1],
            normalized_shape.as_slice(),
        ));
    }
    let backend = B::default();
    let w = weight.cloned().unwrap_or_else(|| {
        Var::new(
            Tensor::ones_on(normalized_shape.as_slice(), &backend),
            false,
        )
    });
    let b = bias.cloned().unwrap_or_else(|| {
        Var::new(
            Tensor::zeros_on(normalized_shape.as_slice(), &backend),
            false,
        )
    });
    for (parameter, actual) in [("weight", w.tensor.shape()), ("bias", b.tensor.shape())] {
        if actual != normalized_shape.as_slice() {
            return Err(validation::shape_mismatch(
                "LayerNorm",
                parameter,
                normalized_shape.as_slice(),
                actual,
            ));
        }
    }
    let layer = LayerNorm::from_parts(w, b, eps);
    layer.forward(input)
}

/// Layer Normalization module.
///
/// Applies Layer Normalization over a configured trailing shape of tensors
/// with rank two or greater.
#[derive(Clone)]
pub struct LayerNorm<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Trainable scale parameter gamma: `normalized_shape`.
    pub weight: Var<T, B>,
    /// Trainable shift parameter beta: `normalized_shape`.
    pub bias: Var<T, B>,
    /// Small value for numerical stability.
    pub eps: f64,
    /// Cached epsilon tensor: `[1]`.
    eps_t: Tensor<T, B>,
    /// Cached normalized-element count: `[1]`.
    d_const: Tensor<T, B>,
    /// Cached ones tensor of shape `[N, 1]`: (N, ones_tensor).
    ones_cache: RefCell<Option<(usize, Tensor<T, B>)>>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> LayerNorm<T, B> {
    /// Create a new LayerNorm layer for a single feature dimension.
    pub fn new(normalized_shape: usize, eps: f64) -> Self {
        Self::from_shape(normalized_shape, eps)
    }

    /// Create a LayerNorm layer for one or more trailing dimensions.
    pub fn from_shape(normalized_shape: impl Into<NormalizedShape>, eps: f64) -> Self {
        let normalized_shape = normalized_shape.into();
        let backend = B::default();
        let weight = Var::new(Tensor::ones_on(normalized_shape.as_slice(), &backend), true);
        let bias = Var::new(
            Tensor::zeros_on(normalized_shape.as_slice(), &backend),
            true,
        );
        let eps_t = Tensor::full_on([1], T::from_f64(eps), &backend);
        let d_const = Tensor::full_on([1], T::from_f64(weight.tensor.numel() as f64), &backend);
        Self {
            weight,
            bias,
            eps,
            eps_t,
            d_const,
            ones_cache: RefCell::new(None),
        }
    }

    /// Create a LayerNorm layer from existing affine parameters.
    pub fn from_parts(weight: Var<T, B>, bias: Var<T, B>, eps: f64) -> Self {
        let backend = B::default();
        let eps_t = Tensor::full_on([1], T::from_f64(eps), &backend);
        let d_const = Tensor::full_on([1], T::from_f64(weight.tensor.numel() as f64), &backend);
        Self {
            weight,
            bias,
            eps,
            eps_t,
            d_const,
            ones_cache: RefCell::new(None),
        }
    }

    fn normalize_flat(
        &self,
        input: &Var<T, B>,
        normalized_size: usize,
    ) -> Result<Var<T, B>, ModuleError<B::Error>> {
        const MODULE: &str = "LayerNorm";
        let batch = input.tensor.shape()[0];
        let backend = B::default();

        let mean_t = coeus_ops::mean_axis(&input.tensor, 1, &backend)
            .map_err(|source| validation::backend(MODULE, source))?;
        let xmu = coeus_ops::sub(&input.tensor, &mean_t, &backend);
        let xmu_sq = coeus_ops::mul(&xmu, &xmu, &backend);
        let mut stdev = coeus_ops::mean_axis(&xmu_sq, 1, &backend)
            .map_err(|source| validation::backend(MODULE, source))?;

        coeus_ops::add_assign(&mut stdev, &self.eps_t, &backend)
            .map_err(|source| validation::backend(MODULE, source))?;
        coeus_ops::sqrt_assign(&mut stdev, &backend)
            .map_err(|source| validation::backend(MODULE, source))?;

        let ones = {
            let mut cache = self
                .ones_cache
                .try_borrow_mut()
                .map_err(|_| validation::state_borrow(MODULE, "ones_cache"))?;
            if let Some((cached_batch, ref cached_ones)) = *cache {
                if cached_batch == batch {
                    cached_ones.clone()
                } else {
                    let ones = Tensor::ones_on([batch, 1], &backend);
                    *cache = Some((batch, ones.clone()));
                    ones
                }
            } else {
                let ones = Tensor::ones_on([batch, 1], &backend);
                *cache = Some((batch, ones.clone()));
                ones
            }
        };
        let mut istdev = ones;
        coeus_ops::div_assign(&mut istdev, &stdev, &backend)
            .map_err(|source| validation::backend(MODULE, source))?;

        let x_hat = coeus_ops::mul(&xmu, &istdev, &backend);
        let w_reshaped = self.weight.tensor.reshape([1, normalized_size]);
        let b_reshaped = self.bias.tensor.reshape([1, normalized_size]);
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

/// Implements the [`crate::module::Module`] interface for [`LayerNorm`].
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for LayerNorm<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    /// Write updated `(weight, bias)` values back to this module.
    fn load_parameters(&mut self, params: &[Var<T, B>]) {
        self.weight = params[0].clone();
        self.bias = params[1].clone();
    }

    /// Forward pass over the configured trailing dimensions.
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        self.forward_nd(input)
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> LayerNorm<T, B> {
    /// Forward pass for any rank ≥ 2 input.
    ///
    /// The configured suffix is flattened into one normalized feature axis for
    /// the provider kernel, then the original input shape is restored through
    /// tracked reshape operations.
    ///
    /// # Errors
    ///
    /// Returns a typed module or backend failure when the input rank, trailing
    /// dimensions, affine parameter shapes, or epsilon violate the contract,
    /// or when normalization execution fails.
    pub fn forward_nd(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        const MODULE: &str = "LayerNorm";
        let input_shape = input.tensor.shape_cloned();
        if input_shape.len() < 2 {
            return Err(validation::invalid_rank(
                MODULE,
                "at least 2",
                input_shape.len(),
            ));
        }

        let normalized_shape = self.weight.tensor.shape_cloned();
        if normalized_shape.is_empty() || normalized_shape.contains(&0) {
            return Err(validation::shape_mismatch(
                MODULE,
                "normalized_shape",
                &[1],
                &normalized_shape,
            ));
        }
        for (parameter, actual) in [
            ("weight", self.weight.tensor.shape()),
            ("bias", self.bias.tensor.shape()),
        ] {
            if actual != &*normalized_shape {
                return Err(validation::shape_mismatch(
                    MODULE,
                    parameter,
                    &normalized_shape,
                    actual,
                ));
            }
        }
        if !self.eps.is_finite() || self.eps < 0.0 {
            return Err(ModuleError::InvalidEpsilon { module: MODULE });
        }

        let Some(suffix_start) = input_shape.len().checked_sub(normalized_shape.len()) else {
            return Err(validation::shape_mismatch(
                MODULE,
                "input trailing dimensions",
                &normalized_shape,
                &input_shape,
            ));
        };
        let input_suffix = &input_shape[suffix_start..];
        if input_suffix != &*normalized_shape {
            return Err(validation::shape_mismatch(
                MODULE,
                "input trailing dimensions",
                &normalized_shape,
                input_suffix,
            ));
        }

        let normalized_size = self.weight.tensor.numel();
        let Some(batch) = input.tensor.numel().checked_div(normalized_size) else {
            return Err(validation::shape_mismatch(
                MODULE,
                "input",
                &normalized_shape,
                &input_shape,
            ));
        };
        if input_shape.len() == 2 && normalized_shape.len() == 1 {
            return self.normalize_flat(input, normalized_size);
        }

        let flat = coeus_autograd::reshape(input, [batch, normalized_size]);
        let normalized = self.normalize_flat(&flat, normalized_size)?;
        Ok(coeus_autograd::reshape(&normalized, input_shape))
    }
}
