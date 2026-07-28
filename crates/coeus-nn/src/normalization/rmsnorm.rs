//! RMS normalization layers and helpers.
//!
//! [`RMSNorm`] rescales activations by their root-mean-square value over the
//! last dimension without subtracting the mean.

use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;

/// Functional (stateless) RMS normalization.
///
/// Applies RMSNorm over the last dimension of a rank-2 input `[N, D]`.
/// `weight` defaults to ones of shape `[D]`.
pub fn rms_norm<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: Option<&Var<T, B>>,
    eps: f64,
) -> Result<Var<T, B>, B::Error> {
    let d = input.tensor.shape().last().copied().unwrap_or(1);
    let backend = B::default();
    let w = match weight {
        Some(weight) => weight.clone(),
        None => Var::new(Tensor::ones_on([d], &backend)?, false)?,
    };
    RMSNorm::from_parts(w, eps)?.forward(input)
}

/// Root Mean Square Normalization (RMSNorm) module.
///
/// Applies RMSNorm over the last dimension of a 2D tensor `[N, D]`.
#[derive(Clone)]
pub struct RMSNorm<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Trainable scale parameter gamma: `[D]`.
    pub weight: Var<T, B>,
    /// Small value for numerical stability.
    pub eps: f64,
    /// Cached epsilon tensor: `[1]`.
    eps_t: Tensor<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> RMSNorm<T, B> {
    /// Create a new RMSNorm layer for a given feature dimension.
    pub fn new(normalized_shape: usize, eps: f64) -> Result<Self, B::Error> {
        let backend = B::default();
        let weight = Var::new(Tensor::ones_on([normalized_shape], &backend)?, true)?;
        let eps_t = Tensor::full_on([1], T::from_f64(eps), &backend)?;
        Ok(Self { weight, eps, eps_t })
    }

    /// Create an RMSNorm layer from existing parameters.
    pub fn from_parts(weight: Var<T, B>, eps: f64) -> Result<Self, B::Error> {
        let backend = B::default();
        let eps_t = Tensor::full_on([1], T::from_f64(eps), &backend)?;
        Ok(Self { weight, eps, eps_t })
    }
}

/// Implements the [`crate::module::Module`] interface for [`RMSNorm`].
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for RMSNorm<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![self.weight.clone()]
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
        let shape = input.tensor.shape_cloned();
        assert_eq!(
            shape.len(),
            2,
            "RMSNorm expects 2D input [batch_size, normalized_shape]"
        );
        let _n = shape[0];
        let d = shape[1];
        let backend = B::default();

        // ── Mean square ──
        let x_sq = coeus_ops::mul(&input.tensor, &input.tensor, &backend)?;
        let mut rms = coeus_ops::mean_axis(&x_sq, 1, &backend)?; // [N, 1]

        // ── RMS ──
        coeus_ops::add_assign(&mut rms, &self.eps_t, &backend)?;
        coeus_ops::sqrt_assign(&mut rms, &backend)?;

        // ── Normalize ──
        let x_hat = coeus_ops::div(&input.tensor, &rms, &backend)?; // [N, D]

        // ── Scale ──
        let w_reshaped = self.weight.tensor.reshape([1, d]);
        let out_tensor = coeus_ops::mul(&x_hat, &w_reshaped, &backend)?;

        coeus_autograd::rmsnorm(input, &self.weight, out_tensor, x_hat, rms)
    }
}
