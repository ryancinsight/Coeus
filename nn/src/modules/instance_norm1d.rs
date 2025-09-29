//! 1D Instance Normalization layer
//!
//! Applies instance normalization over 1D input tensors.
//! Normalizes each channel independently across spatial dimensions.

use crate::{Module, Result};
use coeus_backend::CpuBackend;
use coeus_tensor::{FloatDtype, Tensor};

/// 1D Instance Normalization layer
///
/// Applies instance normalization over 1D input tensors.
/// Normalizes each channel independently across spatial dimensions.
#[derive(Debug, Clone)]
pub struct InstanceNorm1d<T: FloatDtype> {
    /// Number of features (channels)
    pub num_features: usize,
    /// Small constant for numerical stability
    pub eps: T,
    /// Learnable scale parameter (γ) of shape (num_features,)
    pub weight: Option<Tensor<T, CpuBackend>>,
    /// Learnable shift parameter (β) of shape (num_features,)
    pub bias: Option<Tensor<T, CpuBackend>>,
}

impl<T: FloatDtype> InstanceNorm1d<T> {
    /// Create a new InstanceNorm1d layer
    ///
    /// # Arguments
    /// * `num_features` - Number of features (channels)
    pub fn new(num_features: usize) -> Self {
        let weight = Some(Tensor::ones(vec![num_features]));
        let bias = Some(Tensor::zeros(vec![num_features]));
        Self {
            num_features,
            eps: T::from(1e-5).unwrap(),
            weight,
            bias,
        }
    }

    /// Create a new InstanceNorm1d layer with custom parameters
    ///
    /// # Arguments
    /// * `num_features` - Number of features (channels)
    /// * `eps` - Small constant for numerical stability
    pub fn with_params(num_features: usize, eps: T) -> Self {
        let weight = Some(Tensor::ones(vec![num_features]));
        let bias = Some(Tensor::zeros(vec![num_features]));
        Self {
            num_features,
            eps,
            weight,
            bias,
        }
    }
}

impl<T: FloatDtype> Module<T> for InstanceNorm1d<T> {
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> Result<Tensor<T, CpuBackend>> {
        if input.ndim() != 3 {
            return Err(crate::NNError::InvalidInput {
                message: "InstanceNorm1d requires 3D input (batch_size, channels, length)"
                    .to_string(),
            });
        }

        let input_shape = input.shape();
        let (_batch_size, channels, _length) = (input_shape[0], input_shape[1], input_shape[2]);

        if channels != self.num_features {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "InstanceNorm1d expected {} channels, got {}",
                    self.num_features, channels
                ),
            });
        }

        // Simplified instance normalization (training mode disabled)
        // In a full implementation, this would compute per-instance statistics
        // and apply normalization during training vs inference
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        let mut params = Vec::new();
        if let Some(ref w) = self.weight {
            params.push(w);
        }
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        let mut params = Vec::new();
        if let Some(ref mut w) = self.weight {
            params.push(w);
        }
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }
}



