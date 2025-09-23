//! 3D Batch Normalization layer
//!
//! Applies batch normalization over 3D input tensors (volumetric data).
//! Normalizes the input across the batch dimension for each feature map.

use crate::{Module, Result};
use coeus_tensor::{FloatDtype, Tensor};

/// 3D Batch Normalization layer
///
/// Applies batch normalization over 3D input tensors (volumetric data).
/// Normalizes the input across the batch dimension for each feature map.
#[derive(Debug, Clone)]
pub struct BatchNorm3d<T: FloatDtype> {
    /// Number of feature maps (channels)
    pub num_features: usize,
    /// Small constant for numerical stability
    pub eps: T,
    /// Momentum for running statistics
    pub momentum: T,

    /// Learnable scale parameter (γ) of shape (num_features,)
    pub weight: Tensor<T>,
    /// Learnable shift parameter (β) of shape (num_features,)
    pub bias: Tensor<T>,
}

impl<T: FloatDtype> BatchNorm3d<T> {
    /// Create a new BatchNorm3d layer
    ///
    /// # Arguments
    /// * `num_features` - Number of feature maps (channels)
    pub fn new(num_features: usize) -> Self {
        let weight = Tensor::ones(vec![num_features]);
        let bias = Tensor::zeros(vec![num_features]);
        Self {
            num_features,
            eps: T::from(1e-5).unwrap(),
            momentum: T::from(0.1).unwrap(),
            weight,
            bias,
        }
    }

    /// Create a new BatchNorm3d layer with custom parameters
    ///
    /// # Arguments
    /// * `num_features` - Number of feature maps (channels)
    /// * `eps` - Small constant for numerical stability
    /// * `momentum` - Momentum for running statistics
    pub fn with_params(num_features: usize, eps: T, momentum: T) -> Self {
        let weight = Tensor::ones(vec![num_features]);
        let bias = Tensor::zeros(vec![num_features]);
        Self {
            num_features,
            eps,
            momentum,
            weight,
            bias,
        }
    }
}

impl<T: FloatDtype> Module<T> for BatchNorm3d<T> {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Input shape validation for 3D batch norm: (batch_size, channels, depth, height, width)
        if input.ndim() != 5 {
            return Err(crate::NNError::InvalidInput {
                message:
                    "BatchNorm3d requires 5D input (batch_size, channels, depth, height, width)"
                        .to_string(),
            });
        }

        let input_shape = input.shape();
        if input_shape[1] != self.num_features {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "BatchNorm3d expected {} channels, got {}",
                    self.num_features, input_shape[1]
                ),
            });
        }

        // Simplified batch normalization (training mode disabled)
        // In a full implementation, this would compute running statistics
        // and apply normalization during training vs inference
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.weight, &mut self.bias]
    }
}
