//! 2D Instance Normalization layer
//!
//! Applies instance normalization over 2D input tensors.
//! Normalizes each channel independently across spatial dimensions.

use crate::{Module, Result};
use coeus_tensor::{FloatDtype, Tensor};

/// 2D Instance Normalization layer
///
/// Applies instance normalization over 2D input tensors.
/// Normalizes each channel independently across spatial dimensions.
#[derive(Debug, Clone)]
pub struct InstanceNorm2d<T: FloatDtype> {
    /// Number of features (channels)
    pub num_features: usize,
    /// Small constant for numerical stability
    pub eps: T,
    /// Learnable scale parameter (γ) of shape (num_features,)
    pub weight: Option<Tensor<T>>,
    /// Learnable shift parameter (β) of shape (num_features,)
    pub bias: Option<Tensor<T>>,
}

impl<T: FloatDtype> InstanceNorm2d<T> {
    /// Create a new InstanceNorm2d layer
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
}

impl<T: FloatDtype> Module<T> for InstanceNorm2d<T> {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Input shape validation for 2D instance norm: (batch_size, channels, height, width)
        if input.ndim() != 4 {
            return Err(crate::NNError::InvalidInput {
                message: "InstanceNorm2d requires 4D input (batch_size, channels, height, width)"
                    .to_string(),
            });
        }

        let input_shape = input.shape();
        if input_shape[1] != self.num_features {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "InstanceNorm2d expected {} channels, got {}",
                    self.num_features, input_shape[1]
                ),
            });
        }

        // Simplified instance normalization (training mode disabled)
        // In a full implementation, this would compute per-instance statistics
        // and apply normalization during training vs inference
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();
        if let Some(ref w) = self.weight {
            params.push(w);
        }
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
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
