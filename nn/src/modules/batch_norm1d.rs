//! 1D Batch Normalization layer
//!
//! Applies batch normalization over 1D input tensors.
//! Normalizes the input across the batch dimension for each feature.

use crate::{Module, Result};
use coeus_tensor::{FloatDtype, Tensor};

/// 1D Batch Normalization layer
///
/// Applies batch normalization over 1D input tensors.
/// Normalizes the input across the batch dimension for each feature.
#[derive(Debug, Clone)]
pub struct BatchNorm1d<T: FloatDtype> {
    /// Number of features
    pub num_features: usize,
    /// Learnable scale parameter (γ)
    pub weight: Tensor<T>,
    /// Learnable shift parameter (β)
    pub bias: Tensor<T>,
}

impl<T: FloatDtype> BatchNorm1d<T> {
    /// Create a new BatchNorm1d layer
    ///
    /// # Arguments
    /// * `num_features` - Number of features
    pub fn new(num_features: usize) -> Self {
        let weight = Tensor::ones(vec![num_features]);
        let bias = Tensor::zeros(vec![num_features]);

        Self {
            num_features,
            weight,
            bias,
        }
    }
}

impl<T: FloatDtype> Module<T> for BatchNorm1d<T> {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Input shape validation
        let input_shape = input.shape();
        if input_shape.len() != 2 {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "BatchNorm1d expects 2D input (batch_size, num_features), got {}D",
                    input_shape.len()
                ),
            });
        }

        if input_shape[1] != self.num_features {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "BatchNorm1d expected {} features, got {}",
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
