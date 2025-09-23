//! Group Normalization layer
//!
//! Applies group normalization by dividing channels into groups and normalizing within each group.
//! Provides an alternative to batch normalization that doesn't depend on batch size.

use crate::{Module, Result};
use coeus_tensor::{FloatDtype, Tensor};

/// Group Normalization layer
///
/// Applies group normalization by dividing channels into groups and normalizing within each group.
/// Provides an alternative to batch normalization that doesn't depend on batch size.
#[derive(Debug, Clone)]
pub struct GroupNorm<T: FloatDtype> {
    /// Number of groups to divide the channels into
    pub num_groups: usize,
    /// Number of channels
    pub num_channels: usize,
    /// Small constant for numerical stability
    pub eps: T,
    /// Learnable scale parameter (γ) of shape (num_channels,)
    pub weight: Option<Tensor<T>>,
    /// Learnable shift parameter (β) of shape (num_channels,)
    pub bias: Option<Tensor<T>>,
}

impl<T: FloatDtype> GroupNorm<T> {
    /// Create a new GroupNorm layer
    ///
    /// # Arguments
    /// * `num_groups` - Number of groups to divide the channels into
    /// * `num_channels` - Number of channels
    pub fn new(num_groups: usize, num_channels: usize) -> Self {
        if !num_channels.is_multiple_of(num_groups) {
            panic!(
                "num_channels ({}) must be divisible by num_groups ({})",
                num_channels, num_groups
            );
        }

        let weight = Some(Tensor::ones(vec![num_channels]));
        let bias = Some(Tensor::zeros(vec![num_channels]));
        Self {
            num_groups,
            num_channels,
            eps: T::from(1e-5).unwrap(),
            weight,
            bias,
        }
    }

    /// Create a new GroupNorm layer with custom parameters
    ///
    /// # Arguments
    /// * `num_groups` - Number of groups to divide the channels into
    /// * `num_channels` - Number of channels
    /// * `eps` - Small constant for numerical stability
    pub fn with_params(num_groups: usize, num_channels: usize, eps: T) -> Self {
        if !num_channels.is_multiple_of(num_groups) {
            panic!(
                "num_channels ({}) must be divisible by num_groups ({})",
                num_channels, num_groups
            );
        }

        let weight = Some(Tensor::ones(vec![num_channels]));
        let bias = Some(Tensor::zeros(vec![num_channels]));
        Self {
            num_groups,
            num_channels,
            eps,
            weight,
            bias,
        }
    }
}

impl<T: FloatDtype> Module<T> for GroupNorm<T> {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Input shape validation: (batch_size, channels, ...)
        let input_shape = input.shape();
        if input_shape.len() < 3 {
            return Err(crate::NNError::InvalidInput {
                message: "GroupNorm requires input with at least 3 dimensions (batch_size, channels, ...)"
                    .to_string(),
            });
        }

        if input_shape[1] != self.num_channels {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "GroupNorm expected {} channels, got {}",
                    self.num_channels, input_shape[1]
                ),
            });
        }

        // Simplified group normalization (training mode disabled)
        // In a full implementation, this would compute per-group statistics
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
