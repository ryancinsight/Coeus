//! Normalization layers
//!
//! This module provides batch normalization, layer normalization,
//! and other normalization techniques for improving training stability.
//!
//! ## Mathematical Foundation
//!
//! ### Batch Normalization
//! ```math
//! μ_B = (1/m) * Σ xᵢ
//! σ_B² = (1/m) * Σ (xᵢ - μ_B)²
//! x̂ᵢ = (xᵢ - μ_B) / √(σ_B² + ε)
//! yᵢ = γ * x̂ᵢ + β
//! ```
//!
//! ## References
//!
//! - [Ioffe & Szegedy, 2015 - Batch Normalization](https://arxiv.org/abs/1502.03167)

use crate::Module;
use coeus_tensor::{FloatDtype, Tensor};

/// Batch Normalization for 2D inputs (convolutional features)
///
/// Normalizes the input across the batch dimension for each feature map.
/// This helps stabilize training and can act as regularization.
#[derive(Debug, Clone)]
pub struct BatchNorm2d<T: FloatDtype> {
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

impl<T: FloatDtype> BatchNorm2d<T> {
    /// Create a new BatchNorm2d layer
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
}

impl<T: FloatDtype> Module<T> for BatchNorm2d<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // Simplified implementation: just return input unchanged
        // Full implementation would require mutable access for training statistics
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.weight, &mut self.bias]
    }
}

/// Batch Normalization for 1D inputs (fully connected layers)
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
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // Simplified implementation: just return input unchanged
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.weight, &mut self.bias]
    }
}

/// Batch Normalization for 3D inputs (volumetric data)
///
/// Normalizes the input across the batch dimension for each feature map.
/// This helps stabilize training and can act as regularization.
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
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        if input.ndim() != 5 {
            return Err(crate::NNError::InvalidInput {
                message:
                    "BatchNorm3d requires 5D input (batch_size, channels, depth, height, width)"
                        .to_string(),
            });
        }

        if input.shape()[1] != self.num_features {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "Expected {} channels, got {}",
                    self.num_features,
                    input.shape()[1]
                ),
            });
        }

        // Simplified implementation: just return input unchanged
        // In a full implementation, this would compute running statistics and normalize
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.weight, &mut self.bias]
    }
}

/// Layer Normalization
///
/// Normalizes the input across the feature dimension.
#[derive(Debug, Clone)]
pub struct LayerNorm<T: FloatDtype> {
    /// Normalized shape
    pub normalized_shape: Vec<usize>,
    /// Learnable scale parameter (γ)
    pub weight: Option<Tensor<T>>,
    /// Learnable shift parameter (β)
    pub bias: Option<Tensor<T>>,
}

impl<T: FloatDtype> LayerNorm<T> {
    /// Create a new LayerNorm layer
    ///
    /// # Arguments
    /// * `normalized_shape` - Shape of the input to normalize
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        let weight = Some(Tensor::ones(normalized_shape.clone()));
        let bias = Some(Tensor::zeros(normalized_shape.clone()));

        Self {
            normalized_shape,
            weight,
            bias,
        }
    }
}

impl<T: FloatDtype> Module<T> for LayerNorm<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // Layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta

        let input_shape = input.shape();
        let feature_size = input_shape[input_shape.len() - 1];

        // Normalize over the last dimension (features)
        // Works for both 2D (batch_size, feature_size) and 3D (batch_size, seq_len, feature_size) inputs
        if input_shape.is_empty() {
            return Err(crate::NNError::ShapeMismatch {
                expected: vec![feature_size],
                actual: input_shape.to_vec(),
            });
        }

        let batch_seq_size = input_shape.iter().take(input_shape.len() - 1).product();
        let mut output_data = Vec::with_capacity(input.data().len());

        for b in 0..batch_seq_size {
            // Calculate mean and variance for this sample
            let mut sum = T::zero();
            let mut sum_sq = T::zero();

            for f in 0..feature_size {
                let idx = b * feature_size + f;
                let val = input.data()[idx];
                sum = sum + val;
                sum_sq = sum_sq + val * val;
            }

            let mean = sum / T::from(feature_size as f64).unwrap();
            let var = (sum_sq / T::from(feature_size as f64).unwrap()) - mean * mean;

            // Normalize and apply affine transformation
            for f in 0..feature_size {
                let idx = b * feature_size + f;
                let val = input.data()[idx];
                let normalized = (val - mean) / (var + T::from(1e-5).unwrap()).sqrt();

                // Apply gamma and beta if they exist
                let gamma = if let Some(ref w) = self.weight {
                    w.data()[f]
                } else {
                    T::one()
                };

                let beta = if let Some(ref b) = self.bias {
                    b.data()[f]
                } else {
                    T::zero()
                };

                output_data.push(normalized * gamma + beta);
            }
        }

        let mut output = Tensor::from_vec(output_data, input_shape.to_vec());

        // Propagate requires_grad flag
        if input.requires_grad() || self.weight.as_ref().is_some_and(|w| w.requires_grad()) {
            output.set_requires_grad(true);
        }

        Ok(output)
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

/// Instance Normalization for 1D inputs
///
/// Normalizes each channel separately across spatial dimensions.
/// Useful for style transfer and generative models.
#[derive(Debug, Clone)]
pub struct InstanceNorm1d<T: FloatDtype> {
    /// Number of features (channels)
    pub num_features: usize,
    /// Small constant for numerical stability
    pub eps: T,
    /// Learnable scale parameter (γ) of shape (num_features,)
    pub weight: Option<Tensor<T>>,
    /// Learnable shift parameter (β) of shape (num_features,)
    pub bias: Option<Tensor<T>>,
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
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        if input.ndim() != 3 {
            return Err(crate::NNError::InvalidInput {
                message: "InstanceNorm1d requires 3D input (batch_size, channels, length)"
                    .to_string(),
            });
        }

        if input.shape()[1] != self.num_features {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "Expected {} channels, got {}",
                    self.num_features,
                    input.shape()[1]
                ),
            });
        }

        // Simplified implementation: just return input unchanged
        // In a full implementation, this would normalize each channel separately
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

/// Instance Normalization for 2D inputs
///
/// Normalizes each channel separately across spatial dimensions.
/// Useful for style transfer and generative models.
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

    /// Create a new InstanceNorm2d layer with custom parameters
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

impl<T: FloatDtype> Module<T> for InstanceNorm2d<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        if input.ndim() != 4 {
            return Err(crate::NNError::InvalidInput {
                message: "InstanceNorm2d requires 4D input (batch_size, channels, height, width)"
                    .to_string(),
            });
        }

        if input.shape()[1] != self.num_features {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "Expected {} channels, got {}",
                    self.num_features,
                    input.shape()[1]
                ),
            });
        }

        // Simplified implementation: just return input unchanged
        // In a full implementation, this would normalize each channel separately
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

/// Instance Normalization for 3D inputs
///
/// Normalizes each channel separately across spatial dimensions.
/// Useful for 3D data processing and volumetric data.
#[derive(Debug, Clone)]
pub struct InstanceNorm3d<T: FloatDtype> {
    /// Number of features (channels)
    pub num_features: usize,
    /// Small constant for numerical stability
    pub eps: T,
    /// Learnable scale parameter (γ) of shape (num_features,)
    pub weight: Option<Tensor<T>>,
    /// Learnable shift parameter (β) of shape (num_features,)
    pub bias: Option<Tensor<T>>,
}

impl<T: FloatDtype> InstanceNorm3d<T> {
    /// Create a new InstanceNorm3d layer
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

    /// Create a new InstanceNorm3d layer with custom parameters
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

impl<T: FloatDtype> Module<T> for InstanceNorm3d<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        if input.ndim() != 5 {
            return Err(crate::NNError::InvalidInput {
                message:
                    "InstanceNorm3d requires 5D input (batch_size, channels, depth, height, width)"
                        .to_string(),
            });
        }

        if input.shape()[1] != self.num_features {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "Expected {} channels, got {}",
                    self.num_features,
                    input.shape()[1]
                ),
            });
        }

        // Simplified implementation: just return input unchanged
        // In a full implementation, this would normalize each channel separately
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

/// Group Normalization
///
/// Divides the channels into groups and normalizes within each group.
/// This provides an alternative to batch normalization that doesn't depend on batch size.
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
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        if input.ndim() < 3 {
            return Err(crate::NNError::InvalidInput {
                message: "GroupNorm requires at least 3D input (..., channels, ...)".to_string(),
            });
        }

        if input.shape()[1] != self.num_channels {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "Expected {} channels, got {}",
                    self.num_channels,
                    input.shape()[1]
                ),
            });
        }

        // Simplified implementation: just return input unchanged
        // In a full implementation, this would normalize within each group
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_norm_2d_creation() {
        let bn: BatchNorm2d<f64> = BatchNorm2d::new(64);

        assert_eq!(bn.num_features, 64);
        assert_eq!(bn.weight.shape(), &[64]);
        assert_eq!(bn.bias.shape(), &[64]);
    }

    #[test]
    fn test_batch_norm_1d_creation() {
        let bn: BatchNorm1d<f64> = BatchNorm1d::new(128);

        assert_eq!(bn.num_features, 128);
        assert_eq!(bn.weight.shape(), &[128]);
    }

    #[test]
    fn test_layer_norm_creation() {
        let ln: LayerNorm<f64> = LayerNorm::new(vec![64]);

        assert_eq!(ln.normalized_shape, vec![64]);
        assert!(ln.weight.is_some());
    }

    #[test]
    fn test_batch_norm_2d_parameters() {
        let bn: BatchNorm2d<f64> = BatchNorm2d::new(32);

        assert_eq!(bn.parameters().len(), 2); // weight and bias

        let mut bn_mut: BatchNorm2d<f64> = BatchNorm2d::new(32);
        assert_eq!(bn_mut.parameters_mut().len(), 2);
    }

    #[test]
    fn test_batch_norm_3d_creation() {
        let bn: BatchNorm3d<f64> = BatchNorm3d::new(64);

        assert_eq!(bn.num_features, 64);
        assert_eq!(bn.weight.shape(), &[64]);
        assert_eq!(bn.bias.shape(), &[64]);
    }

    #[test]
    fn test_instance_norm_1d_creation() {
        let in1d: InstanceNorm1d<f64> = InstanceNorm1d::new(32);

        assert_eq!(in1d.num_features, 32);
        assert!(in1d.weight.is_some());
        assert!(in1d.bias.is_some());
    }

    #[test]
    fn test_instance_norm_2d_creation() {
        let in2d: InstanceNorm2d<f64> = InstanceNorm2d::new(64);

        assert_eq!(in2d.num_features, 64);
        assert!(in2d.weight.is_some());
        assert!(in2d.bias.is_some());
    }

    #[test]
    fn test_instance_norm_3d_creation() {
        let in3d: InstanceNorm3d<f64> = InstanceNorm3d::new(16);

        assert_eq!(in3d.num_features, 16);
        assert!(in3d.weight.is_some());
        assert!(in3d.bias.is_some());
    }

    #[test]
    fn test_group_norm_creation() {
        let gn: GroupNorm<f64> = GroupNorm::new(4, 32);

        assert_eq!(gn.num_groups, 4);
        assert_eq!(gn.num_channels, 32);
        assert!(gn.weight.is_some());
        assert!(gn.bias.is_some());
    }

    #[test]
    #[should_panic]
    fn test_group_norm_invalid_groups() {
        // This should panic because 32 is not divisible by 7
        let _gn: GroupNorm<f64> = GroupNorm::new(7, 32);
    }

    #[test]
    fn test_normalization_parameters() {
        let bn3d: BatchNorm3d<f64> = BatchNorm3d::new(16);
        assert_eq!(bn3d.parameters().len(), 2); // weight and bias

        let in1d: InstanceNorm1d<f64> = InstanceNorm1d::new(16);
        assert_eq!(in1d.parameters().len(), 2); // weight and bias

        let gn: GroupNorm<f64> = GroupNorm::new(4, 16);
        assert_eq!(gn.parameters().len(), 2); // weight and bias
    }
}
