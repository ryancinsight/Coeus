//! RMSNorm (Root Mean Square Layer Normalization)
//!
//! RMSNorm is a simplified version of LayerNorm that only normalizes by the root mean square
//! of the input, without centering (subtracting mean). This provides better performance
//! and is used in modern transformer architectures like GPT-NeoX, PaLM, and LLaMA.

use crate::core::error::{NNError, Result};
use crate::core::parameter::Parameter;
use autograd::ops::mean;
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{ops::arithmetic::*, FloatExt, Tensor};

/// RMSNorm layer
///
/// RMSNorm(x) = (x / sqrt(mean(x^2) + ε)) * g
/// where g is a learnable parameter and ε is a small constant for numerical stability
pub struct RMSNorm<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync,
    T: DataType
        + 'static
        + FloatExt
        + num_traits::Bounded
        + std::cmp::PartialOrd
        + num_traits::FromPrimitive,
{
    /// Learnable scaling parameter (gamma)
    weight: Parameter<B, S, T>,
    /// Normalized shape
    normalized_shape: Vec<usize>,
    /// Small constant for numerical stability
    eps: f64,
    /// Element-wise scaling factor for initialization
    elementwise_affine: bool,
}

impl<B, S, T> RMSNorm<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync,
    T: DataType
        + 'static
        + FloatExt
        + num_traits::Bounded
        + std::cmp::PartialOrd
        + num_traits::FromPrimitive,
{
    /// Create a new RMSNorm layer
    ///
    /// # Arguments
    /// * `normalized_shape` - Shape of the input to normalize (e.g., [embed_dim])
    /// * `eps` - Small constant for numerical stability (default: 1e-5)
    /// * `elementwise_affine` - Whether to use learnable affine parameters (default: true)
    pub fn new(normalized_shape: Vec<usize>, eps: f64, elementwise_affine: bool) -> Result<Self> {
        let weight = if elementwise_affine {
            // Initialize weight to ones (identity transformation initially)
            let weight_size = normalized_shape.iter().product();
            let weight_data = vec![T::one(); weight_size];
            let weight_tensor = Tensor::from_vec(weight_data, &normalized_shape)?;
            Parameter::new(weight_tensor, "weight".to_string())
        } else {
            // For non-elementwise affine, we still need a dummy parameter
            // Create a minimal tensor for the parameter structure
            let dummy_data = vec![T::one(); 1];
            let dummy_tensor = Tensor::from_vec(dummy_data, &[1])?;
            Parameter::new(dummy_tensor, "weight".to_string())
        };

        Ok(Self {
            weight,
            normalized_shape,
            eps,
            elementwise_affine,
        })
    }

    /// Create RMSNorm with default parameters
    pub fn new_default(normalized_shape: Vec<usize>) -> Result<Self> {
        Self::new(normalized_shape, 1e-5, true)
    }

    /// Forward pass through RMSNorm
    ///
    /// # Arguments
    /// * `input` - Input tensor to normalize
    ///
    /// # Returns
    /// Normalized tensor with same shape as input
    pub fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let input_shape = input.shape().dims();

        // Validate input shape is compatible with normalized_shape
        self.validate_input_shape(input_shape)?;

        // Compute RMS normalization
        let normalized = self.rms_normalize(input)?;

        // Apply elementwise affine transformation if enabled
        if self.elementwise_affine {
            Ok(mul(&normalized, self.weight.data())?)
        } else {
            Ok(normalized)
        }
    }

    /// Compute RMS normalization without affine transformation
    fn rms_normalize(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // RMSNorm(x) = x / sqrt(mean(x^2) + ε)

        // Compute x^2 element-wise
        let squared = mul(input, input)?;

        // Compute mean along normalized dimensions
        let mean_squared = self.compute_mean_along_dims(&squared)?;

        // Add epsilon for numerical stability
        let eps_tensor = Tensor::full_like(&mean_squared, T::from(self.eps).unwrap())?;
        let variance = add(&mean_squared, &eps_tensor)?;

        // Compute sqrt(variance)
        let rms = sqrt(&variance)?;

        // Broadcast RMS back to input shape for element-wise division
        let rms_broadcast = self.broadcast_to_input_shape(&rms, input.shape().dims())?;

        // Normalize: x / rms
        Ok(div(input, &rms_broadcast)?)
    }

    /// Compute mean along the normalized dimensions
    fn compute_mean_along_dims(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // For simplicity, assume we're normalizing the last dimension (most common case)
        // This is the typical case for transformer layers
        if self.normalized_shape.len() == 1 {
            // Normalize along the last dimension
            self.mean_along_last_dim(input)
        } else {
            // For more complex normalization shapes, we'd need more sophisticated reduction
            // For now, fall back to last dimension normalization
            self.mean_along_last_dim(input)
        }
    }

    /// Compute mean along the last dimension
    fn mean_along_last_dim(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let shape = input.shape().dims();
        let last_dim_idx = shape.len() - 1;

        // Use tensor reduction to compute mean along last dimension
        Ok(mean(input, Some(&[last_dim_idx]), false)?)
    }

    /// Broadcast RMS tensor back to input shape
    fn broadcast_to_input_shape(
        &self,
        rms: &Tensor<B, S, T>,
        input_shape: &[usize],
    ) -> Result<Tensor<B, S, T>> {
        let rms_shape = rms.shape().dims();

        // For last-dimension normalization, we need to broadcast along the last axis
        if rms_shape.len() + 1 == input_shape.len() {
            // Expand dimensions to match input shape
            let mut expanded_shape = rms_shape.to_vec();
            expanded_shape.push(input_shape[input_shape.len() - 1]);

            // Create tensor with repeated values along last dimension
            let mut expanded_data = Vec::new();

            for &rms_val in rms.as_slice() {
                for _ in 0..input_shape[input_shape.len() - 1] {
                    expanded_data.push(rms_val);
                }
            }

            Ok(Tensor::from_vec(expanded_data, &expanded_shape)?)
        } else {
            // For now, assume shapes are compatible
            Ok(rms.clone())
        }
    }

    /// Validate that input shape is compatible with normalized shape
    fn validate_input_shape(&self, input_shape: &[usize]) -> Result<()> {
        if input_shape.is_empty() {
            return Err(NNError::InvalidInput {
                message: "Input tensor cannot be empty".to_string(),
            });
        }

        // For last-dimension normalization (most common case)
        if self.normalized_shape.len() == 1 {
            let expected_last_dim = self.normalized_shape[0];
            let actual_last_dim = *input_shape.last().unwrap();

            if actual_last_dim != expected_last_dim {
                return Err(NNError::InvalidInput {
                    message: format!(
                        "Input last dimension {} does not match normalized shape {}",
                        actual_last_dim, expected_last_dim
                    ),
                });
            }
        }

        Ok(())
    }

    /// Get the weight parameter (for training/fine-tuning)
    pub fn weight(&self) -> &Parameter<B, S, T> {
        &self.weight
    }

    /// Get the normalized shape
    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }

    /// Get epsilon value
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Check if elementwise affine is enabled
    pub fn elementwise_affine(&self) -> bool {
        self.elementwise_affine
    }
}

/// RMSNorm configuration for different use cases
pub struct RMSNormConfig {
    pub normalized_shape: Vec<usize>,
    pub eps: f64,
    pub elementwise_affine: bool,
}

impl RMSNormConfig {
    /// Standard RMSNorm configuration for transformer layers
    pub fn standard(embed_dim: usize) -> Self {
        Self {
            normalized_shape: vec![embed_dim],
            eps: 1e-5,
            elementwise_affine: true,
        }
    }

    /// RMSNorm without learnable parameters (for efficiency)
    pub fn no_affine(embed_dim: usize) -> Self {
        Self {
            normalized_shape: vec![embed_dim],
            eps: 1e-5,
            elementwise_affine: false,
        }
    }

    /// RMSNorm with custom epsilon for numerical stability
    pub fn with_eps(embed_dim: usize, eps: f64) -> Self {
        Self {
            normalized_shape: vec![embed_dim],
            eps,
            elementwise_affine: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use num_traits::Float;
    use storage::DenseStorage;

    type TestBackend = CpuBackend<Float32>;
    type TestStorage = DenseStorage<Float32>;
    type TestDataType = Float32;

    #[test]
    fn test_rms_norm_creation() {
        let config = RMSNormConfig::standard(64);
        let rms_norm = RMSNorm::<TestBackend, TestStorage, TestDataType>::new(
            config.normalized_shape,
            config.eps,
            config.elementwise_affine,
        )
        .unwrap();

        assert_eq!(rms_norm.normalized_shape(), &[64]);
        assert_eq!(rms_norm.eps(), 1e-5);
        assert!(rms_norm.elementwise_affine());
    }

    #[test]
    fn test_rms_norm_forward() {
        let rms_norm =
            RMSNorm::<TestBackend, TestStorage, TestDataType>::new_default(vec![4]).unwrap();

        // Create test input
        let input_data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ]; // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let input = Tensor::from_vec(input_data, &[4]).unwrap();

        let output = rms_norm.forward(&input).unwrap();

        // Output should have same shape
        assert_eq!(output.shape().dims(), &[4]);

        // Check that RMS normalization was applied
        let output_data = output.as_slice();

        // First element: 1.0 / 2.7386 ≈ 0.365
        assert!(output_data[0] > Float32::new(0.3) && output_data[0] < Float32::new(0.4));

        // Last element: 4.0 / 2.7386 ≈ 1.46
        assert!(output_data[3] > Float32::new(1.4) && output_data[3] < Float32::new(1.5));
    }

    #[test]
    fn test_rms_norm_preserves_zero_mean() {
        let rms_norm =
            RMSNorm::<TestBackend, TestStorage, TestDataType>::new_default(vec![3]).unwrap();

        // Create input with different scales
        let input_data = vec![Float32::new(0.1), Float32::new(0.2), Float32::new(0.3)];
        let input = Tensor::from_vec(input_data, &[3]).unwrap();

        let output = rms_norm.forward(&input).unwrap();
        let output_data = output.as_slice();

        // RMS norm should preserve relative magnitudes
        // All values should be scaled by the same factor
        let scale = output_data[0] / Float32::new(0.1);
        assert!((output_data[1] / Float32::new(0.2) - scale).abs() < Float32::new(1e-6));
        assert!((output_data[2] / Float32::new(0.3) - scale).abs() < Float32::new(1e-6));
    }

    #[test]
    fn test_rms_norm_shape_validation() {
        let rms_norm =
            RMSNorm::<TestBackend, TestStorage, TestDataType>::new_default(vec![64]).unwrap();

        // Wrong shape should fail
        let wrong_input = Tensor::from_vec(vec![Float32::new(1.0); 32], &[32]).unwrap();
        let result = rms_norm.forward(&wrong_input);
        assert!(result.is_err());
    }

    #[test]
    fn test_rms_norm_no_affine() {
        let rms_norm =
            RMSNorm::<TestBackend, TestStorage, TestDataType>::new(vec![4], 1e-5, false).unwrap();

        let input_data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ];
        let input = Tensor::from_vec(input_data, &[4]).unwrap();

        let output = rms_norm.forward(&input).unwrap();

        // Without affine transformation, weight should be 1.0
        let output_data = output.as_slice();

        // Should be same as with affine but weight=1
        let expected_rms = ((1.0_f32 + 4.0 + 9.0 + 16.0) / 4.0).sqrt();
        assert!(
            (output_data[0] - Float32::new(1.0) / Float32::new(expected_rms)).abs()
                < Float32::new(1e-5)
        );
    }

    #[test]
    fn test_rms_norm_config() {
        let standard = RMSNormConfig::standard(128);
        assert_eq!(standard.normalized_shape, vec![128]);
        assert_eq!(standard.eps, 1e-5);
        assert!(standard.elementwise_affine);

        let no_affine = RMSNormConfig::no_affine(64);
        assert!(!no_affine.elementwise_affine);

        let custom_eps = RMSNormConfig::with_eps(256, 1e-6);
        assert_eq!(custom_eps.eps, 1e-6);
    }
}
