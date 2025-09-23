//! Layer Normalization
//!
//! Applies layer normalization over the last dimension of input tensors.
//! Normalizes the input across the feature dimension for each sample.

use crate::{Module, Result};
use coeus_tensor::{FloatDtype, Tensor};

/// Layer Normalization
///
/// Applies layer normalization over the last dimension of input tensors.
/// Normalizes the input across the feature dimension for each sample.
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
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
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

        Ok(Tensor::from_vec(output_data, input_shape.to_vec()))
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
