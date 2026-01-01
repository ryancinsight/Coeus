//! Layer Normalization for neural networks.

use std::marker::PhantomData;

use backend::{Backend, CpuBackend};
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::error::Result;
use crate::module::Module;
use crate::parameter::Parameter;

/// Layer Normalization layer.
///
/// Normalizes the input across the feature dimension(s) and applies a learnable
/// affine transformation. Unlike Batch Normalization, Layer Normalization computes
/// statistics independently for each sample, making it suitable for variable-length
/// sequences and transformers.
///
/// Formula:
/// ```text
/// mean = Σ(x) / D
/// var = Σ((x - mean)²) / D
/// output = γ * (x - mean) / √(var + ε) + β
/// ```
///
/// Where:
/// - `D` = product of normalized_shape dimensions
/// - `γ` = learnable scale parameter (initialized to 1)
/// - `β` = learnable shift parameter (initialized to 0)
/// - `ε` = numerical stability constant (default: 1e-5)
///
/// # Examples
/// ```rust
/// use nn::{LayerNorm, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// // Create LayerNorm for [batch_size, seq_len, hidden_dim=128]
/// let layer_norm = LayerNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(vec![128], 1e-5);
///
/// // Input: [2, 10, 128]
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2, 10, 128]).unwrap();
///
/// // Output: [2, 10, 128] (normalized across hidden_dim)
/// let output = layer_norm.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[2, 10, 128]);
/// ```
#[derive(Debug, Clone)]
pub struct LayerNorm<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType,
{
    /// Shape to normalize over (e.g., [hidden_dim] for transformers)
    pub normalized_shape: Vec<usize>,
    /// Scale parameter γ (initialized to 1)
    pub weight: Parameter<B, S, T>,
    /// Shift parameter β (initialized to 0)
    pub bias: Parameter<B, S, T>,
    /// Numerical stability constant ε
    pub eps: f64,
    _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> LayerNorm<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Create a new LayerNorm layer.
    ///
    /// # Arguments
    /// * `normalized_shape` - Shape to normalize over (e.g., [hidden_dim])
    /// * `eps` - Numerical stability constant (default: 1e-5)
    ///
    /// # Weight Initialization
    /// - `weight` (γ): Initialized to 1
    /// - `bias` (β): Initialized to 0
    pub fn new(normalized_shape: Vec<usize>, eps: f64) -> Self {
        assert!(
            !normalized_shape.is_empty(),
            "normalized_shape cannot be empty"
        );
        assert!(eps > 0.0, "eps must be > 0");

        let num_features: usize = normalized_shape.iter().product();

        // Initialize weight (γ) to 1
        let weight_data = vec![T::one(); num_features];
        let weight_tensor = Tensor::<B, S, T>::from_vec(weight_data, &normalized_shape).unwrap();
        let weight = Parameter::new(weight_tensor.requires_grad_(true), "weight".to_string());

        // Initialize bias (β) to 0
        let bias_data = vec![T::zero(); num_features];
        let bias_tensor = Tensor::<B, S, T>::from_vec(bias_data, &normalized_shape).unwrap();
        let bias = Parameter::new(bias_tensor.requires_grad_(true), "bias".to_string());

        Self {
            normalized_shape,
            weight,
            bias,
            eps,
            _phantom: PhantomData,
        }
    }
}

impl<B, S, T> Module<B, S, T> for LayerNorm<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + 'static,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Input: [..., *normalized_shape]
        // Output: Same shape as input

        let input_shape = input.shape().dims();
        let normalized_size: usize = self.normalized_shape.iter().product();

        // Verify input shape ends with normalized_shape
        let input_size: usize = input_shape.iter().product();
        assert_eq!(
            input_size % normalized_size,
            0,
            "Input size must be divisible by normalized_size"
        );

        let batch_size = input_size / normalized_size;
        let input_data = input.as_slice();
        let weight_data = self.weight.data().as_slice();
        let bias_data = self.bias.data().as_slice();

        let eps = T::from(self.eps).unwrap();
        let normalized_size_f = T::from(normalized_size as f64).unwrap();

        let mut output_data = Vec::with_capacity(input_size);

        // Process each batch element independently
        for batch_idx in 0..batch_size {
            let start = batch_idx * normalized_size;
            let end = start + normalized_size;
            let batch_data = &input_data[start..end];

            // Compute mean: Σ(x) / D
            let sum = batch_data.iter().copied().fold(T::zero(), |acc, x| acc + x);
            let mean = sum / normalized_size_f;

            // Compute variance: Σ((x - mean)²) / D
            let var_sum = batch_data
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .fold(T::zero(), |acc, x| acc + x);
            let var = var_sum / normalized_size_f;

            // Compute std: √(var + ε)
            let std = (var + eps).sqrt();

            // Normalize and apply affine transform: γ * (x - mean) / std + β
            for (i, &x) in batch_data.iter().enumerate() {
                let normalized = (x - mean) / std;
                let output = weight_data[i] * normalized + bias_data[i];
                output_data.push(output);
            }
        }

        Tensor::from_vec(output_data, input_shape).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn zero_grad(&mut self) {
        self.weight.zero_grad();
        self.bias.zero_grad();
    }

    fn train(&mut self, _mode: bool) {
        // No-op: LayerNorm behavior doesn't change between train/eval
    }

    fn name(&self) -> &str {
        "LayerNorm"
    }
}

// ============================================================================
// TYPE ALIASES FOR BACKWARD COMPATIBILITY
// ============================================================================

/// Type alias for LayerNorm layer with CPU backend and dense storage.
/// This provides backward compatibility with existing code.
pub type LayerNormCpu<T> = LayerNorm<CpuBackend<T>, DenseStorage<T>, T>;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use dtype::float::Float32;

    #[test]
    fn test_layernorm_forward() {
        let layer_norm =
            LayerNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(vec![4], 1e-5);

        // Input: [2, 4]
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
                Float32::new(5.0),
                Float32::new(6.0),
                Float32::new(7.0),
                Float32::new(8.0),
            ],
            &[2, 4],
        )
        .unwrap();

        let output = layer_norm.forward(&input).unwrap();

        // Output shape: [2, 4]
        assert_eq!(output.shape().dims(), &[2, 4]);

        // Check that each row is normalized (mean ≈ 0, std ≈ 1)
        let output_data: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        // First row: [1, 2, 3, 4]
        let row1 = &output_data[0..4];
        let mean1: f32 = row1.iter().sum::<f32>() / 4.0;
        assert_relative_eq!(mean1, 0.0, epsilon = 1e-5);

        // Second row: [5, 6, 7, 8]
        let row2 = &output_data[4..8];
        let mean2: f32 = row2.iter().sum::<f32>() / 4.0;
        assert_relative_eq!(mean2, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_layernorm_numerical_stability() {
        let layer_norm =
            LayerNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(vec![3], 1e-5);

        // Input with large values
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(1000.0),
                Float32::new(1001.0),
                Float32::new(1002.0),
            ],
            &[3],
        )
        .unwrap();

        let output = layer_norm.forward(&input).unwrap();
        let output_data: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        // Should still normalize correctly
        let mean: f32 = output_data.iter().sum::<f32>() / 3.0;
        assert_relative_eq!(mean, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_layernorm_affine_transform() {
        let mut layer_norm =
            LayerNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(vec![3], 1e-5);

        // Set weight to 2.0 and bias to 1.0
        let weight_data = vec![Float32::new(2.0); 3];
        let weight_tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                weight_data,
                &[3],
            )
            .unwrap();
        layer_norm.weight =
            Parameter::new(weight_tensor.requires_grad_(true), "weight".to_string());

        let bias_data = vec![Float32::new(1.0); 3];
        let bias_tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            bias_data,
            &[3],
        )
        .unwrap();
        layer_norm.bias = Parameter::new(bias_tensor.requires_grad_(true), "bias".to_string());

        // Input: [1, 2, 3]
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        let output = layer_norm.forward(&input).unwrap();
        let output_data: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        // After normalization, mean should be 0, then scaled by 2 and shifted by 1
        // So mean should be 1.0
        let mean: f32 = output_data.iter().sum::<f32>() / 3.0;
        assert_relative_eq!(mean, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_layernorm_parameters() {
        let layer_norm = LayerNormCpu::<Float32>::new(vec![4], 1e-5);

        let params = layer_norm.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].name(), "weight");
        assert_eq!(params[1].name(), "bias");
        assert!(params[0].requires_grad());
        assert!(params[1].requires_grad());
    }

    #[test]
    fn test_layernorm_3d_input() {
        let layer_norm = LayerNormCpu::<Float32>::new(vec![4], 1e-5);

        // Input: [2, 3, 4] (batch_size=2, seq_len=3, hidden_dim=4)
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0); 24],
            &[2, 3, 4],
        )
        .unwrap();

        let output = layer_norm.forward(&input).unwrap();

        // Output shape: [2, 3, 4]
        assert_eq!(output.shape().dims(), &[2, 3, 4]);
    }
}
