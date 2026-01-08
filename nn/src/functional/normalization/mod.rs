//! Normalization functions for neural networks.
//!
//! This module provides stateless normalization operations for stabilizing
//! and improving the training of neural networks.

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
#[allow(unused_imports)]
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::core::error::{NNError, Result};

/// Applies Layer Normalization over the last dimensions of the input.
///
/// Normalizes across the feature dimension for each sample in a batch.
/// This is different from Batch Normalization which normalizes across the batch dimension.
///
/// Formula:
/// ```text
/// normalized = (input - mean) / sqrt(variance + eps)
/// output = normalized * weight + bias
/// ```
///
/// # Arguments
/// * `input` - Input tensor of shape `(..., C)` where C is the feature dimension
/// * `normalized_shape` - Shape of the normalized dimensions (typically `[C]`)
/// * `weight` - Optional scale parameter of shape `normalized_shape`
/// * `bias` - Optional shift parameter of shape `normalized_shape`
/// * `eps` - Small constant for numerical stability (default: 1e-5)
///
/// # Returns
/// Layer-normalized tensor with the same shape as input
///
/// # Examples
/// ```rust
/// use nn::functional_normalization::layer_norm;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
///     &[1, 3]
/// ).unwrap();
///
/// let output = layer_norm(&input, &[3], None, None, 1e-5).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 3]);
/// ```
#[allow(clippy::multiple_bound_locations)]
pub fn layer_norm<
    B: Backend<Data = T>,
    S: StorageToDense<T> + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
>(
    input: &Tensor<B, S, T>,
    normalized_shape: &[usize],
    weight: Option<&Tensor<B, S, T>>,
    bias: Option<&Tensor<B, S, T>>,
    eps: f64,
) -> Result<Tensor<B, S, T>>
where
    T: Clone
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let input_dense = input.to_dense_generic()?;
    let input_shape = input_dense.shape().dims();

    // Validate normalized_shape
    let normalized_size: usize = normalized_shape.iter().product();
    let feature_dim = *input_shape.last().unwrap();

    if normalized_size != feature_dim {
        return Err(NNError::InvalidInput {
            message: format!(
                "normalized_shape product ({}) must equal last input dimension ({})",
                normalized_size, feature_dim
            ),
        });
    }

    // Validate weight and bias shapes if provided
    if let Some(w) = weight {
        if w.shape().dims() != normalized_shape {
            return Err(NNError::InvalidInput {
                message: format!(
                    "weight shape {:?} must match normalized_shape {:?}",
                    w.shape().dims(),
                    normalized_shape
                ),
            });
        }
    }

    if let Some(b) = bias {
        if b.shape().dims() != normalized_shape {
            return Err(NNError::InvalidInput {
                message: format!(
                    "bias shape {:?} must match normalized_shape {:?}",
                    b.shape().dims(),
                    normalized_shape
                ),
            });
        }
    }

    let input_data = input_dense.as_slice();
    let mut output_data = Vec::with_capacity(input_data.len());

    // Calculate dimensions
    let batch_size: usize = input_shape[..input_shape.len() - normalized_shape.len()]
        .iter()
        .product();
    let feature_size = normalized_size;

    for batch in 0..batch_size {
        let start_idx = batch * feature_size;
        let end_idx = start_idx + feature_size;
        let features = &input_data[start_idx..end_idx];

        // Compute mean
        let mut sum = T::from(0.0).unwrap();
        for &val in features {
            sum = sum + val;
        }
        let mean = sum / T::from(feature_size as f64).unwrap();

        // Compute variance
        let mut var_sum = T::from(0.0).unwrap();
        for &val in features {
            let diff = val - mean;
            var_sum = var_sum + (diff * diff);
        }
        let variance = var_sum / T::from(feature_size as f64).unwrap();

        // Normalize
        let eps_t = T::from(eps).unwrap();
        let std = (variance + eps_t).sqrt();

        for (i, &val) in features.iter().enumerate() {
            let normalized = (val - mean) / std;

            // Apply weight and bias if provided
            let mut result = normalized;
            if let Some(w) = weight {
                let weight_data = w.as_slice();
                result = result * weight_data[i];
            }
            if let Some(b) = bias {
                let bias_data = b.as_slice();
                result = result + bias_data[i];
            }

            output_data.push(result);
        }
    }

    Ok(Tensor::from_vec_with_backend(
        output_data,
        input_shape,
        input.backend().clone(),
    )?)
}

/// Applies Batch Normalization over the batch dimension.
///
/// This is a functional version that performs batch normalization without
/// maintaining running statistics. For training with running statistics,
/// use the BatchNorm modules instead.
///
/// Formula:
/// ```text
/// normalized = (input - batch_mean) / sqrt(batch_var + eps)
/// output = normalized * weight + bias
/// ```
///
/// # Arguments
/// * `input` - Input tensor of shape `(N, C, ...)`
/// * `weight` - Optional scale parameter of shape `(C,)`
/// * `bias` - Optional shift parameter of shape `(C,)`
/// * `eps` - Small constant for numerical stability (default: 1e-5)
///
/// # Returns
/// Batch-normalized tensor with the same shape as input
///
/// # Examples
/// ```rust
/// use nn::functional_normalization::batch_norm;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)],
///     &[2, 2]
/// ).unwrap();
///
/// let output = batch_norm(&input, None, None, 1e-5).unwrap();
/// assert_eq!(output.shape().dims(), &[2, 2]);
/// ```
#[allow(clippy::multiple_bound_locations)]
pub fn batch_norm<
    B: Backend<Data = T>,
    S: StorageToDense<T> + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
>(
    input: &Tensor<B, S, T>,
    weight: Option<&Tensor<B, S, T>>,
    bias: Option<&Tensor<B, S, T>>,
    eps: f64,
) -> Result<Tensor<B, S, T>>
where
    T: Clone
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let input_dense = input.to_dense_generic()?;
    let input_shape = input_dense.shape().dims();

    if input_shape.len() < 2 {
        return Err(NNError::InvalidInput {
            message: "Input must have at least 2 dimensions [N, C, ...]".to_string(),
        });
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let spatial_size: usize = input_shape[2..].iter().product();

    // Validate weight and bias shapes if provided
    if let Some(w) = weight {
        if w.shape().dims() != [channels] {
            return Err(NNError::InvalidInput {
                message: format!(
                    "weight shape {:?} must be [{:?}]",
                    w.shape().dims(),
                    channels
                ),
            });
        }
    }

    if let Some(b) = bias {
        if b.shape().dims() != [channels] {
            return Err(NNError::InvalidInput {
                message: format!("bias shape {:?} must be [{:?}]", b.shape().dims(), channels),
            });
        }
    }

    let input_data = input_dense.as_slice();
    let mut output_data = vec![T::from(0.0).unwrap(); input_data.len()];
    let elements_per_channel = batch_size * spatial_size;
    let total_elements = T::from(elements_per_channel as f64).unwrap();
    let eps_t = T::from(eps).unwrap();

    // Process each channel independently
    for c in 0..channels {
        // Compute batch statistics for this channel
        let mut sum = T::from(0.0).unwrap();
        for n in 0..batch_size {
            for s in 0..spatial_size {
                let idx = (n * channels + c) * spatial_size + s;
                sum = sum + input_data[idx];
            }
        }
        let mean = sum / total_elements;

        let mut var_sum = T::from(0.0).unwrap();
        for n in 0..batch_size {
            for s in 0..spatial_size {
                let idx = (n * channels + c) * spatial_size + s;
                let diff = input_data[idx] - mean;
                var_sum = var_sum + (diff * diff);
            }
        }
        let variance = var_sum / total_elements;

        // Normalize channel data
        let std = (variance + eps_t).sqrt();

        let weight_val = weight.map(|w| w.as_slice()[c]);
        let bias_val = bias.map(|b| b.as_slice()[c]);

        for n in 0..batch_size {
            for s in 0..spatial_size {
                let idx = (n * channels + c) * spatial_size + s;
                let normalized = (input_data[idx] - mean) / std;

                let mut result = normalized;
                if let Some(w) = weight_val {
                    result = result * w;
                }
                if let Some(b) = bias_val {
                    result = result + b;
                }

                output_data[idx] = result;
            }
        }
    }

    Ok(Tensor::from_vec_with_backend(
        output_data,
        input_shape,
        input.backend().clone(),
    )?)
}
