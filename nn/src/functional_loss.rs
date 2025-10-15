//! Loss functions for neural networks.
//!
//! This module provides stateless loss function computations
//! for training neural networks.

use coeus_backend::Backend;
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use coeus_tensor::Tensor;

use crate::error::{NNError, Result};

/// Computes the mean squared error (MSE) loss.
///
/// Formula: `MSE = mean((predicted - target)^2)`
///
/// # Arguments
/// * `input` - Predicted tensor of any shape
/// * `target` - Target tensor with the same shape as input
///
/// # Returns
/// Scalar tensor containing the MSE loss value
///
/// # Examples
/// ```rust
/// use coeus_nn::functional_loss::mse_loss;
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let pred = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
///     &[3]
/// ).unwrap();
///
/// let target = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.5), Float32::new(2.5), Float32::new(3.5)],
///     &[3]
/// ).unwrap();
///
/// let loss = mse_loss(&pred, &target).unwrap();
/// let loss_val = loss.as_slice()[0];
/// // loss ≈ 0.25 (mean of (0.5² + 0.5² + 0.5²) = mean of (0.25, 0.25, 0.25))
/// ```
pub fn mse_loss<B, S, T>(
    input: &Tensor<B, S, T>,
    target: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    let input_shape = input.shape().dims();
    let target_shape = target.shape().dims();

    if input_shape != target_shape {
        return Err(NNError::InvalidInput {
            message: format!(
                "Input shape {:?} does not match target shape {:?}",
                input_shape, target_shape
            ),
        });
    }

    // Convert to dense for computation
    let input_dense = input.to_dense_generic()?;
    let target_dense = target.to_dense_generic()?;

    let input_data = input_dense.as_slice();
    let target_data = target_dense.as_slice();

    let mut squared_diff_sum = T::from(0.0).unwrap();
    let total_elements = input_data.len() as f64;

    for (pred, targ) in input_data.iter().zip(target_data.iter()) {
        let diff = pred.clone() - targ.clone();
        let squared_diff = diff * diff.clone();
        squared_diff_sum = squared_diff_sum + squared_diff;
    }

    let mean_squared_error = squared_diff_sum / T::from(total_elements).unwrap();

    // Return as scalar tensor
    Tensor::from_vec(vec![mean_squared_error], &[1])
        .map_err(Into::into)
        .and_then(|t| t.to_generic())
}

/// Computes cross-entropy loss for classification tasks.
///
/// Formula: `CE = -mean(sum(target * log(softmax(pred))))`
///
/// # Arguments
/// * `input` - Predicted logits tensor of shape `(..., num_classes)`
/// * `target` - Target tensor of shape `(..., num_classes)` with class probabilities
///
/// # Returns
/// Scalar tensor containing the cross-entropy loss value
///
/// # Examples
/// ```rust
/// use coeus_nn::functional_loss::cross_entropy;
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let pred = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(2.0), Float32::new(1.0), Float32::new(0.1)],
///     &[1, 3]
/// ).unwrap();
///
/// let target = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.0), Float32::new(0.0), Float32::new(0.0)],
///     &[1, 3]
/// ).unwrap();
///
/// let loss = cross_entropy(&pred, &target).unwrap();
/// // loss = -log(softmax([2.0, 1.0, 0.1])[0])
/// ```
pub fn cross_entropy<T: DataType + FloatExt + std::ops::Neg<Output = T> + PartialOrd>(
    input: &Tensor<impl Backend, impl Storage<T>, T>,
    target: &Tensor<impl Backend, impl Storage<T>, T>,
) -> Result<Tensor<impl Backend, impl Storage<T>, T>>
where
    T: Clone,
{
    let input_shape = input.shape().dims();
    let target_shape = target.shape().dims();

    if input_shape != target_shape {
        return Err(NNError::InvalidInput {
            message: format!(
                "Input shape {:?} does not match target shape {:?}",
                input_shape, target_shape
            ),
        });
    }

    if *input_shape.last().unwrap() < 2 {
        return Err(NNError::InvalidInput {
            message: "Cross-entropy requires at least 2 classes".to_string(),
        });
    }

    // Apply softmax to predictions
    let softmax_output = crate::functional_attention::softmax(input)?;

    // Convert to dense for computation
    let softmax_dense = softmax_output.to_dense_generic()?;
    let target_dense = target.to_dense_generic()?;

    let softmax_data = softmax_dense.as_slice();
    let target_data = target_dense.as_slice();

    let last_dim_size = *input_shape.last().unwrap();
    let batch_size: usize = softmax_data.len() / last_dim_size;

    let mut total_loss = T::from(0.0).unwrap();

    for batch in 0..batch_size {
        for class in 0..last_dim_size {
            let idx = batch * last_dim_size + class;
            let softmax_val = softmax_data[idx].clone();
            let target_val = target_data[idx].clone();

            // Add small epsilon to prevent log(0)
            let epsilon = T::from(1e-12).unwrap();
            let safe_softmax = softmax_val + epsilon;

            // Compute -target * log(softmax)
            let log_softmax = safe_softmax.ln();
            let weighted_log = target_val * log_softmax;
            total_loss = total_loss - weighted_log;
        }
    }

    // Compute mean loss
    let total_elements = T::from((batch_size * last_dim_size) as f64).unwrap();
    let mean_loss = total_loss / total_elements;

    // Return as scalar tensor
    Tensor::from_vec(vec![mean_loss], &[1])
        .map_err(Into::into)
        .and_then(|t| t.to_generic())
}

/// Computes negative log likelihood (NLL) loss.
///
/// Formula: `NLL = -mean(target * log(input))`
///
/// This is typically used with log-probabilities as input.
///
/// # Arguments
/// * `input` - Log-probabilities tensor of shape `(..., num_classes)`
/// * `target` - Target tensor of shape `(..., num_classes)` with one-hot encoded classes
///
/// # Returns
/// Scalar tensor containing the NLL loss value
pub fn nll_loss<T: DataType + FloatExt + std::ops::Neg<Output = T>>(
    input: &Tensor<impl Backend, impl Storage<T>, T>,
    target: &Tensor<impl Backend, impl Storage<T>, T>,
) -> Result<Tensor<impl Backend, impl Storage<T>, T>>
where
    T: Clone,
{
    let input_shape = input.shape().dims();
    let target_shape = target.shape().dims();

    if input_shape != target_shape {
        return Err(NNError::InvalidInput {
            message: format!(
                "Input shape {:?} does not match target shape {:?}",
                input_shape, target_shape
            ),
        });
    }

    // Convert to dense for computation
    let input_dense = input.to_dense_generic()?;
    let target_dense = target.to_dense_generic()?;

    let input_data = input_dense.as_slice();
    let target_data = target_dense.as_slice();

    let last_dim_size = *input_shape.last().unwrap();
    let batch_size: usize = input_data.len() / last_dim_size;

    let mut total_loss = T::from(0.0).unwrap();

    for batch in 0..batch_size {
        for class in 0..last_dim_size {
            let idx = batch * last_dim_size + class;
            let log_prob = input_data[idx].clone();
            let target_val = target_data[idx].clone();

            // Compute -target * log_prob
            let weighted_loss = target_val * log_prob;
            total_loss = total_loss - weighted_loss;
        }
    }

    // Compute mean loss
    let total_elements = T::from((batch_size * last_dim_size) as f64).unwrap();
    let mean_loss = total_loss / total_elements;

    // Return as scalar tensor
    Tensor::from_vec(vec![mean_loss], &[1])
        .map_err(Into::into)
        .and_then(|t| t.to_generic())
}
