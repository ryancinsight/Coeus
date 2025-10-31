//! Loss functions for neural networks.
//!
//! This module provides stateless loss function computations
//! for training neural networks.

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
#[allow(unused_imports)]
use storage::{DenseStorage, Storage, StorageToDense};
use tensor::Tensor;

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
/// use nn::functional_loss::mse_loss;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let pred = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
///     &[3]
/// ).unwrap();
///
/// let target = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.5), Float32::new(2.5), Float32::new(3.5)],
///     &[3]
/// ).unwrap();
///
/// let loss = mse_loss(&pred, &target).unwrap();
/// let loss_val = loss.as_slice()[0];
/// // loss ≈ 0.25 (mean of (0.5² + 0.5² + 0.5²) = mean of (0.25, 0.25, 0.25))
/// ```
pub fn mse_loss<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + 'static, T>,
    target: &Tensor<B, impl StorageToDense<T> + 'static, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default,
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
        let diff = *pred - *targ;
        let squared_diff = diff * diff;
        squared_diff_sum = squared_diff_sum + squared_diff;
    }

    let mean_squared_error = squared_diff_sum / T::from(total_elements).unwrap();

    // Return as scalar tensor
    Ok(Tensor::from_vec_with_backend(
        vec![mean_squared_error],
        &[1],
        input.backend().clone(),
    )?)
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
/// use nn::functional_loss::cross_entropy;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let pred = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(2.0), Float32::new(1.0), Float32::new(0.1)],
///     &[1, 3]
/// ).unwrap();
///
/// let target = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.0), Float32::new(0.0), Float32::new(0.0)],
///     &[1, 3]
/// ).unwrap();
///
/// let loss = cross_entropy(&pred, &target).unwrap();
/// // loss = -log(softmax([2.0, 1.0, 0.1])[0])
/// ```
pub fn cross_entropy<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + 'static, T>,
    target: &Tensor<B, impl StorageToDense<T> + 'static, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + PartialOrd + Clone,
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

    // Apply softmax to predictions (local implementation)
    let input_dense = input.to_dense_generic()?;
    let input_data = input_dense.as_slice();
    let mut softmax_data = Vec::with_capacity(input_data.len());

    // Simple softmax: exp(x) / sum(exp(x)) along last dimension
    let last_dim = *input_shape.last().unwrap();
    let batch_size = input_data.len() / last_dim;

    for b in 0..batch_size {
        let start = b * last_dim;
        let end = start + last_dim;
        let batch_data = &input_data[start..end];

        // Find max for numerical stability
        let max_val = batch_data
            .iter()
            .fold(T::from(f64::NEG_INFINITY).unwrap(), |a, &b| {
                if a > b {
                    a
                } else {
                    b
                }
            });

        // Compute exp(x - max) and sum
        let mut exp_vals = Vec::with_capacity(last_dim);
        let mut sum_exp = T::from(0.0).unwrap();

        for &val in batch_data {
            let exp_val = (val - max_val).exp();
            exp_vals.push(exp_val);
            sum_exp = sum_exp + exp_val;
        }

        // Normalize by sum
        for exp_val in exp_vals {
            softmax_data.push(exp_val / sum_exp);
        }
    }

    let softmax_output: Tensor<B, DenseStorage<T>, T> =
        Tensor::from_vec_with_backend(softmax_data, input.shape().dims(), input.backend().clone())?;

    // Convert target to dense for computation
    let target_dense = target.to_dense_generic()?;

    let softmax_data = softmax_output.as_slice();
    let target_data = target_dense.as_slice();

    let last_dim_size = *input_shape.last().unwrap();
    let batch_size: usize = softmax_data.len() / last_dim_size;

    let mut total_loss = T::from(0.0).unwrap();

    for batch in 0..batch_size {
        for class in 0..last_dim_size {
            let idx = batch * last_dim_size + class;
            let softmax_val = softmax_data[idx];
            let target_val = target_data[idx];

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
    let total_elements = T::from(softmax_data.len() as f64).unwrap();
    let mean_loss = total_loss / total_elements;

    // Return as scalar tensor
    Ok(Tensor::from_vec_with_backend(
        vec![mean_loss],
        &[1],
        input.backend().clone(),
    )?)
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
pub fn nll_loss<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + 'static, T>,
    target: &Tensor<B, impl StorageToDense<T> + 'static, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + Clone,
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
            let log_prob = input_data[idx];
            let target_val = target_data[idx];

            // Compute -target * log_prob
            let weighted_loss = target_val * log_prob;
            total_loss = total_loss - weighted_loss;
        }
    }

    // Compute mean loss
    let total_elements = T::from(input_data.len() as f64).unwrap();
    let mean_loss = total_loss / total_elements;

    // Return as scalar tensor
    Ok(Tensor::from_vec_with_backend(
        vec![mean_loss],
        &[1],
        input.backend().clone(),
    )?)
}
