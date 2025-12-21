//! Loss functions for neural networks.
//!
//! This module provides stateless loss function computations
//! for training neural networks.

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
#[allow(unused_imports)]
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use num_traits::FromPrimitive;
use crate::error::{NNError, Result};

#[cfg(feature = "autograd")]
use autograd::{loss, tensor_ops};

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
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static, T>,
    target: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    T: DataType + FloatExt + FromPrimitive + Copy + Send + Sync + 'static,
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

    // Convert to dense for computation (and for autograd compatibility)
    let input_dense = input.to_dense_generic()?;
    let target_dense = target.to_dense_generic()?;

    #[cfg(feature = "autograd")]
    {
        Ok(autograd::loss::mse_loss(&input_dense, &target_dense)?)
    }

    #[cfg(not(feature = "autograd"))]
    {
        let input_data = input_dense.as_slice();
        let target_data = target_dense.as_slice();

        let mut squared_diff_sum = T::from_f64(0.0).unwrap();
        let total_elements = input_data.len() as f64;

        for (pred, targ) in input_data.iter().zip(target_data.iter()) {
            let diff = *pred - *targ;
            let squared_diff = diff * diff;
            squared_diff_sum = squared_diff_sum + squared_diff;
        }

        let mean_squared_error = squared_diff_sum / T::from_f64(total_elements).unwrap();

        // Return as scalar tensor
        Ok(Tensor::from_vec_with_backend(
            vec![mean_squared_error],
            &[1],
            input.backend().clone(),
        )?)
    }
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
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static, T>,
    target: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + PartialOrd + Clone + FromPrimitive + Send + Sync + 'static,
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

    let input_dense = input.to_dense_generic()?;
    let target_dense = target.to_dense_generic()?;

    let input_shape = input_dense.shape().dims();
    let target_shape = target_dense.shape().dims();
    let last_dim = *input_shape.last().unwrap();

    // Check if target is class indices (ndim - 1) or probabilities (same ndim)
    let is_indices = target_shape.len() == input_shape.len() - 1;

    #[cfg(feature = "autograd")]
    {
        // ... (existing autograd block might need update for indices, but keep it simple for now)
        // For now, only support probabilities in autograd if not easily changed
        if is_indices {
             return Err(NNError::InvalidInput {
                message: "Autograd implementation currently only supports probability targets in cross_entropy".to_string(),
            });
        }

        // 1. Log Softmax
        let log_probs = autograd::loss::log_softmax_stable(&input_dense)?;

        // 2. Multiply with targets (element-wise)
        let weighted = autograd::tensor_ops::mul(&target_dense, &log_probs)?;

        // 3. Mean over all elements (to match non-autograd behavior)
        let mean_loss = autograd::tensor_ops::mean(&weighted, None, false)?;

        // 4. Negate
        let neg_one = Tensor::from_vec_with_backend(vec![T::from_f64(-1.0).unwrap()], &[1], input.backend().clone())?;
        let loss = autograd::tensor_ops::mul(&mean_loss, &neg_one)?;

        Ok(loss)
    }

    #[cfg(not(feature = "autograd"))]
    {
        let input_data = input_dense.as_slice();
        let target_data = target_dense.as_slice();
        let batch_size = input_data.len() / last_dim;

        let mut total_loss = T::from_f64(0.0).unwrap();

        for b in 0..batch_size {
            let start = b * last_dim;
            let batch_data = &input_data[start..last_dim * (b + 1)];

            // Softmax for current batch
            let max_val = batch_data.iter().fold(T::from_f64(f64::NEG_INFINITY).unwrap(), |a, &b| if a > b { a } else { b });
            let mut sum_exp = T::from_f64(0.0).unwrap();
            let mut exp_vals = Vec::with_capacity(last_dim);
            for &val in batch_data {
                let ev = (val - max_val).exp();
                exp_vals.push(ev);
                sum_exp = sum_exp + ev;
            }

            if is_indices {
                // target_data[b] is the index of the correct class
                let target_idx = target_data[b].to_f64().unwrap_or(0.0) as usize;
                if target_idx < last_dim {
                    let softmax_val = exp_vals[target_idx] / sum_exp;
                    let epsilon = T::from_f64(1e-12).unwrap();
                    total_loss = total_loss - (softmax_val + epsilon).ln();
                }
            } else {
                // target_data has same shape as input
                for c in 0..last_dim {
                    let softmax_val = exp_vals[c] / sum_exp;
                    let target_val = target_data[b * last_dim + c];
                    let epsilon = T::from_f64(1e-12).unwrap();
                    total_loss = total_loss - target_val * (softmax_val + epsilon).ln();
                }
            }
        }

        let mean_loss = if is_indices {
            total_loss / T::from_f64(batch_size as f64).unwrap()
        } else {
            total_loss / T::from_f64(input_data.len() as f64).unwrap()
        };

        Ok(Tensor::from_vec_with_backend(vec![mean_loss], &[1], input.backend().clone())?)
    }
}

/// Computes binary cross-entropy loss with logits.
///
/// This function combines a Sigmoid layer and the BCE loss in one single class.
/// This version is more numerically stable than using a plain Sigmoid followed
/// by a BCE loss as, by combining the operations into one layer, we take advantage
/// of the log-sum-exp trick for numerical stability.
///
/// Formula: `L = -[target * log(σ(input)) + (1 - target) * log(1 - σ(input))]`
///
/// # Arguments
/// * `input` - Predicted logits tensor
/// * `target` - Target tensor with the same shape as input
///
/// # Returns
/// Scalar tensor containing the BCEWithLogits loss value
pub fn bce_with_logits_loss<B, T>(
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static, T>,
    target: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    T: DataType + FloatExt + FromPrimitive + PartialOrd + Copy + Send + Sync + 'static,
{
    let input_shape = input.shape().dims();
    let target_shape = target.shape().dims();

    if input_shape != target_shape {
        return Err(NNError::InvalidInput {
            message: format!("Shape mismatch: input {:?}, target {:?}", input_shape, target_shape),
        });
    }

    let input_dense = input.to_dense_generic()?;
    let target_dense = target.to_dense_generic()?;

    let input_data = input_dense.as_slice();
    let target_data = target_dense.as_slice();

    let mut total_loss = T::from_f64(0.0).unwrap();
    let zero = T::from_f64(0.0).unwrap();

    for (&x, &y) in input_data.iter().zip(target_data.iter()) {
        // Stable implementation of BCE with logits:
        // max(x, 0) - x * y + log(1 + exp(-|x|))
        let max_x_0 = if x > zero { x } else { zero };
        let abs_x = if x > zero { x } else { -x };
        let term1 = max_x_0 - x * y;
        let term2 = (zero + (-abs_x).exp() + T::from_f64(1.0).unwrap()).ln();
        total_loss = total_loss + term1 + term2;
    }

    let mean_loss = total_loss / T::from_f64(input_data.len() as f64).unwrap();

    Ok(Tensor::from_vec_with_backend(vec![mean_loss], &[1], input.backend().clone())?)
}

/// Computes negative log likelihood (NLL) loss.
///
/// Formula: `NLL = -mean(target * input)`
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
    input: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static, T>,
    target: &Tensor<B, impl StorageToDense<T> + StorageFromVec<T> + 'static, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + Clone + FromPrimitive + Send + Sync + 'static,
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

    let input_dense = input.to_dense_generic()?;
    let target_dense = target.to_dense_generic()?;

    #[cfg(feature = "autograd")]
    {
        // 1. Multiply
        let weighted = autograd::tensor_ops::mul(&target_dense, &input_dense)?;

        // 2. Mean
        let mean_loss = autograd::tensor_ops::mean(&weighted, None, false)?;

        // 3. Negate
        let neg_one = Tensor::from_vec_with_backend(vec![T::from_f64(-1.0).unwrap()], &[1], input.backend().clone())?;
        let loss = autograd::tensor_ops::mul(&mean_loss, &neg_one)?;

        Ok(loss)
    }

    #[cfg(not(feature = "autograd"))]
    {
        let input_data = input_dense.as_slice();
        let target_data = target_dense.as_slice();

        let last_dim_size = *input_shape.last().unwrap();
        let batch_size: usize = input_data.len() / last_dim_size;

        let mut total_loss = T::from_f64(0.0).unwrap();

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
        let total_elements = T::from_f64(input_data.len() as f64).unwrap();
        let mean_loss = total_loss / total_elements;

        // Return as scalar tensor
        Ok(Tensor::from_vec_with_backend(
            vec![mean_loss],
            &[1],
            input.backend().clone(),
        )?)
    }
}
