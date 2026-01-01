//! Loss functions for automatic differentiation
//!
//! This module provides autograd-aware loss functions that participate in the computation graph.
//!
//! ## Mathematical Foundations
//!
//! ### Cross-Entropy Loss
//! The cross-entropy loss for multi-class classification is defined as:
//!
//! `L = -∑ᵢ yᵢ log(softmax(x)ᵢ)`
//!
//! where `softmax(x)ᵢ = exp(xᵢ) / ∑ⱼ exp(xⱼ)`
//!
//! ### Log-Sum-Exp Trick
//! For numerical stability, we use the log-sum-exp trick:
//!
//! `log(∑ᵢ exp(xᵢ)) = max(x) + log(∑ᵢ exp(xᵢ - max(x)))`
//!
//! This prevents overflow when computing softmax probabilities.

use crate::tensor_ops;
use crate::tensor_ops::{mul, sub};
use std::sync::Arc;
use tensor::{Backend, DataType, Storage, StorageFromVec, StorageToDense, Tensor};

use dtype::traits::FloatExt;
use num_traits::{FromPrimitive, ToPrimitive};

/// Compute Mean Squared Error (MSE) loss with automatic differentiation support.
///
/// Computes the mean squared error between predictions and targets:
/// `loss = mean((predictions - targets)²)`
///
/// # Arguments
/// * `predictions` - Predicted values tensor
/// * `targets` - Target values tensor
///
/// # Returns
/// A scalar tensor containing the MSE loss value with gradient computation support.
///
/// # Examples
/// ```
/// use autograd::loss::mse_loss;
/// use tensor::Tensor;
/// use dtype::float::Float32;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
///
/// let predictions = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.0), Float32::new(2.0)],
///     &[2]
/// ).unwrap().requires_grad_(true);
///
/// let targets = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.5), Float32::new(2.5)],
///     &[2]
/// ).unwrap();
///
/// let loss = mse_loss(&predictions, &targets).unwrap();
/// // Loss: mean((1.0-1.5)² + (2.0-2.5)²) = mean(0.25 + 0.25) = 0.25
/// ```
///
/// # Gradient Formula
/// For MSE loss `L = mean((y_pred - y_true)²)`:
/// - `∂L/∂y_pred = 2 * (y_pred - y_true) / n`
///
/// where `n` is the number of elements.
#[allow(clippy::missing_errors_doc)]
pub fn mse_loss<B, S, T>(
    predictions: &Tensor<B, S, T>,
    targets: &Tensor<B, S, T>,
) -> crate::Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + 'static,
    T: DataType + FloatExt + FromPrimitive + Copy + Send + Sync + 'static,
{
    // Compute (predictions - targets)
    let diff = sub(predictions, targets)?;

    // Square the differences: diff²
    let squared = mul(&diff, &diff)?;

    // Compute sum of squared errors
    // Use autograd-aware sum to ensure gradient propagation
    let sse_loss = tensor_ops::sum(&squared, None, false)?;

    // Divide by number of elements to get Mean Squared Error
    let num_elements = squared.len();
    let num_elements_t = T::from_usize(num_elements).ok_or_else(|| {
        crate::AutogradError::TensorError(tensor::TensorError::ShapeError {
            expected: 0,
            actual: 0,
            message: "Failed to convert scale to T".to_string(),
        })
    })?;
    let scale_val = T::one() / num_elements_t;
    let scale_tensor = Tensor::from_vec(vec![scale_val], &[])?;

    // We need to reshape sse_loss to [] if it is [1] to allow multiplication with scalar
    let sse_loss = if sse_loss.shape().ndim() == 1 {
        // Use autograd-aware reshape to preserve computation graph
        tensor_ops::reshape(&sse_loss, &[])?
    } else {
        sse_loss
    };

    // Multiply by 1/N
    // Note: We use mul (element-wise multiplication)
    // The result should be scalar if inputs are scalar
    let mse_loss = mul(&sse_loss, &scale_tensor)?;

    Ok(mse_loss)
}

/// Compute Negative Log Likelihood (NLL) loss with automatic differentiation support.
///
/// Computes the negative log likelihood loss between log-probabilities and targets:
/// `loss = mean(-log_probs[target])`
///
/// # Arguments
/// * `log_probs` - Log probabilities `[batch_size, num_classes]` tensor
/// * `targets` - Class indices `[batch_size]` tensor (integer indices)
///
/// # Returns
/// A scalar tensor containing the NLL loss value with gradient computation support.
#[allow(clippy::missing_errors_doc)]
pub fn nll_loss<B, S, T>(
    log_probs: &Tensor<B, S, T>,
    targets: &Tensor<B, S, T>,
) -> crate::Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + 'static,
    T: DataType + FloatExt + FromPrimitive + ToPrimitive + Copy + Send + Sync + 'static,
{
    // Validate input dimensions
    let logits_shape = log_probs.shape().dims();
    let targets_shape = targets.shape().dims();

    if logits_shape.len() != 2 {
        return Err(crate::AutogradError::InvalidInput {
            message: format!(
                "Logits must be 2D tensor [batch_size, num_classes], got shape {logits_shape:?}"
            ),
        });
    }
    if targets_shape.len() != 1 {
        return Err(crate::AutogradError::InvalidInput {
            message: format!("Targets must be 1D tensor [batch_size], got shape {targets_shape:?}"),
        });
    }
    let logits_batch = logits_shape[0];
    let targets_batch = targets_shape[0];
    if logits_batch != targets_batch {
        return Err(crate::AutogradError::InvalidInput {
            message: format!("Batch size mismatch: logits has batch_size={logits_batch}, targets has batch_size={targets_batch}"),
        });
    }

    let batch_size = logits_batch;
    let num_classes = logits_shape[1];

    // Convert to dense for manual indexing
    let targets_dense = targets
        .to_dense_generic()
        .map_err(crate::AutogradError::TensorError)?;
    let log_probs_dense = log_probs
        .to_dense_generic()
        .map_err(crate::AutogradError::TensorError)?;

    let targets_slice = targets_dense.storage_ref().as_slice();
    let log_probs_slice = log_probs_dense.storage_ref().as_slice();

    let mut nll_loss_vals = Vec::new();

    for (batch_idx, &target_val) in targets_slice.iter().enumerate().take(batch_size) {
        // Validate target is integer
        if let Some(val_f64) = target_val.to_f64() {
            if (val_f64 - val_f64.round()).abs() > 1e-5 {
                return Err(crate::AutogradError::InvalidInput {
                    message: format!("Target value {val_f64} is not an integer"),
                });
            }
            if val_f64 < 0.0 {
                return Err(crate::AutogradError::InvalidInput {
                    message: format!("Target class index {val_f64} is out of range"),
                });
            }
        }

        let target_idx = target_val.to_usize().unwrap_or(0);

        if target_idx >= num_classes {
            return Err(crate::AutogradError::InvalidInput {
                message: format!(
                    "Target class index {target_idx} is out of range for {num_classes} classes"
                ),
            });
        }

        // Get log_prob value for the target class
        let log_prob_idx = batch_idx * num_classes + target_idx;
        let log_prob = log_probs_slice[log_prob_idx];

        // Validate log_prob
        if let Some(lp_f64) = log_prob.to_f64() {
            if lp_f64.is_nan() {
                return Err(crate::AutogradError::InvalidInput {
                    message: format!("Invalid log probability: encountered NaN at batch {batch_idx}, class {target_idx}"),
                });
            }
        }

        // Negative log likelihood: -log_prob
        let neg_log_prob = T::zero() - log_prob;
        nll_loss_vals.push(neg_log_prob);
    }

    // Compute mean loss across batch
    let mut total_loss = T::zero();
    for x in &nll_loss_vals {
        total_loss = total_loss + *x;
    }

    let batch_size_t = T::from_usize(batch_size).ok_or_else(|| {
        crate::AutogradError::TensorError(tensor::TensorError::ShapeError {
            expected: 0,
            actual: 0,
            message: "Failed to convert batch size to T".to_string(),
        })
    })?;

    let mean_loss = total_loss / batch_size_t;

    // Create result tensor
    let result_data = vec![mean_loss];
    let result_tensor = Tensor::from_vec(result_data, &[1])?;

    // Attach gradient function if needed
    if log_probs.requires_grad() {
        Ok(result_tensor
            .with_grad_fn(Some(Arc::new(crate::functions::NLLLossFunction::new(
                Arc::new(log_probs.clone()),
                Arc::new(targets.clone()),
            ))))
            .requires_grad_(true))
    } else {
        Ok(result_tensor)
    }
}

/// Compute Cross-Entropy loss with automatic differentiation support.
///
/// Computes the cross-entropy loss between logits and class targets using the numerically stable
/// log-sum-exp trick for softmax computation. The loss is computed as:
/// `loss = mean(-log_softmax(logits)[target_class])`
///
/// This implementation properly uses the log-sum-exp trick:
/// `log(∑ᵢ exp(xᵢ)) = max(x) + log(∑ᵢ exp(xᵢ - max(x)))`
///
/// # Arguments
/// * `logits` - Unnormalized predictions `[batch_size, num_classes]` tensor
/// * `targets` - Class indices `[batch_size]` tensor (integer indices)
///
/// # Returns
/// A scalar tensor containing the cross-entropy loss value with gradient computation support.
///
/// # Examples
/// ```
/// use autograd::loss::cross_entropy_loss;
/// use tensor::Tensor;
/// use dtype::float::Float32;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
///
/// // 3 classes, 2 samples
/// let logits = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![
///         Float32::new(1.0), Float32::new(0.5), Float32::new(0.2),  // sample 1
///         Float32::new(0.1), Float32::new(2.0), Float32::new(0.3),  // sample 2
///     ],
///     &[2, 3]
/// ).unwrap().requires_grad_(true);
///
/// let targets = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(0.0), Float32::new(1.0)],  // class 0 for sample 1, class 1 for sample 2
///     &[2]
/// ).unwrap();
///
/// let loss = cross_entropy_loss(&logits, &targets).unwrap();
/// ```
///
/// # Mathematical Foundation
/// The cross-entropy loss uses the log-sum-exp trick for numerical stability:
/// - `log_softmax(x)_i = x_i - log(∑ⱼ exp(x_j))`
/// - `log(∑ⱼ exp(x_j)) = max(x) + log(∑ⱼ exp(x_j - max(x)))`
///
/// This prevents overflow when computing softmax probabilities.
///
/// # Gradient Formula
/// For cross-entropy loss with softmax:
/// - `∂L/∂logits[i] = (softmax(logits)[i] - 1{i == target}) / batch_size`
///
/// where `1{condition}` is the indicator function (1 if true, 0 if false).
#[allow(clippy::missing_errors_doc)]
pub fn cross_entropy_loss<B, S, T>(
    logits: &Tensor<B, S, T>,
    targets: &Tensor<B, S, T>,
) -> crate::Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + 'static,
    T: DataType + FloatExt + FromPrimitive + ToPrimitive + Copy + Send + Sync + 'static,
{
    // Validate input dimensions
    let logits_shape = logits.shape().dims();
    let targets_shape = targets.shape().dims();

    if logits_shape.len() != 2 {
        return Err(crate::AutogradError::InvalidInput {
            message: format!(
                "Logits must be 2D tensor [batch_size, num_classes], got shape {logits_shape:?}"
            ),
        });
    }
    if targets_shape.len() != 1 {
        return Err(crate::AutogradError::InvalidInput {
            message: format!("Targets must be 1D tensor [batch_size], got shape {targets_shape:?}"),
        });
    }
    let logits_batch = logits_shape[0];
    let targets_batch = targets_shape[0];
    if logits_batch != targets_batch {
        return Err(crate::AutogradError::InvalidInput {
            message: format!("Batch size mismatch: logits has batch_size={logits_batch}, targets has batch_size={targets_batch}"),
        });
    }

    let batch_size = logits_batch;
    let num_classes = logits_shape[1];

    // Compute log-softmax using the numerically stable log-sum-exp trick
    let log_softmax = log_softmax_stable(logits)?;

    // Gather negative log probabilities at target indices
    // This is a simplified implementation - in practice would use advanced indexing

    // Convert to dense for manual indexing
    let targets_dense = targets
        .to_dense_generic()
        .map_err(crate::AutogradError::TensorError)?;
    let log_softmax_dense = log_softmax
        .to_dense_generic()
        .map_err(crate::AutogradError::TensorError)?;

    let targets_slice = targets_dense.storage_ref().as_slice();
    let log_softmax_slice = log_softmax_dense.storage_ref().as_slice();

    let mut nll_loss = Vec::new();

    for (batch_idx, &target_val) in targets_slice.iter().enumerate().take(batch_size) {
        // Get target class index (assume targets contain integer indices)
        let target_idx = target_val.to_usize().unwrap_or(0);

        if target_idx >= num_classes {
            return Err(crate::AutogradError::InvalidInput {
                message: format!(
                    "Target class index {target_idx} is out of range for {num_classes} classes"
                ),
            });
        }

        // Get log-softmax value for the target class
        let log_prob_idx = batch_idx * num_classes + target_idx;
        let log_prob = log_softmax_slice[log_prob_idx];

        // Negative log likelihood: -log_prob
        let neg_log_prob = T::zero() - log_prob;
        nll_loss.push(neg_log_prob);
    }

    // Compute mean loss across batch
    let mut total_loss = T::zero();
    for x in &nll_loss {
        total_loss = total_loss + *x;
    }

    let batch_size_t = T::from_usize(batch_size).ok_or_else(|| {
        crate::AutogradError::TensorError(tensor::TensorError::ShapeError {
            expected: 0,
            actual: 0,
            message: "Failed to convert batch size to T".to_string(),
        })
    })?;

    let mean_loss = total_loss / batch_size_t;

    // Create result tensor
    let result_data = vec![mean_loss];
    let result_tensor = Tensor::from_vec(result_data, &[1])?;

    // Attach gradient function if needed
    if logits.requires_grad() {
        Ok(
            result_tensor.with_grad_fn(Some(Arc::new(CrossEntropyFunction::new(
                Arc::new(logits.clone()),
                Arc::new(targets.clone()),
            )))),
        )
    } else {
        Ok(result_tensor)
    }
}

/// `CrossEntropy` function for automatic differentiation
#[derive(Debug)]
pub struct CrossEntropyFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Input tensors: [logits, targets]
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> CrossEntropyFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new `CrossEntropy` function
    pub fn new(logits: Arc<Tensor<B, S, T>>, targets: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![logits, targets],
        }
    }
}

impl<B, S, T> tensor::AsAny for CrossEntropyFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl<B, S, T> tensor::DifferentiableFunction<B, S, T> for CrossEntropyFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + 'static,
    T: DataType + FloatExt + FromPrimitive + ToPrimitive + Copy + Send + Sync + 'static,
{
    fn name(&self) -> &'static str {
        "CrossEntropyBackward"
    }
}

impl<B, S, T> tensor::Function<B, S, T> for CrossEntropyFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + 'static,
    T: DataType + FloatExt + FromPrimitive + ToPrimitive + Copy + Send + Sync + 'static,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, tensor::DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // Implementation of CrossEntropy backward pass
        // dL/dlogits = (softmax(logits) - targets_one_hot) / batch_size
        // We need to recompute softmax(logits) here because we don't save it

        // 1. Recompute log_softmax (stable)
        let logits = &*self.inputs[0];
        let targets = &*self.inputs[1];

        let log_softmax =
            log_softmax_stable(logits).map_err(|e| anyhow::anyhow!("Autograd error: {e:?}"))?;

        // 2. Compute softmax = exp(log_softmax)
        let softmax = log_softmax.exp();

        // 3. Create one-hot targets
        // This requires converting targets (indices) to one-hot vectors
        // We'll do this manually for now as we lack a one_hot op
        let batch_size = logits.shape().dims()[0];
        let num_classes = logits.shape().dims()[1];

        // We need to work with dense data for manipulation
        let softmax_dense = softmax
            .to_dense_generic()
            .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;
        let softmax_data = softmax_dense.storage_ref().as_slice();

        let targets_dense = targets
            .to_dense_generic()
            .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;
        let targets_data = targets_dense.storage_ref().as_slice();

        let mut grad_data = Vec::with_capacity(batch_size * num_classes);
        let batch_size_t = T::from_usize(batch_size)
            .ok_or_else(|| anyhow::anyhow!("Failed to convert batch_size"))?;
        let scale = T::one() / batch_size_t;

        // Assuming scalar grad_output for loss (usually 1.0)
        // If grad_output is not 1.0, we multiply by it.
        // grad_output is dense.
        let grad_scale = if grad_output.numel() == 1 {
            grad_output.storage_ref().as_slice()[0]
        } else {
            return Err(anyhow::anyhow!(
                "Expected scalar grad_output for NLL loss backward, got numel={}",
                grad_output.numel()
            ));
        };

        let final_scale = scale * grad_scale;

        for (b, target) in targets_data.iter().enumerate().take(batch_size) {
            let target_idx = target.to_usize().ok_or_else(|| {
                anyhow::anyhow!("NLL loss backward: target index at batch {b} not representable")
            })?;
            for c in 0..num_classes {
                let idx = b * num_classes + c;
                let s = softmax_data[idx];

                let val = if c == target_idx { s - T::one() } else { s };

                grad_data.push(val * final_scale);
            }
        }

        let grad_logits = Tensor::from_vec_with_backend(
            grad_data,
            logits.shape().dims(),
            logits.backend().clone(),
        )
        .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

        // Targets don't require gradients
        let grad_targets = Tensor::<B, S, T>::zeros(targets.shape().dims())
            .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

        Ok(vec![grad_logits, grad_targets])
    }
}
///
/// This implements: `log_softmax(x)_i = x_i - log(∑ⱼ exp(x_j))`
/// where `log(∑ⱼ exp(x_j)) = max(x) + log(∑ⱼ exp(x_j - max(x)))`
#[allow(clippy::missing_errors_doc)]
pub fn log_softmax_stable<B, S, T>(input: &Tensor<B, S, T>) -> crate::Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + 'static,
    T: DataType + FloatExt + FromPrimitive + Copy + Send + Sync + 'static,
{
    let shape = input.shape().dims();
    if shape.len() != 2 {
        return Err(crate::AutogradError::InvalidInput {
            message: format!("log_softmax_stable expects 2D tensor, got shape {shape:?}"),
        });
    }

    let batch_size = shape[0];
    let num_classes = shape[1];

    let input_dense = input
        .to_dense_generic()
        .map_err(crate::AutogradError::TensorError)?;
    let input_slice = input_dense.storage_ref().as_slice();

    let mut result_data = Vec::with_capacity(input_slice.len());

    // Process each batch element separately
    for batch_idx in 0..batch_size {
        // Find max value in this row for numerical stability
        let mut max_val = T::neg_infinity();
        for class_idx in 0..num_classes {
            let val = input_slice[batch_idx * num_classes + class_idx];
            if val > max_val {
                max_val = val;
            }
        }

        // Compute log-sum-exp: log(∑ exp(x_j - max)) + max
        let mut sum_exp = T::zero();
        for class_idx in 0..num_classes {
            let val = input_slice[batch_idx * num_classes + class_idx];
            let diff = val - max_val;
            sum_exp = sum_exp + diff.exp();
        }
        let log_sum_exp = max_val + sum_exp.ln();

        // Compute log-softmax: x_i - log_sum_exp
        for class_idx in 0..num_classes {
            let val = input_slice[batch_idx * num_classes + class_idx];
            result_data.push(val - log_sum_exp);
        }
    }

    Tensor::from_vec(result_data, shape).map_err(crate::AutogradError::TensorError)
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;
    use tensor::Tensor;

    #[test]
    fn test_mse_loss_forward() {
        // Test MSE loss computation
        let predictions = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        let targets = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.5), Float32::new(2.5), Float32::new(3.5)],
            &[3],
        )
        .unwrap();

        let loss = mse_loss(&predictions, &targets).unwrap();

        // Expected: mean((0.5)² + (0.5)² + (0.5)²) = mean(0.25 + 0.25 + 0.25) = 0.25
        let loss_value = loss.as_slice()[0].get();
        assert!(
            (loss_value - 0.25).abs() < 1e-6,
            "MSE loss mismatch: expected 0.25, got {loss_value}",
        );
    }

    #[test]
    fn test_mse_loss_gradient() {
        // Test MSE loss gradient computation
        let predictions = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        )
        .unwrap()
        .requires_grad_(true);

        let targets = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.5), Float32::new(2.5)],
            &[2],
        )
        .unwrap();

        // Use mse_loss function which properly implements division by N
        let loss = mse_loss(&predictions, &targets).unwrap();

        // Create a scalar gradient output (shape [])
        let grad_output = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0)],
            &[],
        )
        .unwrap();

        // Backward pass using backward_with_grad
        crate::ops::backward_with_grad(&loss, grad_output).unwrap();

        // Check gradient: ∂L/∂pred = 2 * (pred - target) / n
        // For pred=[1.0, 2.0], target=[1.5, 2.5]: grad = 2 * [-0.5, -0.5] / 2 = [-0.5, -0.5]
        let pred_grad = predictions.grad().unwrap();
        let expected_grad = [-0.5, -0.5];
        for (i, expected) in expected_grad.iter().copied().enumerate() {
            let actual = pred_grad.as_slice()[i].get();
            assert!(
                (actual - expected).abs() < 1e-2,
                "Gradient mismatch at index {i}: expected {expected}, got {actual}",
            );
        }
    }

    // TODO: Re-enable when numerical gradient validation is implemented
    // #[test]
    // fn test_mse_loss_numerical_gradient() {
    //     use crate::backward;
    //     use crate::numerical::numerical_gradient;
    //
    //     // Test MSE loss with numerical gradient validation
    //     let predictions = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
    //         vec![Float32::new(1.0), Float32::new(2.0)],
    //         &[2]
    //     ).unwrap().requires_grad_(true);
    //
    //     let targets_data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
    //         vec![Float32::new(1.5), Float32::new(2.5)],
    //         &[2]
    //     ).unwrap();
    //
    //     // Compute numerical gradient
    //     let f = |pred: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>| {
    //         mse_loss(pred, &targets_data).unwrap()
    //     };
    //
    //     let numerical_grad = numerical_gradient(f, &predictions, Float32::new(1e-5)).unwrap();
    //
    //     // Compute analytical gradient
    //     let loss = mse_loss(&predictions, &targets_data).unwrap();
    //
    //     // Backward pass using the backward() function
    //     backward(&loss).unwrap();
    //
    //     let analytical_grad = predictions.grad().unwrap();
    //
    //     // Compare gradients
    //     for i in 0..2 {
    //         let num_val = numerical_grad.as_slice()[i].get();
    //         let ana_val = analytical_grad.as_slice()[i].get();
    //         let diff = (num_val - ana_val).abs();
    //         assert!(
    //             diff < 1e-2,
    //             "Gradient mismatch at index {}: numerical={}, analytical={}",
    //             i,
    //             num_val,
    //             ana_val
    //         );
    //     }
    // }

    #[test]
    fn test_cross_entropy_loss_forward() {
        // Test cross-entropy loss computation (simplified version)
        // Logits must be 2D [batch_size=1, num_classes=3]
        let logits = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[1, 3],
        )
        .unwrap();

        // Targets must be 1D [batch_size=1]
        let targets = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.0)], // class 0
            &[1],
        )
        .unwrap();

        let loss = cross_entropy_loss(&logits, &targets).unwrap();

        // Just verify it computes without error (simplified implementation)
        let loss_value = loss.as_slice()[0].get();
        assert!(
            loss_value.is_finite(),
            "Cross-entropy loss should be finite, got {loss_value}",
        );
    }

    #[test]
    fn test_simple_subtraction_backward() {
        use crate::backward;

        // Simplified test: just test subtraction and mean
        let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        )
        .unwrap()
        .requires_grad_(true);

        let b = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.5), Float32::new(2.5)],
            &[2],
        )
        .unwrap();

        // Compute diff = a - b
        let diff = sub(&a, &b).unwrap();

        // Compute mean using autograd mean operation
        // Note: mean reduction usually results in a scalar if keepdim is false
        let result = crate::ops::mean(&diff, None, false).unwrap();

        // Reshape to scalar if needed for backward()
        let result = if result.shape().ndim() == 1 && result.shape().dims()[0] == 1 {
            crate::ops::reshape(&result, &[]).unwrap()
        } else {
            result
        };

        // Backward pass
        backward(&result, None, false, false).unwrap();

        // Check gradient
        assert!(a.grad().is_ok(), "a should have a gradient after backward");
    }
}
