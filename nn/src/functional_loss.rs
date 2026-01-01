//! Loss functions for neural networks.
//!
//! This module provides stateless loss function computations
//! for training neural networks.

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
#[allow(unused_imports)]
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::error::{NNError, Result};
use num_traits::FromPrimitive;

#[cfg(feature = "autograd")]
use std::sync::Arc;

#[cfg(feature = "autograd")]
#[derive(Debug)]
struct SoftCrossEntropyFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    inputs: Vec<Arc<Tensor<B, S, T>>>,
}

#[cfg(feature = "autograd")]
impl<B, S, T> SoftCrossEntropyFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    fn new(logits: Arc<Tensor<B, S, T>>, targets: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![logits, targets],
        }
    }
}

#[cfg(feature = "autograd")]
impl<B, S, T> tensor::AsAny for SoftCrossEntropyFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(feature = "autograd")]
impl<B, S, T> tensor::DifferentiableFunction<B, S, T> for SoftCrossEntropyFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + 'static,
    T: DataType + FloatExt + FromPrimitive + Copy + Send + Sync + 'static,
{
    fn name(&self) -> &'static str {
        "SoftCrossEntropyBackward"
    }
}

#[cfg(feature = "autograd")]
impl<B, S, T> tensor::Function<B, S, T> for SoftCrossEntropyFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + 'static,
    T: DataType + FloatExt + FromPrimitive + Copy + Send + Sync + 'static,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        let logits = &*self.inputs[0];
        let targets = &*self.inputs[1];

        let logits_shape = logits.shape().dims();
        if logits_shape.is_empty() {
            return Err(anyhow::anyhow!(
                "Soft cross-entropy requires at least 1 dimension"
            ));
        }
        let num_classes = *logits_shape.last().unwrap();
        if num_classes < 2 {
            return Err(anyhow::anyhow!(
                "Soft cross-entropy requires at least 2 classes"
            ));
        }

        let logits_dense = logits.to_dense_generic()?;
        let targets_dense = targets.to_dense_generic()?;

        let logits_data = logits_dense.storage_ref().as_slice();
        let targets_data = targets_dense.storage_ref().as_slice();

        if logits_data.len() != targets_data.len() {
            return Err(anyhow::anyhow!("Soft cross-entropy shape mismatch"));
        }

        let batch_size = logits_data.len() / num_classes;
        if batch_size == 0 {
            return Err(anyhow::anyhow!("Soft cross-entropy invalid empty batch"));
        }

        let go = grad_output.as_slice();
        if go.is_empty() {
            return Err(anyhow::anyhow!("Soft cross-entropy grad_output is empty"));
        }

        let inv_batch = T::from_f64(1.0 / batch_size as f64)
            .ok_or_else(|| anyhow::anyhow!("Failed to convert batch scale"))?;
        let grad_scale = go[0] * inv_batch;

        let mut grad_logits = Vec::with_capacity(logits_data.len());
        let mut grad_targets = Vec::with_capacity(targets_data.len());

        for b in 0..batch_size {
            let start = b * num_classes;
            let row_logits = &logits_data[start..start + num_classes];
            let row_targets = &targets_data[start..start + num_classes];

            let mut max_val = T::neg_infinity();
            for &v in row_logits {
                if v > max_val {
                    max_val = v;
                }
            }

            let mut sum_exp =
                T::from_f64(0.0).ok_or_else(|| anyhow::anyhow!("Failed to init sum_exp"))?;
            for &v in row_logits {
                sum_exp = sum_exp + (v - max_val).exp();
            }
            let log_sum_exp = max_val + sum_exp.ln();

            let mut sum_targets =
                T::from_f64(0.0).ok_or_else(|| anyhow::anyhow!("Failed to init sum_targets"))?;
            for &y in row_targets {
                sum_targets = sum_targets + y;
            }

            for c in 0..num_classes {
                let logit = row_logits[c];
                let target = row_targets[c];
                let exp_val = (logit - max_val).exp();
                let softmax = exp_val / sum_exp;
                grad_logits.push((softmax * sum_targets - target) * grad_scale);
                grad_targets.push(-(logit - log_sum_exp) * grad_scale);
            }
        }

        let logits_grad =
            Tensor::from_vec_with_backend(grad_logits, logits_shape, logits.backend().clone())?;
        let targets_grad =
            Tensor::from_vec_with_backend(grad_targets, logits_shape, logits.backend().clone())?;

        Ok(vec![logits_grad, targets_grad])
    }
}

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
pub fn mse_loss<B, S, T>(
    input: &Tensor<B, S, T>,
    target: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
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

    #[cfg(feature = "autograd")]
    {
        Ok(autograd::loss::mse_loss(input, target)?)
    }

    #[cfg(not(feature = "autograd"))]
    {
        let input_dense = input.to_dense_generic()?;
        let target_dense = target.to_dense_generic()?;
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
        Ok(Tensor::<B, S, T>::from_vec_with_backend(
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
pub fn cross_entropy<B, S, T>(
    input: &Tensor<B, S, T>,
    target: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
    T: DataType
        + FloatExt
        + std::ops::Neg<Output = T>
        + PartialOrd
        + Clone
        + FromPrimitive
        + Send
        + Sync
        + 'static,
{
    let input_shape = input.shape().dims();
    let target_shape = target.shape().dims();

    let is_probs = input_shape == target_shape;
    let is_indices = target_shape.len() + 1 == input_shape.len()
        && target_shape == &input_shape[..input_shape.len() - 1];

    if !is_probs && !is_indices {
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

    let last_dim = *input_shape.last().unwrap();

    #[cfg(feature = "autograd")]
    {
        if is_indices {
            return Ok(autograd::loss::cross_entropy_loss(input, target)?);
        }

        let input_dense = input.to_dense_generic()?;
        let target_dense = target.to_dense_generic()?;

        let input_data = input_dense.as_slice();
        let target_data = target_dense.as_slice();

        let batch_size = input_data.len() / last_dim;
        if batch_size == 0 {
            return Err(NNError::InvalidInput {
                message: "Cross-entropy invalid empty batch".to_string(),
            });
        }

        let mut total_loss = T::from_f64(0.0).unwrap();
        for b in 0..batch_size {
            let start = b * last_dim;
            let row_logits = &input_data[start..start + last_dim];
            let row_targets = &target_data[start..start + last_dim];

            let mut max_val = T::neg_infinity();
            for &v in row_logits {
                if v > max_val {
                    max_val = v;
                }
            }

            let mut sum_exp = T::from_f64(0.0).unwrap();
            for &v in row_logits {
                sum_exp = sum_exp + (v - max_val).exp();
            }
            let log_sum_exp = max_val + sum_exp.ln();

            for c in 0..last_dim {
                let log_softmax = row_logits[c] - log_sum_exp;
                total_loss = total_loss - row_targets[c] * log_softmax;
            }
        }

        let mean_loss = total_loss / T::from_f64(batch_size as f64).unwrap();
        let mut out = Tensor::<B, S, T>::from_vec_with_backend(
            vec![mean_loss],
            &[1],
            input.backend().clone(),
        )?;

        if input.requires_grad() || target.requires_grad() {
            out = out
                .with_grad_fn(Some(Arc::new(SoftCrossEntropyFunction::new(
                    Arc::new(input.clone()),
                    Arc::new(target.clone()),
                ))))
                .requires_grad_(true);
        }

        Ok(out)
    }

    #[cfg(not(feature = "autograd"))]
    {
        let input_dense = input.to_dense_generic()?;
        let target_dense = target.to_dense_generic()?;
        let input_data = input_dense.as_slice();
        let target_data = target_dense.as_slice();
        let batch_size = input_data.len() / last_dim;

        let mut total_loss = T::from_f64(0.0).unwrap();

        for b in 0..batch_size {
            let start = b * last_dim;
            let batch_data = &input_data[start..last_dim * (b + 1)];

            // Softmax for current batch
            let max_val =
                batch_data
                    .iter()
                    .fold(T::from_f64(f64::NEG_INFINITY).unwrap(), |a, &b| {
                        if a > b {
                            a
                        } else {
                            b
                        }
                    });
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

        let mean_loss = total_loss / T::from_f64(batch_size as f64).unwrap();

        Ok(Tensor::<B, S, T>::from_vec_with_backend(
            vec![mean_loss],
            &[1],
            input.backend().clone(),
        )?)
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
            message: format!(
                "Shape mismatch: input {:?}, target {:?}",
                input_shape, target_shape
            ),
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

    Ok(Tensor::from_vec_with_backend(
        vec![mean_loss],
        &[1],
        input.backend().clone(),
    )?)
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
    T: DataType
        + FloatExt
        + std::ops::Neg<Output = T>
        + Clone
        + FromPrimitive
        + Send
        + Sync
        + 'static,
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
        let neg_one = Tensor::from_vec_with_backend(
            vec![T::from_f64(-1.0).unwrap()],
            &[1],
            input.backend().clone(),
        )?;
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
