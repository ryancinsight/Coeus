//! Loss functions for neural networks.
//!
//! This module provides stateless loss function computations
//! for training neural networks.

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
#[allow(unused_imports)]
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::core::error::{NNError, Result};
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

#[cfg(feature = "autograd")]
#[derive(Debug)]
struct NLLLossFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    inputs: Vec<Arc<Tensor<B, S, T>>>,
    is_indices: bool,
}

#[cfg(feature = "autograd")]
impl<B, S, T> NLLLossFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    fn new(
        log_probs: Arc<Tensor<B, S, T>>,
        targets: Arc<Tensor<B, S, T>>,
        is_indices: bool,
    ) -> Self {
        Self {
            inputs: vec![log_probs, targets],
            is_indices,
        }
    }
}

#[cfg(feature = "autograd")]
impl<B, S, T> tensor::AsAny for NLLLossFunction<B, S, T>
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
impl<B, S, T> tensor::DifferentiableFunction<B, S, T> for NLLLossFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + 'static,
    T: DataType + FloatExt + FromPrimitive + Copy + Send + Sync + 'static,
{
    fn name(&self) -> &'static str {
        "NLLLossBackward"
    }
}

#[cfg(feature = "autograd")]
impl<B, S, T> tensor::Function<B, S, T> for NLLLossFunction<B, S, T>
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
        let log_probs = &*self.inputs[0];
        let targets = &*self.inputs[1];

        let logits_shape = log_probs.shape().dims();
        if logits_shape.is_empty() {
            return Err(anyhow::anyhow!("NLL loss requires at least 1 dimension"));
        }
        let num_classes = *logits_shape.last().unwrap();
        if num_classes == 0 {
            return Err(anyhow::anyhow!("NLL loss requires num_classes > 0"));
        }

        let log_probs_dense = log_probs.to_dense_generic()?;
        let targets_dense = targets.to_dense_generic()?;

        let log_probs_data = log_probs_dense.storage_ref().as_slice();
        let targets_data = targets_dense.storage_ref().as_slice();

        let batch_elems = log_probs_data
            .len()
            .checked_div(num_classes)
            .ok_or_else(|| anyhow::anyhow!("NLL loss invalid batch size computation"))?;
        if batch_elems == 0 {
            return Err(anyhow::anyhow!("NLL loss invalid empty batch"));
        }

        let go = grad_output.as_slice();
        if go.len() != 1 {
            return Err(anyhow::anyhow!(
                "Expected scalar grad_output for NLL loss backward, got numel={}",
                go.len()
            ));
        }
        let grad_scale = go[0];

        let inv_batch = T::from_f64(1.0 / batch_elems as f64)
            .ok_or_else(|| anyhow::anyhow!("Failed to convert batch scale"))?;
        let scaled = grad_scale * inv_batch;

        if self.is_indices {
            if targets_data.len() != batch_elems {
                return Err(anyhow::anyhow!(
                    "NLL loss backward: target length {} does not match batch_elems {}",
                    targets_data.len(),
                    batch_elems
                ));
            }

            let neg_scaled = T::zero() - scaled;
            let mut grad_logits = vec![T::zero(); log_probs_data.len()];

            for (b, &target_val) in targets_data.iter().enumerate() {
                let target_f64 = target_val.to_f64().ok_or_else(|| {
                    anyhow::anyhow!("NLL loss backward: target not representable")
                })?;
                if !target_f64.is_finite() {
                    return Err(anyhow::anyhow!(
                        "NLL loss backward: target index is not finite at batch {b}"
                    ));
                }
                if target_f64.fract() != 0.0 {
                    return Err(anyhow::anyhow!(
                        "NLL loss backward: target index is not integral at batch {b}"
                    ));
                }
                if target_f64 < 0.0 || target_f64 >= num_classes as f64 {
                    return Err(anyhow::anyhow!(
                        "NLL loss backward: target index {} out of range [0, {}) at batch {b}",
                        target_f64,
                        num_classes
                    ));
                }
                let target_idx = target_f64 as usize;
                grad_logits[b * num_classes + target_idx] = neg_scaled;
            }

            let grad_log_probs = Tensor::from_vec_with_backend(
                grad_logits,
                logits_shape,
                log_probs.backend().clone(),
            )?;
            let grad_targets = Tensor::from_vec_with_backend(
                vec![T::zero(); targets_data.len()],
                targets.shape().dims(),
                targets.backend().clone(),
            )?;
            Ok(vec![grad_log_probs, grad_targets])
        } else {
            if targets_data.len() != log_probs_data.len() {
                return Err(anyhow::anyhow!(
                    "NLL loss backward: target length {} does not match logits length {}",
                    targets_data.len(),
                    log_probs_data.len()
                ));
            }

            let mut grad_logits = Vec::with_capacity(log_probs_data.len());
            let mut grad_targets = Vec::with_capacity(targets_data.len());

            let neg_scaled = T::zero() - scaled;
            for (&lp, &t) in log_probs_data.iter().zip(targets_data.iter()) {
                grad_logits.push(t * neg_scaled);
                grad_targets.push(lp * neg_scaled);
            }

            let grad_log_probs = Tensor::from_vec_with_backend(
                grad_logits,
                logits_shape,
                log_probs.backend().clone(),
            )?;
            let grad_targets = Tensor::from_vec_with_backend(
                grad_targets,
                targets.shape().dims(),
                targets.backend().clone(),
            )?;
            Ok(vec![grad_log_probs, grad_targets])
        }
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
            let batch_data = &input_data[b * last_dim..last_dim * (b + 1)];

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
                let target_idx = target_data[b].to_f64().unwrap_or(0.0) as usize;
                if target_idx < last_dim {
                    let softmax_val = exp_vals[target_idx] / sum_exp;
                    let epsilon = T::from_f64(1e-12).unwrap();
                    total_loss = total_loss - (softmax_val + epsilon).ln();
                }
            } else {
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
pub fn bce_with_logits_loss<B, S, T>(
    input: &Tensor<B, S, T>,
    target: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
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
        let max_x_0 = if x > zero { x } else { zero };
        let abs_x = if x > zero { x } else { -x };
        let term1 = max_x_0 - x * y;
        let term2 = (zero + (-abs_x).exp() + T::from_f64(1.0).unwrap()).ln();
        total_loss = total_loss + term1 + term2;
    }

    let mean_loss = total_loss / T::from_f64(input_data.len() as f64).unwrap();

    Ok(Tensor::<B, S, T>::from_vec_with_backend(
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
pub fn nll_loss<B, S, T>(
    input: &Tensor<B, S, T>,
    target: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
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

    if input_shape.is_empty() {
        return Err(NNError::InvalidInput {
            message: "Input must have at least 1 dimension (..., num_classes)".to_string(),
        });
    }

    let num_classes = *input_shape.last().unwrap();
    if num_classes == 0 {
        return Err(NNError::InvalidInput {
            message: "nll_loss requires num_classes > 0".to_string(),
        });
    }

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

    let input_dense = input.to_dense_generic()?;
    let target_dense = target.to_dense_generic()?;

    let input_data = input_dense.as_slice();
    let target_data = target_dense.as_slice();

    let batch_elems =
        input_data
            .len()
            .checked_div(num_classes)
            .ok_or_else(|| NNError::InvalidInput {
                message: "nll_loss invalid batch size computation".to_string(),
            })?;
    if batch_elems == 0 {
        return Err(NNError::InvalidInput {
            message: "nll_loss invalid empty batch".to_string(),
        });
    }

    let mut total_loss = T::from_f64(0.0).ok_or_else(|| NNError::NumericalError {
        message: "nll_loss failed to initialize loss accumulator".to_string(),
    })?;

    if is_indices {
        if target_data.len() != batch_elems {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Target length {} does not match batch size {}",
                    target_data.len(),
                    batch_elems
                ),
            });
        }

        for (b, &target_val) in target_data.iter().enumerate() {
            let target_f64 = target_val.to_f64().ok_or_else(|| NNError::InvalidInput {
                message: "Target values must be representable as f64".to_string(),
            })?;
            if !target_f64.is_finite() {
                return Err(NNError::InvalidInput {
                    message: format!("Target index is not finite at batch {b}"),
                });
            }
            if target_f64.fract() != 0.0 {
                return Err(NNError::InvalidInput {
                    message: format!("Target index is not integral at batch {b}"),
                });
            }
            if target_f64 < 0.0 || target_f64 >= num_classes as f64 {
                return Err(NNError::InvalidInput {
                    message: format!(
                        "Target index {} out of range [0, {}) at batch {b}",
                        target_f64, num_classes
                    ),
                });
            }
            let target_idx = target_f64 as usize;
            let log_prob = input_data[b * num_classes + target_idx];
            total_loss = total_loss - log_prob;
        }
    } else {
        if target_data.len() != input_data.len() {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Input length {} does not match target length {}",
                    input_data.len(),
                    target_data.len()
                ),
            });
        }

        for b in 0..batch_elems {
            let start = b * num_classes;
            let row_input = &input_data[start..start + num_classes];
            let row_target = &target_data[start..start + num_classes];

            let mut dot = T::from_f64(0.0).ok_or_else(|| NNError::NumericalError {
                message: "nll_loss failed to initialize row accumulator".to_string(),
            })?;
            for (&lp, &t) in row_input.iter().zip(row_target.iter()) {
                dot = dot + t * lp;
            }
            total_loss = total_loss - dot;
        }
    }

    let batch_t = T::from_f64(batch_elems as f64).ok_or_else(|| NNError::NumericalError {
        message: "nll_loss failed to convert batch size".to_string(),
    })?;
    let mean_loss = total_loss / batch_t;

    let mut out = Tensor::<B, S, T>::from_vec_with_backend(
        vec![mean_loss],
        &[1],
        input.backend().clone(),
    )?;

    #[cfg(feature = "autograd")]
    {
        if input.requires_grad() || target.requires_grad() {
            out = out
                .with_grad_fn(Some(Arc::new(NLLLossFunction::new(
                    Arc::new(input.clone()),
                    Arc::new(target.clone()),
                    is_indices,
                ))))
                .requires_grad_(true);
        }
    }

    Ok(out)
}
