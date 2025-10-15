//! Negative Log Likelihood (NLL) Loss implementation.
//!
//! This module provides the NLL loss function commonly used with log-softmax outputs.

use std::fmt;

use coeus_backend::CpuBackend;
use coeus_dtype::traits::FloatExt;
use coeus_dtype::DataType;
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;

use crate::error::{NNError, Result};

/// Negative Log Likelihood (NLL) Loss function.
///
/// Computes the negative log likelihood loss between log probabilities and targets.
/// Typically used with log-softmax outputs.
///
/// # Examples
/// ```rust
/// use coeus_nn::loss::NLLLoss;
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let loss_fn = NLLLoss::new();
///
/// // Log probabilities from log-softmax (already log probabilities)
/// let log_probs = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
///     vec![
///         Float32::new(-0.5), Float32::new(-1.0), Float32::new(-2.0),  // sample 1
///         Float32::new(-2.0), Float32::new(-0.5), Float32::new(-1.0),  // sample 2
///     ],
///     &[2, 3]  // [batch_size, num_classes]
/// ).unwrap();
///
/// let targets = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(0.0), Float32::new(1.0)],  // class 0 for sample 1, class 1 for sample 2
///     &[2]  // [batch_size]
/// ).unwrap();
///
/// let loss = loss_fn.forward(&log_probs, &targets).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct NLLLoss;

impl NLLLoss {
    /// Create a new NLL loss function.
    pub fn new() -> Self {
        Self
    }

    /// Compute NLL loss between log probabilities and targets.
    ///
    /// # Arguments
    /// * `log_probs` - Log probabilities from log-softmax (shape: [batch_size, num_classes])
    /// * `targets` - Target class indices (shape: [batch_size])
    ///
    /// # Returns
    /// Scalar tensor containing the mean NLL loss.
    pub fn forward<T>(
        &self,
        log_probs: &Tensor<CpuBackend, DenseStorage<T>, T>,
        targets: &Tensor<CpuBackend, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>>
    where
        T: DataType + FloatExt,
    {
        nll_loss(log_probs, targets)
    }
}

impl Default for NLLLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for NLLLoss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NLLLoss")
    }
}

/// Compute negative log likelihood loss.
///
/// # Arguments
/// * `log_probs` - Log probabilities from log-softmax (shape: [batch_size, num_classes])
/// * `targets` - Target class indices (shape: [batch_size])
///
/// # Returns
/// Scalar tensor containing the mean NLL loss.
pub fn nll_loss<T>(
    log_probs: &Tensor<CpuBackend, DenseStorage<T>, T>,
    targets: &Tensor<CpuBackend, DenseStorage<T>, T>,
) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>>
where
    T: DataType + FloatExt,
{
    let log_probs_shape = log_probs.shape().dims();
    let targets_shape = targets.shape().dims();

    // Validate shapes
    if log_probs_shape.len() != 2 {
        return Err(NNError::InvalidInput {
            message: format!("log_probs must be 2D, got {}D", log_probs_shape.len()),
        });
    }

    if targets_shape.len() != 1 {
        return Err(NNError::InvalidInput {
            message: format!("targets must be 1D, got {}D", targets_shape.len()),
        });
    }

    let batch_size = log_probs_shape[0];
    let num_classes = log_probs_shape[1];

    if targets_shape[0] != batch_size {
        return Err(NNError::InvalidInput {
            message: format!(
                "Batch size mismatch: log_probs has {}, targets has {}",
                batch_size, targets_shape[0]
            ),
        });
    }

    let log_probs_data = log_probs.as_slice();
    let targets_data = targets.as_slice();

    // Compute NLL: -mean(log_probs[i, targets[i]])
    let mut total_loss = T::zero();

    for (i, &target) in targets_data.iter().enumerate().take(batch_size) {
        let target_class = target.to_f64().unwrap() as usize;

        if target_class >= num_classes {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Target class {} out of range [0, {})",
                    target_class, num_classes
                ),
            });
        }

        let idx = i * num_classes + target_class;
        total_loss = total_loss - log_probs_data[idx];
    }

    let mean_loss = total_loss / T::from(batch_size).unwrap();

    Tensor::from_vec(vec![mean_loss], &[]).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_dtype::float::Float32;

    #[test]
    fn test_nll_loss_basic() {
        let log_probs = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(-0.5), Float32::new(-1.0), Float32::new(-2.0),  // sample 1
                Float32::new(-2.0), Float32::new(-0.5), Float32::new(-1.0),  // sample 2
            ],
            &[2, 3],
        )
        .unwrap();

        let targets = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.0), Float32::new(1.0)],  // class 0 for sample 1, class 1 for sample 2
            &[2],
        )
        .unwrap();

        let loss_fn = NLLLoss::new();
        let loss = loss_fn.forward(&log_probs, &targets).unwrap();

        // Expected: -mean(log_probs[0,0] + log_probs[1,1]) = -mean(-0.5 + -0.5) = -mean(-1.0) = 1.0
        let loss_value = loss.as_slice()[0].get();
        assert!((loss_value - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_nll_loss_perfect_prediction() {
        let log_probs = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(0.0), Float32::new(-100.0), Float32::new(-100.0),  // very confident prediction of class 0
            ],
            &[1, 3],
        )
        .unwrap();

        let targets = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.0)],  // target class 0
            &[1],
        )
        .unwrap();

        let loss_fn = NLLLoss::new();
        let loss = loss_fn.forward(&log_probs, &targets).unwrap();

        // Loss should be 0 for perfect prediction (log prob = 0, -log prob = 0)
        let loss_value = loss.as_slice()[0].get();
        assert!(loss_value.abs() < 1e-6);
    }

    #[test]
    fn test_nll_loss_shape_mismatch() {
        let log_probs = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(-0.5), Float32::new(-1.0)],
            &[1, 2],
        )
        .unwrap();

        let targets = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.0), Float32::new(1.0)],  // wrong shape
            &[2],
        )
        .unwrap();

        let loss_fn = NLLLoss::new();
        let result = loss_fn.forward(&log_probs, &targets);
        assert!(result.is_err());
    }

    #[test]
    fn test_nll_loss_invalid_class() {
        let log_probs = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(-0.5), Float32::new(-1.0)],
            &[1, 2],
        )
        .unwrap();

        let targets = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(5.0)],  // invalid class index
            &[1],
        )
        .unwrap();

        let loss_fn = NLLLoss::new();
        let result = loss_fn.forward(&log_probs, &targets);
        assert!(result.is_err());
    }
}
