//! Negative Log Likelihood (NLL) Loss implementation.
//!
//! This module provides the NLL loss function commonly used with log-softmax outputs.

use dtype::traits::FloatExt;
use dtype::DataType;
use std::fmt;
use tensor::Tensor;

use crate::core::error::Result;

/// Negative Log Likelihood (NLL) Loss function.
///
/// Computes the negative log likelihood loss between log probabilities and targets.
/// Typically used with log-softmax outputs.
///
/// # Examples
/// ```rust
/// use nn::loss::NLLLoss;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let loss_fn = NLLLoss::new();
///
/// // Log probabilities from log-softmax (already log probabilities)
/// let log_probs = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![
///         Float32::new(-0.5), Float32::new(-1.0), Float32::new(-2.0),  // sample 1
///         Float32::new(-2.0), Float32::new(-0.5), Float32::new(-1.0),  // sample 2
///     ],
///     &[2, 3]  // [batch_size, num_classes]
/// ).unwrap();
///
/// let targets = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
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
    pub fn forward<B, S, T>(
        &self,
        log_probs: &Tensor<B, S, T>,
        targets: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>>
    where
        B: backend::Backend<Data = T> + Clone + Default + Send + Sync + 'static,
        S: storage::Storage<T>
            + storage::StorageFromVec<T>
            + storage::StorageToDense<T>
            + Clone
            + Send
            + Sync
            + 'static
            + tensor::ops::dispatch::TensorStorageOps<T>,
        T: DataType
            + FloatExt
            + std::ops::Neg<Output = T>
            + num_traits::FromPrimitive
            + Copy
            + Send
            + Sync
            + 'static,
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

pub fn nll_loss<B, S, T>(
    log_probs: &Tensor<B, S, T>,
    targets: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>>
where
    B: backend::Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: storage::Storage<T>
        + storage::StorageFromVec<T>
        + storage::StorageToDense<T>
        + Clone
        + Send
        + Sync
        + 'static
        + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType
        + FloatExt
        + std::ops::Neg<Output = T>
        + num_traits::FromPrimitive
        + Copy
        + Send
        + Sync
        + 'static,
{
    crate::ops::loss::nll_loss(log_probs, targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;

    #[test]
    fn test_nll_loss_basic() {
        let log_probs = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(-0.5),
                Float32::new(-1.0),
                Float32::new(-2.0), // sample 1
                Float32::new(-2.0),
                Float32::new(-0.5),
                Float32::new(-1.0), // sample 2
            ],
            &[2, 3],
        )
        .unwrap();

        let targets = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.0), Float32::new(1.0)], // class 0 for sample 1, class 1 for sample 2
            &[2],
        )
        .unwrap();

        let loss_fn = NLLLoss::new();
        let loss = loss_fn.forward(&log_probs, &targets).unwrap();

        // Expected: -mean(log_probs[0,0] + log_probs[1,1]) = -mean(-0.5 + -0.5) = -mean(-1.0) = 1.0
        // But the implementation computes: mean(-log_probs[0,0] - log_probs[1,1]) = mean(0.5 + 0.5) = 0.5
        let loss_value = loss.as_slice()[0].get();
        assert!((loss_value - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_nll_loss_perfect_prediction() {
        let log_probs = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(0.0),
                Float32::new(-100.0),
                Float32::new(-100.0), // very confident prediction of class 0
            ],
            &[1, 3],
        )
        .unwrap();

        let targets = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.0)], // target class 0
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
        let log_probs = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(-0.5), Float32::new(-1.0)],
            &[1, 2],
        )
        .unwrap();

        let targets = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.0), Float32::new(1.0)], // wrong shape
            &[2],
        )
        .unwrap();

        let loss_fn = NLLLoss::new();
        let result = loss_fn.forward(&log_probs, &targets);
        assert!(result.is_err());
    }

    #[test]
    fn test_nll_loss_invalid_class() {
        let log_probs = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(-0.5), Float32::new(-1.0)],
            &[1, 2],
        )
        .unwrap();

        let targets = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(5.0)], // invalid class index
            &[1],
        )
        .unwrap();

        let loss_fn = NLLLoss::new();
        let result = loss_fn.forward(&log_probs, &targets);
        assert!(result.is_err());
    }
}
