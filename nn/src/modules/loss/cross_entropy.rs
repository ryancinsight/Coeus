//! Cross-Entropy Loss implementation.
//!
//! This module provides cross-entropy loss for classification tasks.

use std::fmt;

use dtype::traits::FloatExt;
use dtype::DataType;
use tensor::Tensor;

use crate::core::error::Result;

/// Cross-entropy loss function for classification.
///
/// Computes the cross-entropy loss between logits and class targets.
/// Assumes logits are unnormalized and applies softmax internally.
///
/// # Examples
/// ```rust
/// use nn::loss::CrossEntropyLoss;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let loss_fn = CrossEntropyLoss::new();
///
/// // 3 classes, 2 samples
/// let logits = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![
///         Float32::new(1.0), Float32::new(0.5), Float32::new(0.2),  // sample 1
///         Float32::new(0.1), Float32::new(2.0), Float32::new(0.3),  // sample 2
///     ],
///     &[2, 3]
/// ).unwrap();
///
/// let targets = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(0.0), Float32::new(1.0)],  // class 0 for sample 1, class 1 for sample 2
///     &[2]
/// ).unwrap();
///
/// let loss = loss_fn.forward(&logits, &targets).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// Create a new cross-entropy loss function.
    pub fn new() -> Self {
        Self
    }

    /// Compute cross-entropy loss between logits and targets.
    ///
    /// # Arguments
    /// * `logits` - Unnormalized predictions [batch_size, num_classes]
    /// * `targets` - Class indices [batch_size] (as integers stored in T)
    ///
    /// # Returns
    /// Scalar tensor containing the cross-entropy loss value.
    pub fn forward<B, S, T>(
        &self,
        logits: &Tensor<B, S, T>,
        targets: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>>
    where
        B: backend::Backend<Data = T> + Clone + Default + Send + Sync + 'static,
        S: storage::Storage<T>
            + storage::StorageFromVec<T>
            + storage::StorageToDense<T>
            + Clone
            + 'static
            + tensor::ops::dispatch::TensorStorageOps<T>,
        T: DataType
            + FloatExt
            + std::ops::Neg<Output = T>
            + PartialOrd
            + num_traits::FromPrimitive
            + Copy
            + Send
            + Sync
            + 'static,
    {
        cross_entropy_loss(logits, targets)
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CrossEntropyLoss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CrossEntropyLoss")
    }
}

/// Compute cross-entropy loss between logits and targets.
///
/// # Arguments
/// * `logits` - Unnormalized predictions [batch_size, num_classes]
/// * `targets` - Class indices [batch_size] (as integers stored in T)
///
/// # Returns
/// Scalar tensor containing the cross-entropy loss value.
pub fn cross_entropy_loss<B, S, T>(
    logits: &Tensor<B, S, T>,
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
        + PartialOrd
        + num_traits::FromPrimitive
        + Copy
        + Send
        + Sync
        + 'static,
{
    crate::ops::loss::cross_entropy(logits, targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;

    #[test]
    fn test_cross_entropy_loss_basic() {
        let logits = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(0.5),
                Float32::new(0.2), // sample 1
                Float32::new(0.1),
                Float32::new(2.0),
                Float32::new(0.3), // sample 2
            ],
            &[2, 3],
        )
        .unwrap();

        let targets = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.0), Float32::new(1.0)], // class 0 for sample 1, class 1 for sample 2
            &[2],
        )
        .unwrap();

        let loss_fn = CrossEntropyLoss::new();
        let loss = loss_fn.forward(&logits, &targets).unwrap();

        // Loss should be positive
        let loss_value = loss.as_slice()[0].get();
        assert!(loss_value > 0.0);
    }

    #[test]
    fn test_cross_entropy_loss_perfect_prediction() {
        let logits = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(10.0),
                Float32::new(0.0),
                Float32::new(0.0), // very confident prediction of class 0
            ],
            &[1, 3],
        )
        .unwrap();

        let targets = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.0)], // target class 0
            &[1],
        )
        .unwrap();

        let loss_fn = CrossEntropyLoss::new();
        let loss = loss_fn.forward(&logits, &targets).unwrap();

        // Loss should be very close to 0 for perfect prediction
        let loss_value = loss.as_slice()[0].get();
        assert!(loss_value < 0.01);
    }

    #[test]
    fn test_cross_entropy_loss_shape_mismatch() {
        let logits = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(0.5)],
            &[1, 2],
        )
        .unwrap();

        let targets = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.0), Float32::new(1.0)], // wrong shape
            &[2],
        )
        .unwrap();

        let loss_fn = CrossEntropyLoss::new();
        let result = loss_fn.forward(&logits, &targets);
        assert!(result.is_err());
    }
}
