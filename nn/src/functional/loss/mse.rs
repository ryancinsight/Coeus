//! Mean Squared Error (MSE) Loss implementation.
//!
//! This module provides the MSE loss function commonly used for regression tasks.

use std::fmt;

use backend::Backend;
use dtype::traits::FloatExt;
use dtype::DataType;

use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::core::error::Result;

/// Mean Squared Error (MSE) Loss function.
///
/// Computes the mean squared error between predictions and targets:
/// `loss = mean((predictions - targets)²)`
///
/// # Examples
/// ```rust
/// use nn::loss::MSELoss;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let loss_fn = MSELoss::new();
///
/// let predictions = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
///     &[3]
/// ).unwrap();
///
/// let targets = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.5), Float32::new(2.0), Float32::new(2.5)],
///     &[3]
/// ).unwrap();
///
/// let loss = loss_fn.forward(&predictions, &targets).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MSELoss;

impl MSELoss {
    /// Create a new MSE loss function.
    pub fn new() -> Self {
        Self
    }

    /// Compute MSE loss between predictions and targets.
    ///
    /// # Arguments
    /// * `predictions` - Predicted values
    /// * `targets` - Target values
    ///
    /// # Returns
    /// Scalar tensor containing the MSE loss value.
    pub fn forward<B, S, T>(
        &self,
        predictions: &Tensor<B, S, T>,
        targets: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
        S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + Clone + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
        T: DataType + FloatExt + num_traits::FromPrimitive + Copy + Send + Sync + 'static,
    {
        mse_loss(predictions, targets)
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MSELoss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MSELoss")
    }
}

/// Compute MSE loss between predictions and targets.
///
/// # Arguments
/// * `predictions` - Predicted values
/// * `targets` - Target values
///
/// # Returns
/// Scalar tensor containing the MSE loss value.
pub fn mse_loss<B, S, T>(
    predictions: &Tensor<B, S, T>,
    targets: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + Clone + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::FromPrimitive + Copy + Send + Sync + 'static,
{
    crate::ops::loss::mse_loss(predictions, targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;
    use tensor::Tensor;

    #[test]
    fn test_mse_loss_basic() {
        let predictions = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        let targets = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.5), Float32::new(2.0), Float32::new(2.5)],
            &[3],
        )
        .unwrap();

        let loss_fn = MSELoss::new();
        let loss = loss_fn.forward(&predictions, &targets).unwrap();

        // Expected: mean((1-1.5)² + (2-2)² + (3-2.5)²) = mean(0.25 + 0 + 0.25) = 0.5/3 = 0.166...
        let loss_value = loss.as_slice()[0].get();
        assert!(loss_value > 0.166 && loss_value < 0.167);
    }

    #[test]
    fn test_mse_loss_perfect_prediction() {
        let predictions = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        let targets = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        let loss_fn = MSELoss::new();
        let loss = loss_fn.forward(&predictions, &targets).unwrap();

        // Perfect prediction should give zero loss
        let loss_value = loss.as_slice()[0].get();
        assert!(loss_value.abs() < 1e-6);
    }

    #[test]
    fn test_mse_loss_shape_mismatch() {
        let predictions = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        )
        .unwrap();

        let targets = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0)],
            &[1],
        )
        .unwrap();

        let loss_fn = MSELoss::new();
        let result = loss_fn.forward(&predictions, &targets);
        assert!(result.is_err());
    }
}
