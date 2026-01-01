//! Binary Cross-Entropy with Logits (BCEWithLogits) Loss implementation.

use std::fmt;

use backend::Backend;
use dtype::traits::FloatExt;
use dtype::DataType;
use tensor::Tensor;

use crate::error::Result;
use crate::functional_loss::bce_with_logits_loss;

/// Binary Cross-Entropy with Logits Loss function.
///
/// This loss combines a Sigmoid layer and the BCE loss in one single class.
/// This version is more numerically stable than using a plain Sigmoid followed
/// by a BCE loss as, by combining the operations into one layer, we take advantage
/// of the log-sum-exp trick for numerical stability.
///
/// Formula: `loss = -[target * log(sigmoid(input)) + (1 - target) * log(1 - sigmoid(input))]`
///
/// # Examples
/// ```rust
/// use nn::loss::BCEWithLogitsLoss;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let loss_fn = BCEWithLogitsLoss::new();
///
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(0.0), Float32::new(1.0), Float32::new(-1.0)],
///     &[3]
/// ).unwrap();
///
/// let target = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(0.0), Float32::new(1.0), Float32::new(0.0)],
///     &[3]
/// ).unwrap();
///
/// let loss = loss_fn.forward(&input, &target).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct BCEWithLogitsLoss;

impl BCEWithLogitsLoss {
    /// Create a new BCEWithLogits loss function.
    pub fn new() -> Self {
        Self
    }

    /// Compute BCEWithLogits loss between input and target.
    ///
    /// # Arguments
    /// * `input` - Predicted logits
    /// * `target` - Target values
    ///
    /// # Returns
    /// Scalar tensor containing the loss value.
    pub fn forward<B, S, T>(
        &self,
        input: &Tensor<B, S, T>,
        target: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, storage::DenseStorage<T>, T>>
    where
        B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
        S: storage::StorageToDense<T> + storage::StorageFromVec<T> + 'static,
        T: DataType
            + FloatExt
            + num_traits::FromPrimitive
            + PartialOrd
            + Copy
            + Send
            + Sync
            + 'static,
    {
        bce_with_logits_loss(input, target)
    }
}

impl Default for BCEWithLogitsLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for BCEWithLogitsLoss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BCEWithLogitsLoss")
    }
}
