//! Binary Cross-Entropy with Logits (BCEWithLogits) Loss implementation.

use std::fmt;

use backend::Backend;
use dtype::traits::FloatExt;
use dtype::DataType;
use tensor::Tensor;

use crate::core::error::Result;
use crate::ops::loss::bce_with_logits_loss;
use crate::{Module, Parameter};

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
    pub fn forward<B, S, T>(&self, input: &Tensor<B, S, T>, target: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default,
        S: storage::Storage<T> + storage::StorageFromVec<T> + storage::StorageToDense<T> + Clone + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
        T: DataType + FloatExt + num_traits::Zero + num_traits::One + num_traits::FromPrimitive
            + PartialOrd
            + Copy
            + Send
            + Sync
            + 'static,
    {
        bce_with_logits_loss(input, target)
    }
}

impl<B, S, T> Module<B, S, T> for BCEWithLogitsLoss
where
    B: Backend<Data = T> + Clone + Default,
    S: storage::Storage<T> + storage::StorageFromVec<T> + storage::StorageToDense<T> + Clone + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::Zero + num_traits::One + num_traits::FromPrimitive
        + PartialOrd
        + Copy
        + Send
        + Sync
        + 'static,
{
    type Input = (Tensor<B, S, T>, Tensor<B, S, T>);
    type Output = Tensor<B, S, T>;

    fn forward(&self, input: &(Tensor<B, S, T>, Tensor<B, S, T>)) -> Result<Tensor<B, S, T>> {
        let (logits, target) = input;
        bce_with_logits_loss(logits, target)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![]
    }

    fn zero_grad(&mut self) {}

    fn train(&mut self, _mode: bool) {}

    fn name(&self) -> &str {
        "BCEWithLogitsLoss"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
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
