//! Instance Normalization layer.

use backend::{Backend, CpuBackend};
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use super::GroupNorm;
use crate::core::error::Result;
use crate::core::module::Module;
use crate::core::parameter::Parameter;

/// Instance Normalization layer.
///
/// Normalizes each channel independently for each instance in the batch.
/// This is equivalent to GroupNorm with num_groups = num_channels.
#[derive(Debug, Clone)]
pub struct InstanceNorm<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType,
{
    /// Underlying GroupNorm with num_groups = num_channels
    group_norm: GroupNorm<B, S, T>,
}

impl<B, S, T> InstanceNorm<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + num_traits::One + num_traits::Zero,
{
    /// Create a new InstanceNorm layer.
    pub fn new(num_channels: usize, eps: f64, affine: bool) -> Result<Self> {
        Ok(Self {
            group_norm: GroupNorm::new(num_channels, num_channels, eps, affine)?,
        })
    }

    /// Get the number of channels
    pub fn num_channels(&self) -> usize {
        self.group_norm.num_channels
    }

    /// Get the epsilon value
    pub fn eps(&self) -> f64 {
        self.group_norm.eps
    }

    /// Get the affine flag
    pub fn affine(&self) -> bool {
        self.group_norm.affine
    }

    /// Get the weight parameter
    pub fn weight(&self) -> &Parameter<B, S, T> {
        &self.group_norm.weight
    }

    /// Get the bias parameter
    pub fn bias(&self) -> &Parameter<B, S, T> {
        &self.group_norm.bias
    }
}

impl<T> Module<CpuBackend<T>, DenseStorage<T>, T>
    for InstanceNorm<CpuBackend<T>, DenseStorage<T>, T>
where
    T: DataType + FloatExt + PartialOrd,
{
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        self.group_norm.forward(input)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        self.group_norm.parameters()
    }

    fn zero_grad(&mut self) {
        self.group_norm.zero_grad();
    }

    fn train(&mut self, mode: bool) {
        self.group_norm.train(mode);
    }

    fn name(&self) -> &str {
        "InstanceNorm"
    }

    fn clone_box(&self) -> Box<dyn Module<CpuBackend<T>, DenseStorage<T>, T>> {
        Box::new(self.clone())
    }
}
