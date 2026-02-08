use backend::{Backend, CpuBackend};
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
use tensor::{Tensor, ops::TensorStorageOps};

use crate::core::error::Result;
use crate::{Module, Parameter};

/// 3D Adaptive Max Pooling layer.
///
/// Applies a 3D adaptive max pooling over an input signal composed of several input planes.
///
/// # Shape
/// - Input: `(N, C, D_in, H_in, W_in)`
/// - Output: `(N, C, D_out, H_out, W_out)` where `(D_out, H_out, W_out)` is specified by `output_size`
#[derive(Debug, Clone)]
pub struct AdaptiveMaxPool3d {
    /// Output size (depth, height, width)
    pub output_size: (usize, usize, usize),
}

impl AdaptiveMaxPool3d {
    /// Create a new AdaptiveMaxPool3d layer.
    pub fn new(output_size: (usize, usize, usize)) -> Self {
        assert!(
            output_size.0 > 0 && output_size.1 > 0 && output_size.2 > 0,
            "output_size must be > 0"
        );
        Self { output_size }
    }
}

impl<B, S, T> Module<B, S, T> for AdaptiveMaxPool3d
where
    B: Backend<Data = T> + Clone + Default,
    S: storage::Storage<T> + storage::StorageFromVec<T> + storage::StorageToDense<T> + TensorStorageOps<T> + Clone + 'static,
    T: DataType + FloatExt + PartialOrd + Clone,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(
        &self,
        input: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>> {
        let output = crate::functional::ops::pooling::adaptive_max_pool3d(
            input,
            self.output_size,
        )?;
        let dense = output.to_dense_generic()?;
        let storage = S::from_vec(dense.as_slice().to_vec(), dense.shape().dims())?;
        Ok(Tensor::from_storage(storage, input.backend().clone()))
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        Vec::new()
    }

    fn zero_grad(&mut self) {}

    fn train(&mut self, _mode: bool) {}

    fn name(&self) -> &str {
        "AdaptiveMaxPool3d"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}
