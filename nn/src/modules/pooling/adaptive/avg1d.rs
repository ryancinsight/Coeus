use backend::{Backend, CpuBackend};
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
use tensor::{Tensor, ops::TensorStorageOps};

use crate::core::error::{NNError, Result};
use crate::{Module, Parameter};

/// Adaptive Average Pooling 1D layer.
///
/// Applies adaptive 1D average pooling. The output size is specified,
/// and the layer automatically computes the kernel size and stride.
///
/// # Shape
/// - Input: `(N, C, L_in)`
/// - Output: `(N, C, L_out)` where L_out is specified by output_size
///
/// # Examples
/// ```rust
/// use nn::{AdaptiveAvgPool1d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let pool = AdaptiveAvgPool1d::new(10); // Output length = 10
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 100]).unwrap();
/// let output = pool.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 64, 10]);
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveAvgPool1d {
    /// Output size
    pub output_size: usize,
}

impl AdaptiveAvgPool1d {
    /// Create a new AdaptiveAvgPool1d layer.
    pub fn new(output_size: usize) -> Self {
        Self { output_size }
    }

    /// Compute adaptive pooling parameters.
    pub(crate) fn compute_adaptive_params(
        input_size: usize,
        output_size: usize,
        output_idx: usize,
    ) -> (usize, usize) {
        let start = (output_idx * input_size) / output_size;
        let end = ((output_idx + 1) * input_size) / output_size;
        (start, end)
    }
}

impl<B, S, T> Module<B, S, T> for AdaptiveAvgPool1d
where
    B: Backend<Data = T> + Clone + Default,
    S: storage::Storage<T> + storage::StorageFromVec<T> + storage::StorageToDense<T> + TensorStorageOps<T> + Clone + 'static,
    T: DataType + FloatExt + Clone,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(
        &self,
        input: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>> {
        let output = crate::functional::ops::pooling::adaptive_avg_pool1d(
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

    fn zero_grad(&mut self) {
        // No-op: no parameters
    }

    fn train(&mut self, _mode: bool) {
        // No-op: behavior doesn't change
    }

    fn name(&self) -> &str {
        "AdaptiveAvgPool1d"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}
