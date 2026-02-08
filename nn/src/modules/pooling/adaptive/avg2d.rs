use backend::{Backend, CpuBackend};
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
use tensor::{Tensor, ops::TensorStorageOps};

use crate::core::error::{NNError, Result};
use crate::{Module, Parameter};

/// Adaptive Average Pooling 2D layer.
///
/// Applies adaptive 2D average pooling over an input signal composed of several input planes.
/// The output size is specified, and the layer automatically computes the kernel size and stride
/// to produce the desired output dimensions.
///
/// This is essential for modern CNN architectures like ResNet and EfficientNet that need
/// resolution-agnostic feature extraction.
///
/// # Shape
/// - Input: `(N, C, H_in, W_in)` or `(C, H_in, W_in)`
/// - Output: `(N, C, H_out, W_out)` or `(C, H_out, W_out)` where `H_out` and `W_out` are specified by `output_size`
///
/// # Examples
/// ```rust
/// use nn::{AdaptiveAvgPool2d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let pool = AdaptiveAvgPool2d::new((1, 1)); // Global average pooling
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 3, 224, 224]).unwrap();
/// let output = pool.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 3, 1, 1]);
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveAvgPool2d {
    /// Output size (height, width)
    pub output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    /// Create a new AdaptiveAvgPool2d layer.
    ///
    /// # Arguments
    /// * `output_size` - The target output size (height, width)
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }

    /// Compute adaptive pooling parameters for a given dimension.
    ///
    /// Returns (start_index, end_index) for the pooling window.
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

impl<B, S, T> Module<B, S, T> for AdaptiveAvgPool2d
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
        let output = crate::functional::ops::pooling::adaptive_avg_pool2d(
            input,
            self.output_size,
        )?;
        let dense = output.to_dense_generic()?;
        let storage = S::from_vec(dense.as_slice().to_vec(), dense.shape().dims())?;
        Ok(Tensor::from_storage(storage, input.backend().clone()))
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        Vec::new() // No learnable parameters
    }

    fn zero_grad(&mut self) {
        // No-op: no parameters
    }

    fn train(&mut self, _mode: bool) {
        // No-op: behavior doesn't change
    }

    fn name(&self) -> &str {
        "AdaptiveAvgPool2d"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}
