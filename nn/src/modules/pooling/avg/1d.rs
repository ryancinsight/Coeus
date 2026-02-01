use backend::Backend;
use dtype::{DataType};
use tensor::Tensor;

use crate::core::error::{Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;

/// 1D Average Pooling layer.
///
/// Applies 1D average pooling over an input signal.
///
/// # Shape
/// - Input: `(N, C, L_in)`
/// - Output: `(N, C, L_out)` where L_out = floor((L_in + 2*padding - kernel_size) / stride + 1)
///
/// # Examples
/// ```rust
/// use nn::{AvgPool1d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let pool = AvgPool1d::new(2, Some(2), 0);
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 100]).unwrap();
/// let output = pool.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 64, 50]);
/// ```
#[derive(Debug, Clone)]
pub struct AvgPool1d {
    /// Kernel size
    pub kernel_size: usize,
    /// Stride. If None, defaults to kernel_size
    pub stride: Option<usize>,
    /// Padding
    pub padding: usize,
    /// Whether to include padding in average calculation
    pub count_include_pad: bool,
    /// Ciel mode
    pub ceil_mode: bool,
}

impl AvgPool1d {
    /// Create a new AvgPool1d layer.
    pub fn new(
        kernel_size: usize,
        stride: Option<usize>,
        padding: usize,
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> Self {
        assert!(kernel_size > 0, "kernel_size must be > 0");
        Self {
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode
        }
    }
}

impl<B, S, T> Module<B, S, T> for AvgPool1d
where
    B: Backend<Data = T> + Clone + Default + tensor::tensor_backend_dispatch::TensorBackendDispatcher<B, S, T>,
    S:  storage::Storage<T> + storage::StorageFromVec<T> + Clone + tensor::ops::TensorStorageOps<T> + 'static + storage::StorageToDense<T>,
    T: DataType + dtype::traits::FloatExt + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
{
    fn forward(
        &self,
        input: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>> {
         let stride = self.stride.unwrap_or(self.kernel_size);
         tensor::ops::pooling::avg_pool::avg_pool1d(
             input,
             self.kernel_size,
             stride,
             self.padding,
             self.ceil_mode,
             self.count_include_pad
         ).map_err(Into::into)
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
        "AvgPool1d"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}
