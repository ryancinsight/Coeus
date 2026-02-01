use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{Tensor, tensor_backend_dispatch::TensorBackendDispatcher, ops::TensorStorageOps};

use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;

/// 1D Max Pooling layer.
///
/// Applies 1D max pooling over an input signal composed of several input planes.
/// Downsamples the input by taking the maximum value in each pooling window.
///
/// # Shape
/// - Input: `(N, C, L_in)` where N is batch size, C is channels, L_in is input length
/// - Output: `(N, C, L_out)` where L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
///
/// # Examples
/// ```rust
/// use nn::{MaxPool1d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let pool = MaxPool1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(2, Some(2), 0, None, false);
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 100]).unwrap();
/// let output = pool.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 64, 50]);
/// ```
#[derive(Debug, Clone)]
pub struct MaxPool1d<B, S, T> {
    /// Kernel size
    pub kernel_size: usize,
    /// Stride. If None, defaults to kernel_size
    pub stride: Option<usize>,
    /// Padding
    pub padding: usize,
    /// Dilation
    pub dilation: usize,
    /// Ceil mode
    pub ceil_mode: bool,
    
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> MaxPool1d<B, S, T> {
    /// Create a new MaxPool1d layer.
    pub fn new(
        kernel_size: usize,
        stride: Option<usize>,
        padding: usize,
        dilation: Option<usize>,
        ceil_mode: bool
    ) -> Self {
        assert!(kernel_size > 0, "kernel_size must be > 0");
        Self {
            kernel_size,
            stride,
            padding,
            dilation: dilation.unwrap_or(1),
            ceil_mode,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B, S, T> Module<B, S, T> for MaxPool1d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + PartialOrd + num_traits::Float + num_traits::FromPrimitive + std::fmt::Debug + 'static,
{
    fn forward(
        &self,
        input: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>> {
        let input_shape = input.shape().dims();

        if input_shape.len() != 3usize {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Expected 3D input (batch, channels, length), got {}D",
                    input_shape.len()
                ),
            });
        }

        let stride = self.stride.unwrap_or(self.kernel_size);
        
        tensor::ops::pooling::max_pool::max_pool1d(
            input,
            self.kernel_size,
            stride,
            self.padding,
            self.dilation,
            self.ceil_mode
        )
        .map_err(Into::into)
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
        "MaxPool1d"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}
