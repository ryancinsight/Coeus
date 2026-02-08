use backend::Backend;
use dtype::{traits, DataType};
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{Tensor, tensor_backend_dispatch::TensorBackendDispatcher, ops::TensorStorageOps};

use crate::core::error::{NNError, Result};
use crate::{Module, Parameter};

/// 2D Max Pooling layer.
///
/// Applies a 2D max pooling over an input signal composed of several input planes.
/// Downsamples the input by taking the maximum value in each pooling window.
///
/// Input shape: [N, C, H_in, W_in]
/// Output shape: [N, C, H_out, W_out]
///
/// where:
/// - H_out = floor((H_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
/// - W_out = floor((W_in + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1) / stride[1] + 1)
///
/// # Examples
/// ```rust
/// use nn::{MaxPool2d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// // Create MaxPool2d with 2x2 kernel, stride 2
/// let pool = MaxPool2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new((2, 2), Some((2, 2)), (0, 0), None, false);
///
/// // Input: [batch_size=2, channels=64, height=32, width=32]
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 64, 32, 32]).unwrap();
///
/// // Output: [2, 64, 16, 16] (downsampled by 2x)
/// let output = pool.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[2, 64, 16, 16]);
/// ```
#[derive(Debug, Clone)]
pub struct MaxPool2d<B, S, T> {
    /// Kernel size (height, width)
    pub kernel_size: (usize, usize),
    /// Stride (height, width). If None, defaults to kernel_size
    pub stride: Option<(usize, usize)>,
    /// Padding (height, width)
    pub padding: (usize, usize),
    /// Dilation (height, width)
    pub dilation: (usize, usize),
    /// Ceil mode
    pub ceil_mode: bool,
    
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> MaxPool2d<B, S, T> {
    /// Create a new MaxPool2d layer.
    ///
    /// # Arguments
    /// * `kernel_size` - Kernel size (height, width)
    /// * `stride` - Stride (height, width). If None, defaults to kernel_size
    /// * `padding` - Padding (height, width)
    /// * `dilation` - Dilation (height, width)
    /// * `ceil_mode` - Ceil mode
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
        dilation: Option<(usize, usize)>,
        ceil_mode: bool,
    ) -> Self {
        assert!(
            kernel_size.0 > 0 && kernel_size.1 > 0,
            "kernel_size must be > 0"
        );
        if let Some(s) = stride {
            assert!(s.0 > 0 && s.1 > 0, "stride must be > 0");
        }

        Self {
            kernel_size,
            stride,
            padding,
            dilation: dilation.unwrap_or((1, 1)),
            ceil_mode,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B, S, T> Module<B, S, T> for MaxPool2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: storage::Storage<T> + storage::StorageFromVec<T> + storage::StorageToDense<T> + TensorStorageOps<T> + Clone + 'static,
    T: DataType + traits::FloatExt + Clone,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(
        &self,
        input: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>> {
        let output = crate::functional::ops::pooling::max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
        )?;
        let dense = output.to_dense_generic()?;
        let storage = S::from_vec(dense.as_slice().to_vec(), dense.shape().dims())?;
        Ok(Tensor::from_storage(storage, input.backend().clone()))
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![] // No learnable parameters
    }

    fn zero_grad(&mut self) {
        // No parameters to zero
    }

    fn train(&mut self, _mode: bool) {
        // No training-specific behavior
    }

    fn name(&self) -> &str {
        "MaxPool2d"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}
