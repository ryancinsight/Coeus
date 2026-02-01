use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{Tensor, tensor_backend_dispatch::TensorBackendDispatcher, ops::TensorStorageOps};

use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;

/// 3D Max Pooling layer.
///
/// Applies a 3D max pooling over an input signal composed of several input planes.
/// Downsamples the input by taking the maximum value in each pooling window.
/// Essential for video processing and 3D medical imaging.
///
/// # Shape
/// - Input: `(N, C, D_in, H_in, W_in)` where N is batch size, C is channels
/// - Output: `(N, C, D_out, H_out, W_out)` where:
///   - D_out = floor((D_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
///   - H_out = floor((H_in + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1) / stride[1] + 1)
///   - W_out = floor((W_in + 2*padding[2] - dilation[2]*(kernel_size[2]-1) - 1) / stride[2] + 1)
///
/// # Examples
/// ```rust
/// use nn::{MaxPool3d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// // Create MaxPool3d with 2x2x2 kernel, stride 2
/// let pool = MaxPool3d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0), None, false);
///
/// // Input: [batch_size=1, channels=64, depth=16, height=32, width=32]
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 16, 32, 32]).unwrap();
///
/// // Output: [1, 64, 8, 16, 16] (downsampled by 2x in all dimensions)
/// let output = pool.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 64, 8, 16, 16]);
/// ```
#[derive(Debug, Clone)]
pub struct MaxPool3d<B, S, T> {
    /// Kernel size (depth, height, width)
    pub kernel_size: (usize, usize, usize),
    /// Stride (depth, height, width). If None, defaults to kernel_size
    pub stride: Option<(usize, usize, usize)>,
    /// Padding (depth, height, width)
    pub padding: (usize, usize, usize),
    /// Dilation (depth, height, width)
    pub dilation: (usize, usize, usize),
    /// Ceil mode
    pub ceil_mode: bool,

    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> MaxPool3d<B, S, T> {
    /// Create a new MaxPool3d layer.
    pub fn new(
        kernel_size: (usize, usize, usize),
        stride: Option<(usize, usize, usize)>,
        padding: (usize, usize, usize),
        dilation: Option<(usize, usize, usize)>,
        ceil_mode: bool,
    ) -> Self {
        assert!(
            kernel_size.0 > 0 && kernel_size.1 > 0 && kernel_size.2 > 0,
            "kernel_size must be > 0"
        );
        if let Some(s) = stride {
            assert!(s.0 > 0 && s.1 > 0 && s.2 > 0, "stride must be > 0");
        }

        Self {
            kernel_size,
            stride,
            padding,
            dilation: dilation.unwrap_or((1, 1, 1)),
            ceil_mode,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B, S, T> Module<B, S, T> for MaxPool3d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + PartialOrd + num_traits::Float + num_traits::FromPrimitive + std::fmt::Debug + 'static,
{
    fn forward(
        &self,
        input: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>> {
        // Input: [N, C, D_in, H_in, W_in]
        let input_shape = input.shape().dims();
        if input_shape.len() != 5 {
             return Err(NNError::InvalidInput {
                message: format!("Expected 5D input (N, C, D, H, W), got {}D", input_shape.len()),
            });
        }

        let stride = self.stride.unwrap_or(self.kernel_size);
        
        tensor::ops::pooling::max_pool::max_pool3d(
            input,
            self.kernel_size,
            stride,
            self.padding,
            self.dilation,
            self.ceil_mode
        ).map_err(Into::into)
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
        "MaxPool3d"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}
