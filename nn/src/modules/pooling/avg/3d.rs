use backend::Backend;
use dtype::{DataType};
use tensor::Tensor;

use crate::core::error::Result;
use crate::core::module::Module;
use crate::core::parameter::Parameter;

/// 3D Average Pooling layer.
///
/// Applies a 3D average pooling over an input signal composed of several input planes.
/// Downsamples the input by taking the average value in each pooling window.
/// Essential for video processing and 3D medical imaging.
///
/// # Shape
/// - Input: `(N, C, D_in, H_in, W_in)` where N is batch size, C is channels
/// - Output: `(N, C, D_out, H_out, W_out)` where:
///   - D_out = floor((D_in + 2*padding[0] - kernel_size[0]) / stride[0] + 1)
///   - H_out = floor((H_in + 2*padding[1] - kernel_size[1]) / stride[1] + 1)
///   - W_out = floor((W_in + 2*padding[2] - kernel_size[2]) / stride[2] + 1)
///
/// # Examples
/// ```rust
/// use nn::{AvgPool3d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// // Create AvgPool3d with 2x2x2 kernel, stride 2
/// let pool = AvgPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));
///
/// // Input: [batch_size=1, channels=64, depth=16, height=32, width=32]
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 16, 32, 32]).unwrap();
///
/// // Output: [1, 64, 8, 16, 16] (downsampled by 2x in all dimensions)
/// let output = <AvgPool3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 64, 8, 16, 16]);
/// ```
#[derive(Debug, Clone)]
pub struct AvgPool3d {
    /// Kernel size (depth, height, width)
    pub kernel_size: (usize, usize, usize),
    /// Stride (depth, height, width). If None, defaults to kernel_size
    pub stride: Option<(usize, usize, usize)>,
    /// Padding (depth, height, width)
    pub padding: (usize, usize, usize),
    /// Whether to include padding in average calculation
    pub count_include_pad: bool,
    /// Ceil mode
    pub ceil_mode: bool,
}

impl AvgPool3d {
    /// Create a new AvgPool3d layer.
    pub fn new(
        kernel_size: (usize, usize, usize),
        stride: Option<(usize, usize, usize)>,
        padding: (usize, usize, usize),
        count_include_pad: bool,
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
            count_include_pad,
            ceil_mode
        }
    }
}

impl<B, S, T> Module<B, S, T> for AvgPool3d
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
        tensor::ops::pooling::avg_pool::avg_pool3d(
            input,
            self.kernel_size,
            stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad
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
        "AvgPool3d"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}
