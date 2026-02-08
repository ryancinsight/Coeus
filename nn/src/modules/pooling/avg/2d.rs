use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{Tensor, ops::TensorStorageOps};

use crate::core::error::Result;
use crate::core::module::Module;
use crate::core::parameter::Parameter;

/// 2D Average Pooling layer.
///
/// Applies a 2D average pooling over an input signal composed of several input planes.
/// Downsamples the input by taking the average value in each pooling window.
///
/// Input shape: [N, C, H_in, W_in]
/// Output shape: [N, C, H_out, W_out]
///
/// where:
/// - H_out = floor((H_in + 2*padding[0] - kernel_size[0]) / stride[0] + 1)
/// - W_out = floor((W_in + 2*padding[1] - kernel_size[1]) / stride[1] + 1)
///
/// # Examples
/// ```rust
/// use nn::{AvgPool2d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// // Create AvgPool2d with 2x2 kernel, stride 2
/// let pool = AvgPool2d::new((2, 2), Some((2, 2)), (0, 0));
///
/// // Input: [batch_size=2, channels=64, height=32, width=32]
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 64, 32, 32]).unwrap();
///
/// // Output: [2, 64, 16, 16] (downsampled by 2x)
/// let output = <AvgPool2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(&pool, &input).unwrap();
/// assert_eq!(output.shape().dims(), &[2, 64, 16, 16]);
/// ```
#[derive(Debug, Clone)]
pub struct AvgPool2d {
    /// Kernel size (height, width)
    pub kernel_size: (usize, usize),
    /// Stride (height, width). If None, defaults to kernel_size
    pub stride: Option<(usize, usize)>,
    /// Padding (height, width)
    pub padding: (usize, usize),
    /// Whether to include padding in average calculation
    pub count_include_pad: bool,
    /// Ceil mode
    pub ceil_mode: bool,
}

impl AvgPool2d {
    /// Create a new AvgPool2d layer.
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
        count_include_pad: bool,
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
            count_include_pad,
            ceil_mode,
        }
    }
}

impl<B, S, T> Module<B, S, T> for AvgPool2d
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
        let output = crate::functional::ops::pooling::avg_pool2d(
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
        "AvgPool2d"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}
