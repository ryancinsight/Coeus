//! Backend dispatch system for tensor operations.

use crate::{
    ops::dispatch::TensorStorageOps, Backend, DataType, DenseStorage, Result, Storage,
    StorageFromVec, StorageToDense, Tensor,
};

pub mod arithmetic;
pub mod linalg;
pub mod neural;
pub mod activation;

pub use arithmetic::ArithmeticDispatcher;
pub use linalg::LinalgDispatcher;
pub use neural::NeuralDispatcher;
pub use activation::ActivationDispatcher;

/// Backend operations dispatcher using associated types pattern.
pub trait TensorBackendDispatcher<B, S, T>:
    ArithmeticDispatcher<B, S, T> +
    LinalgDispatcher<B, S, T> +
    NeuralDispatcher<B, S, T> +
    ActivationDispatcher<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T>,
    T: DataType,
{
    /// Dispatch cross-backend tensor transfer
    fn dispatch_to_backend<NewB>(
        &self,
        tensor: &Tensor<B, S, T>,
        target_backend: NewB,
    ) -> Result<Tensor<NewB, DenseStorage<T>, T>>
    where
        NewB: Backend<Data = T> + Clone + Send + Sync,
        S: StorageToDense<T>;
}

/// Default implementation for any backend that implements the required operations
impl<B, S, T> TensorBackendDispatcher<B, S, T> for B
where
    B: Backend<Data = T> + Clone + Default +
       ArithmeticDispatcher<B, S, T> +
       LinalgDispatcher<B, S, T> +
       NeuralDispatcher<B, S, T> +
       ActivationDispatcher<B, S, T>,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
    T: DataType + Clone + Copy + num_traits::Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    fn dispatch_to_backend<NewB>(
        &self,
        tensor: &Tensor<B, S, T>,
        target_backend: NewB,
    ) -> Result<Tensor<NewB, DenseStorage<T>, T>>
    where
        NewB: Backend<Data = T> + Clone + Send + Sync,
        S: StorageToDense<T>,
    {
        tensor.to_backend(target_backend)
    }
}

/// High-level dispatch interface for tensor operations.
pub struct TensorDispatcher;

impl TensorDispatcher {
    // Arithmetic
    pub fn add<B, S, T>(lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + ArithmeticDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType,
    {
        lhs.backend().dispatch_add(lhs, rhs)
    }

    pub fn mul<B, S, T>(lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + ArithmeticDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType,
    {
        lhs.backend().dispatch_mul(lhs, rhs)
    }

    pub fn sum<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        B: Backend<Data = T> + Clone + Default + ArithmeticDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType,
    {
        input.backend().dispatch_sum(input)
    }

    // Linalg
    pub fn matmul<B, S, T>(lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + LinalgDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType,
    {
        lhs.backend().dispatch_matmul(lhs, rhs)
    }

    // Neural
    pub fn conv1d<B, S, T>(input: &Tensor<B, S, T>, weight: &Tensor<B, S, T>, bias: Option<&Tensor<B, S, T>>, stride: usize, padding: usize) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + NeuralDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
    {
        input.backend().dispatch_conv1d(input, weight, bias, stride, padding)
    }

    pub fn conv_transpose1d<B, S, T>(input: &Tensor<B, S, T>, weight: &Tensor<B, S, T>, bias: Option<&Tensor<B, S, T>>, stride: usize, padding: usize, output_padding: usize, groups: usize, dilation: usize) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + NeuralDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
    {
        input.backend().dispatch_conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation)
    }

    pub fn conv2d<B, S, T>(input: &Tensor<B, S, T>, weight: &Tensor<B, S, T>, bias: Option<&Tensor<B, S, T>>, stride_h: usize, stride_w: usize, padding_h: usize, padding_w: usize) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + NeuralDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
    {
        input.backend().dispatch_conv2d(input, weight, bias, stride_h, stride_w, padding_h, padding_w)
    }

    pub fn conv_transpose2d<B, S, T>(input: &Tensor<B, S, T>, weight: &Tensor<B, S, T>, bias: Option<&Tensor<B, S, T>>, stride_h: usize, stride_w: usize, padding_h: usize, padding_w: usize, output_padding_h: usize, output_padding_w: usize) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + NeuralDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
    {
        input.backend().dispatch_conv_transpose2d(input, weight, bias, stride_h, stride_w, padding_h, padding_w, output_padding_h, output_padding_w)
    }

    pub fn conv3d<B, S, T>(input: &Tensor<B, S, T>, weight: &Tensor<B, S, T>, bias: Option<&Tensor<B, S, T>>, stride_d: usize, stride_h: usize, stride_w: usize, padding_d: usize, padding_h: usize, padding_w: usize) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + NeuralDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
    {
        input.backend().dispatch_conv3d(input, weight, bias, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w)
    }

    pub fn conv_transpose3d<B, S, T>(input: &Tensor<B, S, T>, weight: &Tensor<B, S, T>, bias: Option<&Tensor<B, S, T>>, stride: (usize, usize, usize), padding: (usize, usize, usize), output_padding: (usize, usize, usize)) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + NeuralDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
    {
        input.backend().dispatch_conv_transpose3d(input, weight, bias, stride, padding, output_padding)
    }

    pub fn max_pool1d<B, S, T>(input: &Tensor<B, S, T>, kernel_size: usize, stride: usize, padding: usize, dilation: usize, ceil_mode: bool) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + NeuralDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
    {
        input.backend().dispatch_max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode)
    }

    pub fn max_pool2d<B, S, T>(input: &Tensor<B, S, T>, kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize), dilation: (usize, usize), ceil_mode: bool) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + NeuralDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
    {
        input.backend().dispatch_max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
    }

    pub fn max_pool3d<B, S, T>(input: &Tensor<B, S, T>, kernel_size: (usize, usize, usize), stride: (usize, usize, usize), padding: (usize, usize, usize), dilation: (usize, usize, usize), ceil_mode: bool) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + NeuralDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
    {
        input.backend().dispatch_max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode)
    }

    pub fn avg_pool1d<B, S, T>(input: &Tensor<B, S, T>, kernel_size: usize, stride: usize, padding: usize, ceil_mode: bool, count_include_pad: bool) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + NeuralDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
    {
        input.backend().dispatch_avg_pool1d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
    }

    pub fn avg_pool2d<B, S, T>(input: &Tensor<B, S, T>, kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize), ceil_mode: bool, count_include_pad: bool) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + NeuralDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
    {
        input.backend().dispatch_avg_pool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
    }

    pub fn avg_pool3d<B, S, T>(input: &Tensor<B, S, T>, kernel_size: (usize, usize, usize), stride: (usize, usize, usize), padding: (usize, usize, usize), ceil_mode: bool, count_include_pad: bool) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + NeuralDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
    {
        input.backend().dispatch_avg_pool3d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
    }
}
