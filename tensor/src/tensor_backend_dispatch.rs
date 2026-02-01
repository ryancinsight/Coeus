//! Backend dispatch system for tensor operations.
//!
//! This module implements efficient backend dispatching using associated types pattern
//! for compile-time resolution of tensor operations across different backends.
//!
//! Uses the Backend trait's Clone bounds established in sprint MS-43.

use crate::{
    ops::dispatch::TensorStorageOps, Backend, DataType, DenseStorage, Result, Storage,
    StorageFromVec, StorageToDense, Tensor,
};

/// Backend operations dispatcher using associated types pattern.
///
/// This trait enables compile-time dispatch of operations to specific backends
/// based on the associated types defined in the Backend trait.
pub trait TensorBackendDispatcher<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T>,
    T: DataType,
{
    /// Dispatch tensor addition operation
    fn dispatch_add(&self, lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>;

    /// Dispatch tensor multiplication operation
    fn dispatch_mul(&self, lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>;

    /// Dispatch matrix multiplication between tensors
    fn dispatch_matmul(
        &self,
        lhs: &Tensor<B, S, T>,
        rhs: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>>;

    /// Dispatch ReLU activation
    fn dispatch_relu(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
    where
        T: PartialOrd + Default;

    /// Dispatch sum reduction
    fn dispatch_sum(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, DenseStorage<T>, T>>;

    /// Dispatch cross-backend tensor transfer
    fn dispatch_to_backend<NewB>(
        &self,
        tensor: &Tensor<B, S, T>,
        target_backend: NewB,
    ) -> Result<Tensor<NewB, DenseStorage<T>, T>>
    where
        NewB: Backend<Data = T> + Clone + Send + Sync,
        S: StorageToDense<T>;

    /// Dispatch 1D Convolution
    fn dispatch_conv1d(
        &self,
        input: &Tensor<B, S, T>,
        weight: &Tensor<B, S, T>,
        bias: Option<&Tensor<B, S, T>>,
        stride: usize,
        padding: usize,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero;

    /// Dispatch 1D Max Pooling
    fn dispatch_max_pool1d(
        &self,
        input: &Tensor<B, S, T>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug;

    /// Dispatch 2D Max Pooling
    fn dispatch_max_pool2d(
        &self,
        input: &Tensor<B, S, T>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        ceil_mode: bool,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug;

    /// Dispatch 3D Max Pooling
    fn dispatch_max_pool3d(
        &self,
        input: &Tensor<B, S, T>,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
        ceil_mode: bool,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug;

    /// Dispatch 1D Average Pooling
    fn dispatch_avg_pool1d(
        &self,
        input: &Tensor<B, S, T>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug;

    /// Dispatch 2D Average Pooling
    fn dispatch_avg_pool2d(
        &self,
        input: &Tensor<B, S, T>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug;

    /// Dispatch 3D Average Pooling
    fn dispatch_avg_pool3d(
        &self,
        input: &Tensor<B, S, T>,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug;

    /// Dispatch 1D Transposed Convolution
    fn dispatch_conv_transpose1d(
        &self,
        input: &Tensor<B, S, T>,
        weight: &Tensor<B, S, T>,
        bias: Option<&Tensor<B, S, T>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
        dilation: usize,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero;

    /// Dispatch 2D Convolution
    fn dispatch_conv2d(
        &self,
        input: &Tensor<B, S, T>,
        weight: &Tensor<B, S, T>,
        bias: Option<&Tensor<B, S, T>>,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero;

    /// Dispatch 2D Transposed Convolution
    fn dispatch_conv_transpose2d(
        &self,
        input: &Tensor<B, S, T>,
        weight: &Tensor<B, S, T>,
        bias: Option<&Tensor<B, S, T>>,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        output_padding_h: usize,
        output_padding_w: usize,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero;

    /// Dispatch 3D Convolution
    fn dispatch_conv3d(
        &self,
        input: &Tensor<B, S, T>,
        weight: &Tensor<B, S, T>,
        bias: Option<&Tensor<B, S, T>>,
        stride_d: usize,
        stride_h: usize,
        stride_w: usize,
        padding_d: usize,
        padding_h: usize,
        padding_w: usize,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero;

    /// Dispatch 3D Transposed Convolution
    fn dispatch_conv_transpose3d(
        &self,
        input: &Tensor<B, S, T>,
        weight: &Tensor<B, S, T>,
        bias: Option<&Tensor<B, S, T>>,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        output_padding: (usize, usize, usize),
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero;


}

/// Default implementation for any backend that implements the required operations
impl<B, S, T> TensorBackendDispatcher<B, S, T> for B
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
    T: DataType
        + Clone
        + Copy
        + num_traits::Zero
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>,
{
    fn dispatch_add(
        &self,
        lhs: &Tensor<B, S, T>,
        rhs: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>> {
        let result_storage = lhs.storage.storage_add(&rhs.storage, self)?;
        Ok(Tensor::from_storage(result_storage, self.clone()))
    }

    fn dispatch_mul(
        &self,
        lhs: &Tensor<B, S, T>,
        rhs: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>> {
        let result_storage = lhs.storage.storage_mul(&rhs.storage, self)?;
        Ok(Tensor::from_storage(result_storage, self.clone()))
    }

    fn dispatch_matmul(
        &self,
        lhs: &Tensor<B, S, T>,
        rhs: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>> {
        let result_storage = lhs.storage.storage_matmul(&rhs.storage, self)?;
        Ok(Tensor::from_storage(result_storage, self.clone()))
    }

    fn dispatch_relu(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
    where
        T: PartialOrd + Default,
    {
        let result_storage = input.storage.storage_relu(self)?;
        Ok(Tensor::from_storage(result_storage, self.clone()))
    }

    fn dispatch_sum(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, DenseStorage<T>, T>> {
        let sum_value = input.storage.storage_sum(self)?;
        let scalar_data = vec![sum_value];
        let scalar_storage =
            DenseStorage::from_vec(scalar_data, &[1]).map_err(crate::TensorError::StorageError)?;
        Ok(Tensor::from_storage(scalar_storage, self.clone()))
    }

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

    fn dispatch_conv1d(
        &self,
        input: &Tensor<B, S, T>,
        weight: &Tensor<B, S, T>,
        bias: Option<&Tensor<B, S, T>>,
        stride: usize,
        padding: usize,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
    {
        let input_dense = input.to_dense_generic()?;
        let input_cpu = input_dense.to_cpu_dense()?;
        let weight_dense = weight.to_dense_generic()?;
        let weight_cpu = weight_dense.to_cpu_dense()?;
        let bias_cpu = match bias {
            Some(b) => {
                let b_dense = b.to_dense_generic()?;
                Some(b_dense.to_cpu_dense()?)
            }
            None => None,
        };

        let out_cpu = crate::ops::conv::kernels::conv1d_cpu_dense(
            &input_cpu,
            &weight_cpu,
            bias_cpu.as_ref(),
            stride,
            padding,
        )?;

        let shape = out_cpu.shape().dims();
        let data = out_cpu.as_slice().to_vec();
        Tensor::from_vec_with_backend(data, shape, self.clone())
    }

    fn dispatch_conv_transpose1d(
        &self,
        input: &Tensor<B, S, T>,
        weight: &Tensor<B, S, T>,
        bias: Option<&Tensor<B, S, T>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
        dilation: usize,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
    {
        let input_dense = input.to_dense_generic()?;
        let input_cpu = input_dense.to_cpu_dense()?;
        let weight_dense = weight.to_dense_generic()?;
        let weight_cpu = weight_dense.to_cpu_dense()?;
        let bias_cpu = match bias {
            Some(b) => {
                let b_dense = b.to_dense_generic()?;
                Some(b_dense.to_cpu_dense()?)
            }
            None => None,
        };

        let out_cpu = crate::ops::conv::kernels::conv_transpose_1d_cpu_dense(
            &input_cpu,
            &weight_cpu,
            bias_cpu.as_ref(),
            stride,
            padding,
            output_padding,
            groups,
            dilation,
        )?;

        let shape = out_cpu.shape().dims();
        let data = out_cpu.as_slice().to_vec();
        Tensor::from_vec_with_backend(data, shape, self.clone())
    }

    fn dispatch_conv2d(
        &self,
        input: &Tensor<B, S, T>,
        weight: &Tensor<B, S, T>,
        bias: Option<&Tensor<B, S, T>>,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
    {
        let input_dense = input.to_dense_generic()?;
        let input_cpu = input_dense.to_cpu_dense()?;
        let weight_dense = weight.to_dense_generic()?;
        let weight_cpu = weight_dense.to_cpu_dense()?;
        let bias_cpu = match bias {
            Some(b) => {
                let b_dense = b.to_dense_generic()?;
                Some(b_dense.to_cpu_dense()?)
            }
            None => None,
        };

        let out_cpu = crate::ops::conv::kernels::conv2d_cpu_dense(
            &input_cpu,
            &weight_cpu,
            bias_cpu.as_ref(),
            stride_h,
            stride_w,
            padding_h,
            padding_w,
        )?;

        let shape = out_cpu.shape().dims();
        let data = out_cpu.as_slice().to_vec();
        Tensor::from_vec_with_backend(data, shape, self.clone())
    }

    fn dispatch_conv_transpose2d(
        &self,
        input: &Tensor<B, S, T>,
        weight: &Tensor<B, S, T>,
        bias: Option<&Tensor<B, S, T>>,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        output_padding_h: usize,
        output_padding_w: usize,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
    {
        let input_dense = input.to_dense_generic()?;
        let input_cpu = input_dense.to_cpu_dense()?;
        let weight_dense = weight.to_dense_generic()?;
        let weight_cpu = weight_dense.to_cpu_dense()?;
        let bias_cpu = match bias {
            Some(b) => {
                let b_dense = b.to_dense_generic()?;
                Some(b_dense.to_cpu_dense()?)
            }
            None => None,
        };

        let out_cpu = crate::ops::conv::kernels::conv_transpose_2d_cpu_dense(
            &input_cpu,
            &weight_cpu,
            bias_cpu.as_ref(),
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            output_padding_h,
            output_padding_w,
        )?;

        let shape = out_cpu.shape().dims();
        let data = out_cpu.as_slice().to_vec();
        Tensor::from_vec_with_backend(data, shape, self.clone())
    }

    fn dispatch_conv3d(
        &self,
        input: &Tensor<B, S, T>,
        weight: &Tensor<B, S, T>,
        bias: Option<&Tensor<B, S, T>>,
        stride_d: usize,
        stride_h: usize,
        stride_w: usize,
        padding_d: usize,
        padding_h: usize,
        padding_w: usize,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
    {
        let input_dense = input.to_dense_generic()?;
        let input_cpu = input_dense.to_cpu_dense()?;
        let weight_dense = weight.to_dense_generic()?;
        let weight_cpu = weight_dense.to_cpu_dense()?;
        let bias_cpu = match bias {
            Some(b) => {
                let b_dense = b.to_dense_generic()?;
                Some(b_dense.to_cpu_dense()?)
            }
            None => None,
        };

        let out_cpu = crate::ops::conv::kernels::conv3d_cpu_dense(
            &input_cpu,
            &weight_cpu,
            bias_cpu.as_ref(),
            stride_d,
            stride_h,
            stride_w,
            padding_d,
            padding_h,
            padding_w,
        )?;

        let shape = out_cpu.shape().dims();
        let data = out_cpu.as_slice().to_vec();
        Tensor::from_vec_with_backend(data, shape, self.clone())
    }

    fn dispatch_conv_transpose3d(
        &self,
        input: &Tensor<B, S, T>,
        weight: &Tensor<B, S, T>,
        bias: Option<&Tensor<B, S, T>>,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        output_padding: (usize, usize, usize),
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
    {
        let input_cpu = input.to_cpu_dense()?;
        let weight_cpu = weight.to_cpu_dense()?;
        let bias_cpu = match bias {
            Some(b) => Some(b.to_cpu_dense()?),
            None => None,
        };

        let out_cpu = crate::ops::conv::kernels::conv_transpose_3d_cpu_dense(
            &input_cpu,
            &weight_cpu,
            bias_cpu.as_ref(),
            stride,
            padding,
            output_padding,
        )?;

        let shape = out_cpu.shape().dims();
        let data = out_cpu.as_slice().to_vec();
        Tensor::from_vec_with_backend(data, shape, self.clone())
    }

    fn dispatch_max_pool1d(
        &self,
        input: &Tensor<B, S, T>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
    {
        let input_dense = input.to_dense_generic()?;
        let input_cpu = input_dense.to_cpu_dense()?;
        
        let out_cpu = crate::ops::pooling::kernels::max_pool1d_cpu_dense(
            &input_cpu,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode
        )?;

        let shape = out_cpu.shape().dims();
        let data = out_cpu.as_slice().to_vec();
        Tensor::from_vec_with_backend(data, shape, self.clone())
    }

    fn dispatch_max_pool2d(
        &self,
        input: &Tensor<B, S, T>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        ceil_mode: bool,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
    {
        let input_dense = input.to_dense_generic()?;
        let input_cpu = input_dense.to_cpu_dense()?;
        
        let out_cpu = crate::ops::pooling::kernels::max_pool2d_cpu_dense(
            &input_cpu,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode
        )?;

        let shape = out_cpu.shape().dims();
        let data = out_cpu.as_slice().to_vec();
        Tensor::from_vec_with_backend(data, shape, self.clone())
    }

    fn dispatch_max_pool3d(
        &self,
        input: &Tensor<B, S, T>,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
        ceil_mode: bool,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
    {
        let input_dense = input.to_dense_generic()?;
        let input_cpu = input_dense.to_cpu_dense()?;
        
        let out_cpu = crate::ops::pooling::kernels::max_pool3d_cpu_dense(
            &input_cpu,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode
        )?;

        let shape = out_cpu.shape().dims();
        let data = out_cpu.as_slice().to_vec();
        Tensor::from_vec_with_backend(data, shape, self.clone())
    }

    fn dispatch_avg_pool1d(
        &self,
        input: &Tensor<B, S, T>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
    {
        let input_dense = input.to_dense_generic()?;
        let input_cpu = input_dense.to_cpu_dense()?;
        
        let out_cpu = crate::ops::pooling::kernels::avg_pool1d_cpu_dense(
            &input_cpu,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad
        )?;

        let shape = out_cpu.shape().dims();
        let data = out_cpu.as_slice().to_vec();
        Tensor::from_vec_with_backend(data, shape, self.clone())
    }

    fn dispatch_avg_pool2d(
        &self,
        input: &Tensor<B, S, T>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
    {
        let input_dense = input.to_dense_generic()?;
        let input_cpu = input_dense.to_cpu_dense()?;
        
        let out_cpu = crate::ops::pooling::kernels::avg_pool2d_cpu_dense(
            &input_cpu,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad
        )?;

        let shape = out_cpu.shape().dims();
        let data = out_cpu.as_slice().to_vec();
        Tensor::from_vec_with_backend(data, shape, self.clone())
    }

    fn dispatch_avg_pool3d(
        &self,
        input: &Tensor<B, S, T>,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Tensor<B, S, T>>
    where
        T: num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
    {
        let input_dense = input.to_dense_generic()?;
        let input_cpu = input_dense.to_cpu_dense()?;
        
        let out_cpu = crate::ops::pooling::kernels::avg_pool3d_cpu_dense(
            &input_cpu,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad
        )?;

        let shape = out_cpu.shape().dims();
        let data = out_cpu.as_slice().to_vec();
        Tensor::from_vec_with_backend(data, shape, self.clone())
    }
}

/// High-level dispatch interface for tensor operations.
///
/// Provides a clean API for backend-agnostic tensor operations.
/// Uses the associated types pattern for efficient dispatch.
pub struct TensorDispatcher;

impl TensorDispatcher {
    /// Dispatch addition operation between tensors
    pub fn add<B, S, T>(lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType,
    {
        lhs.backend().dispatch_add(lhs, rhs)
    }

    /// Dispatch multiplication operation between tensors
    pub fn mul<B, S, T>(lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType,
    {
        lhs.backend().dispatch_mul(lhs, rhs)
    }

    /// Dispatch matrix multiplication between tensors
    pub fn matmul<B, S, T>(lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType,
    {
        lhs.backend().dispatch_matmul(lhs, rhs)
    }

    /// Dispatch ReLU activation
    pub fn relu<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType
            + Clone
            + Copy
            + num_traits::Zero
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>
            + PartialOrd
            + Default,
    {
        input.backend().dispatch_relu(input)
    }

    /// Dispatch 1D Max Pooling
    pub fn max_pool1d<B, S, T>(
        input: &Tensor<B, S, T>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
    {
        input.backend().dispatch_max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode)
    }

    /// Dispatch 2D Max Pooling
    pub fn max_pool2d<B, S, T>(
        input: &Tensor<B, S, T>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        ceil_mode: bool,
    ) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
    {
        input.backend().dispatch_max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
    }

    /// Dispatch 3D Max Pooling
    pub fn max_pool3d<B, S, T>(
        input: &Tensor<B, S, T>,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
        ceil_mode: bool,
    ) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
    {
        input.backend().dispatch_max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode)
    }

    /// Dispatch sum reduction
    pub fn sum<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType,
    {
        input.backend().dispatch_sum(input)
    }

    /// Dispatch cross-backend tensor transfer
    pub fn to_backend<B, S, T, NewB>(
        tensor: &Tensor<B, S, T>,
        target_backend: NewB,
    ) -> Result<Tensor<NewB, DenseStorage<T>, T>>
    where
        NewB: Backend<Data = T> + Clone + Send + Sync,
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static + StorageToDense<T>,
        T: DataType + Clone,
    {
        tensor.backend().dispatch_to_backend(tensor, target_backend)
    }

    /// Dispatch 1D Convolution
    pub fn conv1d<B, S, T>(
        input: &Tensor<B, S, T>,
        weight: &Tensor<B, S, T>,
        bias: Option<&Tensor<B, S, T>>,
        stride: usize,
        padding: usize,
    ) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
    {
        input
            .backend()
            .dispatch_conv1d(input, weight, bias, stride, padding)
    }

    /// Dispatch 1D Transposed Convolution
    pub fn conv_transpose1d<B, S, T>(
        input: &Tensor<B, S, T>,
        weight: &Tensor<B, S, T>,
        bias: Option<&Tensor<B, S, T>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
        dilation: usize,
    ) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
    {
        input.backend().dispatch_conv_transpose1d(
            input,
            weight,
            bias,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
        )
    }

    /// Dispatch 2D Convolution
    pub fn conv2d<B, S, T>(
        input: &Tensor<B, S, T>,
        weight: &Tensor<B, S, T>,
        bias: Option<&Tensor<B, S, T>>,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
    ) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
    {
        input.backend().dispatch_conv2d(
            input, weight, bias, stride_h, stride_w, padding_h, padding_w,
        )
    }

    /// Dispatch 2D Transposed Convolution
    pub fn conv_transpose2d<B, S, T>(
        input: &Tensor<B, S, T>,
        weight: &Tensor<B, S, T>,
        bias: Option<&Tensor<B, S, T>>,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        output_padding_h: usize,
        output_padding_w: usize,
    ) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
    {
        input.backend().dispatch_conv_transpose2d(
            input,
            weight,
            bias,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            output_padding_h,
            output_padding_w,
        )
    }

    /// Dispatch 1D Average Pooling
    pub fn avg_pool1d<B, S, T>(
        input: &Tensor<B, S, T>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
    {
        input.backend().dispatch_avg_pool1d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
    }

    /// Dispatch 2D Average Pooling
    pub fn avg_pool2d<B, S, T>(
        input: &Tensor<B, S, T>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
    {
        input.backend().dispatch_avg_pool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
    }

    /// Dispatch 3D Average Pooling
    pub fn avg_pool3d<B, S, T>(
        input: &Tensor<B, S, T>,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero + PartialOrd + std::fmt::Debug,
    {
        input.backend().dispatch_avg_pool3d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
    }

    /// Dispatch 3D Convolution
    pub fn conv3d<B, S, T>(
        input: &Tensor<B, S, T>,
        weight: &Tensor<B, S, T>,
        bias: Option<&Tensor<B, S, T>>,
        stride_d: usize,
        stride_h: usize,
        stride_w: usize,
        padding_d: usize,
        padding_h: usize,
        padding_w: usize,
    ) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
    {
        input.backend().dispatch_conv3d(
            input, weight, bias, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w,
        )
    }

    /// Dispatch 3D Transposed Convolution
    pub fn conv_transpose3d<B, S, T>(
        input: &Tensor<B, S, T>,
        weight: &Tensor<B, S, T>,
        bias: Option<&Tensor<B, S, T>>,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        output_padding: (usize, usize, usize),
    ) -> Result<Tensor<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
        T: DataType + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
    {
        input.backend().dispatch_conv_transpose3d(
            input,
            weight,
            bias,
            stride,
            padding,
            output_padding,
        )
    }
}

/// Memory transfer operations for cross-backend tensor sharing.
///
/// Implements distributed tensor sharing via Clone bounds as required for sprint MS-44.
pub struct MemoryTransfer;

impl MemoryTransfer {
    /// Transfer tensor between backends with potential zero-copy operations
    pub fn transfer<B, S, T, NewB>(
        tensor: &Tensor<B, S, T>,
        target_backend: NewB,
    ) -> Result<Tensor<NewB, DenseStorage<T>, T>>
    where
        NewB: Backend<Data = T> + Clone + Send + Sync,
        B: Backend<Data = T> + Clone,
        S: Storage<T> + StorageFromVec<T> + Clone + StorageToDense<T> + crate::ops::TensorStorageOps<T>,
        T: DataType + Clone,
    {
        // Use Clone bounds for efficient transfer
        // Backends can implement zero-copy transfers via Clone trait
        tensor.to_backend(target_backend)
    }

    /// Check if backends support zero-copy transfer between them
    pub fn can_zero_copy_transfer<B, NewB, T>(source_backend: &B, target_backend: &NewB) -> bool
    where
        B: Backend<Data = T>,
        NewB: Backend<Data = T>,
        T: DataType,
    {
        // Check device compatibility (same device type and memory space)
        // This is a placeholder - actual implementation depends on backend capabilities
        source_backend.device_name() == target_backend.device_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuBackend, DenseStorage, Tensor};
    use dtype::float::Float32;

    #[test]
    fn test_dispatcher_add() {
        let lhs_data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let rhs_data = vec![Float32::new(4.0), Float32::new(5.0), Float32::new(6.0)];

        let lhs: Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
            Tensor::from_vec(lhs_data, &[3]).unwrap();
        let rhs: Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
            Tensor::from_vec(rhs_data, &[3]).unwrap();

        let result = TensorDispatcher::add(&lhs, &rhs).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.as_slice()[0].get(), 5.0);
        assert_eq!(result.as_slice()[1].get(), 7.0);
        assert_eq!(result.as_slice()[2].get(), 9.0);
    }

    #[test]
    fn test_backend_supports_operation() {
        let tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[5]).unwrap();
        assert!(tensor.backend_supports("arithmetic"));
    }

    #[test]
    fn test_device_access() {
        let tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[3]).unwrap();
        let device = tensor.device();
        assert_eq!(device.name(), "cpu");
    }
}
