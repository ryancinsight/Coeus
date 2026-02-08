use crate::{
    ops::dispatch::TensorStorageOps, Backend, DataType, Result, Storage,
    StorageFromVec, Tensor,
};

pub trait NeuralDispatcher<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T>,
    T: DataType,
{
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

impl<B, S, T> NeuralDispatcher<B, S, T> for B
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + TensorStorageOps<T> + 'static,
    T: DataType + Clone + Copy + num_traits::Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
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
