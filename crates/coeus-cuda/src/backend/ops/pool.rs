use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::get_cuda_context;
use crate::error::CudaBackendError;
use crate::kernels;
use crate::storage::CudaStorage;
use coeus_core::Layout;

impl CudaBackend {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_max_pool1d<T: CudaScalar>(
        &self,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut CudaStorage<T>,
        output_layout: &Layout,
    ) -> Result<(), CudaBackendError> {
        if kernels::dispatch_max_pool1d(
            input,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            output,
            output_layout,
        ) {
            return Ok(());
        }
        Err(CudaBackendError::dispatch_unavailable(
            "max_pool1d",
            "native CUDA kernel compilation or launch failed",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_max_pool1d_backward<T: CudaScalar>(
        &self,
        grad_out: &CudaStorage<T>,
        grad_out_layout: &Layout,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut CudaStorage<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), CudaBackendError> {
        if kernels::dispatch_max_pool1d_backward(
            grad_out,
            grad_out_layout,
            input,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            grad_input,
            grad_input_layout,
        ) {
            return Ok(());
        }
        Err(CudaBackendError::dispatch_unavailable(
            "max_pool1d_backward",
            "native CUDA kernel compilation or launch failed",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_avg_pool1d<T: CudaScalar>(
        &self,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut CudaStorage<T>,
        output_layout: &Layout,
    ) -> Result<(), CudaBackendError> {
        if kernels::dispatch_avg_pool1d(
            input,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            output,
            output_layout,
        ) {
            return Ok(());
        }
        Err(CudaBackendError::dispatch_unavailable(
            "avg_pool1d",
            "native CUDA kernel compilation or launch failed",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_avg_pool1d_backward<T: CudaScalar>(
        &self,
        grad_out: &CudaStorage<T>,
        grad_out_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut CudaStorage<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), CudaBackendError> {
        if kernels::dispatch_avg_pool1d_backward(
            grad_out,
            grad_out_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            grad_input,
            grad_input_layout,
        ) {
            return Ok(());
        }
        Err(CudaBackendError::dispatch_unavailable(
            "avg_pool1d_backward",
            "native CUDA kernel compilation or launch failed",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_max_pool2d<T: CudaScalar>(
        &self,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut CudaStorage<T>,
        output_layout: &Layout,
    ) -> Result<(), CudaBackendError> {
        if get_cuda_context().is_some()
            && kernels::dispatch_max_pool2d::<T>(
                input,
                input_layout,
                kernel_size,
                stride,
                padding,
                dilation,
                output,
                output_layout,
            )
        {
            return Ok(());
        }
        Err(CudaBackendError::dispatch_unavailable(
            "max_pool2d",
            "native CUDA kernel compilation or launch failed",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_max_pool2d_backward<T: CudaScalar>(
        &self,
        grad_out: &CudaStorage<T>,
        grad_out_layout: &Layout,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut CudaStorage<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), CudaBackendError> {
        if get_cuda_context().is_some()
            && kernels::dispatch_max_pool2d_backward::<T>(
                grad_out,
                grad_out_layout,
                input,
                input_layout,
                kernel_size,
                stride,
                padding,
                dilation,
                grad_input,
                grad_input_layout,
            )
        {
            return Ok(());
        }
        Err(CudaBackendError::dispatch_unavailable(
            "max_pool2d_backward",
            "native CUDA kernel compilation or launch failed",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_avg_pool2d<T: CudaScalar>(
        &self,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut CudaStorage<T>,
        output_layout: &Layout,
    ) -> Result<(), CudaBackendError> {
        if get_cuda_context().is_some()
            && kernels::dispatch_avg_pool2d::<T>(
                input,
                input_layout,
                kernel_size,
                stride,
                padding,
                dilation,
                output,
                output_layout,
            )
        {
            return Ok(());
        }
        Err(CudaBackendError::dispatch_unavailable(
            "avg_pool2d",
            "native CUDA kernel compilation or launch failed",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_avg_pool2d_backward<T: CudaScalar>(
        &self,
        grad_out: &CudaStorage<T>,
        grad_out_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut CudaStorage<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), CudaBackendError> {
        if get_cuda_context().is_some()
            && kernels::dispatch_avg_pool2d_backward::<T>(
                grad_out,
                grad_out_layout,
                kernel_size,
                stride,
                padding,
                dilation,
                grad_input,
                grad_input_layout,
            )
        {
            return Ok(());
        }
        Err(CudaBackendError::dispatch_unavailable(
            "avg_pool2d_backward",
            "native CUDA kernel compilation or launch failed",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_max_pool3d<T: CudaScalar>(
        &self,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut CudaStorage<T>,
        output_layout: &Layout,
    ) -> Result<(), CudaBackendError> {
        if get_cuda_context().is_some()
            && kernels::dispatch_max_pool3d::<T>(
                input,
                input_layout,
                kernel_size,
                stride,
                padding,
                dilation,
                output,
                output_layout,
            )
        {
            return Ok(());
        }
        Err(CudaBackendError::dispatch_unavailable(
            "max_pool3d",
            "native CUDA kernel compilation or launch failed",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_max_pool3d_backward<T: CudaScalar>(
        &self,
        grad_out: &CudaStorage<T>,
        grad_out_layout: &Layout,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut CudaStorage<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), CudaBackendError> {
        if get_cuda_context().is_some()
            && kernels::dispatch_max_pool3d_backward::<T>(
                grad_out,
                grad_out_layout,
                input,
                input_layout,
                kernel_size,
                stride,
                padding,
                dilation,
                grad_input,
                grad_input_layout,
            )
        {
            return Ok(());
        }
        Err(CudaBackendError::dispatch_unavailable(
            "max_pool3d_backward",
            "native CUDA kernel compilation or launch failed",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_avg_pool3d<T: CudaScalar>(
        &self,
        input: &CudaStorage<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut CudaStorage<T>,
        output_layout: &Layout,
    ) -> Result<(), CudaBackendError> {
        if get_cuda_context().is_some()
            && kernels::dispatch_avg_pool3d::<T>(
                input,
                input_layout,
                kernel_size,
                stride,
                padding,
                dilation,
                output,
                output_layout,
            )
        {
            return Ok(());
        }
        Err(CudaBackendError::dispatch_unavailable(
            "avg_pool3d",
            "native CUDA kernel compilation or launch failed",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_avg_pool3d_backward<T: CudaScalar>(
        &self,
        grad_out: &CudaStorage<T>,
        grad_out_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut CudaStorage<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), CudaBackendError> {
        if get_cuda_context().is_some()
            && kernels::dispatch_avg_pool3d_backward::<T>(
                grad_out,
                grad_out_layout,
                kernel_size,
                stride,
                padding,
                dilation,
                grad_input,
                grad_input_layout,
            )
        {
            return Ok(());
        }
        Err(CudaBackendError::dispatch_unavailable(
            "avg_pool3d_backward",
            "native CUDA kernel compilation or launch failed",
        ))
    }
}
