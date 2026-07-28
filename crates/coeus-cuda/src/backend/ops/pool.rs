use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::get_cuda_context;
use crate::kernels;
use crate::storage::CudaStorage;
use coeus_core::Layout;

fn dispatch_native_pool(
    operation: &'static str,
    dispatch: impl FnOnce() -> bool,
) -> Result<(), crate::CudaBackendError> {
    let _context = get_cuda_context()
        .ok_or_else(|| crate::CudaBackendError::kernel(operation, "CUDA context is unavailable"))?;
    if dispatch() {
        Ok(())
    } else {
        Err(crate::CudaBackendError::kernel(
            operation,
            "native CUDA kernel rejected the launch contract",
        ))
    }
}

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
    ) -> Result<(), crate::CudaBackendError> {
        kernels::dispatch_max_pool1d(
            input,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            output,
            output_layout,
        )
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
    ) -> Result<(), crate::CudaBackendError> {
        kernels::dispatch_max_pool1d_backward(
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
    ) -> Result<(), crate::CudaBackendError> {
        kernels::dispatch_avg_pool1d(
            input,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            output,
            output_layout,
        )
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
    ) -> Result<(), crate::CudaBackendError> {
        kernels::dispatch_avg_pool1d_backward(
            grad_out,
            grad_out_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            grad_input,
            grad_input_layout,
        )
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
    ) -> Result<(), crate::CudaBackendError> {
        dispatch_native_pool("max pool2d", || {
            kernels::dispatch_max_pool2d::<T>(
                input,
                input_layout,
                kernel_size,
                stride,
                padding,
                dilation,
                output,
                output_layout,
            )
        })
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
    ) -> Result<(), crate::CudaBackendError> {
        dispatch_native_pool("max pool2d backward", || {
            kernels::dispatch_max_pool2d_backward::<T>(
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
        })
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
    ) -> Result<(), crate::CudaBackendError> {
        dispatch_native_pool("average pool2d", || {
            kernels::dispatch_avg_pool2d::<T>(
                input,
                input_layout,
                kernel_size,
                stride,
                padding,
                dilation,
                output,
                output_layout,
            )
        })
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
    ) -> Result<(), crate::CudaBackendError> {
        dispatch_native_pool("average pool2d backward", || {
            kernels::dispatch_avg_pool2d_backward::<T>(
                grad_out,
                grad_out_layout,
                kernel_size,
                stride,
                padding,
                dilation,
                grad_input,
                grad_input_layout,
            )
        })
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
    ) -> Result<(), crate::CudaBackendError> {
        dispatch_native_pool("max pool3d", || {
            kernels::dispatch_max_pool3d::<T>(
                input,
                input_layout,
                kernel_size,
                stride,
                padding,
                dilation,
                output,
                output_layout,
            )
        })
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
    ) -> Result<(), crate::CudaBackendError> {
        dispatch_native_pool("max pool3d backward", || {
            kernels::dispatch_max_pool3d_backward::<T>(
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
        })
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
    ) -> Result<(), crate::CudaBackendError> {
        dispatch_native_pool("average pool3d", || {
            kernels::dispatch_avg_pool3d::<T>(
                input,
                input_layout,
                kernel_size,
                stride,
                padding,
                dilation,
                output,
                output_layout,
            )
        })
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
    ) -> Result<(), crate::CudaBackendError> {
        dispatch_native_pool("average pool3d backward", || {
            kernels::dispatch_avg_pool3d_backward::<T>(
                grad_out,
                grad_out_layout,
                kernel_size,
                stride,
                padding,
                dilation,
                grad_input,
                grad_input_layout,
            )
        })
    }
}
