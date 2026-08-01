use crate::backend::{CudaBackend, CudaScalar};
use coeus_core::Layout;

#[allow(clippy::too_many_arguments)]
impl<T: CudaScalar + hephaestus_cuda::DialectScalar<hephaestus_cuda::CudaC>>
    coeus_ops::UnfoldFoldOps<T> for CudaBackend
{
    fn unfold1d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        if crate::kernels::dispatch_unfold1d(
            input,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            output,
            output_layout,
        ) {
            Ok(())
        } else {
            Err(crate::CudaBackendError::kernel(
                "unfold1d",
                "native CUDA kernel rejected the launch contract",
            ))
        }
    }

    fn fold1d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        output_size: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        if output_layout.shape().get(2).copied() != Some(output_size) {
            return Err(crate::CudaBackendError::kernel(
                "fold1d",
                "output_size does not match the output layout",
            ));
        }
        if crate::kernels::dispatch_fold1d(
            input,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            output,
            output_layout,
        ) {
            Ok(())
        } else {
            Err(crate::CudaBackendError::kernel(
                "fold1d",
                "native CUDA kernel rejected the launch contract",
            ))
        }
    }

    fn unfold2d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        if crate::kernels::dispatch_unfold2d(
            input,
            input_layout,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            output,
            output_layout,
        ) {
            Ok(())
        } else {
            Err(crate::CudaBackendError::kernel(
                "unfold2d",
                "native CUDA kernel rejected the launch contract",
            ))
        }
    }

    fn fold2d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        output_h: usize,
        output_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        if output_layout.shape().get(2).copied() != Some(output_h)
            || output_layout.shape().get(3).copied() != Some(output_w)
        {
            return Err(crate::CudaBackendError::kernel(
                "fold2d",
                "output dimensions do not match the output layout",
            ));
        }
        if crate::kernels::dispatch_fold2d(
            input,
            input_layout,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            output,
            output_layout,
        ) {
            Ok(())
        } else {
            Err(crate::CudaBackendError::kernel(
                "fold2d",
                "native CUDA kernel rejected the launch contract",
            ))
        }
    }
}
