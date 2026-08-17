use crate::backend::{WgpuBackend, WgpuScalar};
use crate::kernels;
use coeus_core::Layout;

#[expect(
    clippy::too_many_arguments,
    reason = "ratchet ATLAS-COEUS-LINT-RATCHET-097"
)]
impl<T: WgpuScalar> coeus_ops::UnfoldFoldOps<T> for WgpuBackend {
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
        kernels::dispatch_unfold1d::<T>(
            input.buffer.as_ref(),
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            output.buffer.as_ref(),
            output_layout,
        )
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
        kernels::dispatch_fold1d::<T>(
            input.buffer.as_ref(),
            input_layout,
            output_size,
            kernel_size,
            stride,
            padding,
            dilation,
            output.buffer.as_ref(),
            output_layout,
        )
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
        kernels::dispatch_unfold2d::<T>(
            input.buffer.as_ref(),
            input_layout,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            output.buffer.as_ref(),
            output_layout,
        )
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
        kernels::dispatch_fold2d::<T>(
            input.buffer.as_ref(),
            input_layout,
            output_h,
            output_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            output.buffer.as_ref(),
            output_layout,
        )
    }
}
