use super::super::unfold_fold;
use super::super::CpuBackend;
use crate::backend_ops::traits::UnfoldFoldOps;
use coeus_core::{CpuAddressableStorageMut, Layout, Scalar, Storage};

#[allow(clippy::too_many_arguments)]
impl<T: Scalar, B: CpuBackend> UnfoldFoldOps<T> for B
where
    B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    #[inline]
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
        unfold_fold::validation::unfold1d(
            input_layout,
            input.len(),
            kernel_size,
            stride,
            padding,
            dilation,
            output_layout,
            output.len(),
        )?;
        unfold_fold::unfold1d(
            self,
            input,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            output,
            output_layout,
        );
        Ok(())
    }

    #[inline]
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
        unfold_fold::validation::fold1d(
            input_layout,
            input.len(),
            output_size,
            kernel_size,
            stride,
            padding,
            dilation,
            output_layout,
            output.len(),
        )?;
        unfold_fold::fold1d(
            self,
            input,
            input_layout,
            output_size,
            kernel_size,
            stride,
            padding,
            dilation,
            output,
            output_layout,
        );
        Ok(())
    }

    #[inline]
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
        unfold_fold::validation::unfold2d(
            input_layout,
            input.len(),
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            output_layout,
            output.len(),
        )?;
        unfold_fold::unfold2d(
            self,
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
        );
        Ok(())
    }

    #[inline]
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
        unfold_fold::validation::fold2d(
            input_layout,
            input.len(),
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
            output_layout,
            output.len(),
        )?;
        unfold_fold::fold2d(
            self,
            input,
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
            output,
            output_layout,
        );
        Ok(())
    }
}
