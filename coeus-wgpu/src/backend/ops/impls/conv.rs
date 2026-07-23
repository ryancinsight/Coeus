use super::super::conv;
use crate::backend::{WgpuBackend, WgpuScalar};
use coeus_core::Layout;

#[allow(clippy::too_many_arguments)]
impl<T: WgpuScalar + leto_ops::Scalar + hephaestus_wgpu::DialectScalar<hephaestus_wgpu::Wgsl>>
    coeus_ops::ConvOps<T> for WgpuBackend
{
    #[inline]
    fn conv1d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        bias: Option<&Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) {
        conv::dispatch_conv1d(
            input,
            input_layout,
            weight,
            weight_layout,
            bias,
            stride,
            padding,
            dilation,
            output,
            output_layout,
        );
    }

    #[inline]
    fn conv1d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        grad_input: Option<&mut Self::DeviceBuffer<T>>,
        grad_input_layout: &Layout,
        grad_weight: Option<&mut Self::DeviceBuffer<T>>,
        grad_weight_layout: &Layout,
        grad_bias: Option<&mut Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) {
        conv::dispatch_conv1d_backward(
            grad_out,
            grad_out_layout,
            input,
            input_layout,
            weight,
            weight_layout,
            grad_input,
            grad_input_layout,
            grad_weight,
            grad_weight_layout,
            grad_bias,
            stride,
            padding,
            dilation,
        );
    }

    #[inline]
    fn conv2d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        bias: Option<&Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) {
        conv::dispatch_conv2d(
            input,
            input_layout,
            weight,
            weight_layout,
            bias,
            stride,
            padding,
            dilation,
            output,
            output_layout,
        );
    }

    #[inline]
    fn conv2d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        grad_input: Option<&mut Self::DeviceBuffer<T>>,
        grad_input_layout: &Layout,
        grad_weight: Option<&mut Self::DeviceBuffer<T>>,
        grad_weight_layout: &Layout,
        grad_bias: Option<&mut Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) {
        conv::dispatch_conv2d_backward(
            grad_out,
            grad_out_layout,
            input,
            input_layout,
            weight,
            weight_layout,
            grad_input,
            grad_input_layout,
            grad_weight,
            grad_weight_layout,
            grad_bias,
            stride,
            padding,
            dilation,
        );
    }

    #[inline]
    fn conv3d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        bias: Option<&Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) {
        conv::dispatch_conv3d(
            input,
            input_layout,
            weight,
            weight_layout,
            bias,
            stride,
            padding,
            dilation,
            output,
            output_layout,
        );
    }

    #[inline]
    fn conv3d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        grad_input: Option<&mut Self::DeviceBuffer<T>>,
        grad_input_layout: &Layout,
        grad_weight: Option<&mut Self::DeviceBuffer<T>>,
        grad_weight_layout: &Layout,
        grad_bias: Option<&mut Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) {
        conv::dispatch_conv3d_backward(
            grad_out,
            grad_out_layout,
            input,
            input_layout,
            weight,
            weight_layout,
            grad_input,
            grad_input_layout,
            grad_weight,
            grad_weight_layout,
            grad_bias,
            stride,
            padding,
            dilation,
        );
    }

    #[inline]
    fn conv_transpose1d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        bias: Option<&Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) where
        T: coeus_core::Float,
    {
        conv::dispatch_conv_transpose1d(
            input,
            input_layout,
            weight,
            weight_layout,
            bias,
            stride,
            padding,
            output_padding,
            dilation,
            output,
            output_layout,
        );
    }

    #[inline]
    fn conv_transpose2d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        bias: Option<&Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) where
        T: coeus_core::Float,
    {
        conv::dispatch_conv_transpose2d(
            input,
            input_layout,
            weight,
            weight_layout,
            bias,
            stride,
            padding,
            output_padding,
            dilation,
            output,
            output_layout,
        );
    }
}
