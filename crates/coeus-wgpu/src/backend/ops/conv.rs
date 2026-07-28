// Delegation functions mirror BackendOps trait signatures verbatim; argument
// counts cannot be reduced without breaking the trait contract.
#![allow(clippy::too_many_arguments)]
use super::*;

pub(super) fn dispatch_conv1d<T: WgpuScalar>(
    input: &crate::backend::WgpuStorage<T>,
    input_layout: &Layout,
    weight: &crate::backend::WgpuStorage<T>,
    weight_layout: &Layout,
    bias: Option<&crate::backend::WgpuStorage<T>>,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut crate::backend::WgpuStorage<T>,
    output_layout: &Layout,
) {
    let out_numel = output_layout.shape().iter().product::<usize>();
    kernels::dispatch_conv1d::<T>(kernels::Conv1dDispatch {
        input: &input.buffer,
        weight: &weight.buffer,
        bias: bias.map(|b| b.buffer.raw()),
        output: &output.buffer,
        input_layout,
        weight_layout,
        output_layout,
        stride,
        padding,
        dilation,
        out_numel,
    });
}

pub(super) fn dispatch_conv1d_backward<T: WgpuScalar>(
    grad_out: &crate::backend::WgpuStorage<T>,
    grad_out_layout: &Layout,
    input: &crate::backend::WgpuStorage<T>,
    input_layout: &Layout,
    weight: &crate::backend::WgpuStorage<T>,
    weight_layout: &Layout,
    grad_input: Option<&mut crate::backend::WgpuStorage<T>>,
    grad_input_layout: &Layout,
    grad_weight: Option<&mut crate::backend::WgpuStorage<T>>,
    grad_weight_layout: &Layout,
    grad_bias: Option<&mut crate::backend::WgpuStorage<T>>,
    stride: usize,
    padding: usize,
    dilation: usize,
) {
    kernels::dispatch_conv1d_backward::<T>(kernels::Conv1dBackwardDispatch {
        grad_out: &grad_out.buffer,
        grad_out_layout,
        input: &input.buffer,
        input_layout,
        weight: &weight.buffer,
        weight_layout,
        grad_input: grad_input.map(|gi| gi.buffer.raw()),
        grad_input_layout,
        grad_weight: grad_weight.map(|gw| gw.buffer.raw()),
        grad_weight_layout,
        grad_bias: grad_bias.map(|gb| gb.buffer.raw()),
        stride,
        padding,
        dilation,
    });
}

pub(super) fn dispatch_conv2d<T: WgpuScalar>(
    input: &crate::backend::WgpuStorage<T>,
    input_layout: &Layout,
    weight: &crate::backend::WgpuStorage<T>,
    weight_layout: &Layout,
    bias: Option<&crate::backend::WgpuStorage<T>>,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut crate::backend::WgpuStorage<T>,
    output_layout: &Layout,
) {
    let out_numel = output_layout.shape().iter().product::<usize>();
    kernels::dispatch_conv2d::<T>(kernels::Conv2dDispatch {
        input: &input.buffer,
        weight: &weight.buffer,
        bias: bias.map(|b| b.buffer.raw()),
        output: &output.buffer,
        input_layout,
        weight_layout,
        output_layout,
        stride,
        padding,
        dilation,
        out_numel,
    });
}

pub(super) fn dispatch_conv2d_backward<T: WgpuScalar>(
    grad_out: &crate::backend::WgpuStorage<T>,
    grad_out_layout: &Layout,
    input: &crate::backend::WgpuStorage<T>,
    input_layout: &Layout,
    weight: &crate::backend::WgpuStorage<T>,
    weight_layout: &Layout,
    grad_input: Option<&mut crate::backend::WgpuStorage<T>>,
    grad_input_layout: &Layout,
    grad_weight: Option<&mut crate::backend::WgpuStorage<T>>,
    grad_weight_layout: &Layout,
    grad_bias: Option<&mut crate::backend::WgpuStorage<T>>,
    stride: usize,
    padding: usize,
    dilation: usize,
) {
    kernels::dispatch_conv2d_backward::<T>(kernels::Conv2dBackwardDispatch {
        grad_out: &grad_out.buffer,
        grad_out_layout,
        input: &input.buffer,
        input_layout,
        weight: &weight.buffer,
        weight_layout,
        grad_input: grad_input.map(|gi| gi.buffer.raw()),
        grad_input_layout,
        grad_weight: grad_weight.map(|gw| gw.buffer.raw()),
        grad_weight_layout,
        grad_bias: grad_bias.map(|gb| gb.buffer.raw()),
        stride,
        padding,
        dilation,
    });
}

pub(super) fn dispatch_conv3d<T: WgpuScalar>(
    input: &crate::backend::WgpuStorage<T>,
    input_layout: &Layout,
    weight: &crate::backend::WgpuStorage<T>,
    weight_layout: &Layout,
    bias: Option<&crate::backend::WgpuStorage<T>>,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut crate::backend::WgpuStorage<T>,
    output_layout: &Layout,
) {
    kernels::dispatch_conv3d::<T>(kernels::Conv3dDispatch {
        input: &input.buffer,
        weight: &weight.buffer,
        bias: bias.map(|b| b.buffer.raw()),
        output: &output.buffer,
        input_layout,
        weight_layout,
        output_layout,
        stride,
        padding,
        dilation,
        out_numel: output.len(),
    });
}

pub(super) fn dispatch_conv3d_backward<T: WgpuScalar>(
    grad_out: &crate::backend::WgpuStorage<T>,
    grad_out_layout: &Layout,
    input: &crate::backend::WgpuStorage<T>,
    input_layout: &Layout,
    weight: &crate::backend::WgpuStorage<T>,
    weight_layout: &Layout,
    grad_input: Option<&mut crate::backend::WgpuStorage<T>>,
    grad_input_layout: &Layout,
    grad_weight: Option<&mut crate::backend::WgpuStorage<T>>,
    grad_weight_layout: &Layout,
    grad_bias: Option<&mut crate::backend::WgpuStorage<T>>,
    stride: usize,
    padding: usize,
    dilation: usize,
) {
    kernels::dispatch_conv3d_backward::<T>(kernels::Conv3dBackwardDispatch {
        grad_out: &grad_out.buffer,
        grad_out_layout,
        input: &input.buffer,
        input_layout,
        weight: &weight.buffer,
        weight_layout,
        grad_input: grad_input.map(|gi| gi.buffer.raw()),
        grad_input_layout,
        grad_weight: grad_weight.map(|gw| gw.buffer.raw()),
        grad_weight_layout,
        grad_bias: grad_bias.map(|gb| gb.buffer.raw()),
        stride,
        padding,
        dilation,
    });
}

pub(super) fn dispatch_conv_transpose1d<T: WgpuScalar + coeus_core::Float>(
    input: &crate::backend::WgpuStorage<T>,
    input_layout: &Layout,
    weight: &crate::backend::WgpuStorage<T>,
    weight_layout: &Layout,
    bias: Option<&crate::backend::WgpuStorage<T>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    output: &mut crate::backend::WgpuStorage<T>,
    output_layout: &Layout,
) -> Result<(), WgpuBackendError> {
    let _ = output_padding; // encoded in output_layout shape, not the kernel loop
                            // input [n, c_in, l] / weight [c_in, c_out, k] / output [n, c_out, l_out]
    kernels::dispatch_conv_transpose1d(kernels::ConvTranspose1dDispatch {
        input: input.buffer.raw(),
        weight: weight.buffer.raw(),
        bias: bias.map(|b| b.buffer.raw()),
        output: output.buffer.raw(),
        n: input_layout.shape()[0],
        c_in: input_layout.shape()[1],
        l: input_layout.shape()[2],
        c_out: weight_layout.shape()[1],
        k: weight_layout.shape()[2],
        l_out: output_layout.shape()[2],
        stride,
        padding,
        dilation,
    });
    Ok(())
}

pub(super) fn dispatch_conv_transpose2d<T: WgpuScalar + coeus_core::Float>(
    input: &crate::backend::WgpuStorage<T>,
    input_layout: &Layout,
    weight: &crate::backend::WgpuStorage<T>,
    weight_layout: &Layout,
    bias: Option<&crate::backend::WgpuStorage<T>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    output: &mut crate::backend::WgpuStorage<T>,
    output_layout: &Layout,
) -> Result<(), WgpuBackendError> {
    let _ = output_padding; // encoded in output_layout shape, not the kernel loop
                            // input [n, c_in, h, w] / weight [c_in, c_out, kh, kw] / output [n, c_out, h_out, w_out]
    kernels::dispatch_conv_transpose2d(kernels::ConvTranspose2dDispatch {
        input: input.buffer.raw(),
        weight: weight.buffer.raw(),
        bias: bias.map(|b| b.buffer.raw()),
        output: output.buffer.raw(),
        n: input_layout.shape()[0],
        c_in: input_layout.shape()[1],
        h: input_layout.shape()[2],
        w: input_layout.shape()[3],
        c_out: weight_layout.shape()[1],
        kh: weight_layout.shape()[2],
        kw: weight_layout.shape()[3],
        h_out: output_layout.shape()[2],
        w_out: output_layout.shape()[3],
        stride,
        padding,
        dilation,
    });
    Ok(())
}
