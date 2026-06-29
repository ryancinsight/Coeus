// Delegation functions mirror BackendOps trait signatures verbatim; argument
// counts cannot be reduced without breaking the trait contract.
#![allow(clippy::too_many_arguments)]
use super::*;

pub(super) fn dispatch_max_pool2d<T: WgpuScalar>(
    input: &crate::backend::WgpuStorage<T>,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut crate::backend::WgpuStorage<T>,
    output_layout: &Layout,
) {
    kernels::dispatch_max_pool2d::<T>(
        &input.buffer,
        input_layout,
        kernel_size,
        stride,
        padding,
        dilation,
        &output.buffer,
        output_layout,
        output.len(),
    );
}

pub(super) fn dispatch_max_pool2d_backward<T: WgpuScalar>(
    grad_out: &crate::backend::WgpuStorage<T>,
    grad_out_layout: &Layout,
    input: &crate::backend::WgpuStorage<T>,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    grad_input: &mut crate::backend::WgpuStorage<T>,
    grad_input_layout: &Layout,
) {
    kernels::dispatch_max_pool2d_backward::<T>(
        &grad_out.buffer,
        grad_out_layout,
        &input.buffer,
        input_layout,
        kernel_size,
        stride,
        padding,
        dilation,
        &grad_input.buffer,
        grad_input_layout,
        grad_input.len(),
    );
}

pub(super) fn dispatch_avg_pool2d<T: WgpuScalar>(
    input: &crate::backend::WgpuStorage<T>,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut crate::backend::WgpuStorage<T>,
    output_layout: &Layout,
) {
    kernels::dispatch_avg_pool2d::<T>(
        &input.buffer,
        input_layout,
        kernel_size,
        stride,
        padding,
        dilation,
        &output.buffer,
        output_layout,
        output.len(),
    );
}

pub(super) fn dispatch_avg_pool2d_backward<T: WgpuScalar>(
    grad_out: &crate::backend::WgpuStorage<T>,
    grad_out_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    grad_input: &mut crate::backend::WgpuStorage<T>,
    grad_input_layout: &Layout,
) {
    kernels::dispatch_avg_pool2d_backward::<T>(
        &grad_out.buffer,
        grad_out_layout,
        kernel_size,
        stride,
        padding,
        dilation,
        &grad_input.buffer,
        grad_input_layout,
        grad_input.len(),
    );
}

pub(super) fn dispatch_max_pool3d<T: WgpuScalar>(
    input: &crate::backend::WgpuStorage<T>,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut crate::backend::WgpuStorage<T>,
    output_layout: &Layout,
) {
    kernels::dispatch_max_pool3d::<T>(
        &input.buffer,
        input_layout,
        kernel_size,
        stride,
        padding,
        dilation,
        &output.buffer,
        output_layout,
        output.len(),
    );
}

pub(super) fn dispatch_max_pool3d_backward<T: WgpuScalar>(
    grad_out: &crate::backend::WgpuStorage<T>,
    grad_out_layout: &Layout,
    input: &crate::backend::WgpuStorage<T>,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    grad_input: &mut crate::backend::WgpuStorage<T>,
    grad_input_layout: &Layout,
) {
    kernels::dispatch_max_pool3d_backward::<T>(
        &grad_out.buffer,
        grad_out_layout,
        &input.buffer,
        input_layout,
        kernel_size,
        stride,
        padding,
        dilation,
        &grad_input.buffer,
        grad_input_layout,
        grad_input.len(),
    );
}

pub(super) fn dispatch_avg_pool3d<T: WgpuScalar>(
    input: &crate::backend::WgpuStorage<T>,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut crate::backend::WgpuStorage<T>,
    output_layout: &Layout,
) {
    kernels::dispatch_avg_pool3d::<T>(
        &input.buffer,
        input_layout,
        kernel_size,
        stride,
        padding,
        dilation,
        &output.buffer,
        output_layout,
        output.len(),
    );
}

pub(super) fn dispatch_avg_pool3d_backward<T: WgpuScalar>(
    grad_out: &crate::backend::WgpuStorage<T>,
    grad_out_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    grad_input: &mut crate::backend::WgpuStorage<T>,
    grad_input_layout: &Layout,
) {
    kernels::dispatch_avg_pool3d_backward::<T>(
        &grad_out.buffer,
        grad_out_layout,
        kernel_size,
        stride,
        padding,
        dilation,
        &grad_input.buffer,
        grad_input_layout,
        grad_input.len(),
    );
}

// ── Pool 1D stubs (native WGPU kernel not yet implemented) ───────────────────
// WgpuBackend delegates to these no-ops; the CPU backends route through
// coeus-ops cpu_impl where the real kernels live.

pub(super) fn dispatch_max_pool1d<T: WgpuScalar>(
    _input: &crate::backend::WgpuStorage<T>,
    _input_layout: &Layout,
    _kernel_size: usize,
    _stride: usize,
    _padding: usize,
    _dilation: usize,
    _output: &mut crate::backend::WgpuStorage<T>,
    _output_layout: &Layout,
) { /* TODO: native WGPU 1D max-pool kernel */
}

pub(super) fn dispatch_max_pool1d_backward<T: WgpuScalar>(
    _grad_out: &crate::backend::WgpuStorage<T>,
    _grad_out_layout: &Layout,
    _input: &crate::backend::WgpuStorage<T>,
    _input_layout: &Layout,
    _kernel_size: usize,
    _stride: usize,
    _padding: usize,
    _dilation: usize,
    _grad_input: &mut crate::backend::WgpuStorage<T>,
    _grad_input_layout: &Layout,
) { /* TODO: native WGPU 1D max-pool backward */
}

pub(super) fn dispatch_avg_pool1d<T: WgpuScalar>(
    _input: &crate::backend::WgpuStorage<T>,
    _input_layout: &Layout,
    _kernel_size: usize,
    _stride: usize,
    _padding: usize,
    _dilation: usize,
    _output: &mut crate::backend::WgpuStorage<T>,
    _output_layout: &Layout,
) { /* TODO: native WGPU 1D avg-pool kernel */
}

pub(super) fn dispatch_avg_pool1d_backward<T: WgpuScalar>(
    _grad_out: &crate::backend::WgpuStorage<T>,
    _grad_out_layout: &Layout,
    _kernel_size: usize,
    _stride: usize,
    _padding: usize,
    _dilation: usize,
    _grad_input: &mut crate::backend::WgpuStorage<T>,
    _grad_input_layout: &Layout,
) { /* TODO: native WGPU 1D avg-pool backward */
}
