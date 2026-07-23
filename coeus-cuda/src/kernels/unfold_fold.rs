//! CUDA sliding-window extraction and adjoint accumulation kernels.

use crate::backend::CudaScalar;
use crate::driver::CUdeviceptr;
use crate::kernels::{launch_1d, GpuLayoutInfo};
use crate::storage::CudaStorage;
use coeus_core::Layout;

const SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

struct GpuLayoutInfo {
    unsigned int offset;
    unsigned int ndim;
    unsigned int shape[8];
    unsigned int strides[8];
};

__device__ unsigned int index3(
    const GpuLayoutInfo layout,
    unsigned int i0,
    unsigned int i1,
    unsigned int i2
) {
    return layout.offset + i0 * layout.strides[0]
        + i1 * layout.strides[1] + i2 * layout.strides[2];
}

__device__ unsigned int index4(
    const GpuLayoutInfo layout,
    unsigned int i0,
    unsigned int i1,
    unsigned int i2,
    unsigned int i3
) {
    return layout.offset + i0 * layout.strides[0]
        + i1 * layout.strides[1] + i2 * layout.strides[2]
        + i3 * layout.strides[3];
}

extern "C" __global__ void unfold1d_kernel(
    const {TYPE}* input,
    {TYPE}* output,
    GpuLayoutInfo input_layout,
    GpuLayoutInfo output_layout,
    unsigned int kernel_size,
    unsigned int stride,
    unsigned int padding,
    unsigned int dilation,
    unsigned int total
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const unsigned int out_l = output_layout.shape[2];
    const unsigned int l = idx % out_l;
    const unsigned int channel_kernel = (idx / out_l) % output_layout.shape[1];
    const unsigned int n = idx / (out_l * output_layout.shape[1]);
    const unsigned int channel = channel_kernel / kernel_size;
    const unsigned int kernel = channel_kernel % kernel_size;
    const long long source = (long long)l * stride + (long long)kernel * dilation - padding;
    const unsigned int output_index = index3(output_layout, n, channel_kernel, l);
    output[output_index] = source >= 0 && source < (long long)input_layout.shape[2]
        ? input[index3(input_layout, n, channel, (unsigned int)source)]
        : ({TYPE})0;
}

extern "C" __global__ void fold1d_kernel(
    const {TYPE}* input,
    {TYPE}* output,
    GpuLayoutInfo input_layout,
    GpuLayoutInfo output_layout,
    unsigned int kernel_size,
    unsigned int stride,
    unsigned int padding,
    unsigned int dilation,
    unsigned int total
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const unsigned int output_l = output_layout.shape[2];
    const unsigned int l = idx % output_l;
    const unsigned int channel = (idx / output_l) % output_layout.shape[1];
    const unsigned int n = idx / (output_l * output_layout.shape[1]);
    {TYPE} sum = ({TYPE})0;
    for (unsigned int kernel = 0; kernel < kernel_size; ++kernel) {
        const long long numerator = (long long)l + padding - (long long)kernel * dilation;
        if (numerator >= 0 && numerator % stride == 0) {
            const unsigned int source_l = (unsigned int)(numerator / stride);
            if (source_l < input_layout.shape[2]) {
                sum += input[index3(
                    input_layout, n, channel * kernel_size + kernel, source_l
                )];
            }
        }
    }
    output[index3(output_layout, n, channel, l)] = sum;
}

extern "C" __global__ void unfold2d_kernel(
    const {TYPE}* input,
    {TYPE}* output,
    GpuLayoutInfo input_layout,
    GpuLayoutInfo output_layout,
    unsigned int kernel_h,
    unsigned int kernel_w,
    unsigned int stride_h,
    unsigned int stride_w,
    unsigned int padding_h,
    unsigned int padding_w,
    unsigned int dilation_h,
    unsigned int dilation_w,
    unsigned int output_w,
    unsigned int total
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const unsigned int locations = output_layout.shape[2];
    const unsigned int location = idx % locations;
    const unsigned int channel_kernel = (idx / locations) % output_layout.shape[1];
    const unsigned int n = idx / (locations * output_layout.shape[1]);
    const unsigned int kernel_area = kernel_h * kernel_w;
    const unsigned int channel = channel_kernel / kernel_area;
    const unsigned int kernel_offset = channel_kernel % kernel_area;
    const unsigned int kh = kernel_offset / kernel_w;
    const unsigned int kw = kernel_offset % kernel_w;
    const unsigned int oh = location / output_w;
    const unsigned int ow = location % output_w;
    const long long source_h = (long long)oh * stride_h + (long long)kh * dilation_h - padding_h;
    const long long source_w = (long long)ow * stride_w + (long long)kw * dilation_w - padding_w;
    const unsigned int output_index = index3(output_layout, n, channel_kernel, location);
    output[output_index] = source_h >= 0 && source_h < (long long)input_layout.shape[2]
        && source_w >= 0 && source_w < (long long)input_layout.shape[3]
        ? input[index4(input_layout, n, channel, (unsigned int)source_h, (unsigned int)source_w)]
        : ({TYPE})0;
}

extern "C" __global__ void fold2d_kernel(
    const {TYPE}* input,
    {TYPE}* output,
    GpuLayoutInfo input_layout,
    GpuLayoutInfo output_layout,
    unsigned int kernel_h,
    unsigned int kernel_w,
    unsigned int stride_h,
    unsigned int stride_w,
    unsigned int padding_h,
    unsigned int padding_w,
    unsigned int dilation_h,
    unsigned int dilation_w,
    unsigned int input_w,
    unsigned int total
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const unsigned int output_w = output_layout.shape[3];
    const unsigned int output_h = output_layout.shape[2];
    const unsigned int w = idx % output_w;
    const unsigned int h = (idx / output_w) % output_h;
    const unsigned int channel = (idx / (output_w * output_h)) % output_layout.shape[1];
    const unsigned int n = idx / (output_w * output_h * output_layout.shape[1]);
    const unsigned int input_h = input_layout.shape[2] / input_w;
    {TYPE} sum = ({TYPE})0;
    for (unsigned int kh = 0; kh < kernel_h; ++kh) {
        const long long numerator_h = (long long)h + padding_h - (long long)kh * dilation_h;
        if (numerator_h < 0 || numerator_h % stride_h != 0) continue;
        const unsigned int source_h = (unsigned int)(numerator_h / stride_h);
        if (source_h >= input_h) continue;
        for (unsigned int kw = 0; kw < kernel_w; ++kw) {
            const long long numerator_w = (long long)w + padding_w - (long long)kw * dilation_w;
            if (numerator_w < 0 || numerator_w % stride_w != 0) continue;
            const unsigned int source_w = (unsigned int)(numerator_w / stride_w);
            if (source_w >= input_w) continue;
            const unsigned int location = source_h * input_w + source_w;
            if (location < input_layout.shape[2]) {
                const unsigned int channel_kernel =
                    (channel * kernel_h + kh) * kernel_w + kw;
                sum += input[index3(input_layout, n, channel_kernel, location)];
            }
        }
    }
    output[index4(output_layout, n, channel, h, w)] = sum;
}
"#;

fn kernel<T: CudaScalar>(
    name: &str,
) -> Option<std::sync::Arc<crate::kernels::fuse::SafeCachedKernel>> {
    let source = SOURCE.replace("{TYPE}", T::CUDA_TYPE);
    crate::kernels::fuse::get_or_create_kernel(
        &format!("unfold_fold_{name}_{}", T::CUDA_TYPE),
        &source,
        name,
    )
}

fn scalar(value: usize, name: &'static str) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| panic!("{name} exceeds CUDA u32 index range: {value}"))
}

fn launch(
    name: &str,
    total: usize,
    function: crate::driver::CUfunction,
    args: &mut [*mut std::ffi::c_void],
) -> bool {
    total == 0 || {
        let launched = launch_1d(function, total, args);
        debug_assert!(launched, "{name} CUDA launch failed");
        launched
    }
}

/// Dispatch one-dimensional sliding-window extraction.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_unfold1d<T: CudaScalar>(
    input: &CudaStorage<T>,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut CudaStorage<T>,
    output_layout: &Layout,
) -> bool {
    let Some(compiled) = kernel::<T>("unfold1d_kernel") else {
        return false;
    };
    let mut input_ptr = input.cu_deviceptr();
    let mut output_ptr = output.cu_deviceptr();
    let Ok(input_gpu) = GpuLayoutInfo::try_from(input_layout) else {
        return false;
    };
    let Ok(output_gpu) = GpuLayoutInfo::try_from(output_layout) else {
        return false;
    };
    let mut kernel_size = scalar(kernel_size, "kernel_size");
    let mut stride = scalar(stride, "stride");
    let mut padding = scalar(padding, "padding");
    let mut dilation = scalar(dilation, "dilation");
    let total = output_layout.shape().iter().product();
    let mut total_gpu = scalar(total, "output elements");
    let mut args = [
        &mut input_ptr as *mut CUdeviceptr as *mut _,
        &mut output_ptr as *mut CUdeviceptr as *mut _,
        &input_gpu as *const GpuLayoutInfo as *mut _,
        &output_gpu as *const GpuLayoutInfo as *mut _,
        &mut kernel_size as *mut u32 as *mut _,
        &mut stride as *mut u32 as *mut _,
        &mut padding as *mut u32 as *mut _,
        &mut dilation as *mut u32 as *mut _,
        &mut total_gpu as *mut u32 as *mut _,
    ];
    launch("unfold1d", total, compiled.func, &mut args)
}

/// Dispatch one-dimensional adjoint fold accumulation.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fold1d<T: CudaScalar>(
    input: &CudaStorage<T>,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut CudaStorage<T>,
    output_layout: &Layout,
) -> bool {
    dispatch_unfold_or_fold1d(
        "fold1d_kernel",
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
fn dispatch_unfold_or_fold1d<T: CudaScalar>(
    name: &str,
    input: &CudaStorage<T>,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut CudaStorage<T>,
    output_layout: &Layout,
) -> bool {
    let Some(compiled) = kernel::<T>(name) else {
        return false;
    };
    let mut input_ptr = input.cu_deviceptr();
    let mut output_ptr = output.cu_deviceptr();
    let Ok(input_gpu) = GpuLayoutInfo::try_from(input_layout) else {
        return false;
    };
    let Ok(output_gpu) = GpuLayoutInfo::try_from(output_layout) else {
        return false;
    };
    let mut kernel_size = scalar(kernel_size, "kernel_size");
    let mut stride = scalar(stride, "stride");
    let mut padding = scalar(padding, "padding");
    let mut dilation = scalar(dilation, "dilation");
    let total = output_layout.shape().iter().product();
    let mut total_gpu = scalar(total, "output elements");
    let mut args = [
        &mut input_ptr as *mut CUdeviceptr as *mut _,
        &mut output_ptr as *mut CUdeviceptr as *mut _,
        &input_gpu as *const GpuLayoutInfo as *mut _,
        &output_gpu as *const GpuLayoutInfo as *mut _,
        &mut kernel_size as *mut u32 as *mut _,
        &mut stride as *mut u32 as *mut _,
        &mut padding as *mut u32 as *mut _,
        &mut dilation as *mut u32 as *mut _,
        &mut total_gpu as *mut u32 as *mut _,
    ];
    launch(name, total, compiled.func, &mut args)
}

/// Dispatch two-dimensional sliding-window extraction.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_unfold2d<T: CudaScalar>(
    input: &CudaStorage<T>,
    input_layout: &Layout,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    output: &mut CudaStorage<T>,
    output_layout: &Layout,
) -> bool {
    dispatch_unfold_or_fold2d(
        "unfold2d_kernel",
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
        output_width(input_layout, kernel_w, stride_w, padding_w, dilation_w),
        output,
        output_layout,
    )
}

/// Dispatch two-dimensional adjoint fold accumulation.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fold2d<T: CudaScalar>(
    input: &CudaStorage<T>,
    input_layout: &Layout,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    output: &mut CudaStorage<T>,
    output_layout: &Layout,
) -> bool {
    let input_w = output_width(output_layout, kernel_w, stride_w, padding_w, dilation_w);
    dispatch_unfold_or_fold2d(
        "fold2d_kernel",
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
        input_w,
        output,
        output_layout,
    )
}

fn output_width(
    layout: &Layout,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    (layout.shape()[layout.ndim() - 1] + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
}

#[allow(clippy::too_many_arguments)]
fn dispatch_unfold_or_fold2d<T: CudaScalar>(
    name: &str,
    input: &CudaStorage<T>,
    input_layout: &Layout,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    width: usize,
    output: &mut CudaStorage<T>,
    output_layout: &Layout,
) -> bool {
    let Some(compiled) = kernel::<T>(name) else {
        return false;
    };
    let mut input_ptr = input.cu_deviceptr();
    let mut output_ptr = output.cu_deviceptr();
    let Ok(input_gpu) = GpuLayoutInfo::try_from(input_layout) else {
        return false;
    };
    let Ok(output_gpu) = GpuLayoutInfo::try_from(output_layout) else {
        return false;
    };
    let mut values = [
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, width,
    ]
    .map(|value| scalar(value, "unfold/fold parameter"));
    let total = output_layout.shape().iter().product();
    let mut total_gpu = scalar(total, "output elements");
    let mut args = [
        &mut input_ptr as *mut CUdeviceptr as *mut _,
        &mut output_ptr as *mut CUdeviceptr as *mut _,
        &input_gpu as *const GpuLayoutInfo as *mut _,
        &output_gpu as *const GpuLayoutInfo as *mut _,
        &mut values[0] as *mut u32 as *mut _,
        &mut values[1] as *mut u32 as *mut _,
        &mut values[2] as *mut u32 as *mut _,
        &mut values[3] as *mut u32 as *mut _,
        &mut values[4] as *mut u32 as *mut _,
        &mut values[5] as *mut u32 as *mut _,
        &mut values[6] as *mut u32 as *mut _,
        &mut values[7] as *mut u32 as *mut _,
        &mut values[8] as *mut u32 as *mut _,
        &mut total_gpu as *mut u32 as *mut _,
    ];
    launch(name, total, compiled.func, &mut args)
}

#[cfg(test)]
mod tests {
    use super::SOURCE;

    #[test]
    fn unfold_fold_source_compiles_for_native_float() {
        let source = SOURCE.replace("{TYPE}", "float");
        crate::kernels::fuse::compile_cuda_to_ptx(&source)
            .expect("unfold/fold CUDA source must compile through NVRTC");
    }
}
