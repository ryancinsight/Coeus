#![allow(clippy::too_many_arguments)]

use super::validation::{
    checked_pool_parameters, pool_layouts_are_valid, pool_prefix_matches, pool_shapes_match,
    POOL_BLOCK_SIZE,
};
use super::POOL_COMMON_SRC;
use crate::backend::CudaScalar;
use crate::driver::{get_cuda_context, CUdeviceptr, CudaDriver};
use crate::kernels::fuse::get_or_create_kernel;
use crate::kernels::validation::{checked_numel, cuda_u32, launch_grid_size};
use crate::storage::CudaStorage;
use crate::CudaBackendError;
use coeus_core::Layout;

fn source<T: CudaScalar>() -> String {
    let scalar = T::CUDA_TYPE;
    format!(
        r#"
{common}

extern "C" __global__ void max_pool_forward(
    const {scalar}* input, {scalar}* output,
    GpuLayoutInfo input_layout, GpuLayoutInfo output_layout,
    unsigned int kernel_size, unsigned int stride, unsigned int padding,
    unsigned int dilation, unsigned int out_numel
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_numel) return;
    unsigned int length_out = output_layout.shape[2];
    unsigned int channels = output_layout.shape[1];
    unsigned int position_out = idx % length_out;
    unsigned int channel = (idx / length_out) % channels;
    unsigned int batch = idx / (length_out * channels);
    unsigned int length_in = input_layout.shape[2];
    bool found = false;
    {scalar} maximum = ({scalar})0;
    for (unsigned int window = 0; window < kernel_size; ++window) {{
        int position_in = (int)(position_out * stride + window * dilation) - (int)padding;
        if (position_in >= 0 && position_in < (int)length_in) {{
            {scalar} value = input[get_physical_index_1d(input_layout, batch, channel, (unsigned int)position_in)];
            if (!found || value > maximum) {{ maximum = value; found = true; }}
        }}
    }}
    output[get_physical_index_1d(output_layout, batch, channel, position_out)] =
        found ? maximum : ({scalar})0;
}}

extern "C" __global__ void max_pool_backward(
    const {scalar}* grad_out, const {scalar}* input, {scalar}* grad_input,
    GpuLayoutInfo grad_out_layout, GpuLayoutInfo input_layout,
    GpuLayoutInfo grad_input_layout, unsigned int kernel_size,
    unsigned int stride, unsigned int padding, unsigned int dilation,
    unsigned int in_numel
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= in_numel) return;
    unsigned int length_in = grad_input_layout.shape[2];
    unsigned int channels = grad_input_layout.shape[1];
    unsigned int position_in = idx % length_in;
    unsigned int channel = (idx / length_in) % channels;
    unsigned int batch = idx / (length_in * channels);
    unsigned int length_out = grad_out_layout.shape[2];
    {scalar} gradient = ({scalar})0;
    for (unsigned int window = 0; window < kernel_size; ++window) {{
        int numerator = (int)(position_in + padding) - (int)(window * dilation);
        if (numerator < 0 || numerator % (int)stride != 0) continue;
        unsigned int position_out = (unsigned int)(numerator / (int)stride);
        if (position_out >= length_out) continue;
        bool found = false;
        {scalar} maximum = ({scalar})0;
        unsigned int maximum_position = 0;
        for (unsigned int candidate = 0; candidate < kernel_size; ++candidate) {{
            int source = (int)(position_out * stride + candidate * dilation) - (int)padding;
            if (source >= 0 && source < (int)length_in) {{
                {scalar} value = input[get_physical_index_1d(input_layout, batch, channel, (unsigned int)source)];
                if (!found || value > maximum) {{
                    maximum = value;
                    maximum_position = (unsigned int)source;
                    found = true;
                }}
            }}
        }}
        if (found && maximum_position == position_in) {{
            gradient += grad_out[get_physical_index_1d(grad_out_layout, batch, channel, position_out)];
        }}
    }}
    grad_input[get_physical_index_1d(grad_input_layout, batch, channel, position_in)] += gradient;
}}

extern "C" __global__ void avg_pool_forward(
    const {scalar}* input, {scalar}* output,
    GpuLayoutInfo input_layout, GpuLayoutInfo output_layout,
    unsigned int kernel_size, unsigned int stride, unsigned int padding,
    unsigned int dilation, unsigned int out_numel
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_numel) return;
    unsigned int length_out = output_layout.shape[2];
    unsigned int channels = output_layout.shape[1];
    unsigned int position_out = idx % length_out;
    unsigned int channel = (idx / length_out) % channels;
    unsigned int batch = idx / (length_out * channels);
    unsigned int length_in = input_layout.shape[2];
    {scalar} sum = ({scalar})0;
    unsigned int count = 0;
    for (unsigned int window = 0; window < kernel_size; ++window) {{
        int position_in = (int)(position_out * stride + window * dilation) - (int)padding;
        if (position_in >= 0 && position_in < (int)length_in) {{
            sum += input[get_physical_index_1d(input_layout, batch, channel, (unsigned int)position_in)];
            ++count;
        }}
    }}
    output[get_physical_index_1d(output_layout, batch, channel, position_out)] =
        count == 0 ? ({scalar})0 : sum / ({scalar})count;
}}

extern "C" __global__ void avg_pool_backward(
    const {scalar}* grad_out, {scalar}* grad_input,
    GpuLayoutInfo grad_out_layout, GpuLayoutInfo grad_input_layout,
    unsigned int kernel_size, unsigned int stride, unsigned int padding,
    unsigned int dilation, unsigned int in_numel
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= in_numel) return;
    unsigned int length_in = grad_input_layout.shape[2];
    unsigned int channels = grad_input_layout.shape[1];
    unsigned int position_in = idx % length_in;
    unsigned int channel = (idx / length_in) % channels;
    unsigned int batch = idx / (length_in * channels);
    unsigned int length_out = grad_out_layout.shape[2];
    {scalar} gradient = ({scalar})0;
    for (unsigned int window = 0; window < kernel_size; ++window) {{
        int numerator = (int)(position_in + padding) - (int)(window * dilation);
        if (numerator < 0 || numerator % (int)stride != 0) continue;
        unsigned int position_out = (unsigned int)(numerator / (int)stride);
        if (position_out >= length_out) continue;
        unsigned int count = 0;
        for (unsigned int candidate = 0; candidate < kernel_size; ++candidate) {{
            int source = (int)(position_out * stride + candidate * dilation) - (int)padding;
            if (source >= 0 && source < (int)length_in) ++count;
        }}
        if (count != 0) {{
            gradient += grad_out[get_physical_index_1d(grad_out_layout, batch, channel, position_out)] / ({scalar})count;
        }}
    }}
    grad_input[get_physical_index_1d(grad_input_layout, batch, channel, position_in)] += gradient;
}}
"#,
        common = POOL_COMMON_SRC,
    )
}

fn launch<T: CudaScalar>(
    operation: &'static str,
    buffers: &mut [CUdeviceptr],
    layouts: &[&Layout],
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    thread_count: Option<usize>,
) -> Result<(), CudaBackendError> {
    let Some(driver) = CudaDriver::get() else {
        return Err(CudaBackendError::kernel(
            operation,
            "CUDA driver unavailable",
        ));
    };
    let Some(_context) = get_cuda_context() else {
        return Err(CudaBackendError::kernel(
            operation,
            "CUDA context unavailable",
        ));
    };
    let Some([kernel_size_value, stride_value, padding_value, dilation_value]) =
        checked_pool_parameters(kernel_size, stride, padding, dilation)
    else {
        return Err(CudaBackendError::kernel(
            operation,
            "pool parameters exceed the CUDA u32 ABI",
        ));
    };
    let Some(thread_count) = thread_count else {
        return Err(CudaBackendError::kernel(
            operation,
            "output element count overflowed",
        ));
    };
    let Some(thread_count_value) = cuda_u32(thread_count) else {
        return Err(CudaBackendError::kernel(
            operation,
            "output element count exceeds the CUDA u32 ABI",
        ));
    };
    if !pool_layouts_are_valid(layouts, 3) {
        return Err(CudaBackendError::kernel(
            operation,
            "pool layouts violate the rank or CUDA ABI contract",
        ));
    }
    let Some(grid_size) = launch_grid_size(thread_count) else {
        return if thread_count == 0 {
            Ok(())
        } else {
            Err(CudaBackendError::kernel(
                operation,
                "launch grid exceeds the CUDA u32 ABI",
            ))
        };
    };
    let key = format!("pool1d_{operation}_{}", T::CUDA_TYPE);
    let kernel_source = source::<T>();
    let Some(kernel) = get_or_create_kernel(&key, &kernel_source, operation) else {
        return Err(CudaBackendError::kernel(
            operation,
            "kernel compilation failed",
        ));
    };
    let Ok(mut gpu_layouts) = layouts
        .iter()
        .map(|layout| crate::kernels::GpuLayoutInfo::try_from(*layout))
        .collect::<Result<Vec<_>, _>>()
    else {
        return Err(CudaBackendError::kernel(
            operation,
            "layout conversion failed",
        ));
    };
    let mut kernel_size = kernel_size_value;
    let mut stride = stride_value;
    let mut padding = padding_value;
    let mut dilation = dilation_value;
    let mut thread_count_u32 = thread_count_value;
    let mut args: Vec<*mut std::ffi::c_void> = buffers
        .iter_mut()
        .map(|pointer| pointer as *mut CUdeviceptr as *mut std::ffi::c_void)
        .collect();
    args.extend(
        gpu_layouts
            .iter_mut()
            .map(|layout| layout as *mut crate::kernels::GpuLayoutInfo as *mut std::ffi::c_void),
    );
    args.extend([
        &mut kernel_size as *mut u32 as *mut std::ffi::c_void,
        &mut stride as *mut u32 as *mut std::ffi::c_void,
        &mut padding as *mut u32 as *mut std::ffi::c_void,
        &mut dilation as *mut u32 as *mut std::ffi::c_void,
        &mut thread_count_u32 as *mut u32 as *mut std::ffi::c_void,
    ]);
    // SAFETY: `kernel` belongs to the current Hephaestus context; every
    // pointer references a live device allocation or stack argument through
    // the synchronous launch call, and the grid covers exactly `thread_count`.
    let status = unsafe {
        (driver.cu_launch_kernel)(
            kernel.func,
            grid_size,
            1,
            1,
            POOL_BLOCK_SIZE,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaBackendError::kernel(
            operation,
            "CUDA kernel launch failed",
        ))
    }
}

/// Dispatch max-pooling over `[N, C, L]`.
pub fn dispatch_max_pool1d<T: CudaScalar>(
    input: &CudaStorage<T>,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut CudaStorage<T>,
    output_layout: &Layout,
) -> Result<(), CudaBackendError> {
    if !pool_prefix_matches(input_layout, output_layout) {
        return Err(CudaBackendError::kernel(
            "max_pool_forward",
            "input and output layouts do not share the batch/channel prefix",
        ));
    }
    let Some(thread_count) = checked_numel(output_layout) else {
        return Err(CudaBackendError::kernel(
            "max_pool_forward",
            "output element count overflowed",
        ));
    };
    launch::<T>(
        "max_pool_forward",
        &mut [input.cu_deviceptr(), output.cu_deviceptr()],
        &[input_layout, output_layout],
        kernel_size,
        stride,
        padding,
        dilation,
        Some(thread_count),
    )
}

/// Dispatch the max-pooling input adjoint over `[N, C, L]`.
pub fn dispatch_max_pool1d_backward<T: CudaScalar>(
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
    if !pool_prefix_matches(grad_out_layout, grad_input_layout)
        || !pool_shapes_match(input_layout, grad_input_layout)
    {
        return Err(CudaBackendError::kernel(
            "max_pool_backward",
            "pool layouts do not satisfy the shape contract",
        ));
    }
    let Some(thread_count) = checked_numel(grad_input_layout) else {
        return Err(CudaBackendError::kernel(
            "max_pool_backward",
            "input element count overflowed",
        ));
    };
    launch::<T>(
        "max_pool_backward",
        &mut [
            grad_out.cu_deviceptr(),
            input.cu_deviceptr(),
            grad_input.cu_deviceptr(),
        ],
        &[grad_out_layout, input_layout, grad_input_layout],
        kernel_size,
        stride,
        padding,
        dilation,
        Some(thread_count),
    )
}

/// Dispatch average pooling over `[N, C, L]`.
pub fn dispatch_avg_pool1d<T: CudaScalar>(
    input: &CudaStorage<T>,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut CudaStorage<T>,
    output_layout: &Layout,
) -> Result<(), CudaBackendError> {
    if !pool_prefix_matches(input_layout, output_layout) {
        return Err(CudaBackendError::kernel(
            "avg_pool_forward",
            "input and output layouts do not share the batch/channel prefix",
        ));
    }
    let Some(thread_count) = checked_numel(output_layout) else {
        return Err(CudaBackendError::kernel(
            "avg_pool_forward",
            "output element count overflowed",
        ));
    };
    launch::<T>(
        "avg_pool_forward",
        &mut [input.cu_deviceptr(), output.cu_deviceptr()],
        &[input_layout, output_layout],
        kernel_size,
        stride,
        padding,
        dilation,
        Some(thread_count),
    )
}

/// Dispatch the average-pooling input adjoint over `[N, C, L]`.
pub fn dispatch_avg_pool1d_backward<T: CudaScalar>(
    grad_out: &CudaStorage<T>,
    grad_out_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    grad_input: &mut CudaStorage<T>,
    grad_input_layout: &Layout,
) -> Result<(), CudaBackendError> {
    if !pool_prefix_matches(grad_out_layout, grad_input_layout) {
        return Err(CudaBackendError::kernel(
            "avg_pool_backward",
            "gradient layouts do not share the batch/channel prefix",
        ));
    }
    let Some(thread_count) = checked_numel(grad_input_layout) else {
        return Err(CudaBackendError::kernel(
            "avg_pool_backward",
            "input element count overflowed",
        ));
    };
    launch::<T>(
        "avg_pool_backward",
        &mut [grad_out.cu_deviceptr(), grad_input.cu_deviceptr()],
        &[grad_out_layout, grad_input_layout],
        kernel_size,
        stride,
        padding,
        dilation,
        Some(thread_count),
    )
}
