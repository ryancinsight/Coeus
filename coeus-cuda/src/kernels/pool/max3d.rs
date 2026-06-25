#![allow(clippy::too_many_arguments)]

use super::POOL_COMMON_SRC;
use crate::backend::CudaScalar;
use crate::driver::{get_cuda_context, CUdeviceptr, CudaDriver};
use crate::kernels::fuse::get_or_create_kernel;
use crate::storage::CudaStorage;
use coeus_core::Layout;

pub fn dispatch_max_pool3d<T: CudaScalar>(
    input: &CudaStorage<T>,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut CudaStorage<T>,
    output_layout: &Layout,
) -> bool {
    let Some(drv) = CudaDriver::get() else {
        return false;
    };
    let Some(_ctx) = get_cuda_context() else {
        return false;
    };

    let cuda_type = T::CUDA_TYPE;
    let min_val = if cuda_type == "double" {
        "-1e300"
    } else if cuda_type == "int" {
        "-2147483648"
    } else if cuda_type == "__half" {
        "-65504.0f"
    } else if cuda_type == "__nv_bfloat16" {
        "-3.38953e38f"
    } else {
        "-1e38f"
    };

    let cuda_src = format!(
        r#"
{common}

extern "C" __global__ void max_pool3d_kernel(
    const {cuda_type}* input,
    {cuda_type}* output,
    GpuLayoutInfo input_layout,
    GpuLayoutInfo output_layout,
    unsigned int kernel_size,
    unsigned int stride,
    unsigned int padding,
    unsigned int dilation,
    unsigned int out_numel
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_numel) {{
        return;
    }}

    unsigned int w_out = output_layout.shape[4];
    unsigned int h_out = output_layout.shape[3];
    unsigned int d_out = output_layout.shape[2];
    unsigned int c = output_layout.shape[1];

    unsigned int ow = idx % w_out;
    unsigned int temp1 = idx / w_out;
    unsigned int oh = temp1 % h_out;
    unsigned int temp2 = temp1 / h_out;
    unsigned int od = temp2 % d_out;
    unsigned int temp3 = temp2 / d_out;
    unsigned int ci = temp3 % c;
    unsigned int ni = temp3 / c;

    unsigned int d_in_limit = input_layout.shape[2];
    unsigned int h_in_limit = input_layout.shape[3];
    unsigned int w_in_limit = input_layout.shape[4];

    int pad_s = (int)padding;
    int stride_s = (int)stride;
    int dil_s = (int)dilation;

    {cuda_type} max_val = ({cuda_type}){min_val};
    bool has_val = false;

    for (unsigned int ikd = 0; ikd < kernel_size; ++ikd) {{
        int d_in = (int)od * stride_s + (int)ikd * dil_s - pad_s;
        if (d_in >= 0 && d_in < (int)d_in_limit) {{
            for (unsigned int ikh = 0; ikh < kernel_size; ++ikh) {{
                int h_in = (int)oh * stride_s + (int)ikh * dil_s - pad_s;
                if (h_in >= 0 && h_in < (int)h_in_limit) {{
                    for (unsigned int ikw = 0; ikw < kernel_size; ++ikw) {{
                        int w_in = (int)ow * stride_s + (int)ikw * dil_s - pad_s;
                        if (w_in >= 0 && w_in < (int)w_in_limit) {{
                            unsigned int input_idx = get_physical_index_5d(input_layout, ni, ci, (unsigned int)d_in, (unsigned int)h_in, (unsigned int)w_in);
                            {cuda_type} val = input[input_idx];
                            if (!has_val) {{
                                max_val = val;
                                has_val = true;
                            }} else if (val > max_val) {{
                                max_val = val;
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}

    unsigned int output_idx = get_physical_index_5d(output_layout, ni, ci, od, oh, ow);
    output[output_idx] = has_val ? max_val : ({cuda_type})0.0f;
}}
"#,
        common = POOL_COMMON_SRC,
        cuda_type = cuda_type,
        min_val = min_val,
    );

    let key = format!("max_pool3d_{}", cuda_type);
    let Some(kernel) = get_or_create_kernel(&key, &cuda_src, "max_pool3d_kernel") else {
        return false;
    };

    let gpu_input_layout = crate::kernels::GpuLayoutInfo::from_layout(input_layout);
    let gpu_output_layout = crate::kernels::GpuLayoutInfo::from_layout(output_layout);

    let out_numel = output_layout.shape().iter().product::<usize>();
    let mut input_ptr = input.cu_deviceptr();
    let mut output_ptr = output.cu_deviceptr();
    let mut k_size = kernel_size as u32;
    let mut s_size = stride as u32;
    let mut p_size = padding as u32;
    let mut d_size = dilation as u32;
    let mut out_n = out_numel as u32;

    let mut args: [*mut std::ffi::c_void; 9] = [
        &mut input_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut output_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &gpu_input_layout as *const crate::kernels::GpuLayoutInfo as *mut std::ffi::c_void,
        &gpu_output_layout as *const crate::kernels::GpuLayoutInfo as *mut std::ffi::c_void,
        &mut k_size as *mut u32 as *mut std::ffi::c_void,
        &mut s_size as *mut u32 as *mut std::ffi::c_void,
        &mut p_size as *mut u32 as *mut std::ffi::c_void,
        &mut d_size as *mut u32 as *mut std::ffi::c_void,
        &mut out_n as *mut u32 as *mut std::ffi::c_void,
    ];

    let block_size = 256;
    let grid_size = out_numel.div_ceil(block_size);

    unsafe {
        let res = (drv.cu_launch_kernel)(
            kernel.func,
            grid_size as u32,
            1,
            1,
            block_size as u32,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        res == 0
    }
}

pub fn dispatch_max_pool3d_backward<T: CudaScalar>(
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
) -> bool {
    let Some(drv) = CudaDriver::get() else {
        return false;
    };
    let Some(_ctx) = get_cuda_context() else {
        return false;
    };

    let cuda_type = T::CUDA_TYPE;
    let min_val = if cuda_type == "double" {
        "-1e300"
    } else if cuda_type == "int" {
        "-2147483648"
    } else if cuda_type == "__half" {
        "-65504.0f"
    } else if cuda_type == "__nv_bfloat16" {
        "-3.38953e38f"
    } else {
        "-1e38f"
    };

    let cuda_src = format!(
        r#"
{common}

extern "C" __global__ void max_pool3d_backward_kernel(
    const {cuda_type}* grad_out,
    const {cuda_type}* input,
    {cuda_type}* grad_input,
    GpuLayoutInfo grad_out_layout,
    GpuLayoutInfo input_layout,
    GpuLayoutInfo grad_input_layout,
    unsigned int kernel_size,
    unsigned int stride,
    unsigned int padding,
    unsigned int dilation,
    unsigned int in_numel
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= in_numel) {{
        return;
    }}

    unsigned int w = grad_input_layout.shape[4];
    unsigned int h = grad_input_layout.shape[3];
    unsigned int d = grad_input_layout.shape[2];
    unsigned int c = grad_input_layout.shape[1];

    unsigned int wi = idx % w;
    unsigned int temp1 = idx / w;
    unsigned int hi = temp1 % h;
    unsigned int temp2 = temp1 / h;
    unsigned int di = temp2 % d;
    unsigned int temp3 = temp2 / d;
    unsigned int ci = temp3 % c;
    unsigned int ni = temp3 / c;

    unsigned int my_idx = get_physical_index_5d(input_layout, ni, ci, di, hi, wi);
    {cuda_type} my_val = input[my_idx];

    {cuda_type} sum = ({cuda_type})0.0f;

    int pad_s = (int)padding;
    int stride_s = (int)stride;
    int dil_s = (int)dilation;

    unsigned int d_out_limit = grad_out_layout.shape[2];
    unsigned int h_out_limit = grad_out_layout.shape[3];
    unsigned int w_out_limit = grad_out_layout.shape[4];

    for (unsigned int ikd = 0; ikd < kernel_size; ++ikd) {{
        int numer_d = (int)di + pad_s - (int)ikd * dil_s;
        if (numer_d >= 0 && numer_d % stride_s == 0) {{
            unsigned int od = (unsigned int)(numer_d / stride_s);
            if (od < d_out_limit) {{
                for (unsigned int ikh = 0; ikh < kernel_size; ++ikh) {{
                    int numer_h = (int)hi + pad_s - (int)ikh * dil_s;
                    if (numer_h >= 0 && numer_h % stride_s == 0) {{
                        unsigned int oh = (unsigned int)(numer_h / stride_s);
                        if (oh < h_out_limit) {{
                            for (unsigned int ikw = 0; ikw < kernel_size; ++ikw) {{
                                int numer_w = (int)wi + pad_s - (int)ikw * dil_s;
                                if (numer_w >= 0 && numer_w % stride_s == 0) {{
                                    unsigned int ow = (unsigned int)(numer_w / stride_s);
                                    if (ow < w_out_limit) {{
                                        {cuda_type} max_val = ({cuda_type}){min_val};
                                        bool has_val = false;
                                        unsigned int max_d = 0;
                                        unsigned int max_h = 0;
                                        unsigned int max_w = 0;

                                        for (unsigned int jkd = 0; jkd < kernel_size; ++jkd) {{
                                            int d_in = (int)od * stride_s + (int)jkd * dil_s - pad_s;
                                            if (d_in >= 0 && d_in < (int)d) {{
                                                for (unsigned int jkh = 0; jkh < kernel_size; ++jkh) {{
                                                    int h_in = (int)oh * stride_s + (int)jkh * dil_s - pad_s;
                                                    if (h_in >= 0 && h_in < (int)h) {{
                                                        for (unsigned int jkw = 0; jkw < kernel_size; ++jkw) {{
                                                            int w_in = (int)ow * stride_s + (int)jkw * dil_s - pad_s;
                                                            if (w_in >= 0 && w_in < (int)w) {{
                                                                unsigned int input_idx = get_physical_index_5d(input_layout, ni, ci, (unsigned int)d_in, (unsigned int)h_in, (unsigned int)w_in);
                                                                {cuda_type} val = input[input_idx];
                                                                if (!has_val || val > max_val) {{
                                                                    max_val = val;
                                                                    max_d = (unsigned int)d_in;
                                                                    max_h = (unsigned int)h_in;
                                                                    max_w = (unsigned int)w_in;
                                                                    has_val = true;
                                                                }}
                                                            }}
                                                        }}
                                                    }}
                                                }}
                                            }}
                                        }}

                                        if (has_val && max_d == di && max_h == hi && max_w == wi && my_val == max_val) {{
                                            unsigned int go_idx = get_physical_index_5d(grad_out_layout, ni, ci, od, oh, ow);
                                            sum += grad_out[go_idx];
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}

    unsigned int gi_idx = get_physical_index_5d(grad_input_layout, ni, ci, di, hi, wi);
    grad_input[gi_idx] += sum;
}}
"#,
        common = POOL_COMMON_SRC,
        cuda_type = cuda_type,
        min_val = min_val,
    );

    let key = format!("max_pool3d_backward_{}", cuda_type);
    let Some(kernel) = get_or_create_kernel(&key, &cuda_src, "max_pool3d_backward_kernel") else {
        return false;
    };

    let gpu_go_layout = crate::kernels::GpuLayoutInfo::from_layout(grad_out_layout);
    let gpu_inp_layout = crate::kernels::GpuLayoutInfo::from_layout(input_layout);
    let gpu_gi_layout = crate::kernels::GpuLayoutInfo::from_layout(grad_input_layout);

    let in_numel = grad_input_layout.shape().iter().product::<usize>();
    let mut go_ptr = grad_out.cu_deviceptr();
    let mut inp_ptr = input.cu_deviceptr();
    let mut gi_ptr = grad_input.cu_deviceptr();
    let mut k_size = kernel_size as u32;
    let mut s_size = stride as u32;
    let mut p_size = padding as u32;
    let mut d_size = dilation as u32;
    let mut in_n = in_numel as u32;

    let mut args: [*mut std::ffi::c_void; 11] = [
        &mut go_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut inp_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &mut gi_ptr as *mut CUdeviceptr as *mut std::ffi::c_void,
        &gpu_go_layout as *const crate::kernels::GpuLayoutInfo as *mut std::ffi::c_void,
        &gpu_inp_layout as *const crate::kernels::GpuLayoutInfo as *mut std::ffi::c_void,
        &gpu_gi_layout as *const crate::kernels::GpuLayoutInfo as *mut std::ffi::c_void,
        &mut k_size as *mut u32 as *mut std::ffi::c_void,
        &mut s_size as *mut u32 as *mut std::ffi::c_void,
        &mut p_size as *mut u32 as *mut std::ffi::c_void,
        &mut d_size as *mut u32 as *mut std::ffi::c_void,
        &mut in_n as *mut u32 as *mut std::ffi::c_void,
    ];

    let block_size = 256;
    let grid_size = in_numel.div_ceil(block_size);

    unsafe {
        let res = (drv.cu_launch_kernel)(
            kernel.func,
            grid_size as u32,
            1,
            1,
            block_size as u32,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        res == 0
    }
}
