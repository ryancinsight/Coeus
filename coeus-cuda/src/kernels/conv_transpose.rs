// ── On-device transposed convolution forward (NVRTC CUDA C kernels) ──
//
// Mirrors the verified CPU reference in `coeus-ops` (`BackendOps::
// conv_transpose1d`/`conv_transpose2d`). The CPU path expresses the op as a
// scatter (each input element scatters to outputs); these kernels use the
// equivalent gather (one thread per output element accumulates its
// contributions), which is conflict-free on the GPU — no atomics needed.
//
// Tensors are contiguous with offset 0:
//   1d: input [n, c_in, l],  weight [c_in, c_out, k],  output [n, c_out, l_out]
//   2d: input [n, c_in, h, w], weight [c_in, c_out, kh, kw], output [n, c_out, h_out, w_out]
//
// For an output position p, the contributing input index is
//   t*stride = p + padding - ki*dilation   (must be exact and in-range),
// the gather inverse of the scatter `p = t*stride + ki*dilation - padding`.

use super::launch_1d;
use crate::driver::get_cuda_context;
use crate::storage::CudaStorage;

const FWD1D_SRC: &str = r#"
extern "C" __global__ void conv_transpose1d_fwd_kernel(
    const float* input, const float* weight, const float* bias, float* out,
    unsigned int n, unsigned int c_in, unsigned int l,
    unsigned int c_out, unsigned int k, unsigned int l_out,
    unsigned int stride, unsigned int padding, unsigned int dilation,
    unsigned int has_bias, unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    // Decode output index (ni, oc, t_out) from [n, c_out, l_out].
    unsigned int t_out = idx % l_out;
    unsigned int tmp = idx / l_out;
    unsigned int oc = tmp % c_out;
    unsigned int ni = tmp / c_out;

    float acc = 0.0f;
    for (unsigned int ic = 0; ic < c_in; ++ic) {
        const float* in_row = input + (size_t)(ni * c_in + ic) * l;
        const float* w_row = weight + (size_t)(ic * c_out + oc) * k;
        for (unsigned int ki = 0; ki < k; ++ki) {
            int num = (int)(t_out + padding) - (int)(ki * dilation);
            if (num < 0) continue;
            if ((unsigned int)num % stride != 0u) continue;
            unsigned int ti = (unsigned int)num / stride;
            if (ti >= l) continue;
            acc = fmaf(in_row[ti], w_row[ki], acc);
        }
    }
    if (has_bias) acc += bias[oc];
    out[(size_t)(ni * c_out + oc) * l_out + t_out] = acc;
}
"#;

const FWD2D_SRC: &str = r#"
extern "C" __global__ void conv_transpose2d_fwd_kernel(
    const float* input, const float* weight, const float* bias, float* out,
    unsigned int n, unsigned int c_in, unsigned int h, unsigned int w,
    unsigned int c_out, unsigned int kh, unsigned int kw,
    unsigned int h_out, unsigned int w_out,
    unsigned int stride, unsigned int padding, unsigned int dilation,
    unsigned int has_bias, unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    // Decode (ni, oc, ho, wo) from [n, c_out, h_out, w_out].
    unsigned int wo = idx % w_out;
    unsigned int t1 = idx / w_out;
    unsigned int ho = t1 % h_out;
    unsigned int t2 = t1 / h_out;
    unsigned int oc = t2 % c_out;
    unsigned int ni = t2 / c_out;

    float acc = 0.0f;
    for (unsigned int ic = 0; ic < c_in; ++ic) {
        const float* in_plane = input + (size_t)(ni * c_in + ic) * h * w;
        const float* w_plane = weight + (size_t)(ic * c_out + oc) * kh * kw;
        for (unsigned int ki = 0; ki < kh; ++ki) {
            int numh = (int)(ho + padding) - (int)(ki * dilation);
            if (numh < 0) continue;
            if ((unsigned int)numh % stride != 0u) continue;
            unsigned int hi = (unsigned int)numh / stride;
            if (hi >= h) continue;
            for (unsigned int kj = 0; kj < kw; ++kj) {
                int numw = (int)(wo + padding) - (int)(kj * dilation);
                if (numw < 0) continue;
                if ((unsigned int)numw % stride != 0u) continue;
                unsigned int wi = (unsigned int)numw / stride;
                if (wi >= w) continue;
                acc = fmaf(in_plane[hi * w + wi], w_plane[ki * kw + kj], acc);
            }
        }
    }
    if (has_bias) acc += bias[oc];
    out[(size_t)(ni * c_out + oc) * h_out * w_out + (size_t)ho * w_out + wo] = acc;
}
"#;

/// On-device transposed conv1d forward. Returns `false` (caller falls back) if
/// no CUDA context or kernel compilation/launch fails.
#[allow(clippy::too_many_arguments)]
pub fn launch_conv_transpose1d(
    input: &CudaStorage<f32>,
    weight: &CudaStorage<f32>,
    bias: Option<&CudaStorage<f32>>,
    output: &mut CudaStorage<f32>,
    n: usize,
    c_in: usize,
    l: usize,
    c_out: usize,
    k: usize,
    l_out: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> bool {
    if get_cuda_context().is_none() {
        return false;
    }
    let Some(kernel) = super::fuse::get_or_create_kernel(
        "conv_transpose1d_fwd",
        FWD1D_SRC,
        "conv_transpose1d_fwd_kernel",
    ) else {
        return false;
    };

    let mut in_ptr = input.cu_deviceptr();
    let mut w_ptr = weight.cu_deviceptr();
    let mut b_ptr = bias.map(|b| b.cu_deviceptr()).unwrap_or(0);
    let mut out_ptr = output.cu_deviceptr();
    let mut n_v = n as u32;
    let mut c_in_v = c_in as u32;
    let mut l_v = l as u32;
    let mut c_out_v = c_out as u32;
    let mut k_v = k as u32;
    let mut l_out_v = l_out as u32;
    let mut stride_v = stride as u32;
    let mut pad_v = padding as u32;
    let mut dil_v = dilation as u32;
    let mut has_bias_v = u32::from(bias.is_some());
    let total = n * c_out * l_out;
    let mut total_v = total as u32;

    let mut args: [*mut std::ffi::c_void; 15] = [
        &mut in_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut w_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut b_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut n_v as *mut u32 as *mut std::ffi::c_void,
        &mut c_in_v as *mut u32 as *mut std::ffi::c_void,
        &mut l_v as *mut u32 as *mut std::ffi::c_void,
        &mut c_out_v as *mut u32 as *mut std::ffi::c_void,
        &mut k_v as *mut u32 as *mut std::ffi::c_void,
        &mut l_out_v as *mut u32 as *mut std::ffi::c_void,
        &mut stride_v as *mut u32 as *mut std::ffi::c_void,
        &mut pad_v as *mut u32 as *mut std::ffi::c_void,
        &mut dil_v as *mut u32 as *mut std::ffi::c_void,
        &mut has_bias_v as *mut u32 as *mut std::ffi::c_void,
        &mut total_v as *mut u32 as *mut std::ffi::c_void,
    ];
    launch_1d(kernel.func, total, &mut args)
}

/// On-device transposed conv2d forward. Returns `false` (caller falls back) if
/// no CUDA context or kernel compilation/launch fails.
#[allow(clippy::too_many_arguments)]
pub fn launch_conv_transpose2d(
    input: &CudaStorage<f32>,
    weight: &CudaStorage<f32>,
    bias: Option<&CudaStorage<f32>>,
    output: &mut CudaStorage<f32>,
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    h_out: usize,
    w_out: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> bool {
    if get_cuda_context().is_none() {
        return false;
    }
    let Some(kernel) = super::fuse::get_or_create_kernel(
        "conv_transpose2d_fwd",
        FWD2D_SRC,
        "conv_transpose2d_fwd_kernel",
    ) else {
        return false;
    };

    let mut in_ptr = input.cu_deviceptr();
    let mut w_ptr = weight.cu_deviceptr();
    let mut b_ptr = bias.map(|b| b.cu_deviceptr()).unwrap_or(0);
    let mut out_ptr = output.cu_deviceptr();
    let mut n_v = n as u32;
    let mut c_in_v = c_in as u32;
    let mut h_v = h as u32;
    let mut w_v = w as u32;
    let mut c_out_v = c_out as u32;
    let mut kh_v = kh as u32;
    let mut kw_v = kw as u32;
    let mut h_out_v = h_out as u32;
    let mut w_out_v = w_out as u32;
    let mut stride_v = stride as u32;
    let mut pad_v = padding as u32;
    let mut dil_v = dilation as u32;
    let mut has_bias_v = u32::from(bias.is_some());
    let total = n * c_out * h_out * w_out;
    let mut total_v = total as u32;

    let mut args: [*mut std::ffi::c_void; 18] = [
        &mut in_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut w_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut b_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut n_v as *mut u32 as *mut std::ffi::c_void,
        &mut c_in_v as *mut u32 as *mut std::ffi::c_void,
        &mut h_v as *mut u32 as *mut std::ffi::c_void,
        &mut w_v as *mut u32 as *mut std::ffi::c_void,
        &mut c_out_v as *mut u32 as *mut std::ffi::c_void,
        &mut kh_v as *mut u32 as *mut std::ffi::c_void,
        &mut kw_v as *mut u32 as *mut std::ffi::c_void,
        &mut h_out_v as *mut u32 as *mut std::ffi::c_void,
        &mut w_out_v as *mut u32 as *mut std::ffi::c_void,
        &mut stride_v as *mut u32 as *mut std::ffi::c_void,
        &mut pad_v as *mut u32 as *mut std::ffi::c_void,
        &mut dil_v as *mut u32 as *mut std::ffi::c_void,
        &mut has_bias_v as *mut u32 as *mut std::ffi::c_void,
        &mut total_v as *mut u32 as *mut std::ffi::c_void,
    ];
    launch_1d(kernel.func, total, &mut args)
}
