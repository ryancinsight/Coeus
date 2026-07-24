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
use crate::kernels::validation::cuda_u32;
use crate::storage::CudaStorage;
use coeus_core::Storage;

#[derive(Clone, Copy)]
struct ConvTranspose1dParameters {
    n: usize,
    c_in: usize,
    l: usize,
    c_out: usize,
    k: usize,
    l_out: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

#[derive(Clone, Copy)]
struct ConvTranspose1dLaunch {
    values: [u32; 9],
    total: u32,
    total_elements: usize,
}

#[derive(Clone, Copy)]
struct ConvTranspose2dParameters {
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
}

#[derive(Clone, Copy)]
struct ConvTranspose2dLaunch {
    values: [u32; 12],
    total: u32,
    total_elements: usize,
}

fn checked_product(values: &[usize]) -> Option<usize> {
    values.iter().copied().try_fold(1, usize::checked_mul)
}

fn checked_conv_transpose1d_launch(
    input: &CudaStorage<f32>,
    weight: &CudaStorage<f32>,
    bias: Option<&CudaStorage<f32>>,
    output: &CudaStorage<f32>,
    parameters: ConvTranspose1dParameters,
) -> Option<ConvTranspose1dLaunch> {
    let ConvTranspose1dParameters {
        n,
        c_in,
        l,
        c_out,
        k,
        l_out,
        stride,
        padding,
        dilation,
    } = parameters;
    if [n, c_in, l, c_out, k, l_out, stride, dilation]
        .into_iter()
        .any(|dimension| dimension == 0)
    {
        return None;
    }
    let input_elements = checked_product(&[n, c_in, l])?;
    let weight_elements = checked_product(&[c_in, c_out, k])?;
    let total_elements = checked_product(&[n, c_out, l_out])?;
    cuda_u32(input_elements)?;
    cuda_u32(weight_elements)?;
    if input.len() < input_elements
        || weight.len() < weight_elements
        || output.len() < total_elements
        || bias.is_some_and(|storage| storage.len() < c_out)
    {
        return None;
    }
    let values = [n, c_in, l, c_out, k, l_out, stride, padding, dilation].map(cuda_u32);
    let [Some(n), Some(c_in), Some(l), Some(c_out), Some(k), Some(l_out), Some(stride), Some(padding), Some(dilation)] =
        values
    else {
        return None;
    };
    Some(ConvTranspose1dLaunch {
        values: [n, c_in, l, c_out, k, l_out, stride, padding, dilation],
        total: cuda_u32(total_elements)?,
        total_elements,
    })
}

fn checked_conv_transpose2d_launch(
    input: &CudaStorage<f32>,
    weight: &CudaStorage<f32>,
    bias: Option<&CudaStorage<f32>>,
    output: &CudaStorage<f32>,
    parameters: ConvTranspose2dParameters,
) -> Option<ConvTranspose2dLaunch> {
    let ConvTranspose2dParameters {
        n,
        c_in,
        h,
        w,
        c_out,
        kh,
        kw,
        h_out,
        w_out,
        stride,
        padding,
        dilation,
    } = parameters;
    if [n, c_in, h, w, c_out, kh, kw, h_out, w_out, stride, dilation]
        .into_iter()
        .any(|dimension| dimension == 0)
    {
        return None;
    }
    let input_elements = checked_product(&[n, c_in, h, w])?;
    let weight_elements = checked_product(&[c_in, c_out, kh, kw])?;
    let total_elements = checked_product(&[n, c_out, h_out, w_out])?;
    cuda_u32(input_elements)?;
    cuda_u32(weight_elements)?;
    if input.len() < input_elements
        || weight.len() < weight_elements
        || output.len() < total_elements
        || bias.is_some_and(|storage| storage.len() < c_out)
    {
        return None;
    }
    let values = [
        n, c_in, h, w, c_out, kh, kw, h_out, w_out, stride, padding, dilation,
    ]
    .map(cuda_u32);
    let [Some(n), Some(c_in), Some(h), Some(w), Some(c_out), Some(kh), Some(kw), Some(h_out), Some(w_out), Some(stride), Some(padding), Some(dilation)] =
        values
    else {
        return None;
    };
    Some(ConvTranspose2dLaunch {
        values: [
            n, c_in, h, w, c_out, kh, kw, h_out, w_out, stride, padding, dilation,
        ],
        total: cuda_u32(total_elements)?,
        total_elements,
    })
}

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
            unsigned long long numerator = (unsigned long long)t_out + padding;
            unsigned long long kernel_offset = (unsigned long long)ki * dilation;
            if (numerator < kernel_offset) continue;
            numerator -= kernel_offset;
            if (numerator % stride != 0u) continue;
            unsigned int ti = (unsigned int)(numerator / stride);
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
            unsigned long long numerator_h = (unsigned long long)ho + padding;
            unsigned long long kernel_offset_h = (unsigned long long)ki * dilation;
            if (numerator_h < kernel_offset_h) continue;
            numerator_h -= kernel_offset_h;
            if (numerator_h % stride != 0u) continue;
            unsigned int hi = (unsigned int)(numerator_h / stride);
            if (hi >= h) continue;
            for (unsigned int kj = 0; kj < kw; ++kj) {
                unsigned long long numerator_w = (unsigned long long)wo + padding;
                unsigned long long kernel_offset_w = (unsigned long long)kj * dilation;
                if (numerator_w < kernel_offset_w) continue;
                numerator_w -= kernel_offset_w;
                if (numerator_w % stride != 0u) continue;
                unsigned int wi = (unsigned int)(numerator_w / stride);
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
    let Some(dimensions) = checked_conv_transpose1d_launch(
        input,
        weight,
        bias,
        output,
        ConvTranspose1dParameters {
            n,
            c_in,
            l,
            c_out,
            k,
            l_out,
            stride,
            padding,
            dilation,
        },
    ) else {
        return false;
    };
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
    let [mut n_v, mut c_in_v, mut l_v, mut c_out_v, mut k_v, mut l_out_v, mut stride_v, mut pad_v, mut dil_v] =
        dimensions.values;
    let mut has_bias_v = u32::from(bias.is_some());
    let total = dimensions.total_elements;
    let mut total_v = dimensions.total;

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
    let Some(dimensions) = checked_conv_transpose2d_launch(
        input,
        weight,
        bias,
        output,
        ConvTranspose2dParameters {
            n,
            c_in,
            h,
            w,
            c_out,
            kh,
            kw,
            h_out,
            w_out,
            stride,
            padding,
            dilation,
        },
    ) else {
        return false;
    };
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
    let [mut n_v, mut c_in_v, mut h_v, mut w_v, mut c_out_v, mut kh_v, mut kw_v, mut h_out_v, mut w_out_v, mut stride_v, mut pad_v, mut dil_v] =
        dimensions.values;
    let mut has_bias_v = u32::from(bias.is_some());
    let total = dimensions.total_elements;
    let mut total_v = dimensions.total;

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

#[cfg(test)]
mod tests {
    use super::checked_product;

    #[test]
    fn checked_product_preserves_representable_work_sizes() {
        assert_eq!(checked_product(&[2, 3, 4]), Some(24));
    }

    #[test]
    fn checked_product_rejects_overflow() {
        assert_eq!(checked_product(&[usize::MAX, 2]), None);
    }
}
