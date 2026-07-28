use super::super::launch_1d;
use super::source::FWD_SRC;
use super::{AttentionMask, AttentionShape, checked_attention_dimensions};
use crate::driver::get_cuda_context;
use crate::storage::CudaStorage;
use coeus_core::Storage;

/// On-device SDP forward. Handles causal/unmasked and a contiguous
/// key-padding mask (`mask` = `None` for the unmasked case). Returns `false`
/// if no CUDA context or kernel compilation/launch fails, so the caller can
/// fall back.
#[allow(clippy::too_many_arguments)]
pub fn launch_sdp_attention(
    query: &CudaStorage<f32>,
    key: &CudaStorage<f32>,
    value: &CudaStorage<f32>,
    mask: Option<&CudaStorage<f32>>,
    output: &mut CudaStorage<f32>,
    attn_weights: &mut CudaStorage<f32>,
    batch: usize,
    seq_q: usize,
    seq_k: usize,
    d_k: usize,
    d_v: usize,
    is_causal: bool,
    scale: f32,
    mask_ndim: usize,
    num_heads: usize,
) -> bool {
    let Some(dimensions) = checked_attention_dimensions(
        AttentionShape {
            batch,
            seq_q,
            seq_k,
            d_k,
            d_v,
        },
        AttentionMask {
            has_mask: mask.is_some(),
            ndim: mask_ndim,
            num_heads,
        },
    ) else {
        return false;
    };
    if query.len() < dimensions.query_elements
        || key.len() < dimensions.key_elements
        || value.len() < dimensions.value_elements
        || output.len() < dimensions.output_elements
        || attn_weights.len() < dimensions.attention_elements
        || mask.is_some_and(|storage| storage.len() < dimensions.mask_elements)
    {
        return false;
    }
    if get_cuda_context().is_none() {
        return false;
    }
    let Some(kernel) =
        super::super::fuse::get_or_create_kernel("sdp_attn_fwd", FWD_SRC, "sdp_attn_fwd_kernel")
    else {
        return false;
    };

    let mut q_ptr = query.cu_deviceptr();
    let mut k_ptr = key.cu_deviceptr();
    let mut v_ptr = value.cu_deviceptr();
    let mut mask_ptr = mask.map(|m| m.cu_deviceptr()).unwrap_or(0);
    let mut out_ptr = output.cu_deviceptr();
    let mut aw_ptr = attn_weights.cu_deviceptr();
    let mut seq_q_v = dimensions.seq_q;
    let mut seq_k_v = dimensions.seq_k;
    let mut d_k_v = dimensions.d_k;
    let mut d_v_v = dimensions.d_v;
    let mut causal_v = u32::from(is_causal);
    let mut scale_v = scale;
    let total = dimensions.total_q_elements;
    let mut total_v = dimensions.total_q;
    let mut has_mask_v = u32::from(mask.is_some());
    let mut mask_ndim_v = dimensions.mask_ndim;
    let mut num_heads_v = dimensions.num_heads;

    let mut args: [*mut std::ffi::c_void; 16] = [
        &mut q_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut k_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut v_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut mask_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut aw_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut seq_q_v as *mut u32 as *mut std::ffi::c_void,
        &mut seq_k_v as *mut u32 as *mut std::ffi::c_void,
        &mut d_k_v as *mut u32 as *mut std::ffi::c_void,
        &mut d_v_v as *mut u32 as *mut std::ffi::c_void,
        &mut causal_v as *mut u32 as *mut std::ffi::c_void,
        &mut scale_v as *mut f32 as *mut std::ffi::c_void,
        &mut total_v as *mut u32 as *mut std::ffi::c_void,
        &mut has_mask_v as *mut u32 as *mut std::ffi::c_void,
        &mut mask_ndim_v as *mut u32 as *mut std::ffi::c_void,
        &mut num_heads_v as *mut u32 as *mut std::ffi::c_void,
    ];
    launch_1d(kernel.func, total, &mut args)
}
