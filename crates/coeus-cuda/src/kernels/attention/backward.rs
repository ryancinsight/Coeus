use super::super::launch_1d;
use super::source::{BWD_DKV_SRC, BWD_DQ_SRC};
use super::{checked_attention_dimensions, AttentionMask, AttentionShape};
use crate::driver::get_cuda_context;
use crate::storage::CudaStorage;
use coeus_core::Storage;

/// On-device SDP backward (causal/unmasked). Allocates a transient `d_scores`
/// device buffer, then runs the dQ pass and the dK/dV pass; the two launches are
/// stream-ordered so the dK/dV pass observes the completed `d_scores`.
#[allow(clippy::too_many_arguments)]
pub fn launch_sdp_attention_backward(
    grad_out: &CudaStorage<f32>,
    query: &CudaStorage<f32>,
    key: &CudaStorage<f32>,
    value: &CudaStorage<f32>,
    attn_weights: &CudaStorage<f32>,
    grad_q: Option<&mut CudaStorage<f32>>,
    grad_k: Option<&mut CudaStorage<f32>>,
    grad_v: Option<&mut CudaStorage<f32>>,
    batch: usize,
    seq_q: usize,
    seq_k: usize,
    d_k: usize,
    d_v: usize,
    scale: f32,
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
            has_mask: false,
            ndim: 0,
            num_heads: 1,
        },
    ) else {
        return false;
    };
    if grad_out.len() < dimensions.output_elements
        || query.len() < dimensions.query_elements
        || key.len() < dimensions.key_elements
        || value.len() < dimensions.value_elements
        || attn_weights.len() < dimensions.attention_elements
        || grad_q
            .as_ref()
            .is_some_and(|storage| storage.len() < dimensions.query_elements)
        || grad_k
            .as_ref()
            .is_some_and(|storage| storage.len() < dimensions.key_elements)
        || grad_v
            .as_ref()
            .is_some_and(|storage| storage.len() < dimensions.value_elements)
    {
        return false;
    }
    if get_cuda_context().is_none() {
        return false;
    }
    let Some(dq_kernel) = super::super::fuse::get_or_create_kernel(
        "sdp_attn_bwd_dq",
        BWD_DQ_SRC,
        "sdp_attn_bwd_dq_kernel",
    ) else {
        return false;
    };
    let Some(dkv_kernel) = super::super::fuse::get_or_create_kernel(
        "sdp_attn_bwd_dkv",
        BWD_DKV_SRC,
        "sdp_attn_bwd_dkv_kernel",
    ) else {
        return false;
    };

    let d_scores = CudaStorage::<f32>::new(dimensions.attention_elements);

    let mut go_ptr = grad_out.cu_deviceptr();
    let mut q_ptr = query.cu_deviceptr();
    let mut k_ptr = key.cu_deviceptr();
    let mut v_ptr = value.cu_deviceptr();
    let mut aw_ptr = attn_weights.cu_deviceptr();
    let mut ds_ptr = d_scores.cu_deviceptr();

    let mut seq_q_v = dimensions.seq_q;
    let mut seq_k_v = dimensions.seq_k;
    let mut d_k_v = dimensions.d_k;
    let mut d_v_v = dimensions.d_v;
    let mut scale_v = scale;

    // ── Pass 1: fill d_scores, accumulate dQ. One thread per (b, i). ──
    let mut grad_q = grad_q;
    let mut gq_ptr = grad_q.as_mut().map(|g| g.cu_deviceptr()).unwrap_or(0);
    let mut has_gq = u32::from(grad_q.is_some());
    let total_q = dimensions.total_q_elements;
    let mut total_q_v = dimensions.total_q;
    {
        let mut args: [*mut std::ffi::c_void; 13] = [
            &mut go_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut k_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut v_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut aw_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut ds_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut gq_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut seq_q_v as *mut u32 as *mut std::ffi::c_void,
            &mut seq_k_v as *mut u32 as *mut std::ffi::c_void,
            &mut d_k_v as *mut u32 as *mut std::ffi::c_void,
            &mut d_v_v as *mut u32 as *mut std::ffi::c_void,
            &mut has_gq as *mut u32 as *mut std::ffi::c_void,
            &mut scale_v as *mut f32 as *mut std::ffi::c_void,
            &mut total_q_v as *mut u32 as *mut std::ffi::c_void,
        ];
        if !launch_1d(dq_kernel.func, total_q, &mut args) {
            return false;
        }
    }

    // ── Pass 2: accumulate dK and dV. One thread per (b, j). ──
    let mut grad_k = grad_k;
    let mut grad_v = grad_v;
    let mut gk_ptr = grad_k.as_mut().map(|g| g.cu_deviceptr()).unwrap_or(0);
    let mut gv_ptr = grad_v.as_mut().map(|g| g.cu_deviceptr()).unwrap_or(0);
    let mut has_gk = u32::from(grad_k.is_some());
    let mut has_gv = u32::from(grad_v.is_some());
    let total_k = dimensions.total_k_elements;
    let mut total_k_v = dimensions.total_k;
    {
        let mut args: [*mut std::ffi::c_void; 14] = [
            &mut go_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut q_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut aw_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut ds_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut gk_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut gv_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut seq_q_v as *mut u32 as *mut std::ffi::c_void,
            &mut seq_k_v as *mut u32 as *mut std::ffi::c_void,
            &mut d_k_v as *mut u32 as *mut std::ffi::c_void,
            &mut d_v_v as *mut u32 as *mut std::ffi::c_void,
            &mut has_gk as *mut u32 as *mut std::ffi::c_void,
            &mut has_gv as *mut u32 as *mut std::ffi::c_void,
            &mut scale_v as *mut f32 as *mut std::ffi::c_void,
            &mut total_k_v as *mut u32 as *mut std::ffi::c_void,
        ];
        launch_1d(dkv_kernel.func, total_k, &mut args)
    }
}
