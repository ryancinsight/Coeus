// ── On-device scaled dot-product attention (NVRTC CUDA C kernels) ──
//
// Mirrors the verified CPU reference in
// `coeus-ops/src/backend_ops/cpu_impl/attention.rs`. Tensors are contiguous
// `[batch, seq, dim]` with offset 0 (multi-head attention folds heads into the
// batch dimension before dispatch). The attention matrix is materialized into
// `attn_weights` (standard attention, not flash), matching the CPU contract and
// the `attn_weights` output buffer the backward pass consumes.
//
// Scope: causal, unmasked, and supported contiguous key-padding-mask cases run
// on-device. Other layouts route to the CPU reference path (see
// `backend/ops/attention.rs`); this is an explicit capability boundary, not a
// silent fallback.

use super::launch_1d;
use crate::driver::get_cuda_context;
use crate::kernels::validation::cuda_u32;
use crate::storage::CudaStorage;
use coeus_core::Storage;

#[derive(Clone, Copy)]
struct AttentionShape {
    batch: usize,
    seq_q: usize,
    seq_k: usize,
    d_k: usize,
    d_v: usize,
}

#[derive(Clone, Copy)]
struct AttentionMask {
    has_mask: bool,
    ndim: usize,
    num_heads: usize,
}

#[derive(Clone, Copy)]
struct AttentionLaunchDimensions {
    seq_q: u32,
    seq_k: u32,
    d_k: u32,
    d_v: u32,
    total_q: u32,
    total_k: u32,
    mask_ndim: u32,
    num_heads: u32,
    total_q_elements: usize,
    total_k_elements: usize,
    query_elements: usize,
    key_elements: usize,
    value_elements: usize,
    output_elements: usize,
    attention_elements: usize,
    mask_elements: usize,
}

fn checked_attention_dimensions(
    shape: AttentionShape,
    mask: AttentionMask,
) -> Option<AttentionLaunchDimensions> {
    if [
        shape.batch,
        shape.seq_q,
        shape.seq_k,
        shape.d_k,
        shape.d_v,
        mask.num_heads,
    ]
    .into_iter()
    .any(|dimension| dimension == 0)
        || (mask.has_mask && !matches!(mask.ndim, 1 | 2))
        || (!mask.has_mask && mask.ndim != 0)
        || (mask.ndim == 2 && !shape.batch.is_multiple_of(mask.num_heads))
    {
        return None;
    }

    let total_q = shape.batch.checked_mul(shape.seq_q)?;
    let total_k = shape.batch.checked_mul(shape.seq_k)?;
    let attention_elements = total_q.checked_mul(shape.seq_k)?;
    let query_elements = total_q.checked_mul(shape.d_k)?;
    let key_elements = total_k.checked_mul(shape.d_k)?;
    let value_elements = total_k.checked_mul(shape.d_v)?;
    let output_elements = total_q.checked_mul(shape.d_v)?;
    let mask_elements = match mask.ndim {
        0 => 0,
        1 => shape.seq_k,
        2 => (shape.batch / mask.num_heads).checked_mul(shape.seq_k)?,
        _ => return None,
    };

    Some(AttentionLaunchDimensions {
        seq_q: cuda_u32(shape.seq_q)?,
        seq_k: cuda_u32(shape.seq_k)?,
        d_k: cuda_u32(shape.d_k)?,
        d_v: cuda_u32(shape.d_v)?,
        total_q: cuda_u32(total_q)?,
        total_k: cuda_u32(total_k)?,
        mask_ndim: cuda_u32(mask.ndim)?,
        num_heads: cuda_u32(mask.num_heads)?,
        total_q_elements: total_q,
        total_k_elements: total_k,
        query_elements,
        key_elements,
        value_elements,
        output_elements,
        attention_elements,
        mask_elements,
    })
}

const FWD_SRC: &str = r#"
extern "C" __global__ void sdp_attn_fwd_kernel(
    const float* q, const float* k, const float* v, const float* mask,
    float* out, float* aw,
    unsigned int seq_q, unsigned int seq_k,
    unsigned int d_k, unsigned int d_v,
    unsigned int is_causal, float scale, unsigned int total,
    unsigned int has_mask, unsigned int mask_ndim, unsigned int num_heads
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int b = idx / seq_q;
    unsigned int i = idx % seq_q;
    const float* q_bi = q + (size_t)(b * seq_q + i) * d_k;
    float* aw_bi = aw + (size_t)(b * seq_q + i) * seq_k;
    float* out_bi = out + (size_t)(b * seq_q + i) * d_v;
    const float* k_b = k + (size_t)b * seq_k * d_k;
    const float* v_b = v + (size_t)b * seq_k * d_v;
    // Contiguous key-padding mask base: 2-D [batch_mask, seq_k] folds heads.
    size_t mask_row = (mask_ndim == 2u) ? (size_t)(b / num_heads) * seq_k : 0;

    // Phase 1: scores[j] = scale * dot(Q[i,:], K[j,:]); masked/causal -> -inf.
    float mx = -INFINITY;
    for (unsigned int j = 0; j < seq_k; ++j) {
        if (is_causal && j > i) { aw_bi[j] = -INFINITY; continue; }
        if (has_mask && mask[mask_row + j] == 0.0f) { aw_bi[j] = -INFINITY; continue; }
        const float* k_j = k_b + (size_t)j * d_k;
        float dot = 0.0f;
        for (unsigned int d = 0; d < d_k; ++d) dot = fmaf(q_bi[d], k_j[d], dot);
        float s = dot * scale;
        aw_bi[j] = s;
        if (s > mx) mx = s;
    }
    // Phase 2: numerically stable softmax over the row (exp(-inf)=0).
    float sum = 0.0f;
    for (unsigned int j = 0; j < seq_k; ++j) {
        float e = expf(aw_bi[j] - mx);
        aw_bi[j] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (unsigned int j = 0; j < seq_k; ++j) aw_bi[j] *= inv;
    // Phase 3: out[i,l] = sum_j attn[i,j] * V[j,l].
    for (unsigned int l = 0; l < d_v; ++l) {
        float acc = 0.0f;
        for (unsigned int j = 0; j < seq_k; ++j)
            acc = fmaf(aw_bi[j], v_b[(size_t)j * d_v + l], acc);
        out_bi[l] = acc;
    }
}
"#;

const BWD_DQ_SRC: &str = r#"
extern "C" __global__ void sdp_attn_bwd_dq_kernel(
    const float* go, const float* k, const float* v, const float* aw,
    float* d_scores, float* gq,
    unsigned int seq_q, unsigned int seq_k,
    unsigned int d_k, unsigned int d_v,
    unsigned int has_gq, float scale, unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int b = idx / seq_q;
    unsigned int i = idx % seq_q;
    const float* go_bi = go + (size_t)(b * seq_q + i) * d_v;
    const float* aw_bi = aw + (size_t)(b * seq_q + i) * seq_k;
    float* ds_bi = d_scores + (size_t)(b * seq_q + i) * seq_k;
    const float* v_b = v + (size_t)b * seq_k * d_v;
    const float* k_b = k + (size_t)b * seq_k * d_k;

    // d_attn_row[j] = dot(dO[i,:], V[j,:]) -> stash in d_scores row.
    for (unsigned int j = 0; j < seq_k; ++j) {
        const float* v_j = v_b + (size_t)j * d_v;
        float dot = 0.0f;
        for (unsigned int l = 0; l < d_v; ++l) dot = fmaf(go_bi[l], v_j[l], dot);
        ds_bi[j] = dot;
    }
    // rs = dot(A[i,:], d_attn_row).
    float rs = 0.0f;
    for (unsigned int j = 0; j < seq_k; ++j) rs = fmaf(aw_bi[j], ds_bi[j], rs);
    // d_scores[i,j] = A[i,j] * (d_attn_row[j] - rs)  (softmax backward).
    for (unsigned int j = 0; j < seq_k; ++j) ds_bi[j] = aw_bi[j] * (ds_bi[j] - rs);
    // dQ[i,d] += scale * sum_j d_scores[i,j] * K[j,d].
    if (has_gq) {
        float* gq_bi = gq + (size_t)(b * seq_q + i) * d_k;
        for (unsigned int d = 0; d < d_k; ++d) {
            float acc = 0.0f;
            for (unsigned int j = 0; j < seq_k; ++j)
                acc = fmaf(ds_bi[j], k_b[(size_t)j * d_k + d], acc);
            gq_bi[d] += acc * scale;
        }
    }
}
"#;

const BWD_DKV_SRC: &str = r#"
extern "C" __global__ void sdp_attn_bwd_dkv_kernel(
    const float* go, const float* q, const float* aw, const float* d_scores,
    float* gk, float* gv,
    unsigned int seq_q, unsigned int seq_k,
    unsigned int d_k, unsigned int d_v,
    unsigned int has_gk, unsigned int has_gv, float scale, unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int b = idx / seq_k;
    unsigned int j = idx % seq_k;
    // dK[j,d] += scale * sum_i d_scores[i,j] * Q[i,d].
    if (has_gk) {
        float* gk_bj = gk + (size_t)(b * seq_k + j) * d_k;
        for (unsigned int d = 0; d < d_k; ++d) {
            float acc = 0.0f;
            for (unsigned int i = 0; i < seq_q; ++i) {
                float ds = d_scores[(size_t)(b * seq_q + i) * seq_k + j];
                float qv = q[(size_t)(b * seq_q + i) * d_k + d];
                acc = fmaf(ds, qv, acc);
            }
            gk_bj[d] += acc * scale;
        }
    }
    // dV[j,l] += sum_i A[i,j] * dO[i,l].
    if (has_gv) {
        float* gv_bj = gv + (size_t)(b * seq_k + j) * d_v;
        for (unsigned int l = 0; l < d_v; ++l) {
            float acc = 0.0f;
            for (unsigned int i = 0; i < seq_q; ++i) {
                float awv = aw[(size_t)(b * seq_q + i) * seq_k + j];
                float gov = go[(size_t)(b * seq_q + i) * d_v + l];
                acc = fmaf(awv, gov, acc);
            }
            gv_bj[l] += acc;
        }
    }
}
"#;

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
        super::fuse::get_or_create_kernel("sdp_attn_fwd", FWD_SRC, "sdp_attn_fwd_kernel")
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
    let Some(dq_kernel) =
        super::fuse::get_or_create_kernel("sdp_attn_bwd_dq", BWD_DQ_SRC, "sdp_attn_bwd_dq_kernel")
    else {
        return false;
    };
    let Some(dkv_kernel) = super::fuse::get_or_create_kernel(
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

#[cfg(test)]
mod tests {
    use super::{checked_attention_dimensions, AttentionMask, AttentionShape};

    fn shape() -> AttentionShape {
        AttentionShape {
            batch: 2,
            seq_q: 3,
            seq_k: 4,
            d_k: 5,
            d_v: 6,
        }
    }

    #[test]
    fn accepts_contiguous_rank_two_mask_dimensions() {
        let dimensions = checked_attention_dimensions(
            shape(),
            AttentionMask {
                has_mask: true,
                ndim: 2,
                num_heads: 1,
            },
        )
        .expect("valid attention dimensions");

        assert_eq!(dimensions.mask_elements, 8);
        assert_eq!(dimensions.attention_elements, 24);
        assert_eq!(dimensions.total_q, 6);
        assert_eq!(dimensions.total_k, 8);
    }

    #[test]
    fn rejects_zero_dimensions_and_inconsistent_mask_rank() {
        let mut zero_shape = shape();
        zero_shape.seq_q = 0;
        assert!(checked_attention_dimensions(
            zero_shape,
            AttentionMask {
                has_mask: false,
                ndim: 0,
                num_heads: 1,
            },
        )
        .is_none());

        assert!(checked_attention_dimensions(
            shape(),
            AttentionMask {
                has_mask: false,
                ndim: 1,
                num_heads: 1,
            },
        )
        .is_none());
    }

    #[test]
    fn rejects_non_divisible_mask_heads_and_product_overflow() {
        assert!(checked_attention_dimensions(
            shape(),
            AttentionMask {
                has_mask: true,
                ndim: 2,
                num_heads: 3,
            },
        )
        .is_none());

        let overflow_shape = AttentionShape {
            batch: usize::MAX,
            ..shape()
        };
        assert!(checked_attention_dimensions(
            overflow_shape,
            AttentionMask {
                has_mask: false,
                ndim: 0,
                num_heads: 1,
            },
        )
        .is_none());
    }
}
