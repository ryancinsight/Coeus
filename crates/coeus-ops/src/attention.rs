// ── Coeus Ops — free function for scaled dot-product attention ──

use crate::BackendOps;
use coeus_core::{BackendError, Float};
use coeus_tensor::Tensor;

/// Compute scaled dot-product attention.
///
/// Implements `Attention(Q, K, V) = softmax(Q·Kᵀ · scale) · V` with optional
/// causal or key-padding masking.
///
/// # Shapes
/// - `query`:  `[batch, seq_q, d_k]`
/// - `key`:    `[batch, seq_k, d_k]`
/// - `value`:  `[batch, seq_k, d_v]`
///
/// Returns `(output [batch, seq_q, d_v], attn_weights [batch, seq_q, seq_k])`.
///
/// # Memory
/// Both output buffers are allocated uninitialized (`alloc_on`); the CPU kernel
/// writes every position exactly once before returning, so no zero-initialization
/// overhead is incurred.
///
/// # Performance
/// The CPU backend dispatches one task per `(batch, seq_q)` pair via
/// `parallel_for`, enabling SIMD dot products via `Float::dot_slice` on each
/// query–key pair and coarse-grained parallelism over the batch × query matrix.
pub fn scaled_dot_product_attention<T: Float, B: BackendOps<T> + Default>(
    query: &Tensor<T, B>,
    key: &Tensor<T, B>,
    value: &Tensor<T, B>,
    key_padding_mask: Option<&Tensor<T, B>>,
    is_causal: bool,
    scale: T,
    backend: &B,
) -> Result<(Tensor<T, B>, Tensor<T, B>), B::Error> {
    if query.ndim() != 3 || key.ndim() != 3 || value.ndim() != 3 {
        return Err(B::Error::from(BackendError::Storage {
            operation: "scaled_dot_product_attention",
            reason: "query, key, and value must be rank-3 tensors".to_owned(),
        }));
    }
    let q_shape = query.shape();
    let batch = q_shape[0];
    let seq_q = q_shape[1];

    let k_shape = key.shape();
    let seq_k = k_shape[1];

    let v_shape = value.shape();
    let d_v = v_shape[2];
    if k_shape[0] != batch || v_shape[0] != batch || k_shape[2] != q_shape[2] || v_shape[1] != seq_k
    {
        return Err(B::Error::from(BackendError::Storage {
            operation: "scaled_dot_product_attention",
            reason: "query, key, and value dimensions are incompatible".to_owned(),
        }));
    }

    // alloc_on: sdp_attention writes every output/attn_weights position — no zero-init needed.
    let mut output = Tensor::alloc_on([batch, seq_q, d_v], backend)?;
    let mut attn_weights = Tensor::alloc_on([batch, seq_q, seq_k], backend)?;

    let (out_storage, out_layout) = output.storage_mut_and_layout()?;
    let (aw_storage, aw_layout) = attn_weights.storage_mut_and_layout()?;

    let (mask_storage, mask_layout) = match key_padding_mask {
        Some(m) => (Some(m.storage()), Some(m.layout())),
        None => (None, None),
    };

    backend.sdp_attention(
        query.storage(),
        query.layout(),
        key.storage(),
        key.layout(),
        value.storage(),
        value.layout(),
        mask_storage,
        mask_layout,
        is_causal,
        scale,
        out_storage,
        out_layout,
        aw_storage,
        aw_layout,
    )?;

    Ok((output, attn_weights))
}

/// Compute the backward pass of scaled dot-product attention.
///
/// Accumulates gradients into the provided mutable tensors.
///
/// # Memory
/// Each gradient buffer (`grad_q`, `grad_k`, `grad_v`) is **accumulated**
/// (`+=`) by the backward kernel — existing values are preserved and gradient
/// contributions are added on top. The caller is responsible for initialising
/// the buffers to zero before the first backward pass (typically via
/// `GradBuffer::new(Tensor::zeros_on(...))`).
///
/// This means the backward MUST NOT use `alloc_on` for these buffers: reading
/// uninitialized memory before adding would produce incorrect gradients.
#[allow(clippy::too_many_arguments)]
pub fn scaled_dot_product_attention_backward<T: Float, B: BackendOps<T> + Default>(
    grad_out: &Tensor<T, B>,
    query: &Tensor<T, B>,
    key: &Tensor<T, B>,
    value: &Tensor<T, B>,
    attn_weights: &Tensor<T, B>,
    scale: T,
    grad_q: Option<&mut Tensor<T, B>>,
    grad_k: Option<&mut Tensor<T, B>>,
    grad_v: Option<&mut Tensor<T, B>>,
    backend: &B,
) -> Result<(), B::Error> {
    // We need to pass mutable references into BackendOps, but only for the storage of
    // each gradient tensor. Extract them separately to avoid borrow conflicts.
    let go_storage = grad_out.storage();
    let go_layout = grad_out.layout();
    let q_storage = query.storage();
    let q_layout = query.layout();
    let k_storage = key.storage();
    let k_layout = key.layout();
    let v_storage = value.storage();
    let v_layout = value.layout();
    let aw_storage = attn_weights.storage();
    let aw_layout = attn_weights.layout();

    match (grad_q, grad_k, grad_v) {
        (Some(gq), Some(gk), Some(gv)) => {
            let (gq_s, _gq_l) = gq.storage_mut_and_layout()?;
            let (gk_s, _gk_l) = gk.storage_mut_and_layout()?;
            let (gv_s, _gv_l) = gv.storage_mut_and_layout()?;
            backend.sdp_attention_backward(
                go_storage,
                go_layout,
                q_storage,
                q_layout,
                k_storage,
                k_layout,
                v_storage,
                v_layout,
                aw_storage,
                aw_layout,
                scale,
                Some(gq_s),
                Some(gk_s),
                Some(gv_s),
            )?;
        }
        (Some(gq), Some(gk), None) => {
            let (gq_s, _) = gq.storage_mut_and_layout()?;
            let (gk_s, _) = gk.storage_mut_and_layout()?;
            backend.sdp_attention_backward(
                go_storage,
                go_layout,
                q_storage,
                q_layout,
                k_storage,
                k_layout,
                v_storage,
                v_layout,
                aw_storage,
                aw_layout,
                scale,
                Some(gq_s),
                Some(gk_s),
                None,
            )?;
        }
        (Some(gq), None, Some(gv)) => {
            let (gq_s, _) = gq.storage_mut_and_layout()?;
            let (gv_s, _) = gv.storage_mut_and_layout()?;
            backend.sdp_attention_backward(
                go_storage,
                go_layout,
                q_storage,
                q_layout,
                k_storage,
                k_layout,
                v_storage,
                v_layout,
                aw_storage,
                aw_layout,
                scale,
                Some(gq_s),
                None,
                Some(gv_s),
            )?;
        }
        (None, Some(gk), Some(gv)) => {
            let (gk_s, _) = gk.storage_mut_and_layout()?;
            let (gv_s, _) = gv.storage_mut_and_layout()?;
            backend.sdp_attention_backward(
                go_storage,
                go_layout,
                q_storage,
                q_layout,
                k_storage,
                k_layout,
                v_storage,
                v_layout,
                aw_storage,
                aw_layout,
                scale,
                None,
                Some(gk_s),
                Some(gv_s),
            )?;
        }
        (Some(gq), None, None) => {
            let (gq_s, _) = gq.storage_mut_and_layout()?;
            backend.sdp_attention_backward(
                go_storage,
                go_layout,
                q_storage,
                q_layout,
                k_storage,
                k_layout,
                v_storage,
                v_layout,
                aw_storage,
                aw_layout,
                scale,
                Some(gq_s),
                None,
                None,
            )?;
        }
        (None, Some(gk), None) => {
            let (gk_s, _) = gk.storage_mut_and_layout()?;
            backend.sdp_attention_backward(
                go_storage,
                go_layout,
                q_storage,
                q_layout,
                k_storage,
                k_layout,
                v_storage,
                v_layout,
                aw_storage,
                aw_layout,
                scale,
                None,
                Some(gk_s),
                None,
            )?;
        }
        (None, None, Some(gv)) => {
            let (gv_s, _) = gv.storage_mut_and_layout()?;
            backend.sdp_attention_backward(
                go_storage,
                go_layout,
                q_storage,
                q_layout,
                k_storage,
                k_layout,
                v_storage,
                v_layout,
                aw_storage,
                aw_layout,
                scale,
                None,
                None,
                Some(gv_s),
            )?;
        }
        (None, None, None) => {}
    }
    Ok(())
}
