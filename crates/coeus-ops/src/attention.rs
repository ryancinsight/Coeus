// ── Coeus Ops — free function for scaled dot-product attention ──

use crate::{AttentionOps, AttentionScalar};
use coeus_core::BackendError;
use coeus_tensor::Tensor;

#[cfg(test)]
#[path = "attention/tests.rs"]
mod tests;

const FORWARD_OPERATION: &str = "attention forward";

fn rank_three<E>(shape: &[usize]) -> Result<[usize; 3], E>
where
    E: From<BackendError>,
{
    shape.try_into().map_err(|_| {
        BackendError::UnsupportedRank {
            operation: FORWARD_OPERATION,
            rank: shape.len(),
            max_rank: 3,
        }
        .into()
    })
}

fn validate_mask<E>(shape: &[usize], execution_batches: usize, sequence: usize) -> Result<(), E>
where
    E: From<BackendError>,
{
    let valid = match shape {
        [mask_sequence] => *mask_sequence == sequence,
        [mask_batches, mask_sequence] => {
            *mask_sequence == sequence
                && *mask_batches > 0
                && execution_batches.is_multiple_of(*mask_batches)
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(BackendError::IncompatibleBroadcast {
            operation: FORWARD_OPERATION,
            from: shape.to_vec(),
            to: vec![execution_batches, sequence],
        }
        .into())
    }
}

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
/// Both output buffers are allocated uninitialized (`alloc_on`). The selected
/// provider validates all operands before writing every output position, so no
/// zero-initialization or intermediate host copy is required.
///
/// # Errors
///
/// Returns the selected backend's typed validation, preparation, or dispatch
/// failure without returning partially initialized tensors.
#[expect(
    clippy::type_complexity,
    reason = "the public contract returns the output and reusable softmax weights together"
)]
pub fn scaled_dot_product_attention<T: AttentionScalar, B: AttentionOps<T> + Default>(
    query: &Tensor<T, B>,
    key: &Tensor<T, B>,
    value: &Tensor<T, B>,
    key_padding_mask: Option<&Tensor<T, B>>,
    is_causal: bool,
    scale: T,
    backend: &B,
) -> Result<(Tensor<T, B>, Tensor<T, B>), B::Error> {
    let [batch, seq_q, query_width] = rank_three::<B::Error>(query.shape())?;
    let [key_batch, seq_k, key_width] = rank_three::<B::Error>(key.shape())?;
    let [value_batch, value_sequence, d_v] = rank_three::<B::Error>(value.shape())?;
    if batch != key_batch || query_width != key_width {
        return Err(BackendError::ShapeMismatch {
            operation: FORWARD_OPERATION,
            lhs: query.shape().to_vec(),
            rhs: key.shape().to_vec(),
        }
        .into());
    }
    if batch != value_batch || seq_k != value_sequence {
        return Err(BackendError::ShapeMismatch {
            operation: FORWARD_OPERATION,
            lhs: key.shape().to_vec(),
            rhs: value.shape().to_vec(),
        }
        .into());
    }
    if let Some(mask) = key_padding_mask {
        validate_mask::<B::Error>(mask.shape(), batch, seq_k)?;
    }

    // alloc_on: sdp_attention writes every output/attn_weights position — no zero-init needed.
    let mut output = Tensor::alloc_on([batch, seq_q, d_v], backend);
    let mut attn_weights = Tensor::alloc_on([batch, seq_q, seq_k], backend);

    let (out_storage, out_layout) = output.storage_mut_and_layout();
    let (aw_storage, aw_layout) = attn_weights.storage_mut_and_layout();

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
///
/// # Errors
///
/// Returns the selected backend's typed validation, preparation, or dispatch
/// failure. Validation and preparation failures preserve every selected
/// destination; dispatch failures may partially accumulate gradients.
#[expect(
    clippy::too_many_arguments,
    reason = "the function carries the complete differentiable attention contract"
)]
pub fn scaled_dot_product_attention_backward<T: AttentionScalar, B: AttentionOps<T> + Default>(
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

    let grad_q = grad_q.map(Tensor::storage_mut_and_layout);
    let grad_k = grad_k.map(Tensor::storage_mut_and_layout);
    let grad_v = grad_v.map(Tensor::storage_mut_and_layout);

    backend.sdp_attention_backward(
        go_storage, go_layout, q_storage, q_layout, k_storage, k_layout, v_storage, v_layout,
        aw_storage, aw_layout, scale, grad_q, grad_k, grad_v,
    )
}
