use crate::kernels::validation::cuda_u32;

#[derive(Clone, Copy)]
pub(crate) struct AttentionShape {
    pub(crate) batch: usize,
    pub(crate) seq_q: usize,
    pub(crate) seq_k: usize,
    pub(crate) d_k: usize,
    pub(crate) d_v: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct AttentionMask {
    pub(crate) has_mask: bool,
    pub(crate) ndim: usize,
    pub(crate) num_heads: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct AttentionLaunchDimensions {
    pub(crate) seq_q: u32,
    pub(crate) seq_k: u32,
    pub(crate) d_k: u32,
    pub(crate) d_v: u32,
    pub(crate) total_q: u32,
    pub(crate) total_k: u32,
    pub(crate) mask_ndim: u32,
    pub(crate) num_heads: u32,
    pub(crate) total_q_elements: usize,
    pub(crate) total_k_elements: usize,
    pub(crate) query_elements: usize,
    pub(crate) key_elements: usize,
    pub(crate) value_elements: usize,
    pub(crate) output_elements: usize,
    pub(crate) attention_elements: usize,
    pub(crate) mask_elements: usize,
}

pub(crate) fn checked_attention_dimensions(
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
