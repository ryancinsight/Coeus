#![allow(clippy::too_many_arguments)]

use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, Float, Layout};

/// Sealed helper — compute a single row's max for numerically stable softmax.
#[inline(always)]
fn row_max<T: Float>(row: &[T]) -> T {
    row.iter()
        .copied()
        .fold(T::NEG_INFINITY, |a, b| if b > a { b } else { a })
}

/// Forward: scaled dot-product attention.
///
/// # Shapes
/// - `query`:  `[batch, seq_q, d_k]`
/// - `key`:    `[batch, seq_k, d_k]`
/// - `value`:  `[batch, seq_k, d_v]`
/// - `output`: `[batch, seq_q, d_v]`  (pre-allocated, zeroed)
/// - `attn_weights`: `[batch, seq_q, seq_k]` (pre-allocated, zeroed, returned for backward)
///
/// All tensors must be contiguous with offset == 0.
pub(crate) fn sdp_attention<T: Float, B: Backend>(
    _backend: &B,
    query: &B::DeviceBuffer<T>,
    query_layout: &Layout,
    key: &B::DeviceBuffer<T>,
    key_layout: &Layout,
    value: &B::DeviceBuffer<T>,
    value_layout: &Layout,
    key_padding_mask: Option<&B::DeviceBuffer<T>>,
    key_padding_mask_layout: Option<&Layout>,
    is_causal: bool,
    scale: T,
    output: &mut B::DeviceBuffer<T>,
    _output_layout: &Layout,
    attn_weights: &mut B::DeviceBuffer<T>,
    _attn_weights_layout: &Layout,
) where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let q_shape = query_layout.shape();
    let batch = q_shape[0];
    let seq_q = q_shape[1];
    let d_k = q_shape[2];

    let k_shape = key_layout.shape();
    let seq_k = k_shape[1];

    let v_shape = value_layout.shape();
    let d_v = v_shape[2];

    let q_slice = query.as_slice();
    let k_slice = key.as_slice();
    let v_slice = value.as_slice();
    let aw_slice = attn_weights.as_mut_slice();
    let out_slice = output.as_mut_slice();

    // ── Inner-Function Pattern: extract non-generic index math ──
    #[inline(always)]
    fn idx3(b: usize, i: usize, j: usize, dim1: usize, dim2: usize) -> usize {
        b * dim1 * dim2 + i * dim2 + j
    }

    for b in 0..batch {
        // 1. scores[b, i, j] = dot(Q[b,i,:], K[b,j,:]) * scale
        for i in 0..seq_q {
            for j in 0..seq_k {
                // Causal mask: positions j > i are masked to -∞
                if is_causal && j > i {
                    aw_slice[idx3(b, i, j, seq_q, seq_k)] = T::NEG_INFINITY;
                    continue;
                }
                if let Some(mask) = key_padding_mask {
                    let mask_layout = key_padding_mask_layout.unwrap();
                    let mask_shape = mask_layout.shape();
                    let mask_strides = mask_layout.strides();
                    let mask_offset = mask_layout.offset();
                    let m_slice = mask.as_slice();
                    let mask_idx = if mask_shape.len() == 1 {
                        mask_offset + j * mask_strides[0]
                    } else {
                        let batch_mask = mask_shape[0];
                        let num_heads = batch / batch_mask;
                        let b_mask = b / num_heads;
                        mask_offset + b_mask * mask_strides[0] + j * mask_strides[1]
                    };
                    if m_slice[mask_idx] == T::zero() {
                        aw_slice[idx3(b, i, j, seq_q, seq_k)] = T::NEG_INFINITY;
                        continue;
                    }
                }
                let q_start = idx3(b, i, 0, seq_q, d_k);
                let k_start = idx3(b, j, 0, seq_k, d_k);
                let dot = T::dot_slice(
                    &q_slice[q_start..q_start + d_k],
                    &k_slice[k_start..k_start + d_k],
                );
                aw_slice[idx3(b, i, j, seq_q, seq_k)] = dot * scale;
            }
        }

        // 2. Row-wise numerically stable softmax over scores
        for i in 0..seq_q {
            let row_start = idx3(b, i, 0, seq_q, seq_k);
            let row = &aw_slice[row_start..row_start + seq_k];
            let mx = row_max(row);

            // shift and exp
            let mut sum_exp = T::zero();
            for j in 0..seq_k {
                let v = (aw_slice[idx3(b, i, j, seq_q, seq_k)] - mx).exp();
                aw_slice[idx3(b, i, j, seq_q, seq_k)] = v;
                sum_exp = sum_exp + v;
            }
            // normalize
            let inv_sum = T::one() / sum_exp;
            T::scale_slice(&mut aw_slice[row_start..row_start + seq_k], inv_sum);
        }

        // 3. output[b, i, l] = sum_j attn[b, i, j] * V[b, j, l]
        for i in 0..seq_q {
            for l in 0..d_v {
                let mut acc = T::zero();
                for j in 0..seq_k {
                    acc = acc
                        + aw_slice[idx3(b, i, j, seq_q, seq_k)]
                            * v_slice[idx3(b, j, l, seq_k, d_v)];
                }
                out_slice[idx3(b, i, l, seq_q, d_v)] = acc;
            }
        }
    }
}

/// Backward: compute gradients for Q, K, V from the stored attention weights.
///
/// Given stored `attn_weights A` (post-softmax) and `grad_out dO`:
///  - `dV = A^T @ dO`      shape `[B, seq_k, d_v]`
///  - `dA = dO @ V^T`      shape `[B, seq_q, seq_k]`
///  - `dS = A ⊙ (dA - rowsum(A ⊙ dA))`  (softmax backward)
///  - `dQ = dS @ K * scale`
///  - `dK = dS^T @ Q * scale`
///
/// All gradient buffers (grad_q, grad_k, grad_v) are accumulated (+=).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sdp_attention_backward<T: Float, B: Backend>(
    _backend: &B,
    grad_out: &B::DeviceBuffer<T>,
    _grad_out_layout: &Layout,
    query: &B::DeviceBuffer<T>,
    query_layout: &Layout,
    key: &B::DeviceBuffer<T>,
    key_layout: &Layout,
    value: &B::DeviceBuffer<T>,
    value_layout: &Layout,
    attn_weights: &B::DeviceBuffer<T>,
    _attn_weights_layout: &Layout,
    scale: T,
    mut grad_q: Option<&mut B::DeviceBuffer<T>>,
    mut grad_k: Option<&mut B::DeviceBuffer<T>>,
    mut grad_v: Option<&mut B::DeviceBuffer<T>>,
) where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let q_shape = query_layout.shape();
    let batch = q_shape[0];
    let seq_q = q_shape[1];
    let d_k = q_shape[2];

    let k_shape = key_layout.shape();
    let seq_k = k_shape[1];

    let v_shape = value_layout.shape();
    let d_v = v_shape[2];

    let go = grad_out.as_slice();
    let aw = attn_weights.as_slice();
    let q_sl = query.as_slice();
    let k_sl = key.as_slice();
    let v_sl = value.as_slice();

    #[inline(always)]
    fn idx3(b: usize, i: usize, j: usize, dim1: usize, dim2: usize) -> usize {
        b * dim1 * dim2 + i * dim2 + j
    }

    for b in 0..batch {
        // ── dV accumulation ──
        if let Some(ref mut gv) = grad_v.as_deref_mut() {
            let gv_sl = gv.as_mut_slice();
            // dV[b,j,l] += sum_i A[b,i,j] * dO[b,i,l]
            for j in 0..seq_k {
                for l in 0..d_v {
                    let mut acc = T::zero();
                    for i in 0..seq_q {
                        acc = acc + aw[idx3(b, i, j, seq_q, seq_k)] * go[idx3(b, i, l, seq_q, d_v)];
                    }
                    gv_sl[idx3(b, j, l, seq_k, d_v)] = gv_sl[idx3(b, j, l, seq_k, d_v)] + acc;
                }
            }
        }

        // ── dA = dO @ V^T  shape [seq_q, seq_k] ──
        let mut d_attn = vec![T::zero(); seq_q * seq_k];
        for i in 0..seq_q {
            for j in 0..seq_k {
                let mut acc = T::zero();
                for l in 0..d_v {
                    acc = acc + go[idx3(b, i, l, seq_q, d_v)] * v_sl[idx3(b, j, l, seq_k, d_v)];
                }
                d_attn[i * seq_k + j] = acc;
            }
        }

        // ── Softmax backward: dS[i,j] = A[i,j] * (dA[i,j] - rowsum(A[i,:] ⊙ dA[i,:])) ──
        let mut d_scores = vec![T::zero(); seq_q * seq_k];
        for i in 0..seq_q {
            // rowsum(A[i,:] ⊙ dA[i,:])
            let mut rs = T::zero();
            for j in 0..seq_k {
                rs = rs + aw[idx3(b, i, j, seq_q, seq_k)] * d_attn[i * seq_k + j];
            }
            for j in 0..seq_k {
                d_scores[i * seq_k + j] =
                    aw[idx3(b, i, j, seq_q, seq_k)] * (d_attn[i * seq_k + j] - rs);
            }
        }

        // ── dQ accumulation: dQ[b,i,dk] += sum_j dS[b,i,j] * K[b,j,dk] * scale ──
        if let Some(ref mut gq) = grad_q.as_deref_mut() {
            let gq_sl = gq.as_mut_slice();
            for i in 0..seq_q {
                for dk in 0..d_k {
                    let mut acc = T::zero();
                    for j in 0..seq_k {
                        acc = acc + d_scores[i * seq_k + j] * k_sl[idx3(b, j, dk, seq_k, d_k)];
                    }
                    gq_sl[idx3(b, i, dk, seq_q, d_k)] =
                        gq_sl[idx3(b, i, dk, seq_q, d_k)] + acc * scale;
                }
            }
        }

        // ── dK accumulation: dK[b,j,dk] += sum_i dS[b,i,j] * Q[b,i,dk] * scale ──
        if let Some(ref mut gk) = grad_k.as_deref_mut() {
            let gk_sl = gk.as_mut_slice();
            for j in 0..seq_k {
                for dk in 0..d_k {
                    let mut acc = T::zero();
                    for i in 0..seq_q {
                        acc = acc + d_scores[i * seq_k + j] * q_sl[idx3(b, i, dk, seq_q, d_k)];
                    }
                    gk_sl[idx3(b, j, dk, seq_k, d_k)] =
                        gk_sl[idx3(b, j, dk, seq_k, d_k)] + acc * scale;
                }
            }
        }
    }
}
