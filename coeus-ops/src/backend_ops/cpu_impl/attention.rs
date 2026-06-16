#![allow(clippy::too_many_arguments)]

use crate::ptr::{MutPtr, Ptr};
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
    backend: &B,
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

    let q_ptr = Ptr(q_slice.as_ptr());
    let k_ptr = Ptr(k_slice.as_ptr());
    let v_ptr = Ptr(v_slice.as_ptr());
    let aw_ptr = MutPtr(aw_slice.as_mut_ptr());
    let out_ptr = MutPtr(out_slice.as_mut_ptr());

    let mask_ptr = key_padding_mask.map(|m| Ptr(m.as_slice().as_ptr()));
    let mask_layout = key_padding_mask_layout.cloned();

    #[inline(always)]
    fn idx3(b: usize, i: usize, j: usize, dim1: usize, dim2: usize) -> usize {
        b * dim1 * dim2 + i * dim2 + j
    }

    let num_tasks = batch * seq_q;

    backend.parallel_for(0, num_tasks, move |index| {
        let b = index / seq_q;
        let i = index % seq_q;

        // 1. scores[b, i, j] = dot(Q[b,i,:], K[b,j,:]) * scale
        for j in 0..seq_k {
            let aw_idx = idx3(b, i, j, seq_q, seq_k);
            if is_causal && j > i {
                unsafe {
                    aw_ptr.write(aw_idx, T::NEG_INFINITY);
                }
                continue;
            }
            if let Some(ref mp) = mask_ptr {
                let layout = mask_layout.as_ref().unwrap();
                let mask_shape = layout.shape();
                let mask_strides = layout.strides();
                let mask_offset = layout.offset();
                let mask_idx = if mask_shape.len() == 1 {
                    mask_offset + j * mask_strides[0]
                } else {
                    let batch_mask = mask_shape[0];
                    let num_heads = batch / batch_mask;
                    let b_mask = b / num_heads;
                    mask_offset + b_mask * mask_strides[0] + j * mask_strides[1]
                };
                if unsafe { mp.read(mask_idx) } == T::zero() {
                    unsafe {
                        aw_ptr.write(aw_idx, T::NEG_INFINITY);
                    }
                    continue;
                }
            }
            let q_start = idx3(b, i, 0, seq_q, d_k);
            let k_start = idx3(b, j, 0, seq_k, d_k);
            let q_window = unsafe { q_ptr.slice(q_start, d_k) };
            let k_window = unsafe { k_ptr.slice(k_start, d_k) };
            let dot = T::dot_slice(q_window, k_window);
            unsafe {
                aw_ptr.write(aw_idx, dot * scale);
            }
        }

        // 2. Row-wise numerically stable softmax over scores
        let row_start = idx3(b, i, 0, seq_q, seq_k);
        let row = unsafe { aw_ptr.slice_mut(row_start, seq_k) };
        let mx = row_max(row);

        // shift and exp
        let mut sum_exp = T::zero();
        for j in 0..seq_k {
            let aw_idx = idx3(b, i, j, seq_q, seq_k);
            let val = unsafe { aw_ptr.read(aw_idx) };
            let v = (val - mx).exp();
            unsafe {
                aw_ptr.write(aw_idx, v);
            }
            sum_exp = sum_exp + v;
        }

        // normalize
        let inv_sum = T::one() / sum_exp;
        let row_mut = unsafe { aw_ptr.slice_mut(row_start, seq_k) };
        T::scale_slice(row_mut, inv_sum);

        // 3. output[b, i, l] = sum_j attn[b, i, j] * V[b, j, l]
        for l in 0..d_v {
            let mut acc = T::zero();
            for j in 0..seq_k {
                let aw_idx = idx3(b, i, j, seq_q, seq_k);
                let v_idx = idx3(b, j, l, seq_k, d_v);
                let aw_val = unsafe { aw_ptr.read(aw_idx) };
                let v_val = unsafe { v_ptr.read(v_idx) };
                acc = acc + aw_val * v_val;
            }
            unsafe {
                out_ptr.write(idx3(b, i, l, seq_q, d_v), acc);
            }
        }
    });
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
pub(crate) fn sdp_attention_backward<T: Float, B: Backend>(
    backend: &B,
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

    let go_ptr = Ptr(go.as_ptr());
    let aw_ptr = Ptr(aw.as_ptr());
    let q_ptr = Ptr(q_sl.as_ptr());
    let k_ptr = Ptr(k_sl.as_ptr());
    let v_ptr = Ptr(v_sl.as_ptr());

    let gq_ptr = grad_q
        .as_mut()
        .map(|gq| MutPtr(gq.as_mut_slice().as_mut_ptr()));
    let gk_ptr = grad_k
        .as_mut()
        .map(|gk| MutPtr(gk.as_mut_slice().as_mut_ptr()));
    let gv_ptr = grad_v
        .as_mut()
        .map(|gv| MutPtr(gv.as_mut_slice().as_mut_ptr()));

    #[inline(always)]
    fn idx3(b: usize, i: usize, j: usize, dim1: usize, dim2: usize) -> usize {
        b * dim1 * dim2 + i * dim2 + j
    }

    // 1. Allocate a temporary d_scores buffer of size batch * seq_q * seq_k.
    let d_scores_numel = batch * seq_q * seq_k;
    let mut d_scores = vec![T::zero(); d_scores_numel];
    let d_scores_ptr = MutPtr(d_scores.as_mut_ptr());

    let num_tasks_q = batch * seq_q;

    backend.parallel_for(0, num_tasks_q, move |index| {
        let b = index / seq_q;
        let i = index % seq_q;

        // a. Compute d_attn_row of size seq_k.
        let mut d_attn_row = vec![T::zero(); seq_k];
        for j in 0..seq_k {
            let go_start = idx3(b, i, 0, seq_q, d_v);
            let value_start = idx3(b, j, 0, seq_k, d_v);
            let go_window = unsafe { go_ptr.slice(go_start, d_v) };
            let v_window = unsafe { v_ptr.slice(value_start, d_v) };
            d_attn_row[j] = T::dot_slice(go_window, v_window);
        }

        // b. Compute rs = dot_slice(A[b, i, :], d_attn_row)
        let aw_start = idx3(b, i, 0, seq_q, seq_k);
        let aw_window = unsafe { aw_ptr.slice(aw_start, seq_k) };
        let rs = T::dot_slice(aw_window, &d_attn_row);

        // c. Fill d_scores for this row.
        for j in 0..seq_k {
            let aw_idx = idx3(b, i, j, seq_q, seq_k);
            let aw_val = unsafe { aw_ptr.read(aw_idx) };
            let val = aw_val * (d_attn_row[j] - rs);
            unsafe {
                d_scores_ptr.write(aw_idx, val);
            }
        }

        // d. Accumulate into dQ if present:
        // dQ[b, i, dk] += sum_j dS[b, i, j] * K[b, j, dk] * scale
        if let Some(ref gp) = gq_ptr {
            for dk in 0..d_k {
                let mut acc = T::zero();
                for j in 0..seq_k {
                    let ds_idx = idx3(b, i, j, seq_q, seq_k);
                    let k_idx = idx3(b, j, dk, seq_k, d_k);
                    let ds_val = unsafe { d_scores_ptr.read(ds_idx) };
                    let k_val = unsafe { k_ptr.read(k_idx) };
                    acc = acc + ds_val * k_val;
                }
                let gq_idx = idx3(b, i, dk, seq_q, d_k);
                unsafe {
                    let old = gp.read(gq_idx);
                    gp.write(gq_idx, old + acc * scale);
                }
            }
        }
    });

    let num_tasks_k = batch * seq_k;
    let d_scores_const_ptr = Ptr(d_scores.as_ptr());

    backend.parallel_for(0, num_tasks_k, move |index| {
        let b = index / seq_k;
        let j = index % seq_k;

        // a. Accumulate into dK:
        // dK[b, j, dk] += sum_i dS[b, i, j] * Q[b, i, dk] * scale
        if let Some(ref gp) = gk_ptr {
            for dk in 0..d_k {
                let mut acc = T::zero();
                for i in 0..seq_q {
                    let ds_idx = idx3(b, i, j, seq_q, seq_k);
                    let q_idx = idx3(b, i, dk, seq_q, d_k);
                    let ds_val = unsafe { d_scores_const_ptr.read(ds_idx) };
                    let q_val = unsafe { q_ptr.read(q_idx) };
                    acc = acc + ds_val * q_val;
                }
                let gk_idx = idx3(b, j, dk, seq_k, d_k);
                unsafe {
                    let old = gp.read(gk_idx);
                    gp.write(gk_idx, old + acc * scale);
                }
            }
        }

        // b. Accumulate into dV:
        // dV[b, j, l] += sum_i A[b, i, j] * dO[b, i, l]
        if let Some(ref gp) = gv_ptr {
            for l in 0..d_v {
                let mut acc = T::zero();
                for i in 0..seq_q {
                    let aw_idx = idx3(b, i, j, seq_q, seq_k);
                    let go_idx = idx3(b, i, l, seq_q, d_v);
                    let aw_val = unsafe { aw_ptr.read(aw_idx) };
                    let go_val = unsafe { go_ptr.read(go_idx) };
                    acc = acc + aw_val * go_val;
                }
                let gv_idx = idx3(b, j, l, seq_k, d_v);
                unsafe {
                    let old = gp.read(gv_idx);
                    gp.write(gv_idx, old + acc);
                }
            }
        }
    });
}
