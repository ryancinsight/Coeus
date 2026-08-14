// ── gather — index-based element selection along one axis ──
//
// `gather(input, dim, index)` is the element-wise index look-up:
//
//   out[i0, …, id, …, iN] = input[i0, …, index[i0, …, id, …, iN], …, iN]
//
// where `id` is the coordinate along `dim`.
//
// Shape contract:
// - `input` and `index` must have the **same number of dimensions**.
// - Every dimension other than `dim` must have the same size in `input` and `index`.
// - The `dim`-dimension of `index` may be any size k (≥ 1).
// - Output has the same shape as `index`.
//
// This is the exact semantic of `torch.gather(input, dim, index)`.

use crate::backend_ops::BackendOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Index-based element selection along `dim`.
///
/// Returns a tensor with the same shape as `index` where
/// `out[…, k, …] = input[…, index[…, k, …], …]` at position `dim`.
///
/// # Panics
/// - `input` and `index` must have the same number of dimensions.
/// - Every non-dim dimension must match between `input` and `index`.
/// - `dim` must be < `input.ndim()`.
#[inline]
pub fn gather<T: Scalar, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    dim: usize,
    index: &Tensor<T, B>,
    _backend: &B,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = input.ndim();
    assert_eq!(
        ndim,
        index.ndim(),
        "gather: input and index must have the same ndim"
    );
    assert!(
        dim < ndim,
        "gather: dim {dim} out of range for {ndim}-D tensor"
    );

    let in_shape = input.shape();
    let idx_shape = index.shape();
    for d in 0..ndim {
        if d != dim {
            assert_eq!(
                in_shape[d], idx_shape[d],
                "gather: shape mismatch at dim {d}: input={} index={}",
                in_shape[d], idx_shape[d]
            );
        }
    }

    // Zero-copy fast path: gather with an identity index returns the input.
    // This preserves COW semantics and avoids allocating/initializing output.
    if idx_shape == in_shape {
        let idx_cont = index.to_contiguous();
        let idx_s = idx_cont.as_slice();
        let mut idx_strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            idx_strides[d] = idx_strides[d + 1] * idx_shape[d + 1];
        }
        let idx_numel: usize = idx_shape.iter().product();
        let mut is_identity = true;
        for flat in 0..idx_numel {
            let coord_dim = (flat / idx_strides[dim]) % idx_shape[dim];
            if (<T as Scalar>::to_f64(idx_s[flat]) as usize) != coord_dim {
                is_identity = false;
                break;
            }
        }
        if is_identity {
            return input.to_contiguous();
        }
    }

    let in_cont = input.to_contiguous();
    let idx_cont = index.to_contiguous();
    let in_s = in_cont.as_slice();
    let idx_s = idx_cont.as_slice();

    // Compute strides for input (row-major).
    let mut in_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
    }

    let out_numel: usize = idx_shape.iter().product();

    // Compute strides for index/output (row-major).
    let mut idx_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        idx_strides[d] = idx_strides[d + 1] * idx_shape[d + 1];
    }

    // Coordinate decode is fused into the offset accumulation: each coordinate
    // is consumed by the same iteration that produces it, so no per-element
    // coordinate buffer exists. Collecting rather than filling a zeroed vector
    // also drops a full initialising pass over the output.
    let out_data: Vec<T> = (0..out_numel)
        .map(|flat| {
            // Look up the gather index (stored as T, cast to usize).
            let gather_idx = <T as Scalar>::to_f64(idx_s[flat]) as usize;
            assert!(
                gather_idx < in_shape[dim],
                "gather: index {gather_idx} out of bounds for dim {dim} size {}",
                in_shape[dim]
            );

            // Decode `flat` against the index strides and accumulate the input
            // offset in one pass; `dim` takes the gathered coordinate instead.
            let mut rem = flat;
            let mut in_flat = 0usize;
            for d in 0..ndim {
                let coord = rem / idx_strides[d];
                rem %= idx_strides[d];
                let c = if d == dim { gather_idx } else { coord };
                in_flat += c * in_strides[d];
            }

            in_s[in_flat]
        })
        .collect();

    Tensor::from_slice(idx_shape.to_vec(), &out_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn gather_identity_returns_shared_storage() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let idx = Tensor::from_slice(vec![2, 3], &[0.0f32, 1.0, 2.0, 0.0, 1.0, 2.0]);
        let out = gather(&x, 1, &idx, &b);
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.as_slice(), x.as_slice());
        assert_eq!(out.as_slice().as_ptr(), x.as_slice().as_ptr());
    }
}
