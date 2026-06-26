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
    let mut out_data = vec![T::zero(); out_numel];

    // Compute strides for index/output (row-major).
    let mut idx_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        idx_strides[d] = idx_strides[d + 1] * idx_shape[d + 1];
    }

    for flat in 0..out_numel {
        // Decode flat index into multi-dim coordinates.
        let mut coords = vec![0usize; ndim];
        let mut rem = flat;
        for d in 0..ndim {
            coords[d] = rem / idx_strides[d];
            rem %= idx_strides[d];
        }

        // Look up the gather index (stored as T, cast to usize).
        let gather_idx = idx_s[flat].to_f64() as usize;
        assert!(
            gather_idx < in_shape[dim],
            "gather: index {gather_idx} out of bounds for dim {dim} size {}",
            in_shape[dim]
        );

        // Compute the input flat offset.
        let mut in_flat = 0usize;
        for d in 0..ndim {
            let c = if d == dim { gather_idx } else { coords[d] };
            in_flat += c * in_strides[d];
        }

        out_data[flat] = in_s[in_flat];
    }

    Tensor::from_slice(idx_shape.to_vec(), &out_data)
}
