// ── scatter_add — accumulate source values into output at given indices ──
//
// `scatter_add(input, dim, index, src)` builds an output tensor equal to
// `input` with `src` values accumulated at positions specified by `index`:
//
//   out[i0, …, index[i0, …, id, …, iN], …, iN] += src[i0, …, id, …, iN]
//
// This is the exact semantic of `torch.scatter_add(input, dim, index, src)`.
//
// Shape contract:
// - `input`, `index`, and `src` must have the same number of dimensions.
// - `index` and `src` must have the same shape.
// - Every dimension of `index` must be ≤ the corresponding dimension of `src`.
//
// `scatter_add` is the **backward operator** of `gather`: if `out = gather(x, dim, idx)`,
// then `dx = scatter_add(zeros_like(x), dim, idx, grad_out)`.

use crate::backend_ops::BackendOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Scatter-accumulate: `out = input` then `out[…, index[…,k,…], …] += src[…,k,…]`.
///
/// Returns a new tensor (does not mutate `input`).
///
/// # Panics
/// - ndim mismatch between `input`, `index`, `src`.
/// - `index` and `src` shapes must match.
/// - `dim` out of range.
/// - Any index value ≥ `input.shape()[dim]`.
#[inline]
pub fn scatter_add<T: Scalar, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    dim: usize,
    index: &Tensor<T, B>,
    src: &Tensor<T, B>,
    _backend: &B,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = input.ndim();
    assert_eq!(ndim, index.ndim(), "scatter_add: input/index ndim mismatch");
    assert_eq!(ndim, src.ndim(), "scatter_add: input/src ndim mismatch");
    assert_eq!(
        index.shape(),
        src.shape(),
        "scatter_add: index and src shapes must match"
    );
    assert!(
        dim < ndim,
        "scatter_add: dim {dim} out of range for {ndim}-D tensor"
    );

    let out_shape = input.shape().to_vec();
    let idx_shape = index.shape().to_vec();

    let in_cont = input.to_contiguous();
    let idx_cont = index.to_contiguous();
    let src_cont = src.to_contiguous();

    let in_s = in_cont.as_slice();
    let idx_s = idx_cont.as_slice();
    let src_s = src_cont.as_slice();

    // Zero-copy fast path: if src contributes no updates, scatter_add is identity.
    if src_s.iter().all(|v| <T as Scalar>::to_f64(*v) == 0.0) {
        return input.to_contiguous();
    }

    // Start from a copy of input.
    let mut out_data = in_s.to_vec();

    // Compute strides for output (row-major).
    let mut out_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
    }

    // Compute strides for index/src (row-major).
    let idx_numel: usize = idx_shape.iter().product();
    let mut idx_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        idx_strides[d] = idx_strides[d + 1] * idx_shape[d + 1];
    }

    for flat in 0..idx_numel {
        // Decode flat index into multi-dim coordinates.
        let mut coords = vec![0usize; ndim];
        let mut rem = flat;
        for d in 0..ndim {
            coords[d] = rem / idx_strides[d];
            rem %= idx_strides[d];
        }

        let scatter_idx = <T as Scalar>::to_f64(idx_s[flat]) as usize;
        assert!(
            scatter_idx < out_shape[dim],
            "scatter_add: index {scatter_idx} out of bounds for dim {dim} size {}",
            out_shape[dim]
        );

        // Compute the output flat offset.
        let mut out_flat = 0usize;
        for d in 0..ndim {
            let c = if d == dim { scatter_idx } else { coords[d] };
            out_flat += c * out_strides[d];
        }

        out_data[out_flat] = out_data[out_flat].add(src_s[flat]);
    }

    Tensor::from_slice(out_shape, &out_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn scatter_add_zero_src_returns_shared_storage() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![4], &[1.0f32, 2.0, 3.0, 4.0]);
        let idx = Tensor::from_slice(vec![2], &[1.0f32, 3.0]);
        let src = Tensor::from_slice(vec![2], &[0.0f32, 0.0]);
        let out = scatter_add(&x, 0, &idx, &src, &b);
        assert_eq!(out.shape(), &[4]);
        assert_eq!(out.as_slice(), x.as_slice());
        assert_eq!(out.as_slice().as_ptr(), x.as_slice().as_ptr());
    }
}
