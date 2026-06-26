// ── repeat_interleave — repeat each element along a dimension ──
//
// `repeat_interleave(input, repeats, dim)` repeats each element `repeats`
// times along `dim`, interleaving the copies:
//
//   input  = [[1, 2], [3, 4]]  (shape [2,2])
//   output = [[1, 1, 2, 2], [3, 3, 4, 4]]  (shape [2,4]) for dim=1, repeats=2
//
// This matches `torch.repeat_interleave(input, repeats, dim)`.

use crate::backend_ops::BackendOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Repeat each slice of `input` along `dim` exactly `repeats` times,
/// interleaving the copies.
///
/// # Panics
/// - `dim` must be < `input.ndim()`.
/// - `repeats` must be > 0.
#[inline]
pub fn repeat_interleave<T: Scalar, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    repeats: usize,
    dim: usize,
    _backend: &B,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert!(repeats > 0, "repeat_interleave: repeats must be > 0");
    let ndim = input.ndim();
    assert!(
        dim < ndim,
        "repeat_interleave: dim {dim} out of range for {ndim}-D tensor"
    );

    let in_shape = input.shape();
    let mut out_shape = in_shape.to_vec();
    out_shape[dim] *= repeats;

    let in_cont = input.to_contiguous();
    let in_s = in_cont.as_slice();

    // Compute strides for input (row-major).
    let mut in_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
    }

    // Compute strides for output (row-major).
    let mut out_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
    }

    let out_numel: usize = out_shape.iter().product();
    let mut out_data = vec![T::zero(); out_numel];

    for flat_out in 0..out_numel {
        // Decode flat output index into multi-dim coords.
        let mut coords = vec![0usize; ndim];
        let mut rem = flat_out;
        for d in 0..ndim {
            coords[d] = rem / out_strides[d];
            rem %= out_strides[d];
        }

        // Map dim coordinate back to input: element i repeats at positions
        // [i*repeats, i*repeats+1, ..., i*repeats+repeats-1].
        let in_dim_coord = coords[dim] / repeats;

        let mut in_flat = 0usize;
        for d in 0..ndim {
            let c = if d == dim { in_dim_coord } else { coords[d] };
            in_flat += c * in_strides[d];
        }

        out_data[flat_out] = in_s[in_flat];
    }

    Tensor::from_slice(out_shape, &out_data)
}
