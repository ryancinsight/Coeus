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

    // Coordinate decode is fused into the offset accumulation, so no
    // per-element coordinate buffer exists; collecting rather than filling a
    // zeroed vector drops an initialising pass over the output. At `dim` the
    // decoded coordinate maps back to the input by integer division, since
    // element i repeats at positions [i*repeats, .., i*repeats+repeats-1].
    let out_data: Vec<T> = (0..out_numel)
        .map(|flat_out| {
            let mut rem = flat_out;
            let mut in_flat = 0usize;
            for d in 0..ndim {
                let coord = rem / out_strides[d];
                rem %= out_strides[d];
                let c = if d == dim { coord / repeats } else { coord };
                in_flat += c * in_strides[d];
            }
            in_s[in_flat]
        })
        .collect();

    Tensor::from_slice(out_shape, &out_data)
}
