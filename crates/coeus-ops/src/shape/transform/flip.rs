// ── flip — reverse a tensor along an axis ──

use crate::backend_ops::BackendOps;
use coeus_core::{BackendError, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Reverse the elements of `input` along `axis`.
///
/// The returned tensor is a contiguous materialization in row-major order.
///
/// # Errors
/// Returns a backend error when `axis` is out of range or materialization fails.
#[inline]
pub fn flip<T: Scalar, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    axis: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = input.ndim();
    if axis >= ndim {
        return Err(B::Error::from(BackendError::AxisOutOfRange {
            operation: "flip",
            axis,
            rank: ndim,
        }));
    }

    let shape = input.shape();
    let n = shape[axis];

    // Build output by reading in reversed order along `axis`.
    // We materialise into a contiguous output rather than returning a
    // stride-negative view (Rust doesn't support negative strides in safe
    // slice indexing; the contiguous copy is O(numel) and cache-local).
    let numel: usize = shape.iter().product();
    let out_vec: Vec<T> = (0..numel)
        .map(|flat| {
            // Convert flat index → multi-dim, flip axis, → read from input.
            let mut idx = crate::shape::flat_to_nd(flat, shape);
            idx[axis] = n - 1 - idx[axis];
            input.get(&idx)
        })
        .collect();

    Tensor::from_slice_on(shape.to_vec(), &out_vec, backend)
}
