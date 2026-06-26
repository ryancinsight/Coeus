// ── flip — reverse a tensor along an axis ──

use crate::backend_ops::BackendOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Reverse the elements of `input` along `axis`.
///
/// The returned tensor is a contiguous materialization in row-major order.
///
/// # Panics
/// Panics if `axis >= input.ndim()`.
#[inline]
pub fn flip<T: Scalar, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    axis: usize,
    backend: &B,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = input.ndim();
    assert!(
        axis < ndim,
        "flip: axis {axis} out of range for {ndim}-D tensor"
    );

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

    let _ = backend;
    Tensor::from_slice(shape.to_vec(), &out_vec)
}

