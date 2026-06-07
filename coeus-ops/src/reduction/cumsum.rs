// ── Cumulative sum ──

use coeus_core::{Scalar, ComputeBackend, Layout, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_tensor::Tensor;

/// Compute the inclusive cumulative sum of `x` along `dim`.
///
/// Output has the same shape as `x`.
///
/// # Panics
/// - `dim` is out of range.
#[inline]
pub fn cumsum<T: Scalar, B: ComputeBackend + Default>(
    x: &Tensor<T, B>,
    dim: usize,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = x.ndim();
    assert!(dim < ndim, "cumsum: dim {dim} out of range for {ndim}D tensor");

    let backend = B::default();
    let shape = x.shape_cloned();
    let strides = Layout::new(shape.clone()).strides_cloned();
    let x_cont = x.to_contiguous_on(&backend);
    let x_slice = x_cont.as_slice();

    let numel = x_cont.numel();
    let mut out_data = vec![T::zero(); numel];
    let dim_size = shape[dim];

    // Outer loop: all positions except the cumsum dimension.
    // For each "slice" along dim, compute a prefix sum.
    let outer: usize = numel / dim_size;

    for outer_idx in 0..outer {
        // Map outer_idx to a coordinate vector excluding the cumsum dim.
        let mut coords = vec![0usize; ndim];
        let mut rem = outer_idx;
        for d in (0..ndim).rev() {
            if d == dim {
                continue;
            }
            coords[d] = rem % shape[d];
            rem /= shape[d];
        }

        // Compute the base flat index for this slice.
        let base: usize = coords.iter().enumerate().map(|(d, &c)| {
            if d != dim { c * strides[d] } else { 0 }
        }).sum();

        let mut acc = T::zero();
        for k in 0..dim_size {
            let flat_in = base + k * strides[dim];
            acc = acc + x_slice[flat_in];
            out_data[flat_in] = acc;
        }
    }

    Tensor::from_slice_on(shape, &out_data, &backend)
}

/// Compute the inclusive cumulative suffix sum (reverse cumulative sum) of `x` along `dim`.
///
/// Output has the same shape as `x`.
///
/// # Panics
/// - `dim` is out of range.
#[inline]
pub fn suffix_sum<T: Scalar, B: ComputeBackend + Default>(
    x: &Tensor<T, B>,
    dim: usize,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = x.ndim();
    assert!(dim < ndim, "suffix_sum: dim {dim} out of range for {ndim}D tensor");

    let backend = B::default();
    let shape = x.shape_cloned();
    let strides = Layout::new(shape.clone()).strides_cloned();
    let x_cont = x.to_contiguous_on(&backend);
    let x_slice = x_cont.as_slice();

    let numel = x_cont.numel();
    let mut out_data = vec![T::zero(); numel];
    let dim_size = shape[dim];

    let outer: usize = numel / dim_size;

    for outer_idx in 0..outer {
        let mut coords = vec![0usize; ndim];
        let mut rem = outer_idx;
        for d in (0..ndim).rev() {
            if d == dim {
                continue;
            }
            coords[d] = rem % shape[d];
            rem /= shape[d];
        }

        let base: usize = coords.iter().enumerate().map(|(d, &c)| {
            if d != dim { c * strides[d] } else { 0 }
        }).sum();

        let mut acc = T::zero();
        for k in (0..dim_size).rev() {
            let flat_in = base + k * strides[dim];
            acc = acc + x_slice[flat_in];
            out_data[flat_in] = acc;
        }
    }

    Tensor::from_slice_on(shape, &out_data, &backend)
}

