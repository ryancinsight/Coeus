// ── Cumulative sum ──

use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Compute the inclusive cumulative sum of `x` along `dim`.
///
/// Output has the same shape as `x`.
///
/// # Panics
/// - `dim` is out of range.
#[inline]
pub fn cumsum<T: Scalar + leto_ops::Scalar, B: ComputeBackend + Default>(
    x: &Tensor<T, B>,
    dim: usize,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = x.ndim();
    assert!(
        dim < ndim,
        "cumsum: dim {dim} out of range for {ndim}D tensor"
    );

    let backend = B::default();
    let shape = x.shape_cloned();
    let mut out = Tensor::zeros_on(shape, &backend);
    let (out_storage, out_layout) = out.storage_mut_and_layout();
    coeus_leto::cumsum_into(
        x.layout(),
        x.storage().as_slice(),
        dim,
        out_layout,
        out_storage.as_mut_slice(),
    )
    .expect("coeus-leto cumsum failed");
    out
}

/// Compute the inclusive cumulative suffix sum (reverse cumulative sum) of `x` along `dim`.
///
/// Output has the same shape as `x`.
///
/// # Panics
/// - `dim` is out of range.
#[inline]
pub fn suffix_sum<T: Scalar + leto_ops::Scalar, B: ComputeBackend + Default>(
    x: &Tensor<T, B>,
    dim: usize,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = x.ndim();
    assert!(
        dim < ndim,
        "suffix_sum: dim {dim} out of range for {ndim}D tensor"
    );

    let backend = B::default();
    let shape = x.shape_cloned();
    let mut out = Tensor::zeros_on(shape, &backend);
    let (out_storage, out_layout) = out.storage_mut_and_layout();
    coeus_leto::suffix_sum_into(
        x.layout(),
        x.storage().as_slice(),
        dim,
        out_layout,
        out_storage.as_mut_slice(),
    )
    .expect("coeus-leto suffix_sum failed");
    out
}
