// ── Split ──
// Splits a tensor into multiple tensors along a given dimension.

use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Split `x` into chunks of size `chunk_size` along `dim`.
///
/// The last chunk may be smaller if `x.shape()[dim]` is not divisible.
///
/// # Panics
/// - `chunk_size` is zero.
/// - `dim` is out of range.
#[inline]
pub fn split<T: Scalar, B: ComputeBackend + Default>(
    x: &Tensor<T, B>,
    chunk_size: usize,
    dim: usize,
) -> Vec<Tensor<T, B>>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert!(chunk_size > 0, "split: chunk_size must be > 0");
    let ndim = x.ndim();
    assert!(
        dim < ndim,
        "split: dim {dim} out of range for {ndim}D tensor"
    );

    let backend = B::default();
    let dim_size = x.shape()[dim];
    let sizes: Vec<_> = (0..dim_size)
        .step_by(chunk_size)
        .map(|start| (dim_size - start).min(chunk_size))
        .collect();
    let values = coeus_leto::split_values(x.layout(), x.storage().as_slice(), dim, &sizes)
        .expect("coeus-leto split failed");

    values
        .iter()
        .zip(sizes)
        .map(|(chunk, chunk_dim)| {
            let mut out_shape = x.shape_cloned();
            out_shape[dim] = chunk_dim;
            Tensor::from_slice_on(out_shape, chunk, &backend)
        })
        .collect()
}
