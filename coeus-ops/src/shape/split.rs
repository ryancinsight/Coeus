// ── Split ──
// Splits a tensor into multiple tensors along a given dimension.

use coeus_core::{Scalar, ComputeBackend, Layout, CpuAddressableStorage, CpuAddressableStorageMut};
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
    assert!(dim < ndim, "split: dim {dim} out of range for {ndim}D tensor");

    let backend = B::default();
    let dim_size = x.shape()[dim];
    let x_cont = x.to_contiguous_on(&backend);
    let x_slice = x_cont.as_slice();
    let x_strides = Layout::new(x_cont.shape_cloned()).strides_cloned();

    let mut results = Vec::new();
    let mut start = 0;
    while start < dim_size {
        let end = (start + chunk_size).min(dim_size);
        let chunk_dim = end - start;

        let mut out_shape = x_cont.shape_cloned();
        out_shape[dim] = chunk_dim;

        let numel_out: usize = out_shape.iter().product();
        let mut out_data = vec![T::zero(); numel_out];

        // Compute strides for the output shape.
        let out_strides = Layout::new(out_shape.clone()).strides_cloned();

        for flat_out in 0..numel_out {
            let mut rem = flat_out;
            let mut src_phys = 0usize;
            for d in (0..ndim).rev() {
                let coord = rem % out_shape[d];
                rem /= out_shape[d];
                let src_coord = if d == dim { coord + start } else { coord };
                src_phys += src_coord * x_strides[d];
            }
            let _ = out_strides[0]; // ensure computed
            out_data[flat_out] = x_slice[src_phys];
        }

        results.push(Tensor::from_slice_on(out_shape, &out_data, &backend));
        start = end;
    }
    results
}
