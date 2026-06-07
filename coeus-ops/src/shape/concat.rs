// ── Concatenation ──
// Concatenates a list of tensors along a given dimension.
// Zero-copy when a single input is given (returns a clone of the storage view).

use coeus_core::{Scalar, ComputeBackend, Layout, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_tensor::Tensor;

/// Concatenate `tensors` along `dim`.
///
/// All tensors must have the same shape in every dimension except `dim`.
///
/// # Panics
/// - `tensors` is empty.
/// - Any tensor has mismatched shape on a non-cat dimension.
/// - `dim` is out of range for any tensor.
#[inline]
pub fn cat<T: Scalar, B: ComputeBackend + Default>(
    tensors: &[&Tensor<T, B>],
    dim: usize,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert!(!tensors.is_empty(), "cat: input list is empty");
    // Fast path: single input.
    if tensors.len() == 1 {
        return tensors[0].clone();
    }

    let backend = B::default();
    let ndim = tensors[0].ndim();
    assert!(dim < ndim, "cat: dim {dim} out of range for {ndim}D tensor");

    // Validate shapes and compute output dim size.
    let mut out_shape = tensors[0].shape_cloned();
    let out_dim_size = tensors.iter().map(|t| {
        assert_eq!(t.ndim(), ndim, "cat: all tensors must have the same rank");
        for d in 0..ndim {
            if d != dim {
                assert_eq!(
                    t.shape()[d], out_shape[d],
                    "cat: shape mismatch at dimension {d}"
                );
            }
        }
        t.shape()[dim]
    }).sum::<usize>();
    out_shape[dim] = out_dim_size;

    let mut out = Tensor::zeros_on(out_shape.clone(), &backend);

    // Copy each tensor slice into the output.
    let mut offset = 0usize;
    for t in tensors {
        let t_dim_size = t.shape()[dim];
        // Build slice ranges: full range for all dims except cat dim.
        let ranges: Vec<(usize, usize)> = out_shape
            .iter()
            .enumerate()
            .map(|(d, &s)| if d == dim { (offset, offset + t_dim_size) } else { (0, s) })
            .collect();
        // Write into slice view of output.
        let src = t.to_contiguous_on(&backend);
        let src_slice = src.as_slice();
        {
            // We need mutable access to the output storage; collect destination indices.
            let numel_src = src_slice.len();
            let dst_numel = numel_src;
            // Compute destination physical offsets using the output layout.
            let out_strides = Layout::new(out_shape.clone()).strides_cloned();
            let out_raw = out.as_mut_slice();
            // Iterate over logical indices of the source tensor and write to output.
            let src_shape: Vec<usize> = (0..ndim)
                .map(|d| if d == dim { t_dim_size } else { out_shape[d] })
                .collect();
            let _ = (dst_numel, ranges.as_slice()); // suppress unused warnings
            for flat in 0..numel_src {
                let mut rem = flat;
                let mut dst_phys = 0usize;
                for d in (0..ndim).rev() {
                    let coord = rem % src_shape[d];
                    rem /= src_shape[d];
                    let out_coord = if d == dim { coord + offset } else { coord };
                    dst_phys += out_coord * out_strides[d];
                }
                out_raw[dst_phys] = src_slice[flat];
            }
        }
        offset += t_dim_size;
    }
    out
}
