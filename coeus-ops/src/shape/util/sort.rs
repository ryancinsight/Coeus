// ── sort — sort a tensor along an axis ──
//
// Returns (sorted_values, argsort_indices) where:
//   sorted_values[..axis_i..] = input values sorted ascending (or descending)
//   argsort_indices[..axis_i..] = original integer positions before sorting
//
// The sort is stable (preserves relative order of equal elements).

use crate::backend_ops::BackendOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Stably sorts `input` along `axis`, like PyTorch `torch.sort` or NumPy `sort`.
///
/// Returns `(sorted, indices)`, where `indices` stores the original axis
/// positions encoded as `T`.
///
/// # Panics
/// Panics if `axis >= input.ndim()`.
#[inline]
pub fn sort<T: Scalar + PartialOrd, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    axis: usize,
    descending: bool,
    backend: &B,
) -> (Tensor<T, B>, Tensor<T, B>)
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = input.ndim();
    assert!(
        axis < ndim,
        "sort: axis {axis} out of range for {ndim}-D tensor"
    );

    let shape = input.shape().to_vec();
    let axis_len = shape[axis];

    // Outer dimensions: product of all dims except `axis`
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let numel: usize = shape.iter().product();

    let mut out_vals = vec![T::zero(); numel];
    let mut out_idx = vec![T::zero(); numel];

    // Iterate over every (outer, inner) slice and sort along `axis`.
    for o in 0..outer {
        for i in 0..inner {
            // Collect the axis slice.
            let mut pairs: Vec<(T, usize)> = (0..axis_len)
                .map(|a| {
                    // physical flat index for [outer_dims..., a, inner_dims...]
                    let flat = o * (axis_len * inner) + a * inner + i;
                    let val = input.get(&crate::shape::flat_to_nd(flat, &shape));
                    (val, a)
                })
                .collect();

            // Stable sort.
            pairs.sort_by(|(va, _), (vb, _)| {
                let ord = va.partial_cmp(vb).unwrap_or(std::cmp::Ordering::Equal);
                if descending {
                    ord.reverse()
                } else {
                    ord
                }
            });

            // Write back.
            for (a, (val, orig_idx)) in pairs.into_iter().enumerate() {
                let flat = o * (axis_len * inner) + a * inner + i;
                out_vals[flat] = val;
                out_idx[flat] = T::from_f64(orig_idx as f64);
            }
        }
    }

    let _ = backend;
    (
        Tensor::from_slice(shape.clone(), &out_vals),
        Tensor::from_slice(shape, &out_idx),
    )
}
