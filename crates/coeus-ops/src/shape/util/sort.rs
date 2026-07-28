// ── sort — sort a tensor along an axis ──
//
// Returns (sorted_values, argsort_indices) where:
//   sorted_values[..axis_i..] = input values sorted ascending (or descending)
//   argsort_indices[..axis_i..] = original integer positions before sorting
//
// The sort is stable (preserves relative order of equal elements).

use crate::backend_ops::BackendOps;
use coeus_core::{BackendError, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Stably sorts `input` along `axis`, like PyTorch `torch.sort` or NumPy `sort`.
///
/// Returns `(sorted, indices)`, where `indices` stores the original axis
/// positions encoded as `T`.
///
/// # Errors
/// Returns a backend error when `axis` is out of range or materialization
/// fails.
#[inline]
pub fn sort<T: Scalar + PartialOrd, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    axis: usize,
    descending: bool,
    backend: &B,
) -> Result<(Tensor<T, B>, Tensor<T, B>), B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = input.ndim();
    if axis >= ndim {
        return Err(B::Error::from(BackendError::AxisOutOfRange {
            operation: "sort",
            axis,
            rank: ndim,
        }));
    }

    let shape = input.shape().to_vec();
    let axis_len = shape[axis];

    // Outer dimensions: product of all dims except `axis`
    let product = |extents: &[usize], reason: &'static str| {
        extents.iter().try_fold(1usize, |count, &extent| {
            count.checked_mul(extent).ok_or_else(|| {
                B::Error::from(BackendError::Overflow {
                    operation: "sort",
                    reason,
                })
            })
        })
    };
    let outer = product(&shape[..axis], "outer element count")?;
    let inner = product(&shape[axis + 1..], "inner element count")?;
    let numel = product(&shape, "element count")?;

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
                if descending { ord.reverse() } else { ord }
            });

            // Write back.
            for (a, (val, orig_idx)) in pairs.into_iter().enumerate() {
                let flat = o * (axis_len * inner) + a * inner + i;
                out_vals[flat] = val;
                out_idx[flat] = T::from_usize(orig_idx);
            }
        }
    }

    Ok((
        Tensor::from_slice_on(shape.clone(), &out_vals, backend)?,
        Tensor::from_slice_on(shape, &out_idx, backend)?,
    ))
}
