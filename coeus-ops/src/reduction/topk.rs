// ── TopK and ArgMax/ArgMin ──
// Returns the k largest (or smallest) values and their indices along a dimension.

use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};
use coeus_tensor::Tensor;

/// Return the `k` largest (or smallest) values and their flat indices along `dim`.
///
/// Returns `(values, indices)` both with shape equal to `x.shape()` but with
/// `shape[dim] == k`.
///
/// # Panics
/// - `k == 0` or `k > x.shape()[dim]`.
/// - `dim` out of range.
#[inline]
pub fn topk<T: Scalar, B: ComputeBackend + Default>(
    x: &Tensor<T, B>,
    k: usize,
    dim: usize,
    largest: bool,
) -> (Tensor<T, B>, Tensor<i64, B>)
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorageMut<i64>,
{
    let ndim = x.ndim();
    let dim_size = x.shape()[dim];
    assert!(
        k > 0 && k <= dim_size,
        "topk: k={k} invalid for dim_size={dim_size}"
    );
    assert!(dim < ndim, "topk: dim {dim} out of range");

    let backend = B::default();
    let in_shape = x.shape_cloned();
    let in_strides = Layout::new(in_shape.clone()).strides_cloned();
    let x_cont = x.to_contiguous_on(&backend);
    let x_slice = x_cont.as_slice();

    let mut out_shape = in_shape.clone();
    out_shape[dim] = k;
    let out_numel: usize = out_shape.iter().product();
    let outer = out_numel / k;

    let mut val_data = vec![T::zero(); out_numel];
    let mut idx_data = vec![0i64; out_numel];

    for outer_idx in 0..outer {
        // Map outer_idx to coordinates excluding dim.
        let mut coords = vec![0usize; ndim];
        let mut rem = outer_idx;
        for d in (0..ndim).rev() {
            if d == dim {
                continue;
            }
            coords[d] = rem % in_shape[d];
            rem /= in_shape[d];
        }
        let base: usize = coords
            .iter()
            .enumerate()
            .map(|(d, &c)| if d != dim { c * in_strides[d] } else { 0 })
            .sum();

        // Collect (value, index) pairs for this slice.
        let mut pairs: Vec<(T, usize)> = (0..dim_size)
            .map(|k_idx| (x_slice[base + k_idx * in_strides[dim]], k_idx))
            .collect();

        // Partial sort: select k elements.
        if largest {
            pairs.select_nth_unstable_by(k - 1, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            pairs.select_nth_unstable_by(k - 1, |a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        pairs.truncate(k);
        if largest {
            pairs.sort_unstable_by(|a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            pairs.sort_unstable_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        for (rank, (v, orig_idx)) in pairs.iter().enumerate() {
            let flat_out = outer_idx * k + rank;
            val_data[flat_out] = *v;
            idx_data[flat_out] = *orig_idx as i64;
        }
    }

    let val_tensor = Tensor::from_slice_on(out_shape.clone(), &val_data, &backend);
    let idx_backend = B::default();
    let idx_tensor = Tensor::from_slice_on(out_shape, &idx_data, &idx_backend);
    (val_tensor, idx_tensor)
}

/// Argmax along `dim`: returns indices of maximum values, shape `x.shape()[dim] = 1`.
#[inline]
pub fn argmax<T: Scalar + leto_ops::Scalar, B: ComputeBackend + Default>(
    x: &Tensor<T, B>,
    dim: usize,
) -> Tensor<i64, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorageMut<i64>,
{
    assert!(dim < x.ndim(), "argmax: dim {dim} out of range");

    let backend = B::default();
    let mut out_shape = x.shape_cloned();
    out_shape[dim] = 1;
    let mut out = Tensor::zeros_on(out_shape, &backend);
    let (out_storage, out_layout) = out.storage_mut_and_layout();
    coeus_leto::argmax_into(
        x.layout(),
        x.storage().as_slice(),
        dim,
        out_layout,
        out_storage.as_mut_slice(),
    )
    .expect("coeus-leto argmax failed");
    out
}

/// Argmin along `dim`: returns indices of minimum values, shape `x.shape()[dim] = 1`.
#[inline]
pub fn argmin<T: Scalar + leto_ops::Scalar, B: ComputeBackend + Default>(
    x: &Tensor<T, B>,
    dim: usize,
) -> Tensor<i64, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorageMut<i64>,
{
    assert!(dim < x.ndim(), "argmin: dim {dim} out of range");

    let backend = B::default();
    let mut out_shape = x.shape_cloned();
    out_shape[dim] = 1;
    let mut out = Tensor::zeros_on(out_shape, &backend);
    let (out_storage, out_layout) = out.storage_mut_and_layout();
    coeus_leto::argmin_into(
        x.layout(),
        x.storage().as_slice(),
        dim,
        out_layout,
        out_storage.as_mut_slice(),
    )
    .expect("coeus-leto argmin failed");
    out
}
