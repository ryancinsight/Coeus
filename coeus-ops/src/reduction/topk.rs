// ── TopK and ArgMax/ArgMin ──
// Returns the k largest (or smallest) values and their indices along a dimension.

use crate::BackendOps;
use coeus_core::{Layout, Scalar};
use coeus_tensor::Tensor;

/// Helper function to compute topk on contiguous CPU slices.
pub fn topk_impl<T: Scalar>(
    a_slice: &[T],
    a_shape: &[usize],
    k: usize,
    dim: usize,
    largest: bool,
    val_slice: &mut [T],
    idx_slice: &mut [i64],
) {
    let ndim = a_shape.len();
    let dim_size = a_shape[dim];
    let in_strides = Layout::from_shape_strides(coeus_core::Shape::from(a_shape), Layout::new(coeus_core::Shape::from(a_shape)).strides_cloned(), 0).strides_cloned();

    let mut out_shape = a_shape.to_vec();
    out_shape[dim] = k;
    let out_numel: usize = out_shape.iter().product();
    let outer = out_numel / k;

    for outer_idx in 0..outer {
        // Map outer_idx to coordinates excluding dim.
        let mut coords = vec![0usize; ndim];
        let mut rem = outer_idx;
        for d in (0..ndim).rev() {
            if d == dim {
                continue;
            }
            coords[d] = rem % a_shape[d];
            rem /= a_shape[d];
        }
        let base: usize = coords
            .iter()
            .enumerate()
            .map(|(d, &c)| if d != dim { c * in_strides[d] } else { 0 })
            .sum();

        // Collect (value, index) pairs for this slice.
        let mut pairs: Vec<(T, usize)> = (0..dim_size)
            .map(|k_idx| (a_slice[base + k_idx * in_strides[dim]], k_idx))
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
            val_slice[flat_out] = *v;
            idx_slice[flat_out] = *orig_idx as i64;
        }
    }
}

/// Return the `k` largest (or smallest) values and their flat indices along `dim`.
///
/// Returns `(values, indices)` both with shape equal to `x.shape()` but with
/// `shape[dim] == k`.
///
/// # Panics
/// - `k == 0` or `k > x.shape()[dim]`.
/// - `dim` out of range.
#[inline]
pub fn topk<T: Scalar + leto_ops::Scalar, B: BackendOps<T> + BackendOps<i64> + Default>(
    x: &Tensor<T, B>,
    k: usize,
    dim: usize,
    largest: bool,
) -> (Tensor<T, B>, Tensor<i64, B>) {
    let ndim = x.ndim();
    let dim_size = x.shape()[dim];
    assert!(
        k > 0 && k <= dim_size,
        "topk: k={k} invalid for dim_size={dim_size}"
    );
    assert!(dim < ndim, "topk: dim {dim} out of range");

    let backend = B::default();
    let x_cont = x.to_contiguous_on(&backend);

    let mut out_shape = x.shape_cloned();
    out_shape[dim] = k;

    let mut val_tensor = Tensor::zeros_on(out_shape.clone(), &backend);
    let mut idx_tensor = Tensor::zeros_on(out_shape, &backend);

    {
        let (val_storage, val_layout) = val_tensor.storage_mut_and_layout();
        let (idx_storage, idx_layout) = idx_tensor.storage_mut_and_layout();
        backend.topk(
            x_cont.storage(),
            x_cont.layout(),
            k,
            dim,
            largest,
            val_storage,
            val_layout,
            idx_storage,
            idx_layout,
        );
    }

    (val_tensor, idx_tensor)
}

/// Argmax along `dim`: returns indices of maximum values, shape `x.shape()[dim] = 1`.
#[inline]
pub fn argmax<T: Scalar + leto_ops::Scalar, B: BackendOps<T> + BackendOps<i64> + Default>(
    x: &Tensor<T, B>,
    dim: usize,
) -> Tensor<i64, B> {
    assert!(dim < x.ndim(), "argmax: dim {dim} out of range");

    let backend = B::default();
    let mut out_shape = x.shape_cloned();
    out_shape[dim] = 1;
    let mut out = Tensor::zeros_on(out_shape, &backend);
    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.argmax(x.storage(), x.layout(), dim, out_storage, out_layout);
    out
}

/// Argmin along `dim`: returns indices of minimum values, shape `x.shape()[dim] = 1`.
#[inline]
pub fn argmin<T: Scalar + leto_ops::Scalar, B: BackendOps<T> + BackendOps<i64> + Default>(
    x: &Tensor<T, B>,
    dim: usize,
) -> Tensor<i64, B> {
    assert!(dim < x.ndim(), "argmin: dim {dim} out of range");

    let backend = B::default();
    let mut out_shape = x.shape_cloned();
    out_shape[dim] = 1;
    let mut out = Tensor::zeros_on(out_shape, &backend);
    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.argmin(x.storage(), x.layout(), dim, out_storage, out_layout);
    out
}
