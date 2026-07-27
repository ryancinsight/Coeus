// ── TopK and ArgMax/ArgMin ──
// Returns the k largest (or smallest) values and their indices along a dimension.

use crate::{BackendOps, CpuBackend};
use coeus_core::{Layout, Scalar};
use coeus_tensor::Tensor;

/// Computes top-k values for a contiguous slice backing a tensor view.
///
/// This is the CPU helper behind PyTorch-style `topk`, writing both selected
/// values and their source indices into the provided output slices.
///
/// # Panics
/// Panics if `dim` is out of range, or if `k == 0` / `k > a_shape[dim]`.
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
    let in_strides = Layout::from_shape_strides(
        coeus_core::Shape::from(a_shape),
        Layout::new(coeus_core::Shape::from(a_shape)).strides_cloned(),
        0,
    )
    .strides_cloned();

    let mut out_shape = a_shape.to_vec();
    out_shape[dim] = k;
    let out_numel: usize = out_shape.iter().product();
    let outer = out_numel / k;

    // Row-major strides of the output: results write back through `dim`'s
    // output stride, so non-terminal dims (e.g. dim = 0) interleave into the
    // output correctly instead of being laid out line-contiguously (which is
    // only equivalent when `dim` is the last axis).
    let mut out_strides = vec![1usize; ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
    }

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

        let out_base: usize = coords
            .iter()
            .enumerate()
            .map(|(d, &c)| if d != dim { c * out_strides[d] } else { 0 })
            .sum();
        for (rank, (v, orig_idx)) in pairs.iter().enumerate() {
            let flat_out = out_base + rank * out_strides[dim];
            val_slice[flat_out] = *v;
            idx_slice[flat_out] = *orig_idx as i64;
        }
    }
}

/// Returns the `k` largest or smallest values along `dim`, like PyTorch `torch.topk`.
///
/// Returns `(values, indices)` both with shape equal to `x.shape()` but with
/// `shape[dim] == k`; `indices` stores positions along `dim`, similar to NumPy
/// `take_along_axis` consumers.
///
/// # Memory
/// Both output tensors are `alloc_on` (uninitialized); the kernel writes every
/// `(outer, rank)` position exactly once using row-major output strides that
/// account for `dim` being anywhere in the shape (not just the last axis).
///
/// # Panics
/// - If `k == 0` or `k > x.shape()[dim]`.
/// - If `dim` is out of range.
#[inline]
pub fn topk<
    T: Scalar + leto_ops::Scalar,
    B: BackendOps<T> + BackendOps<i64> + CpuBackend + Default,
>(
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

    // alloc_on: backend.topk writes every val/idx position — no zero-init needed.
    let mut val_tensor = Tensor::alloc_on(out_shape.clone(), &backend);
    let mut idx_tensor = Tensor::alloc_on(out_shape, &backend);

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
pub fn argmax<
    T: Scalar + leto_ops::Scalar,
    B: BackendOps<T> + BackendOps<i64> + CpuBackend + Default,
>(
    x: &Tensor<T, B>,
    dim: usize,
) -> Tensor<i64, B> {
    assert!(dim < x.ndim(), "argmax: dim {dim} out of range");

    let backend = B::default();
    let mut out_shape = x.shape_cloned();
    out_shape[dim] = 1;
    // alloc_on: backend.argmax writes every position — no zero-init needed.
    let mut out = Tensor::alloc_on(out_shape, &backend);
    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.argmax(x.storage(), x.layout(), dim, out_storage, out_layout);
    out
}

/// Argmin along `dim`: returns indices of minimum values, shape `x.shape()[dim] = 1`.
#[inline]
pub fn argmin<
    T: Scalar + leto_ops::Scalar,
    B: BackendOps<T> + BackendOps<i64> + CpuBackend + Default,
>(
    x: &Tensor<T, B>,
    dim: usize,
) -> Tensor<i64, B> {
    assert!(dim < x.ndim(), "argmin: dim {dim} out of range");

    let backend = B::default();
    let mut out_shape = x.shape_cloned();
    out_shape[dim] = 1;
    // alloc_on: backend.argmin writes every position — no zero-init needed.
    let mut out = Tensor::alloc_on(out_shape, &backend);
    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.argmin(x.storage(), x.layout(), dim, out_storage, out_layout);
    out
}
