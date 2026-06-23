// ── Cumulative sum ──

use crate::BackendOps;
use coeus_core::Scalar;
use coeus_tensor::Tensor;

/// Compute the inclusive cumulative sum of `x` along `dim`.
///
/// Output has the same shape as `x`.
///
/// # Panics
/// - `dim` is out of range.
#[inline]
pub fn cumsum<T: Scalar + leto_ops::Scalar, B: BackendOps<T> + Default>(
    x: &Tensor<T, B>,
    dim: usize,
) -> Tensor<T, B> {
    let ndim = x.ndim();
    assert!(
        dim < ndim,
        "cumsum: dim {dim} out of range for {ndim}D tensor"
    );

    let backend = B::default();
    let shape = x.shape_cloned();
    let mut out = Tensor::zeros_on(shape, &backend);
    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.cumsum(x.storage(), x.layout(), dim, out_storage, out_layout);
    out
}

/// Compute the inclusive cumulative suffix sum (reverse cumulative sum) of `x` along `dim`.
///
/// Output has the same shape as `x`.
///
/// # Panics
/// - `dim` is out of range.
#[inline]
pub fn suffix_sum<T: Scalar + leto_ops::Scalar, B: BackendOps<T> + Default>(
    x: &Tensor<T, B>,
    dim: usize,
) -> Tensor<T, B> {
    let ndim = x.ndim();
    assert!(
        dim < ndim,
        "suffix_sum: dim {dim} out of range for {ndim}D tensor"
    );

    let backend = B::default();
    let shape = x.shape_cloned();
    let mut out = Tensor::zeros_on(shape, &backend);
    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.suffix_sum(x.storage(), x.layout(), dim, out_storage, out_layout);
    out
}
