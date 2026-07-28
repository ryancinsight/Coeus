// ── Cumulative sum ──

use crate::BackendOps;
use coeus_core::{BackendError, Scalar};
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
) -> Result<Tensor<T, B>, B::Error> {
    let ndim = x.ndim();
    if dim >= ndim {
        return Err(B::Error::from(BackendError::AxisOutOfRange {
            operation: "cumsum",
            axis: dim,
            rank: ndim,
        }));
    }

    let backend = B::default();
    let shape = x.shape_cloned();
    // alloc_on: the cumsum kernel writes every output element in order — no zeros needed.
    let mut out = Tensor::alloc_on(shape, &backend)?;
    let (out_storage, out_layout) = out.storage_mut_and_layout()?;
    backend.cumsum(x.storage(), x.layout(), dim, out_storage, out_layout)?;
    Ok(out)
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
) -> Result<Tensor<T, B>, B::Error> {
    let ndim = x.ndim();
    if dim >= ndim {
        return Err(B::Error::from(BackendError::AxisOutOfRange {
            operation: "suffix_sum",
            axis: dim,
            rank: ndim,
        }));
    }

    let backend = B::default();
    let shape = x.shape_cloned();
    // alloc_on: suffix_sum writes every element — no zeros needed.
    let mut out = Tensor::alloc_on(shape, &backend)?;
    let (out_storage, out_layout) = out.storage_mut_and_layout()?;
    backend.suffix_sum(x.storage(), x.layout(), dim, out_storage, out_layout)?;
    Ok(out)
}
