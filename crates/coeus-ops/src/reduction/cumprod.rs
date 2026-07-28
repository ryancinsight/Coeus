// ── Cumulative product ──

use crate::BackendOps;
use coeus_core::{BackendError, Scalar};
use coeus_tensor::Tensor;

/// Compute the inclusive cumulative product of `x` along `dim`.
///
/// Output has the same shape as `x`. The backend owns execution: CPU
/// implementations delegate to Leto, while accelerator implementations may
/// dispatch a native scan kernel.
///
/// # Panics
///
/// Panics if `dim` is out of range or if backend dispatch rejects the layout.
#[inline]
pub fn cumprod<T: Scalar + leto_ops::Scalar, B: BackendOps<T> + Default>(
    x: &Tensor<T, B>,
    dim: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    let ndim = x.ndim();
    if dim >= ndim {
        return Err(B::Error::from(BackendError::AxisOutOfRange {
            operation: "cumprod",
            axis: dim,
            rank: ndim,
        }));
    }

    let shape = x.shape_cloned();
    // alloc_on: the cumulative product scan writes every output element.
    let mut out = Tensor::alloc_on(shape, backend)?;
    let (out_storage, out_layout) = out.storage_mut_and_layout()?;
    backend.cumprod(x.storage(), x.layout(), dim, out_storage, out_layout)?;
    Ok(out)
}

/// Compute the inclusive cumulative suffix product of `x` along `dim`.
///
/// Output has the same shape as `x`; each value is the product from its
/// position through the end of the selected dimension.
///
/// # Panics
///
/// Panics if `dim` is out of range or if backend dispatch rejects the layout.
#[inline]
pub fn suffix_prod<T: Scalar + leto_ops::Scalar, B: BackendOps<T> + Default>(
    x: &Tensor<T, B>,
    dim: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    let ndim = x.ndim();
    if dim >= ndim {
        return Err(B::Error::from(BackendError::AxisOutOfRange {
            operation: "suffix_prod",
            axis: dim,
            rank: ndim,
        }));
    }

    let shape = x.shape_cloned();
    // alloc_on: the cumulative product scan writes every output element.
    let mut out = Tensor::alloc_on(shape, backend)?;
    let (out_storage, out_layout) = out.storage_mut_and_layout()?;
    backend.suffix_prod(x.storage(), x.layout(), dim, out_storage, out_layout)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    #[test]
    fn cumulative_product_matches_prefix_values() {
        let backend = SequentialBackend::new();
        let input = Tensor::from_slice([2, 3], &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("construct tensor");

        let output = cumprod(&input, 1, &backend).expect("run operation");

        assert_eq!(output.as_slice(), &[1.0, 2.0, 6.0, 4.0, 20.0, 120.0]);
    }

    #[test]
    fn cumulative_product_matches_suffix_values() {
        let backend = SequentialBackend::new();
        let input = Tensor::from_slice([2, 3], &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("construct tensor");

        let output = suffix_prod(&input, 1, &backend).expect("run operation");

        assert_eq!(output.as_slice(), &[6.0, 6.0, 3.0, 120.0, 30.0, 6.0]);
    }
}
