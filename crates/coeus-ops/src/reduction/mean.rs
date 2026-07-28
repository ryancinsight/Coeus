// ── Mean reduction ──

use crate::backend_ops::{BackendOps, ReductionOp};
use coeus_core::{BackendError, Scalar};
use coeus_tensor::Tensor;

/// Mean of all elements.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::mean;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("construct tensor");
/// let result = mean(&a, &backend).expect("valid mean inputs");
/// assert!((result - 3.5).abs() < 1e-5);
/// ```
#[inline]
pub fn mean<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    backend: &B,
) -> Result<T, B::Error> {
    if a.numel() == 0 {
        return Ok(T::zero() / T::from_f64(0.0));
    }
    let reshaped = if a.is_contiguous() && a.layout().offset() == 0 {
        a.reshape([a.numel()])
    } else {
        let contiguous = a.to_contiguous_on(backend)?;
        contiguous.reshape([a.numel()])
    };
    let reduced = mean_axis(&reshaped, 0, backend)?;
    let mut host_scalar = [T::zero()];
    backend.copy_to_host(reduced.storage(), &mut host_scalar)?;
    Ok(host_scalar[0])
}

/// Mean along a specific axis.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::mean_axis;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("construct tensor");
/// let result = mean_axis(&a, 1, &backend).expect("valid reduction axis");
/// assert_eq!(result.shape(), &[2, 1]);
/// let s = result.as_slice();
/// assert!((s[0] - 2.0).abs() < 1e-5);
/// assert!((s[1] - 5.0).abs() < 1e-5);
/// ```
#[inline]
pub fn mean_axis<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    axis: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    if axis >= a.ndim() {
        return Err(B::Error::from(BackendError::AxisOutOfRange {
            operation: "mean_axis",
            axis,
            rank: a.ndim(),
        }));
    }

    let mut out_shape = a.shape_cloned();
    out_shape[axis] = 1;

    let mut out = Tensor::alloc_on(out_shape, backend)?;

    let (out_storage, out_layout) = out.storage_mut_and_layout()?;
    backend.reduce(
        ReductionOp::Mean,
        a.storage(),
        a.layout(),
        axis,
        out_storage,
        out_layout,
    )?;

    Ok(out)
}
