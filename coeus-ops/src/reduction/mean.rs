// ── Mean reduction ──

use crate::backend_ops::{BackendOps, ReductionOp};
use coeus_core::Scalar;
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
/// let a = Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let result = mean(&a, &backend);
/// assert!((result - 3.5).abs() < 1e-5);
/// ```
#[inline]
pub fn mean<T: Scalar, B: BackendOps<T> + Default>(a: &Tensor<T, B>, backend: &B) -> T {
    if a.numel() == 0 {
        return T::zero() / T::from_f64(0.0);
    }
    let reshaped = if a.is_contiguous() && a.layout().offset() == 0 {
        a.reshape([a.numel()])
    } else {
        let contiguous = a.to_contiguous_on(backend);
        contiguous.reshape([a.numel()])
    };
    let reduced = mean_axis(&reshaped, 0, backend);
    let mut host_scalar = [T::zero()];
    backend.copy_to_host(reduced.storage(), &mut host_scalar);
    host_scalar[0]
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
/// let a = Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let result = mean_axis(&a, 1, &backend);
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
) -> Tensor<T, B> {
    assert!(axis < a.ndim(), "mean_axis: axis {axis} out of bounds");

    let mut out_shape = a.shape_cloned();
    out_shape[axis] = 1;

    let mut out = Tensor::alloc_on(out_shape, backend);

    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.reduce(
        ReductionOp::Mean,
        a.storage(),
        a.layout(),
        axis,
        out_storage,
        out_layout,
    );

    out
}
