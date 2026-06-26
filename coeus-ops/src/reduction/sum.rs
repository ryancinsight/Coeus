// ── Sum reduction ──

use crate::backend_ops::{BackendOps, ReductionOp};
use coeus_core::Scalar;
use coeus_tensor::Tensor;

/// Sum all elements.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::sum;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let result = sum(&a, &backend);
/// assert!((result - 21.0).abs() < 1e-5);
/// ```
#[inline]
pub fn sum<T: Scalar, B: BackendOps<T> + Default>(a: &Tensor<T, B>, backend: &B) -> T {
    if a.numel() == 0 {
        return T::zero();
    }
    let reshaped = if a.is_contiguous() && a.layout().offset() == 0 {
        a.reshape([a.numel()])
    } else {
        let contiguous = a.to_contiguous_on(backend);
        contiguous.reshape([a.numel()])
    };
    let reduced = sum_axis(&reshaped, 0, backend);
    let mut host_scalar = [T::zero()];
    backend.copy_to_host(reduced.storage(), &mut host_scalar);
    host_scalar[0]
}

/// Sum along a specific axis, reducing it to size 1.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::sum_axis;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let result = sum_axis(&a, 1, &backend);
/// assert_eq!(result.shape(), &[2, 1]);
/// assert_eq!(result.as_slice(), &[6.0, 15.0]);
/// ```
#[inline]
pub fn sum_axis<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    axis: usize,
    backend: &B,
) -> Tensor<T, B> {
    assert!(axis < a.ndim(), "sum_axis: axis {axis} out of bounds");

    let mut out_shape = a.shape_cloned();
    out_shape[axis] = 1;

    let mut out = Tensor::zeros_on(out_shape, backend);

    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.reduce(
        ReductionOp::Sum,
        a.storage(),
        a.layout(),
        axis,
        out_storage,
        out_layout,
    );

    out
}

/// Maximum along a specific axis, reducing it to size 1.
#[inline]
pub fn max_axis<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    axis: usize,
    backend: &B,
) -> Tensor<T, B> {
    assert!(axis < a.ndim(), "max_axis: axis {axis} out of bounds");

    let mut out_shape = a.shape_cloned();
    out_shape[axis] = 1;

    let mut out = Tensor::zeros_on(out_shape, backend);

    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.reduce(
        ReductionOp::Max,
        a.storage(),
        a.layout(),
        axis,
        out_storage,
        out_layout,
    );

    out
}

/// Minimum along a specific axis, reducing it to size 1.
#[inline]
pub fn min_axis<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    axis: usize,
    backend: &B,
) -> Tensor<T, B> {
    assert!(axis < a.ndim(), "min_axis: axis {axis} out of bounds");

    let mut out_shape = a.shape_cloned();
    out_shape[axis] = 1;

    let mut out = Tensor::zeros_on(out_shape, backend);

    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.reduce(
        ReductionOp::Min,
        a.storage(),
        a.layout(),
        axis,
        out_storage,
        out_layout,
    );

    out
}

/// Global maximum — returns a scalar-shaped `[1]` tensor.
///
/// Equivalent to `torch.amax(input)` with no dim argument.
#[inline]
pub fn amax<T: Scalar, B: BackendOps<T> + Default>(a: &Tensor<T, B>, backend: &B) -> T {
    if a.numel() == 0 {
        panic!("amax: empty tensor has no maximum");
    }
    let flat = if a.is_contiguous() && a.layout().offset() == 0 {
        a.reshape([a.numel()])
    } else {
        a.to_contiguous_on(backend).reshape([a.numel()])
    };
    let reduced = max_axis(&flat, 0, backend);
    let mut host = [T::zero()];
    backend.copy_to_host(reduced.storage(), &mut host);
    host[0]
}

/// Global minimum — returns a scalar value.
///
/// Equivalent to `torch.amin(input)` with no dim argument.
#[inline]
pub fn amin<T: Scalar, B: BackendOps<T> + Default>(a: &Tensor<T, B>, backend: &B) -> T {
    if a.numel() == 0 {
        panic!("amin: empty tensor has no minimum");
    }
    let flat = if a.is_contiguous() && a.layout().offset() == 0 {
        a.reshape([a.numel()])
    } else {
        a.to_contiguous_on(backend).reshape([a.numel()])
    };
    let reduced = min_axis(&flat, 0, backend);
    let mut host = [T::zero()];
    backend.copy_to_host(reduced.storage(), &mut host);
    host[0]
}

/// Global product of all elements.
///
/// Equivalent to `torch.prod(input)`.
#[inline]
pub fn prod<T: Scalar, B: BackendOps<T> + Default>(a: &Tensor<T, B>, backend: &B) -> T {
    let cont = a.to_contiguous_on(backend);
    let mut host = vec![T::zero(); a.numel()];
    backend.copy_to_host(cont.storage(), &mut host);
    host.iter().fold(T::one(), |acc, &x| acc * x)
}
