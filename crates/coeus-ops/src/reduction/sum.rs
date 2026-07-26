// ── Sum reduction ──

use crate::backend_ops::{BackendOps, ReductionOp};
use coeus_core::{BackendError, Scalar};
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
/// let result = sum(&a, &backend).expect("valid sum inputs");
/// assert!((result - 21.0).abs() < 1e-5);
/// ```
#[inline]
pub fn sum<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    backend: &B,
) -> Result<T, B::Error> {
    if a.numel() == 0 {
        return Ok(T::zero());
    }
    let reshaped = if a.is_contiguous() && a.layout().offset() == 0 {
        a.reshape([a.numel()])
    } else {
        let contiguous = a.to_contiguous_on(backend);
        contiguous.reshape([a.numel()])
    };
    let reduced = sum_axis(&reshaped, 0, backend)?;
    let mut host_scalar = [T::zero()];
    backend.copy_to_host(reduced.storage(), &mut host_scalar);
    Ok(host_scalar[0])
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
/// let result = sum_axis(&a, 1, &backend).expect("valid reduction axis");
/// assert_eq!(result.shape(), &[2, 1]);
/// assert_eq!(result.as_slice(), &[6.0, 15.0]);
/// ```
#[inline]
pub fn sum_axis<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    axis: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    if axis >= a.ndim() {
        return Err(B::Error::from(BackendError::AxisOutOfRange {
            operation: "sum_axis",
            axis,
            rank: a.ndim(),
        }));
    }

    let mut out_shape = a.shape_cloned();
    out_shape[axis] = 1;

    let mut out = Tensor::alloc_on(out_shape, backend);

    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.reduce(
        ReductionOp::Sum,
        a.storage(),
        a.layout(),
        axis,
        out_storage,
        out_layout,
    )?;

    Ok(out)
}

/// Product along a specific axis, reducing it to size 1.
///
/// The reduction uses the multiplicative identity for empty axes, matching the
/// CPU Leto contract and the provider kernels.
#[inline]
pub fn prod_axis<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    axis: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    if axis >= a.ndim() {
        return Err(B::Error::from(BackendError::AxisOutOfRange {
            operation: "prod_axis",
            axis,
            rank: a.ndim(),
        }));
    }

    let mut out_shape = a.shape_cloned();
    out_shape[axis] = 1;

    let mut out = Tensor::alloc_on(out_shape, backend);

    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.reduce(
        ReductionOp::Prod,
        a.storage(),
        a.layout(),
        axis,
        out_storage,
        out_layout,
    )?;

    Ok(out)
}

/// Maximum along a specific axis, reducing it to size 1.
#[inline]
pub fn max_axis<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    axis: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    if axis >= a.ndim() {
        return Err(B::Error::from(BackendError::AxisOutOfRange {
            operation: "max_axis",
            axis,
            rank: a.ndim(),
        }));
    }

    let mut out_shape = a.shape_cloned();
    out_shape[axis] = 1;

    let mut out = Tensor::alloc_on(out_shape, backend);

    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.reduce(
        ReductionOp::Max,
        a.storage(),
        a.layout(),
        axis,
        out_storage,
        out_layout,
    )?;

    Ok(out)
}

/// Minimum along a specific axis, reducing it to size 1.
#[inline]
pub fn min_axis<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    axis: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    if axis >= a.ndim() {
        return Err(B::Error::from(BackendError::AxisOutOfRange {
            operation: "min_axis",
            axis,
            rank: a.ndim(),
        }));
    }

    let mut out_shape = a.shape_cloned();
    out_shape[axis] = 1;

    let mut out = Tensor::alloc_on(out_shape, backend);

    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.reduce(
        ReductionOp::Min,
        a.storage(),
        a.layout(),
        axis,
        out_storage,
        out_layout,
    )?;

    Ok(out)
}

/// Global maximum — returns a scalar-shaped `[1]` tensor.
///
/// Equivalent to `torch.amax(input)` with no dim argument.
#[inline]
pub fn amax<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    backend: &B,
) -> Result<T, B::Error> {
    if a.numel() == 0 {
        return Err(B::Error::from(BackendError::Storage {
            operation: "amax",
            reason: "empty tensor has no maximum".to_owned(),
        }));
    }
    let flat = if a.is_contiguous() && a.layout().offset() == 0 {
        a.reshape([a.numel()])
    } else {
        a.to_contiguous_on(backend).reshape([a.numel()])
    };
    let reduced = max_axis(&flat, 0, backend)?;
    let mut host = [T::zero()];
    backend.copy_to_host(reduced.storage(), &mut host);
    Ok(host[0])
}

/// Global minimum — returns a scalar value.
///
/// Equivalent to `torch.amin(input)` with no dim argument.
#[inline]
pub fn amin<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    backend: &B,
) -> Result<T, B::Error> {
    if a.numel() == 0 {
        return Err(B::Error::from(BackendError::Storage {
            operation: "amin",
            reason: "empty tensor has no minimum".to_owned(),
        }));
    }
    let flat = if a.is_contiguous() && a.layout().offset() == 0 {
        a.reshape([a.numel()])
    } else {
        a.to_contiguous_on(backend).reshape([a.numel()])
    };
    let reduced = min_axis(&flat, 0, backend)?;
    let mut host = [T::zero()];
    backend.copy_to_host(reduced.storage(), &mut host);
    Ok(host[0])
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
