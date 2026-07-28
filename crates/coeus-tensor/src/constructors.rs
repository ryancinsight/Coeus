// ── Tensor constructors ──
// Factory functions for creating tensors.

use crate::tensor::Tensor;
use coeus_core::{ComputeBackend, CpuAddressableStorageMut, Scalar, Shape};

impl<T: Scalar, B: ComputeBackend + Default> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    /// Create a new tensor with shape filled using a function `f(index)`.
    #[inline]
    pub fn from_fn<S: Into<Shape>, F>(shape: S, f: F) -> Result<Self, B::Error>
    where
        F: Fn(&[usize]) -> T,
    {
        Self::from_fn_on(shape, &B::default(), f)
    }

    /// Identity matrix of size n×n.
    #[inline]
    pub fn eye(n: usize) -> Result<Self, B::Error> {
        Self::eye_on(n, &B::default())
    }

    /// Linspace: n evenly spaced values from start to end (inclusive).
    #[inline]
    pub fn linspace(start: T, end: T, n: usize) -> Result<Self, B::Error> {
        Self::linspace_on(start, end, n, &B::default())
    }

    /// Arange: values from [0, n) with step 1.
    #[inline]
    pub fn arange(n: usize) -> Result<Self, B::Error> {
        Self::arange_on(n, &B::default())
    }

    /// Logspace: `n` values from `base^start` to `base^end` (inclusive).
    #[inline]
    pub fn logspace(start: T, end: T, n: usize, base: T) -> Result<Self, B::Error> {
        Self::logspace_on(start, end, n, base, &B::default())
    }

    /// Geometric progression: `n` values from `start` to `end` (inclusive).
    #[inline]
    pub fn geomspace(start: T, end: T, n: usize) -> Result<Self, B::Error> {
        Self::geomspace_on(start, end, n, &B::default())
    }
}

impl<T: Scalar, B: ComputeBackend> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    /// Create a new tensor with shape filled using a function `f(index)` on the given backend.
    #[inline]
    pub fn from_fn_on<S: Into<Shape>, F>(shape: S, backend: &B, f: F) -> Result<Self, B::Error>
    where
        F: Fn(&[usize]) -> T,
    {
        let shape = shape.into();
        let values = coeus_leto::from_shape_fn_values(&shape, f).map_err(|source| {
            B::Error::from(coeus_core::BackendError::Storage {
                operation: "tensor from_fn",
                reason: source.to_string(),
            })
        })?;
        Self::from_slice_on(shape, &values, backend)
    }

    /// Identity matrix of size n×n on the given backend.
    #[inline]
    pub fn eye_on(n: usize, backend: &B) -> Result<Self, B::Error> {
        let values = coeus_leto::from_shape_fn_values(&[n, n], |index| {
            if index[0] == index[1] {
                T::one()
            } else {
                T::zero()
            }
        })
        .map_err(|source| {
            B::Error::from(coeus_core::BackendError::Storage {
                operation: "tensor eye",
                reason: source.to_string(),
            })
        })?;
        Self::from_slice_on([n, n], &values, backend)
    }

    /// Linspace: n evenly spaced values from start to end (inclusive) on the given backend.
    #[inline]
    pub fn linspace_on(start: T, end: T, n: usize, backend: &B) -> Result<Self, B::Error> {
        let start_f = <T as Scalar>::to_f64(start);
        let end_f = <T as Scalar>::to_f64(end);
        let step = if n > 1 {
            (end_f - start_f) / (n - 1) as f64
        } else {
            0.0
        };
        let values = coeus_leto::from_shape_fn_values(&[n], |index| {
            <T as Scalar>::from_f64(start_f + step * index[0] as f64)
        })
        .map_err(|source| {
            B::Error::from(coeus_core::BackendError::Storage {
                operation: "tensor linspace",
                reason: source.to_string(),
            })
        })?;
        Self::from_slice_on([n], &values, backend)
    }

    /// Arange: values from [0, n) with step 1 on the given backend.
    #[inline]
    pub fn arange_on(n: usize, backend: &B) -> Result<Self, B::Error> {
        let values = coeus_leto::from_shape_fn_values(&[n], |index| T::from_usize(index[0]))
            .map_err(|source| {
                B::Error::from(coeus_core::BackendError::Storage {
                    operation: "tensor arange",
                    reason: source.to_string(),
                })
            })?;
        Self::from_slice_on([n], &values, backend)
    }

    /// Logspace: `n` values from `base^start` to `base^end` (inclusive)
    /// on the given backend.
    #[inline]
    pub fn logspace_on(start: T, end: T, n: usize, base: T, backend: &B) -> Result<Self, B::Error> {
        let start_f = <T as Scalar>::to_f64(start);
        let end_f = <T as Scalar>::to_f64(end);
        let base_f = <T as Scalar>::to_f64(base);
        let values = coeus_leto::from_shape_fn_values(&[n], |index| {
            let exp = if n > 1 {
                start_f + (end_f - start_f) * index[0] as f64 / (n - 1) as f64
            } else {
                start_f
            };
            <T as Scalar>::from_f64(base_f.powf(exp))
        })
        .map_err(|source| {
            B::Error::from(coeus_core::BackendError::Storage {
                operation: "tensor logspace",
                reason: source.to_string(),
            })
        })?;
        Self::from_slice_on([n], &values, backend)
    }

    /// Geometric progression: `n` values from `start` to `end` (inclusive)
    /// on the given backend.
    ///
    /// Requires non-zero endpoints with the same sign.
    #[inline]
    pub fn geomspace_on(start: T, end: T, n: usize, backend: &B) -> Result<Self, B::Error> {
        let start_f = <T as Scalar>::to_f64(start);
        let end_f = <T as Scalar>::to_f64(end);
        if start_f == 0.0 || end_f == 0.0 {
            return Err(B::Error::from(coeus_core::BackendError::Storage {
                operation: "tensor geomspace",
                reason: "start and end must be non-zero".to_owned(),
            }));
        }
        if start_f.signum() != end_f.signum() {
            return Err(B::Error::from(coeus_core::BackendError::Storage {
                operation: "tensor geomspace",
                reason: "start and end must have the same sign".to_owned(),
            }));
        }
        let sign = start_f.signum();
        let start_abs = start_f.abs();
        let end_abs = end_f.abs();
        let ratio = if n > 1 {
            (end_abs / start_abs).powf(1.0 / (n - 1) as f64)
        } else {
            1.0
        };
        let values = coeus_leto::from_shape_fn_values(&[n], |index| {
            let value = if n > 1 {
                sign * start_abs * ratio.powf(index[0] as f64)
            } else {
                start_f
            };
            <T as Scalar>::from_f64(value)
        })
        .map_err(|source| {
            B::Error::from(coeus_core::BackendError::Storage {
                operation: "tensor geomspace",
                reason: source.to_string(),
            })
        })?;
        Self::from_slice_on([n], &values, backend)
    }
}
