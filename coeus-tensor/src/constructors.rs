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
    pub fn from_fn<S: Into<Shape>, F>(shape: S, f: F) -> Self
    where
        F: Fn(&[usize]) -> T,
    {
        Self::from_fn_on(shape, &B::default(), f)
    }

    /// Identity matrix of size n×n.
    #[inline]
    pub fn eye(n: usize) -> Self {
        Self::eye_on(n, &B::default())
    }

    /// Linspace: n evenly spaced values from start to end (inclusive).
    #[inline]
    pub fn linspace(start: T, end: T, n: usize) -> Self {
        Self::linspace_on(start, end, n, &B::default())
    }

    /// Arange: values from [0, n) with step 1.
    #[inline]
    pub fn arange(n: usize) -> Self {
        Self::arange_on(n, &B::default())
    }
}

impl<T: Scalar, B: ComputeBackend> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    /// Create a new tensor with shape filled using a function `f(index)` on the given backend.
    #[inline]
    pub fn from_fn_on<S: Into<Shape>, F>(shape: S, backend: &B, f: F) -> Self
    where
        F: Fn(&[usize]) -> T,
    {
        let shape = shape.into();
        let values = coeus_leto::from_shape_fn_values(&shape, f)
            .expect("coeus-leto shape function generation failed");
        Self::from_slice_on(shape, &values, backend)
    }

    /// Identity matrix of size n×n on the given backend.
    #[inline]
    pub fn eye_on(n: usize, backend: &B) -> Self {
        let values = coeus_leto::from_shape_fn_values(&[n, n], |index| {
            if index[0] == index[1] {
                T::one()
            } else {
                T::zero()
            }
        })
        .expect("coeus-leto identity generation failed");
        Self::from_slice_on([n, n], &values, backend)
    }

    /// Linspace: n evenly spaced values from start to end (inclusive) on the given backend.
    #[inline]
    pub fn linspace_on(start: T, end: T, n: usize, backend: &B) -> Self {
        let start_f = start.to_f64();
        let end_f = end.to_f64();
        let step = if n > 1 {
            (end_f - start_f) / (n - 1) as f64
        } else {
            0.0
        };
        let values = coeus_leto::from_shape_fn_values(&[n], |index| {
            T::from_f64(start_f + step * index[0] as f64)
        })
        .expect("coeus-leto linspace generation failed");
        Self::from_slice_on([n], &values, backend)
    }

    /// Arange: values from [0, n) with step 1 on the given backend.
    #[inline]
    pub fn arange_on(n: usize, backend: &B) -> Self {
        let values = coeus_leto::from_shape_fn_values(&[n], |index| T::from_usize(index[0]))
            .expect("coeus-leto arange generation failed");
        Self::from_slice_on([n], &values, backend)
    }
}
