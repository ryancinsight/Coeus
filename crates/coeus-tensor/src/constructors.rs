// ── Tensor constructors ──
// Factory functions for creating tensors.

use crate::tensor::Tensor;
use coeus_core::{ComputeBackend, CpuAddressableStorageMut, Float, Scalar, Shape};

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

    /// Arange: values from [0, n) with step 1.
    #[inline]
    pub fn arange(n: usize) -> Self {
        Self::arange_on(n, &B::default())
    }

    /// Linspace: n evenly spaced values from start to end (inclusive).
    #[inline]
    pub fn linspace(start: T, end: T, n: usize) -> Self {
        Self::linspace_on(start, end, n, &B::default())
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

    /// Arange: values from [0, n) with step 1 on the given backend.
    #[inline]
    pub fn arange_on(n: usize, backend: &B) -> Self {
        let values = coeus_leto::from_shape_fn_values(&[n], |index| T::from_usize(index[0]))
            .expect("coeus-leto arange generation failed");
        Self::from_slice_on([n], &values, backend)
    }

    /// Linspace: n evenly spaced values from start to end (inclusive) on the given backend.
    ///
    /// Computes natively in `T` (no `f64` widen-compute-narrow detour); valid
    /// for any [`Scalar`], including integer types (exact-division steps).
    #[inline]
    pub fn linspace_on(start: T, end: T, n: usize, backend: &B) -> Self {
        let step = if n > 1 {
            (end - start) / T::from_usize(n - 1)
        } else {
            T::zero()
        };
        let values =
            coeus_leto::from_shape_fn_values(&[n], |index| start + step * T::from_usize(index[0]))
                .expect("coeus-leto linspace generation failed");
        Self::from_slice_on([n], &values, backend)
    }
}

impl<T: Float, B: ComputeBackend + Default> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    /// Logspace: `n` values from `base^start` to `base^end` (inclusive).
    #[inline]
    pub fn logspace(start: T, end: T, n: usize, base: T) -> Self {
        Self::logspace_on(start, end, n, base, &B::default())
    }

    /// Geometric progression: `n` values from `start` to `end` (inclusive).
    #[inline]
    pub fn geomspace(start: T, end: T, n: usize) -> Self {
        Self::geomspace_on(start, end, n, &B::default())
    }

    /// Logspace: `n` values from `base^start` to `base^end` (inclusive)
    /// on the given backend.
    ///
    /// Computes natively in `T` (no `f64` widen-compute-narrow detour).
    #[inline]
    pub fn logspace_on(start: T, end: T, n: usize, base: T, backend: &B) -> Self {
        let n_minus_1 = T::from_usize(if n > 1 { n - 1 } else { 1 });
        let values = coeus_leto::from_shape_fn_values(&[n], |index| {
            let exp = if n > 1 {
                start + (end - start) * T::from_usize(index[0]) / n_minus_1
            } else {
                start
            };
            base.powf(exp)
        })
        .expect("coeus-leto logspace generation failed");
        Self::from_slice_on([n], &values, backend)
    }

    /// Geometric progression: `n` values from `start` to `end` (inclusive)
    /// on the given backend.
    ///
    /// Requires non-zero endpoints with the same sign. Computes natively in
    /// `T` (no `f64` widen-compute-narrow detour).
    #[inline]
    pub fn geomspace_on(start: T, end: T, n: usize, backend: &B) -> Self {
        let zero = T::zero();
        assert!(
            start != zero && end != zero,
            "geomspace requires non-zero start/end"
        );
        assert!(
            Float::signum(start) == Float::signum(end),
            "geomspace requires start/end to have the same sign"
        );
        let sign = Float::signum(start);
        let start_abs = Float::abs(start);
        let end_abs = Float::abs(end);
        let one = T::one();
        let ratio = if n > 1 {
            Float::powf(end_abs / start_abs, one / T::from_usize(n - 1))
        } else {
            one
        };
        let values = coeus_leto::from_shape_fn_values(&[n], |index| {
            if n > 1 {
                sign * start_abs * ratio.powf(T::from_usize(index[0]))
            } else {
                start
            }
        })
        .expect("coeus-leto geomspace generation failed");
        Self::from_slice_on([n], &values, backend)
    }
}
