// ── Tensor constructors ──
// Factory functions for creating tensors.

use coeus_core::{Scalar, CpuAddressableStorageMut, ComputeBackend, Shape};
use crate::tensor::Tensor;

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
        let ndim = shape.len();
        let numel: usize = shape.iter().product();
        let shape_clone = shape.clone();
        let mut t = Self::zeros_on(shape, backend);
        let slice = t.as_mut_slice();
        let mut index = smallvec::SmallVec::<[usize; 4]>::from_elem(0, ndim);
        for i in 0..numel {
            slice[i] = f(&index);
            for d in (0..ndim).rev() {
                index[d] += 1;
                if index[d] < shape_clone[d] {
                    break;
                }
                index[d] = 0;
            }
        }
        t
    }

    /// Identity matrix of size n×n on the given backend.
    #[inline]
    pub fn eye_on(n: usize, backend: &B) -> Self {
        let mut t = Self::zeros_on([n, n], backend);
        for i in 0..n {
            t.set(&[i, i], T::one());
        }
        t
    }

    /// Linspace: n evenly spaced values from start to end (inclusive) on the given backend.
    #[inline]
    pub fn linspace_on(start: T, end: T, n: usize, backend: &B) -> Self {
        let mut t = Self::zeros_on([n], backend);
        let slice = t.as_mut_slice();
        let start_f = start.to_f64();
        let end_f = end.to_f64();
        let step = if n > 1 { (end_f - start_f) / (n - 1) as f64 } else { 0.0 };
        for i in 0..n {
            slice[i] = T::from_f64(start_f + step * i as f64);
        }
        t
    }

    /// Arange: values from [0, n) with step 1 on the given backend.
    #[inline]
    pub fn arange_on(n: usize, backend: &B) -> Self {
        let mut t = Self::zeros_on([n], backend);
        let slice = t.as_mut_slice();
        for i in 0..n {
            slice[i] = T::from_f64(i as f64);
        }
        t
    }
}
