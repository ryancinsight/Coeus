// ── coeus-ops constructors ──
//
// Free-function alternatives to the `Tensor::linspace_on` / `logspace_on` /
// `geomspace_on` inherent methods.  These accept an explicit backend reference
// and return a device-resident tensor, matching the calling convention used by
// all other `coeus-ops` free functions (`matmul`, `dot`, `topk`, …).

use crate::BackendOps;
use coeus_core::{CpuAddressableStorageMut, Float, FloatElement};
use coeus_tensor::Tensor;

/// `n` evenly-spaced values from `start` to `end` (inclusive) on `backend`.
///
/// Equivalent to `numpy.linspace(start, end, n)` / `torch.linspace(start, end, n)`.
/// Delegates to [`Tensor::linspace_on`], the one native-`T` implementation.
///
/// # Panics
/// Panics if `n == 0` (matches NumPy / PyTorch behaviour — zero-element
/// linspace is meaningless without a keepdim flag).
#[inline]
pub fn linspace<T: Float, B: BackendOps<T> + Default>(
    start: T,
    end: T,
    n: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    assert!(n > 0, "linspace: n must be > 0");
    Tensor::linspace_on(start, end, n, backend)
}

/// `n` values from `base^start` to `base^end` (inclusive) on `backend`.
///
/// Equivalent to `numpy.logspace(start, end, n, base=base)` /
/// `torch.logspace(start, end, n, base)`.
/// Delegates to [`Tensor::logspace_on`], the one native-`T` implementation.
///
/// # Panics
/// Panics if `n == 0`.
#[inline]
pub fn logspace<T: Float + FloatElement, B: BackendOps<T> + Default>(
    start: T,
    end: T,
    n: usize,
    base: T,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    assert!(n > 0, "logspace: n must be > 0");
    Tensor::logspace_on(start, end, n, base, backend)
}

/// `n` geometrically-spaced values from `start` to `end` (inclusive) on `backend`.
///
/// Equivalent to `numpy.geomspace(start, end, n)`.
/// Delegates to [`Tensor::geomspace_on`], the one native-`T` implementation.
///
/// # Panics
/// Panics if `n == 0`, if either endpoint is zero, or if they have opposite signs.
#[inline]
pub fn geomspace<T: Float + FloatElement, B: BackendOps<T> + Default>(
    start: T,
    end: T,
    n: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    assert!(n > 0, "geomspace: n must be > 0");
    Tensor::geomspace_on(start, end, n, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    #[test]
    fn linspace_endpoints_inclusive() {
        let backend = SequentialBackend::new();
        let values =
            linspace(0.0f32, 1.0, 5, &backend).expect("invariant: test operation succeeds");
        assert_eq!(values.as_slice(), &[0.0f32, 0.25, 0.5, 0.75, 1.0]);
        let values =
            linspace(0.0f64, 1.0, 5, &backend).expect("invariant: test operation succeeds");
        assert_eq!(values.as_slice(), &[0.0f64, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn logspace_base10() {
        let backend = SequentialBackend::new();
        let values =
            logspace(0.0f32, 2.0, 3, 10.0, &backend).expect("invariant: test operation succeeds");
        assert_eq!(values.as_slice(), &[1.0f32, 10.0, 100.0]);
        let values =
            logspace(0.0f64, 2.0, 3, 10.0, &backend).expect("invariant: test operation succeeds");
        assert_eq!(values.as_slice(), &[1.0f64, 10.0, 100.0]);
    }

    #[test]
    fn geomspace_doubling() {
        let backend = SequentialBackend::new();
        let values =
            geomspace(1.0f32, 16.0, 5, &backend).expect("invariant: test operation succeeds");
        assert_eq!(values.as_slice(), &[1.0f32, 2.0, 4.0, 8.0, 16.0]);
        let values =
            geomspace(1.0f64, 16.0, 5, &backend).expect("invariant: test operation succeeds");
        assert_eq!(values.as_slice(), &[1.0f64, 2.0, 4.0, 8.0, 16.0]);
    }

    #[test]
    fn linspace_single_sample_returns_start() {
        let backend = SequentialBackend::new();
        // 3.5 is exactly representable in f32 and f64.
        let values =
            linspace(3.5f32, 99.0, 1, &backend).expect("invariant: test operation succeeds");
        assert_eq!(values.as_slice(), &[3.5f32]);
        let values =
            linspace(3.5f64, 99.0, 1, &backend).expect("invariant: test operation succeeds");
        assert_eq!(values.as_slice(), &[3.5f64]);
    }
}
