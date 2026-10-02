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
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    assert!(n > 0, "linspace: n must be > 0");
    Tensor::linspace_on(start, end, n, backend)
        .expect("invariant: float elements represent every count")
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
) -> Tensor<T, B>
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
) -> Tensor<T, B>
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
        let b = SequentialBackend::new();
        let t = linspace(0.0f32, 1.0, 5, &b);
        let s = t.as_slice();
        assert!((s[0] - 0.0).abs() < 1e-6);
        assert!((s[4] - 1.0).abs() < 1e-6);
        assert!((s[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn logspace_base10() {
        let b = SequentialBackend::new();
        let t = logspace(0.0f32, 2.0, 3, 10.0, &b);
        let s = t.as_slice();
        assert!((s[0] - 1.0).abs() < 1e-4);
        assert!((s[1] - 10.0).abs() < 1e-4);
        assert!((s[2] - 100.0).abs() < 1e-4);
    }

    #[test]
    fn geomspace_doubling() {
        let b = SequentialBackend::new();
        let t = geomspace(1.0f32, 16.0, 5, &b);
        let s = t.as_slice();
        for (i, &v) in s.iter().enumerate() {
            let expected = 2.0f32.powi(i as i32);
            assert!(
                (v - expected).abs() < 1e-4,
                "geomspace[{i}]={v} vs {expected}"
            );
        }
    }

    #[test]
    fn linspace_n1_returns_start() {
        let b = SequentialBackend::new();
        // Use 3.5 (exactly representable in f32) to avoid PI-approximation lint.
        let t = linspace(3.5f32, 99.0, 1, &b);
        assert!((t.as_slice()[0] - 3.5_f32).abs() < 1e-5);
    }
}
