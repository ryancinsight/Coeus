// ── coeus-ops constructors ──
//
// Free-function alternatives to the `Tensor::linspace_on` / `logspace_on` /
// `geomspace_on` inherent methods.  These accept an explicit backend reference
// and return a device-resident tensor, matching the calling convention used by
// all other `coeus-ops` free functions (`matmul`, `dot`, `topk`, …).

use crate::BackendOps;
use coeus_core::{BackendError, CpuAddressableStorageMut, Float, Scalar};
use coeus_tensor::Tensor;

/// `n` evenly-spaced values from `start` to `end` (inclusive) on `backend`.
///
/// Equivalent to `numpy.linspace(start, end, n)` / `torch.linspace(start, end, n)`.
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
    if n == 0 {
        return Err(B::Error::from(BackendError::Storage {
            operation: "linspace",
            reason: "number of samples must be greater than zero".to_owned(),
        }));
    }
    let start_f = <T as Scalar>::to_f64(start);
    let end_f = <T as Scalar>::to_f64(end);
    let step = if n > 1 {
        (end_f - start_f) / (n - 1) as f64
    } else {
        0.0
    };
    let values: Vec<T> = (0..n)
        .map(|i| <T as Scalar>::from_f64(start_f + step * i as f64))
        .collect();
    Tensor::from_slice_on(vec![n], &values, backend)
}

/// `n` values from `base^start` to `base^end` (inclusive) on `backend`.
///
/// Equivalent to `numpy.logspace(start, end, n, base=base)` /
/// `torch.logspace(start, end, n, base)`.
///
/// # Panics
/// Panics if `n == 0`.
#[inline]
pub fn logspace<T: Float, B: BackendOps<T> + Default>(
    start: T,
    end: T,
    n: usize,
    base: T,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    if n == 0 {
        return Err(B::Error::from(BackendError::Storage {
            operation: "logspace",
            reason: "number of samples must be greater than zero".to_owned(),
        }));
    }
    let start_f = <T as Scalar>::to_f64(start);
    let end_f = <T as Scalar>::to_f64(end);
    let base_f = <T as Scalar>::to_f64(base);
    let values: Vec<T> = (0..n)
        .map(|i| {
            let exp = if n > 1 {
                start_f + (end_f - start_f) * i as f64 / (n - 1) as f64
            } else {
                start_f
            };
            <T as Scalar>::from_f64(base_f.powf(exp))
        })
        .collect();
    Tensor::from_slice_on(vec![n], &values, backend)
}

/// `n` geometrically-spaced values from `start` to `end` (inclusive) on `backend`.
///
/// Equivalent to `numpy.geomspace(start, end, n)`.
///
/// # Panics
/// Panics if `n == 0`, if either endpoint is zero, or if they have opposite signs.
#[inline]
pub fn geomspace<T: Float, B: BackendOps<T> + Default>(
    start: T,
    end: T,
    n: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    if n == 0 {
        return Err(B::Error::from(BackendError::Storage {
            operation: "geomspace",
            reason: "number of samples must be greater than zero".to_owned(),
        }));
    }
    let start_f = <T as Scalar>::to_f64(start);
    let end_f = <T as Scalar>::to_f64(end);
    if start_f == 0.0 || end_f == 0.0 {
        return Err(B::Error::from(BackendError::Storage {
            operation: "geomspace",
            reason: "start and end must be non-zero".to_owned(),
        }));
    }
    if start_f.signum() != end_f.signum() {
        return Err(B::Error::from(BackendError::Storage {
            operation: "geomspace",
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
    let values: Vec<T> = (0..n)
        .map(|i| {
            if n == 1 {
                start
            } else {
                <T as Scalar>::from_f64(sign * start_abs * ratio.powf(i as f64))
            }
        })
        .collect();
    Tensor::from_slice_on(vec![n], &values, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    #[test]
    fn linspace_endpoints_inclusive() {
        let b = SequentialBackend::new();
        let t = linspace(0.0f32, 1.0, 5, &b).expect("run operation");
        let s = t.as_slice();
        assert!((s[0] - 0.0).abs() < 1e-6);
        assert!((s[4] - 1.0).abs() < 1e-6);
        assert!((s[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn logspace_base10() {
        let b = SequentialBackend::new();
        let t = logspace(0.0f32, 2.0, 3, 10.0, &b).expect("run operation");
        let s = t.as_slice();
        assert!((s[0] - 1.0).abs() < 1e-4);
        assert!((s[1] - 10.0).abs() < 1e-4);
        assert!((s[2] - 100.0).abs() < 1e-4);
    }

    #[test]
    fn geomspace_doubling() {
        let b = SequentialBackend::new();
        let t = geomspace(1.0f32, 16.0, 5, &b).expect("run operation");
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
        let t = linspace(3.5f32, 99.0, 1, &b).expect("run operation");
        assert!((t.as_slice()[0] - 3.5_f32).abs() < 1e-5);
    }
}
