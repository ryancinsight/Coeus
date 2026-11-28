// -- Bessel functions of the first kind: J0, J1 --
//
// There is no provider `Bessel` opcode, so these entry points evaluate
// Eunomia's scalar `RealField::bessel_j0`/`bessel_j1` on the host after
// materializing a contiguous view — the same placement as `digamma` in
// `unary::digamma` and `atan2` in `binary::transcendental`. `f32`/`f64`
// only: `RealField` is implemented for the native precisions, not the
// reduced-storage wrappers.

use crate::backend_ops::BackendOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Float};
use coeus_tensor::Tensor;
use eunomia::RealField;

/// Element-wise Bessel function of the first kind, order zero: `J₀(x)`.
///
/// Even: `J₀(−x) = J₀(x)`.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::bessel_j0;
///
/// let backend = SequentialBackend::new();
/// let x = Tensor::<f64, SequentialBackend>::from_slice([2], &[0.0, 1.0])
///     .expect("example tensor allocation succeeds");
/// let j = bessel_j0(&x, &backend).expect("example Bessel succeeds");
/// assert_eq!(j.as_slice()[0], 1.0);
/// assert!((j.as_slice()[1] - 0.765_197_686_557_966_6).abs() < 1e-12);
/// ```
#[inline]
pub fn bessel_j0<T: Float + RealField, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let host = input.to_contiguous_on(backend)?;
    let slice = host.as_slice();
    let mut out: Vec<T> = Vec::with_capacity(slice.len());
    for &v in slice.iter() {
        out.push(v.bessel_j0());
    }
    Tensor::from_slice_on(input.shape().to_vec(), &out, backend)
}

/// Element-wise Bessel function of the first kind, order one: `J₁(x)`.
///
/// Odd: `J₁(−x) = −J₁(x)`.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::bessel_j1;
///
/// let backend = SequentialBackend::new();
/// let x = Tensor::<f64, SequentialBackend>::from_slice([2], &[0.0, 1.0])
///     .expect("example tensor allocation succeeds");
/// let j = bessel_j1(&x, &backend).expect("example Bessel succeeds");
/// assert_eq!(j.as_slice()[0], 0.0);
/// assert!((j.as_slice()[1] - 0.440_050_585_744_933_6).abs() < 1e-12);
/// ```
#[inline]
pub fn bessel_j1<T: Float + RealField, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let host = input.to_contiguous_on(backend)?;
    let slice = host.as_slice();
    let mut out: Vec<T> = Vec::with_capacity(slice.len());
    for &v in slice.iter() {
        out.push(v.bessel_j1());
    }
    Tensor::from_slice_on(input.shape().to_vec(), &out, backend)
}

/// Element-wise modified Bessel function of the second kind, order zero: `K₀(x)`.
///
/// Diverges at the origin and is undefined for `x ≤ 0`; non-positive and
/// non-finite arguments yield `NaN` (inherited from the scalar contract).
///
/// Forward-only: no tracked autograd wrapper. The derivative is
/// `d/dx K₀(x) = −K₁(x)`, and the provider surface exposes no `K₁`, so a
/// tracked node would need an invented kernel. Revisit when Eunomia gains
/// `bessel_k1`.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::bessel_k0;
///
/// let backend = SequentialBackend::new();
/// let x = Tensor::<f64, SequentialBackend>::from_slice([1], &[1.0])
///     .expect("example tensor allocation succeeds");
/// let k = bessel_k0(&x, &backend).expect("example Bessel succeeds");
/// assert!((k.as_slice()[0] - 0.421_024_438_240_708_4).abs() < 1e-6);
/// ```
#[inline]
pub fn bessel_k0<T: Float + RealField, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let host = input.to_contiguous_on(backend)?;
    let slice = host.as_slice();
    let mut out: Vec<T> = Vec::with_capacity(slice.len());
    for &v in slice.iter() {
        out.push(v.bessel_k0());
    }
    Tensor::from_slice_on(input.shape().to_vec(), &out, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    #[test]
    fn bessel_j0_at_zero_is_one() {
        let backend = SequentialBackend::new();
        let x = Tensor::<f64, SequentialBackend>::from_slice([3], &[0.0, 1.0, -1.0])
            .expect("invariant: test backend operation succeeds");
        let out = bessel_j0(&x, &backend).expect("invariant: test operation succeeds");
        assert_eq!(out.as_slice()[0], 1.0);
        assert!((out.as_slice()[1] - 0.765_197_686_557_966_6).abs() < 1e-12);
        assert_eq!(out.as_slice()[1], out.as_slice()[2]);
    }

    #[test]
    fn bessel_j1_at_zero_is_zero_and_odd() {
        let backend = SequentialBackend::new();
        let x = Tensor::<f64, SequentialBackend>::from_slice([3], &[0.0, 1.0, -1.0])
            .expect("invariant: test backend operation succeeds");
        let out = bessel_j1(&x, &backend).expect("invariant: test operation succeeds");
        assert_eq!(out.as_slice()[0], 0.0);
        assert!((out.as_slice()[1] - 0.440_050_585_744_933_6).abs() < 1e-12);
        assert_eq!(out.as_slice()[1], -out.as_slice()[2]);
    }

    #[test]
    fn bessel_k0_matches_reference_and_rejects_nonpositive() {
        // Tolerance 1e-6: the provider documents absolute error below 1e-7
        // on this domain (A&S 9.8.5), so 1e-6 is an order of magnitude of
        // margin, not a tight pin of the polynomial coefficients.
        let backend = SequentialBackend::new();
        let x = Tensor::<f64, SequentialBackend>::from_slice([3], &[1.0, 2.0, 0.0])
            .expect("invariant: test backend operation succeeds");
        let out = bessel_k0(&x, &backend).expect("invariant: test operation succeeds");
        assert!((out.as_slice()[0] - 0.421_024_438_240_708_4).abs() < 1e-6);
        assert!((out.as_slice()[1] - 0.113_893_872_749_533).abs() < 1e-6);
        assert!(out.as_slice()[2].is_nan());
    }
}
