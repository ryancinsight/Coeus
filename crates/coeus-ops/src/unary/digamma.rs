// -- Digamma: logarithmic derivative of gamma, d/dx ln|Gamma(x)| --
//
// The provider opcode layer has no `Digamma` kernel (neither Leto nor the
// Hephaestus seams expose one), so this entry point evaluates Eunomia's
// scalar `FloatElement::digamma` on the host after materializing a
// contiguous view — the same placement as `atan2` in `binary::transcendental`.
// This closes the `G-049` residual honestly: `lgamma` backward no longer
// raises `NotImplementedError`; it composes `digamma(x) * grad_out` through
// this entry point.
//
// Domain: negative integers and `-inf` give NaN; `+-0` give `-+inf`;
// `+inf` stays `+inf` — all inherited from `FloatElement::digamma`.

use crate::backend_ops::BackendOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Float, FloatElement};
use coeus_tensor::Tensor;

/// Element-wise digamma: `psi(x) = d/dx ln|Gamma(x)|`.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::digamma;
///
/// let backend = SequentialBackend::new();
/// let x = Tensor::<f64, SequentialBackend>::from_slice([2], &[1.0, 2.0])
///     .expect("example tensor allocation succeeds");
/// let p = digamma(&x, &backend).expect("example digamma succeeds");
/// assert!((p.as_slice()[0] - (-0.577_215_664_901_532_9)).abs() < 1e-9);
/// assert!((p.as_slice()[1] - (1.0 - 0.577_215_664_901_532_9)).abs() < 1e-9);
/// ```
#[inline]
pub fn digamma<T: Float + FloatElement, B: BackendOps<T> + Default>(
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
        out.push(v.digamma());
    }
    Tensor::from_slice_on(input.shape().to_vec(), &out, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    #[test]
    fn digamma_at_one_is_neg_euler_mascheroni() {
        let backend = SequentialBackend::new();
        let x = Tensor::<f64, SequentialBackend>::from_slice([2], &[1.0, 2.0])
            .expect("invariant: test backend operation succeeds");
        let out = digamma(&x, &backend).expect("invariant: test operation succeeds");
        assert!((out.as_slice()[0] - (-0.577_215_664_901_532_9)).abs() < 1e-9);
        assert!((out.as_slice()[1] - (1.0 - 0.577_215_664_901_532_9)).abs() < 1e-9);
    }
}
