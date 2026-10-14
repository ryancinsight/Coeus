//! Host staging for small host-consumed scalar reductions.
//!
//! Scalar reductions whose result crosses to the host anyway (`sum`, `mean`,
//! ...) route contiguous inputs below the backend crossover through one
//! bulk download plus the CPU reduction instead of a device kernel plus a
//! scalar readback. Resident results consumed by further device ops never
//! stage: pipelined dispatch has no host round trip to beat.

use coeus_core::{BackendError, ComputeBackend, Scalar, SequentialBackend, Storage};
use coeus_tensor::Tensor;

/// Run a scalar reduction on a host-staged mirror of a device tensor.
///
/// Downloads the full device storage once, mirrors the contiguous input on
/// the host, and runs `reduce` (the same scalar function) against
/// [`SequentialBackend`], whose zero crossover terminates the routing.
/// Zero-copy when the device storage length equals the element count; one
/// slice copy for oversized shared storage.
pub(crate) fn staged_scalar<T, B, F, R>(
    a: &Tensor<T, B>,
    backend: &B,
    reduce: F,
) -> Result<R, B::Error>
where
    T: Scalar,
    B: ComputeBackend,
    F: FnOnce(&Tensor<T, SequentialBackend>, &SequentialBackend) -> Result<R, BackendError>,
{
    let seq = SequentialBackend::new();
    let staged = stage_contiguous_to_host(a, backend)?;
    reduce(&staged, &seq).map_err(B::Error::from)
}

/// Download a contiguous offset-0 tensor into an equivalent host tensor.
///
/// The caller checks contiguity, offset, and the backend crossover.
fn stage_contiguous_to_host<T, B>(
    a: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, SequentialBackend>, B::Error>
where
    T: Scalar,
    B: ComputeBackend,
{
    let n = a.numel();
    let storage_len = a.storage().len();
    let mut buf = vec![T::ZERO; storage_len];
    backend.copy_to_host(a.storage(), &mut buf)?;
    if storage_len == n {
        Tensor::<T, SequentialBackend>::from_vec(buf).map_err(B::Error::from)
    } else {
        let head = buf.get(..n).ok_or_else(|| {
            B::Error::from(BackendError::Storage {
                operation: "stage_contiguous_to_host",
                reason: format!(
                    "contiguous view of {n} elements exceeds {storage_len}-element storage"
                ),
            })
        })?;
        Tensor::<T, SequentialBackend>::from_slice([n], head).map_err(B::Error::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::ComputeBackend;

    #[test]
    fn host_backend_disables_staging() {
        assert_eq!(SequentialBackend::new().host_staging_crossover(), 0);
    }
}
