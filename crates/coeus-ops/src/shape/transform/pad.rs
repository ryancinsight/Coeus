// ── Constant padding ──
// Pads a tensor with a constant value along each dimension.

use coeus_core::{
    BackendError, ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar,
};
use coeus_tensor::Tensor;

/// Pad `x` with `value` along each dimension.
///
/// `pads` is a list of `(before, after)` pairs, one per dimension.
/// Zero padding on all sides is equivalent to a clone.
///
/// # Errors
/// Returns a backend error for invalid padding metadata or materialization
/// failure.
#[inline]
pub fn pad<T: Scalar, B: ComputeBackend + Default>(
    x: &Tensor<T, B>,
    pads: &[(usize, usize)],
    value: T,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = x.ndim();
    if pads.len() != ndim {
        return Err(B::Error::from(BackendError::LayoutRankMismatch {
            operation: "pad",
            lhs: ndim,
            rhs: pads.len(),
        }));
    }

    // Fast path: no padding anywhere.
    if pads.iter().all(|&(b, a)| b == 0 && a == 0) {
        return Ok(x.clone());
    }

    let backend = B::default();
    let mut out_shape = x.shape_cloned();
    for d in 0..ndim {
        out_shape[d] = out_shape[d]
            .checked_add(pads[d].0)
            .and_then(|extent| extent.checked_add(pads[d].1))
            .ok_or_else(|| {
                B::Error::from(BackendError::Overflow {
                    operation: "pad",
                    reason: "output dimension",
                })
            })?;
    }

    let values = coeus_leto::pad_values(x.layout(), x.storage().as_slice(), pads, value).map_err(
        |error| {
            B::Error::from(BackendError::Storage {
                operation: "pad",
                reason: error.to_string(),
            })
        },
    )?;
    Tensor::from_slice_on(out_shape, &values, &backend)
}
