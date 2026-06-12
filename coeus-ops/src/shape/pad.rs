// ── Constant padding ──
// Pads a tensor with a constant value along each dimension.

use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Pad `x` with `value` along each dimension.
///
/// `pads` is a list of `(before, after)` pairs, one per dimension.
/// Zero padding on all sides is equivalent to a clone.
///
/// # Panics
/// - `pads.len() != x.ndim()`.
#[inline]
pub fn pad<T: Scalar, B: ComputeBackend + Default>(
    x: &Tensor<T, B>,
    pads: &[(usize, usize)],
    value: T,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = x.ndim();
    assert_eq!(pads.len(), ndim, "pad: pads.len() must equal tensor ndim");

    // Fast path: no padding anywhere.
    if pads.iter().all(|&(b, a)| b == 0 && a == 0) {
        return x.clone();
    }

    let backend = B::default();
    let mut out_shape = x.shape_cloned();
    for d in 0..ndim {
        out_shape[d] += pads[d].0 + pads[d].1;
    }

    let values = coeus_leto::pad_values(x.layout(), x.storage().as_slice(), pads, value)
        .expect("coeus-leto pad failed");
    Tensor::from_slice_on(out_shape, &values, &backend)
}
