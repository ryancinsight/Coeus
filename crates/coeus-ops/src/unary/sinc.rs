// -- Unnormalized sinc: sin(x)/x with sinc(0) = 1 --
//
// There is no provider `Sinc` opcode, and the `where_cond` composition
// cannot patch the removable singularity: it evaluates
// `on_true * mask + on_false * (1 - mask)`, and at `x = 0` the `on_false`
// lane holds `sin(0)/0 = NaN`, so `NaN * 0 = NaN` poisons the sum. Fixing
// that composition would change existing `where_cond` semantics, so this
// entry point evaluates the scalar sinc on the host after a shape check —
// the same placement as `atan2` in `binary::transcendental` — which is
// NaN-safe by construction.
//
// Matches `numpy.sinc` without the pi scaling (unnormalized `sin(x)/x`).
// Callers needing the `torch.sinc` normalized form compose `sinc(pi * x)`
// themselves.

use crate::backend_ops::BackendOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Float, FloatElement};
use coeus_tensor::Tensor;

/// Element-wise unnormalized sinc: `sin(x)/x`, `sinc(0) = 1`.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::sinc;
///
/// let backend = SequentialBackend::new();
/// let x = Tensor::<f64, SequentialBackend>::from_slice([2], &[0.0, 1.0])
///     .expect("example tensor allocation succeeds");
/// let s = sinc(&x, &backend).expect("example sinc succeeds");
/// assert_eq!(s.as_slice()[0], 1.0);
/// assert!((s.as_slice()[1] - 1.0f64.sin()).abs() < 1e-12);
/// ```
#[inline]
pub fn sinc<T: Float + FloatElement, B: BackendOps<T> + Default>(
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
        let x = v.to_f64();
        let s = if x == 0.0 { 1.0 } else { x.sin() / x };
        out.push(<T as FloatElement>::from_f64(s));
    }
    Tensor::from_slice_on(input.shape().to_vec(), &out, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    #[test]
    fn sinc_removable_singularity_is_one() {
        let backend = SequentialBackend::new();
        let x = Tensor::<f64, SequentialBackend>::from_slice([5], &[0.0, 1.0, -1.0, 2.0, -0.0])
            .expect("invariant: test backend operation succeeds");
        let out = sinc(&x, &backend).expect("invariant: test operation succeeds");
        assert_eq!(out.as_slice()[0], 1.0);
        assert_eq!(out.as_slice()[4], 1.0);
        assert!((out.as_slice()[1] - 1.0f64.sin()).abs() < 1e-12);
        assert!((out.as_slice()[2] - (-1.0f64).sin() / -1.0).abs() < 1e-12);
        assert!((out.as_slice()[3] - 2.0f64.sin() / 2.0).abs() < 1e-12);
    }
}
