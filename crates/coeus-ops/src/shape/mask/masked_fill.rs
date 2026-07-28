// ── masked_fill — replace values under a boolean mask ──

use crate::backend_ops::BackendOps;
use coeus_core::{BackendError, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Return a copy of `input` with elements replaced by `value` wherever
/// `mask[i] != 0`.
///
/// `mask` must have the same shape as `input` and is treated as boolean:
/// non-zero = true, zero = false.
///
/// # Errors
/// Returns a backend error when the input and mask shapes differ.
#[inline]
pub fn masked_fill<T: Scalar, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    mask: &Tensor<T, B>,
    value: T,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    if input.shape() != mask.shape() {
        return Err(B::Error::from(BackendError::ShapeMismatch {
            operation: "masked_fill",
            lhs: input.shape().to_vec(),
            rhs: mask.shape().to_vec(),
        }));
    }

    let shape = input.shape();
    let numel = input.numel();
    let out_vec: Vec<T> = (0..numel)
        .map(|flat| {
            // Inline flat-to-ND index conversion (formerly crate::shape::flat_to_nd).
            let idx: Vec<usize> = {
                let mut f = flat;
                let mut coords = vec![0usize; shape.len()];
                for d in (0..shape.len()).rev() {
                    coords[d] = f % shape[d];
                    f /= shape[d];
                }
                coords
            };
            if mask.get(&idx) != T::zero() {
                value
            } else {
                input.get(&idx)
            }
        })
        .collect();

    Tensor::from_slice_on(shape.to_vec(), &out_vec, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn masked_fill_1d_replaces_masked_elements() {
        let b = SequentialBackend::new();
        let input = Tensor::from_slice(vec![4], &[1.0f32, 2.0, 3.0, 4.0]).expect("construct tensor");
        let mask = Tensor::from_slice(vec![4], &[0.0f32, 1.0, 0.0, -2.0]).expect("construct tensor");
        let out = masked_fill(&input, &mask, 9.0, &b).expect("run operation");
        assert_eq!(out.as_slice(), &[1.0, 9.0, 3.0, 9.0]);
    }

    #[test]
    fn masked_fill_2d_replaces_selected_entries() {
        let b = SequentialBackend::new();
        let input = Tensor::from_slice(vec![2, 2], &[1.0f32, 2.0, 3.0, 4.0]).expect("construct tensor");
        let mask = Tensor::from_slice(vec![2, 2], &[1.0f32, 0.0, 0.0, 1.0]).expect("construct tensor");
        let out = masked_fill(&input, &mask, -1.0, &b).expect("run operation");
        assert_eq!(out.as_slice(), &[-1.0, 2.0, 3.0, -1.0]);
    }

    #[test]
    fn masked_fill_all_zero_mask_is_identity() {
        let b = SequentialBackend::new();
        let input = Tensor::from_slice(vec![3], &[5.0f32, 6.0, 7.0]).expect("construct tensor");
        let mask = Tensor::zeros(vec![3]).expect("construct tensor");
        let out = masked_fill(&input, &mask, 42.0, &b).expect("run operation");
        assert_eq!(out.as_slice(), input.as_slice());
    }

    #[test]
    fn masked_fill_all_one_mask_replaces_everything() {
        let b = SequentialBackend::new();
        let input = Tensor::from_slice(vec![2, 2], &[1.0f32, 2.0, 3.0, 4.0]).expect("construct tensor");
        let mask = Tensor::ones(vec![2, 2]).expect("construct tensor");
        let out = masked_fill(&input, &mask, 8.0, &b).expect("run operation");
        assert_eq!(out.as_slice(), &[8.0, 8.0, 8.0, 8.0]);
    }
}
