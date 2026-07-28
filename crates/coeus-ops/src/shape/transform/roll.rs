// ── roll — circular shift along dimensions ──
//
// Matches `torch.roll(input, shifts, dims)`.
// Elements that would be shifted beyond the boundary wrap around.

use crate::backend_ops::BackendOps;
use coeus_core::{BackendError, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Circular shift `input` along each of `dims` by the corresponding `shifts`.
///
/// For a 1-D tensor `[0, 1, 2, 3]` with `shift=1`:
/// `roll([0,1,2,3], 1, 0)` → `[3, 0, 1, 2]`.
///
/// When `shifts` and `dims` have more than one element, the shifts are applied
/// sequentially from left to right.
///
/// # Errors
/// Returns a backend error when shift/axis metadata is invalid or
/// materialization fails.
#[inline]
pub fn roll<T: Scalar, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    shifts: &[isize],
    dims: &[usize],
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    if shifts.len() != dims.len() {
        return Err(B::Error::from(BackendError::Storage {
            operation: "roll",
            reason: "shifts and dims must have equal length".to_owned(),
        }));
    }

    let ndim = input.ndim();
    let shape = input.shape();
    let numel = shape.iter().try_fold(1usize, |count, &extent| {
        count.checked_mul(extent).ok_or_else(|| {
            B::Error::from(BackendError::Overflow {
                operation: "roll",
                reason: "element count",
            })
        })
    })?;

    for &dim in dims {
        if dim >= ndim {
            return Err(B::Error::from(BackendError::AxisOutOfRange {
                operation: "roll",
                axis: dim,
                rank: ndim,
            }));
        }
        if shape[dim] == 0 {
            return Err(B::Error::from(BackendError::Storage {
                operation: "roll",
                reason: format!("axis {dim} has zero extent"),
            }));
        }
    }

    // Build a lookup table: output_flat_idx → input_flat_idx.
    // We iterate output positions and compute the corresponding input position
    // after reverse-rolling (each element at output_pos came from
    // input_pos = (output_pos - shift + n) % n along that dim).
    let out_vec: Vec<T> = (0..numel)
        .map(|flat| {
            let mut idx = crate::shape::flat_to_nd(flat, shape);
            // Apply each (shift, dim) in reverse order to find source position.
            for (&shift, &dim) in shifts.iter().zip(dims.iter()) {
                let n = shape[dim] as isize;
                // Normalise shift to [0, n) to handle negatives and large values.
                let eff_shift = ((shift % n) + n) % n;
                // Source index: rolled-back by eff_shift (i.e. source is at
                // (out_idx - shift + n) % n in the rolling direction).
                let src = ((idx[dim] as isize - eff_shift + n) % n) as usize;
                idx[dim] = src;
            }
            input.get(&idx)
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
    fn roll_1d_shift_1() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![4], &[0.0f32, 1.0, 2.0, 3.0]).expect("construct tensor");
        let out = roll(&x, &[1], &[0], &b).expect("run operation");
        assert_eq!(out.as_slice(), &[3.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn roll_1d_negative_shift() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![4], &[0.0f32, 1.0, 2.0, 3.0]).expect("construct tensor");
        let out = roll(&x, &[-1], &[0], &b).expect("run operation");
        assert_eq!(out.as_slice(), &[1.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn roll_2d_shift_row() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("construct tensor");
        // roll along rows by 1: row 1 → row 0, row 0 → row 1
        let out = roll(&x, &[1], &[0], &b).expect("run operation");
        assert_eq!(out.as_slice(), &[4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn roll_shift_zero_is_identity() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![3], &[7.0f32, 8.0, 9.0]).expect("construct tensor");
        let out = roll(&x, &[0], &[0], &b).expect("run operation");
        assert_eq!(out.as_slice(), &[7.0, 8.0, 9.0]);
    }
}
