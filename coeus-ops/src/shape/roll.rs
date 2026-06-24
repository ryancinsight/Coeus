// ── roll — circular shift along dimensions ──
//
// Matches `torch.roll(input, shifts, dims)`.
// Elements that would be shifted beyond the boundary wrap around.

use crate::backend_ops::BackendOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Circular shift `input` along each of `dims` by the corresponding `shifts`.
///
/// For a 1-D tensor `[0, 1, 2, 3]` with `shift=1`:
/// `roll([0,1,2,3], 1, 0)` → `[3, 0, 1, 2]`.
///
/// When `shifts` and `dims` have more than one element, the shifts are applied
/// sequentially from left to right.
///
/// # Panics
/// - Panics if `shifts.len() != dims.len()`.
/// - Panics if any `dim >= input.ndim()`.
#[inline]
pub fn roll<T: Scalar, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    shifts: &[isize],
    dims: &[usize],
    _backend: &B,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(
        shifts.len(),
        dims.len(),
        "roll: shifts and dims must have equal length"
    );

    let ndim = input.ndim();
    let shape = input.shape();
    let numel: usize = shape.iter().product();

    // Build a lookup table: output_flat_idx → input_flat_idx.
    // We iterate output positions and compute the corresponding input position
    // after reverse-rolling (each element at output_pos came from
    // input_pos = (output_pos - shift + n) % n along that dim).
    let out_vec: Vec<T> = (0..numel)
        .map(|flat| {
            let mut idx = super::index::flat_to_nd(flat, shape);
            // Apply each (shift, dim) in reverse order to find source position.
            for (&shift, &dim) in shifts.iter().zip(dims.iter()) {
                assert!(dim < ndim, "roll: dim {dim} out of range for {ndim}-D tensor");
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

    Tensor::from_slice(shape.to_vec(), &out_vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn roll_1d_shift_1() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![4], &[0.0f32, 1.0, 2.0, 3.0]);
        let out = roll(&x, &[1], &[0], &b);
        assert_eq!(out.as_slice(), &[3.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn roll_1d_negative_shift() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![4], &[0.0f32, 1.0, 2.0, 3.0]);
        let out = roll(&x, &[-1], &[0], &b);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn roll_2d_shift_row() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // roll along rows by 1: row 1 → row 0, row 0 → row 1
        let out = roll(&x, &[1], &[0], &b);
        assert_eq!(out.as_slice(), &[4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn roll_shift_zero_is_identity() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![3], &[7.0f32, 8.0, 9.0]);
        let out = roll(&x, &[0], &[0], &b);
        assert_eq!(out.as_slice(), &[7.0, 8.0, 9.0]);
    }
}
