// ── repeat / tile — replicate tensor elements or the whole tensor ──
//
// `tile(input, reps)` replicates `input` along each dimension according to
// `reps`, matching `torch.Tensor.repeat(repeats)` / `np.tile(input, reps)`.
//
// The operation conceptually concatenates `reps[d]` copies of `input`
// along dimension `d` for each `d`.
//
// Note: PyTorch calls this `Tensor.repeat(repeats)` (not `tile`); NumPy calls
// it `np.tile`.  We expose both names but implement the same operation.

use crate::backend_ops::BackendOps;
use coeus_core::{BackendError, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Tile `input` by repeating it `reps[d]` times along each dimension `d`.
///
/// Output shape: `[input.shape[d] * reps[d] for d in 0..ndim]`.
///
/// If `reps.len() < input.ndim()`, `reps` is prepended with ones.
/// If `reps.len() > input.ndim()`, `input` is treated as having leading
/// size-1 dimensions.
///
/// # Errors
/// Returns a backend error when `reps` is empty, dimensions overflow, or
/// materialization fails.
#[inline]
pub fn tile<T: Scalar, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    reps: &[usize],
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    if reps.is_empty() {
        return Err(B::Error::from(BackendError::Storage {
            operation: "tile",
            reason: "reps must be non-empty".to_owned(),
        }));
    }
    let in_shape = input.shape();
    let in_ndim = in_shape.len();

    // Pad input shape with leading 1s if reps.len() > in_ndim.
    let ndim = in_ndim.max(reps.len());
    let pad_in = ndim - in_ndim;
    let pad_reps = ndim - reps.len();

    // Effective shapes after padding.
    let eff_in: Vec<usize> = (0..ndim)
        .map(|d| if d < pad_in { 1 } else { in_shape[d - pad_in] })
        .collect();
    let eff_reps: Vec<usize> = (0..ndim)
        .map(|d| if d < pad_reps { 1 } else { reps[d - pad_reps] })
        .collect();

    let mut out_shape = Vec::with_capacity(ndim);
    for d in 0..ndim {
        out_shape.push(eff_in[d].checked_mul(eff_reps[d]).ok_or_else(|| {
            B::Error::from(BackendError::Overflow {
                operation: "tile",
                reason: "output dimension",
            })
        })?);
    }
    let total = out_shape.iter().try_fold(1usize, |count, &extent| {
        count.checked_mul(extent).ok_or_else(|| {
            B::Error::from(BackendError::Overflow {
                operation: "tile",
                reason: "output element count",
            })
        })
    })?;

    let in_cont = if pad_in > 0 {
        // Reshape to eff_in shape.
        let _in_numel: usize = in_shape.iter().product();
        let in_c = input.to_contiguous()?;
        // Build a tensor with eff_in shape pointing to the same data.
        Tensor::from_slice_on(eff_in.clone(), in_c.as_slice(), backend)?
    } else {
        input.to_contiguous()?
    };
    let in_s = in_cont.as_slice();

    // Compute input row-major strides.
    let mut in_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        in_strides[d] = in_strides[d + 1] * eff_in[d + 1];
    }

    // Compute output row-major strides.
    let mut out_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
    }

    let data: Vec<T> = (0..total)
        .map(|flat| {
            let mut in_flat = 0usize;
            let mut rem = flat;
            for d in 0..ndim {
                let out_coord = rem / out_strides[d];
                rem %= out_strides[d];
                let in_coord = out_coord % eff_in[d];
                in_flat += in_coord * in_strides[d];
            }
            in_s[in_flat]
        })
        .collect();

    Tensor::from_slice_on(out_shape, &data, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn tile_1d_repeats_twice() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![3], &[1.0f32, 2.0, 3.0]).expect("construct tensor");
        let out = tile(&x, &[2], &b).expect("run operation");
        assert_eq!(out.shape(), &[6]);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn tile_2d_repeats_both_dims() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![2, 2], &[1.0f32, 2.0, 3.0, 4.0]).expect("construct tensor");
        let out = tile(&x, &[2, 3], &b).expect("run operation");
        assert_eq!(out.shape(), &[4, 6]);
        // Row 0 = [1,2,1,2,1,2], Row 1 = [3,4,3,4,3,4], Row 2=Row 0, Row 3=Row 1
        assert_eq!(out.as_slice()[0..6], [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        assert_eq!(out.as_slice()[6..12], [3.0, 4.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn tile_identity_reps_all_ones() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("construct tensor");
        let out = tile(&x, &[1, 1], &b).expect("run operation");
        assert_eq!(out.shape(), x.shape());
        assert_eq!(out.as_slice(), x.as_slice());
    }

    #[test]
    fn tile_adds_leading_dim_when_reps_longer() {
        let b = SequentialBackend::new();
        let x = Tensor::from_slice(vec![3], &[1.0f32, 2.0, 3.0]).expect("construct tensor");
        // reps=[2,2] is longer than ndim=1: treat input as [1,3], tile → [2,6]
        let out = tile(&x, &[2, 2], &b).expect("run operation");
        assert_eq!(out.shape(), &[2, 6]);
    }
}
