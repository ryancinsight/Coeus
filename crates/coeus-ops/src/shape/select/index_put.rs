// ── index_put ──
//
// Scatter assignment: assigns `values` at positions specified by `indices`.
// Equivalent to `torch.index_put_(input, indices, values, accumulate)`.
//
// This implementation supports 1-D integer index tensors applied to the first
// dimension (the most common use case in embeddings and selection).

use crate::BackendOps;
use coeus_core::{BackendError, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Scatter-assign `values` into `input` at row indices given by `indices`.
///
/// `indices` is a 1-D tensor of non-negative integer row indices (stored as `T`).
/// `values` must have shape `[len(indices), *input.shape[1..]]`.
///
/// When `accumulate = true`, values are added to existing entries instead of
/// replacing them.
///
/// Returns a new tensor with the assignments applied.
///
/// Equivalent to `torch.index_put(input, (indices,), values, accumulate)`.
#[inline]
pub fn index_put<T: Scalar, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    indices: &Tensor<T, B>,
    values: &Tensor<T, B>,
    accumulate: bool,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    if indices.ndim() != 1 {
        return Err(B::Error::from(BackendError::UnsupportedRank {
            operation: "index_put",
            rank: indices.ndim(),
            max_rank: 1,
        }));
    }
    let n_idx = indices.shape()[0];
    if input.ndim() == 0 {
        return Err(B::Error::from(BackendError::UnsupportedRank {
            operation: "index_put",
            rank: 0,
            max_rank: usize::MAX,
        }));
    }

    // Row size: product of all dimensions except dim 0.
    let row_size = input.shape()[1..]
        .iter()
        .try_fold(1usize, |size, &extent| size.checked_mul(extent))
        .ok_or_else(|| {
            B::Error::from(BackendError::Overflow {
                operation: "index_put",
                reason: "row element count",
            })
        })?;
    let expected = n_idx.checked_mul(row_size).ok_or_else(|| {
        B::Error::from(BackendError::Overflow {
            operation: "index_put",
            reason: "values element count",
        })
    })?;
    if values.numel() != expected {
        return Err(B::Error::from(BackendError::ShapeMismatch {
            operation: "index_put",
            lhs: vec![values.numel()],
            rhs: vec![expected],
        }));
    }

    // Copy input to host, apply updates, copy back.
    let numel = input.numel();
    let mut host = vec![T::zero(); numel];
    backend.copy_to_host(input.storage(), &mut host)?;

    let idx_cont = indices.to_contiguous()?;
    let val_cont = values.to_contiguous()?;
    let idx_s = idx_cont.as_slice();
    let val_s = val_cont.as_slice();

    let n_rows = input.shape()[0];
    for (src_row, &idx_val) in idx_s.iter().enumerate() {
        let raw_index = <T as Scalar>::to_f64(idx_val);
        if !raw_index.is_finite() || raw_index < 0.0 || raw_index.fract() != 0.0 {
            return Err(B::Error::from(BackendError::Storage {
                operation: "index_put",
                reason: format!("index {raw_index} is not a non-negative integer"),
            }));
        }
        let row = raw_index as usize;
        if row >= n_rows {
            return Err(B::Error::from(BackendError::Storage {
                operation: "index_put",
                reason: format!("index {row} out of range for axis 0 of size {n_rows}"),
            }));
        }
        let dst_start = row * row_size;
        let src_start = src_row * row_size;
        for k in 0..row_size {
            if accumulate {
                host[dst_start + k] += val_s[src_start + k];
            } else {
                host[dst_start + k] = val_s[src_start + k];
            }
        }
    }

    Tensor::from_slice_on(input.shape().to_vec(), &host, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn index_put_replace() {
        let b = SequentialBackend::new();
        let x = Tensor::<f32, SequentialBackend>::from_slice(vec![4], &[1.0, 2.0, 3.0, 4.0]).expect("construct tensor");
        let idx = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0, 3.0]).expect("construct tensor");
        let vals = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[10.0, 20.0]).expect("construct tensor");
        let out = index_put(&x, &idx, &vals, false, &b).expect("run operation");
        assert_eq!(out.as_slice(), &[1.0, 10.0, 3.0, 20.0]);
    }

    #[test]
    fn index_put_accumulate() {
        let b = SequentialBackend::new();
        let x = Tensor::<f32, SequentialBackend>::from_slice(vec![3], &[1.0, 2.0, 3.0]).expect("construct tensor");
        let idx = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[0.0, 0.0]).expect("construct tensor");
        let vals = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[5.0, 3.0]).expect("construct tensor");
        let out = index_put(&x, &idx, &vals, true, &b).expect("run operation");
        // 1.0 + 5.0 + 3.0 = 9.0
        assert!((out.as_slice()[0] - 9.0).abs() < 1e-6);
    }
}
