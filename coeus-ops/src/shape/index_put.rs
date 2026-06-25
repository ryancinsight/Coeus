// ── index_put ──
//
// Scatter assignment: assigns `values` at positions specified by `indices`.
// Equivalent to `torch.index_put_(input, indices, values, accumulate)`.
//
// This implementation supports 1-D integer index tensors applied to the first
// dimension (the most common use case in embeddings and selection).

use crate::BackendOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
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
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(
        indices.ndim(), 1,
        "index_put: indices must be 1-D, got {}-D", indices.ndim()
    );
    let n_idx = indices.shape()[0];
    assert!(
        input.ndim() >= 1,
        "index_put: input must be at least 1-D"
    );

    // Row size: product of all dimensions except dim 0.
    let row_size: usize = input.shape()[1..].iter().product::<usize>().max(1);
    assert_eq!(
        values.numel(), n_idx * row_size,
        "index_put: values.numel()={} must equal len(indices)*row_size={}*{}",
        values.numel(), n_idx, row_size
    );

    // Copy input to host, apply updates, copy back.
    let numel = input.numel();
    let mut host = vec![T::zero(); numel];
    backend.copy_to_host(input.storage(), &mut host);

    let idx_cont = indices.to_contiguous();
    let val_cont = values.to_contiguous();
    let idx_s = idx_cont.as_slice();
    let val_s = val_cont.as_slice();

    let n_rows = input.shape()[0];
    for (src_row, &idx_val) in idx_s.iter().enumerate() {
        let row = idx_val.to_f64() as usize;
        assert!(row < n_rows, "index_put: index {row} out of range for dim 0 size {n_rows}");
        let dst_start = row * row_size;
        let src_start = src_row * row_size;
        for k in 0..row_size {
            if accumulate {
                host[dst_start + k] = host[dst_start + k] + val_s[src_start + k];
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
        let x = Tensor::<f32, SequentialBackend>::from_slice(vec![4], &[1.0, 2.0, 3.0, 4.0]);
        let idx = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0, 3.0]);
        let vals = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[10.0, 20.0]);
        let out = index_put(&x, &idx, &vals, false, &b);
        assert_eq!(out.as_slice(), &[1.0, 10.0, 3.0, 20.0]);
    }

    #[test]
    fn index_put_accumulate() {
        let b = SequentialBackend::new();
        let x = Tensor::<f32, SequentialBackend>::from_slice(vec![3], &[1.0, 2.0, 3.0]);
        let idx = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[0.0, 0.0]);
        let vals = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[5.0, 3.0]);
        let out = index_put(&x, &idx, &vals, true, &b);
        // 1.0 + 5.0 + 3.0 = 9.0
        assert!((out.as_slice()[0] - 9.0).abs() < 1e-6);
    }
}
