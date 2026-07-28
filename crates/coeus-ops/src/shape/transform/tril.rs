// ── tril / triu — triangular masking ──
//
// Both functions operate on the last two dimensions of any ≥2-D tensor,
// matching `torch.tril(input, diagonal=k)` and `torch.triu(input, diagonal=k)`.

use crate::backend_ops::BackendOps;
use coeus_core::{BackendError, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Return the lower-triangular part of `input`, zeroing elements above the
/// `k`-th diagonal.
///
/// For a matrix element at row `i`, column `j`:
/// - `k = 0` (default) keeps elements where `j <= i`.
/// - `k > 0` keeps elements where `j <= i + k` (keeps `k` super-diagonals).
/// - `k < 0` keeps elements where `j <= i + k` (zeros `|k|` sub-diagonals).
///
/// For ND tensors (`ndim > 2`) the operation is applied independently to each
/// `[rows, cols]` slice formed by the last two dimensions.
///
/// # Errors
/// Returns a backend error when the input rank is less than two or
/// materialization fails.
#[inline]
pub fn tril<T: Scalar, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    k: isize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = input.ndim();
    if ndim < 2 {
        return Err(B::Error::from(BackendError::UnsupportedRank {
            operation: "tril",
            rank: ndim,
            max_rank: usize::MAX,
        }));
    }

    let shape = input.shape();
    let rows = shape[ndim - 2];
    let cols = shape[ndim - 1];
    let numel: usize = shape.iter().product();

    let out_vec: Vec<T> = (0..numel)
        .map(|flat| {
            let idx = crate::shape::flat_to_nd(flat, shape);
            let row = idx[ndim - 2] as isize;
            let col = idx[ndim - 1] as isize;
            // Keep element only when col <= row + k (lower-triangular part).
            if col <= row + k {
                input.get(&idx)
            } else {
                T::zero()
            }
        })
        .collect();

    let _ = (rows, cols);
    Tensor::from_slice_on(shape.to_vec(), &out_vec, backend)
}

/// Return the upper-triangular part of `input`, zeroing elements below the
/// `k`-th diagonal.
///
/// For a matrix element at row `i`, column `j`:
/// - `k = 0` (default) keeps elements where `j >= i`.
/// - `k > 0` keeps elements where `j >= i + k` (zeros `k` super-diagonals below).
/// - `k < 0` keeps elements where `j >= i + k` (keeps `|k|` sub-diagonals).
///
/// # Errors
/// Returns a backend error when the input rank is less than two or
/// materialization fails.
#[inline]
pub fn triu<T: Scalar, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    k: isize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = input.ndim();
    if ndim < 2 {
        return Err(B::Error::from(BackendError::UnsupportedRank {
            operation: "triu",
            rank: ndim,
            max_rank: usize::MAX,
        }));
    }

    let shape = input.shape();
    let rows = shape[ndim - 2];
    let cols = shape[ndim - 1];
    let numel: usize = shape.iter().product();

    let out_vec: Vec<T> = (0..numel)
        .map(|flat| {
            let idx = crate::shape::flat_to_nd(flat, shape);
            let row = idx[ndim - 2] as isize;
            let col = idx[ndim - 1] as isize;
            // Keep element only when col >= row + k (upper-triangular part).
            if col >= row + k {
                input.get(&idx)
            } else {
                T::zero()
            }
        })
        .collect();

    let _ = (rows, cols);
    Tensor::from_slice_on(shape.to_vec(), &out_vec, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor;

    fn mat() -> Tensor<f32, SequentialBackend> {
        Tensor::from_slice(
            vec![3, 3],
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ).expect("construct tensor")
    }

    #[test]
    fn tril_k0_zeroes_above_main_diagonal() {
        let b = SequentialBackend::new();
        let out = tril(&mat(), 0, &b).expect("run operation");
        assert_eq!(
            out.as_slice(),
            &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn tril_k1_keeps_one_superdiagonal() {
        let b = SequentialBackend::new();
        let out = tril(&mat(), 1, &b).expect("run operation");
        assert_eq!(
            out.as_slice(),
            &[1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn tril_k_neg1_zeroes_main_and_above() {
        let b = SequentialBackend::new();
        let out = tril(&mat(), -1, &b).expect("run operation");
        assert_eq!(
            out.as_slice(),
            &[0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0]
        );
    }

    #[test]
    fn triu_k0_zeroes_below_main_diagonal() {
        let b = SequentialBackend::new();
        let out = triu(&mat(), 0, &b).expect("run operation");
        assert_eq!(
            out.as_slice(),
            &[1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]
        );
    }

    #[test]
    fn triu_k1_zeroes_main_and_below() {
        let b = SequentialBackend::new();
        let out = triu(&mat(), 1, &b).expect("run operation");
        assert_eq!(
            out.as_slice(),
            &[0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn tril_triu_together_isolate_main_diagonal() {
        let b = SequentialBackend::new();
        let l = tril(&mat(), 0, &b).expect("run operation");
        // triu(tril(x, 0), 0) picks out only the diagonal
        let diag = triu(&Tensor::from_slice(vec![3, 3], l.as_slice()).expect("construct tensor"), 0, &b).expect("run operation");
        assert_eq!(
            diag.as_slice(),
            &[1.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 9.0]
        );
    }
}
