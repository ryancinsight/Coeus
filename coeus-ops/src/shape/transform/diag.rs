// ── diag / diagonal — diagonal matrix creation and extraction ──
//
// `diag(v, k)` creates a 2-D matrix from 1-D vector `v` placed on diagonal `k`.
// `diagonal(M, k)` extracts diagonal `k` from 2-D matrix `M` as a 1-D vector.
//
// These match the PyTorch `torch.diag(input, diagonal=k)` semantics.

use crate::backend_ops::BackendOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Create a 2-D diagonal matrix from a 1-D vector `v`.
///
/// `v` is placed on diagonal `k`:
/// - `k = 0` (default) — main diagonal.
/// - `k > 0` — `k`-th super-diagonal.
/// - `k < 0` — `|k|`-th sub-diagonal.
///
/// Output shape: `[n + |k|, n + |k|]` where `n = v.len()`.
///
/// # Panics
/// Panics if `v.ndim() != 1`.
#[inline]
pub fn diag<T: Scalar, B: BackendOps<T> + Default>(
    v: &Tensor<T, B>,
    k: isize,
    _backend: &B,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(v.ndim(), 1, "diag: input must be 1-D, got {}-D", v.ndim());
    let n = v.shape()[0];
    let size = n + k.unsigned_abs();
    let v_cont = v.to_contiguous();
    let v_s = v_cont.as_slice();
    let mut data = vec![T::zero(); size * size];
    for (i, &val) in v_s.iter().enumerate() {
        let (row, col) = if k >= 0 {
            (i, i + k as usize)
        } else {
            (i + (-k) as usize, i)
        };
        data[row * size + col] = val;
    }
    Tensor::from_slice(vec![size, size], &data)
}

/// Extract the `k`-th diagonal from a 2-D matrix `M` as a 1-D vector.
///
/// - `k = 0` — main diagonal (length `min(rows, cols)`).
/// - `k > 0` — `k`-th super-diagonal.
/// - `k < 0` — `|k|`-th sub-diagonal.
///
/// # Panics
/// Panics if `M.ndim() != 2` or `k` is out of range.
#[inline]
pub fn diagonal<T: Scalar, B: BackendOps<T> + Default>(
    m: &Tensor<T, B>,
    k: isize,
    _backend: &B,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(
        m.ndim(),
        2,
        "diagonal: input must be 2-D, got {}-D",
        m.ndim()
    );
    let rows = m.shape()[0];
    let cols = m.shape()[1];
    let m_cont = m.to_contiguous();
    let m_s = m_cont.as_slice();

    let diag_len = if k >= 0 {
        let k = k as usize;
        if k >= cols {
            0
        } else {
            rows.min(cols - k)
        }
    } else {
        let k = (-k) as usize;
        if k >= rows {
            0
        } else {
            (rows - k).min(cols)
        }
    };

    let data: Vec<T> = (0..diag_len)
        .map(|i| {
            let (row, col) = if k >= 0 {
                (i, i + k as usize)
            } else {
                (i + (-k) as usize, i)
            };
            m_s[row * cols + col]
        })
        .collect();

    Tensor::from_slice(vec![diag_len], &data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn diag_creates_diagonal_matrix_from_vector() {
        let b = SequentialBackend::new();
        let v = Tensor::from_slice(vec![3], &[1.0f32, 2.0, 3.0]);
        let m = diag(&v, 0, &b);
        assert_eq!(m.shape(), &[3, 3]);
        assert_eq!(m.as_slice(), &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn diag_superdiagonal_k1() {
        let b = SequentialBackend::new();
        let v = Tensor::from_slice(vec![2], &[5.0f32, 6.0]);
        let m = diag(&v, 1, &b);
        assert_eq!(m.shape(), &[3, 3]);
        // 5 at (0,1), 6 at (1,2), rest zero
        assert_eq!(m.as_slice()[1], 5.0);
        assert_eq!(m.as_slice()[5], 6.0);
    }

    #[test]
    fn diagonal_extracts_main_diagonal() {
        let b = SequentialBackend::new();
        let m = Tensor::from_slice(
            vec![3, 3],
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let v = diagonal(&m, 0, &b);
        assert_eq!(v.shape(), &[3]);
        assert_eq!(v.as_slice(), &[1.0, 5.0, 9.0]);
    }

    #[test]
    fn diagonal_superdiagonal_k1() {
        let b = SequentialBackend::new();
        let m = Tensor::from_slice(
            vec![3, 3],
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let v = diagonal(&m, 1, &b);
        assert_eq!(v.shape(), &[2]);
        assert_eq!(v.as_slice(), &[2.0, 6.0]);
    }

    #[test]
    fn diag_diagonal_roundtrip() {
        // diag(diagonal(M, 0)) should recover the diagonal of M.
        let b = SequentialBackend::new();
        let m = Tensor::from_slice(
            vec![3, 3],
            &[1.0f32, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 9.0],
        );
        let d = diagonal(&m, 0, &b);
        let m2 = diag(&d, 0, &b);
        assert_eq!(m2.as_slice(), m.as_slice());
    }
}
