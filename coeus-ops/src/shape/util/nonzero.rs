// ── nonzero — row-major coordinates of non-zero elements ──

use crate::backend_ops::BackendOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Return the row-major coordinates of all non-zero elements.
///
/// The result has shape `[N, ndim]`, where `N` is the number of non-zero
/// elements in `input`. Each row stores one logical index encoded as `T`.
#[inline]
pub fn nonzero<T: Scalar, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    backend: &B,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let shape = input.shape();
    let ndim = input.ndim();
    let mut out_vec = Vec::new();
    let mut count = 0usize;

    for flat in 0..input.numel() {
        let idx = crate::shape::flat_to_nd(flat, shape);
        if input.get(&idx) != T::zero() {
            count += 1;
            for coord in idx {
                out_vec.push(T::from_usize(coord));
            }
        }
    }

    Tensor::from_slice_on(vec![count, ndim], &out_vec, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn nonzero_1d_returns_positions() {
        let b = SequentialBackend::new();
        let input = Tensor::from_slice(vec![4], &[0.0f64, 2.0, 0.0, 3.0]);
        let out = nonzero(&input, &b);
        assert_eq!(out.shape(), &[2, 1]);
        assert_eq!(out.as_slice(), &[1.0, 3.0]);
    }

    #[test]
    fn nonzero_2d_returns_row_major_indices() {
        let b = SequentialBackend::new();
        let input = Tensor::from_slice(vec![2, 3], &[0.0f64, 5.0, 0.0, 6.0, 7.0, 0.0]);
        let out = nonzero(&input, &b);
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(out.as_slice(), &[0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn nonzero_all_zero_returns_empty_rows() {
        let b = SequentialBackend::new();
        let input = Tensor::<f64, SequentialBackend>::zeros(vec![2, 2]);
        let out = nonzero(&input, &b);
        assert_eq!(out.shape(), &[0, 2]);
        assert!(out.as_slice().is_empty());
    }

    #[test]
    fn nonzero_single_element_tensor() {
        let b = SequentialBackend::new();
        let input = Tensor::from_slice(vec![1], &[4.0f64]);
        let out = nonzero(&input, &b);
        assert_eq!(out.shape(), &[1, 1]);
        assert_eq!(out.as_slice(), &[0.0]);
    }
}

