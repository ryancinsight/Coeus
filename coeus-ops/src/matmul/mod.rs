// -- Matmul module --

mod kernel;

pub use kernel::{matmul, matmul_accumulate};

use crate::backend_ops::BackendOps;
use coeus_core::Scalar;
use coeus_tensor::Tensor;

/// Batch matrix multiply: `[B, M, K] x [B, K, N] -> [B, M, N]`.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::bmm;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
/// let b = Tensor::<f32, SequentialBackend>::from_slice([1, 2, 2], &[5.0, 6.0, 7.0, 8.0]);
/// let c = bmm(&a, &b, &backend).expect("valid batched matmul doctest inputs");
/// assert_eq!(c.shape(), &[1, 2, 2]);
/// let expected = [19.0, 22.0, 43.0, 50.0];
/// for (got, want) in c.as_slice().iter().zip(expected.iter()) {
///     assert!((got - want).abs() < 1e-4);
/// }
/// ```
#[inline]
pub fn bmm<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    assert_eq!(a.ndim(), 3, "bmm: a must be 3-D, got {}-D", a.ndim());
    assert_eq!(b.ndim(), 3, "bmm: b must be 3-D, got {}-D", b.ndim());
    let (batch, _m, _k) = (a.shape()[0], a.shape()[1], a.shape()[2]);
    let (_b2, _k2, _n) = (b.shape()[0], b.shape()[1], b.shape()[2]);
    assert_eq!(batch, b.shape()[0], "bmm: batch mismatch");
    assert_eq!(a.shape()[2], b.shape()[1], "bmm: inner dim mismatch");
    kernel::matmul(a, b, backend)
}

/// Outer product: `[M] x [N] -> [M, N]`.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::outer;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([3], &[1.0, 2.0, 3.0]);
/// let b = Tensor::<f32, SequentialBackend>::from_slice([2], &[4.0, 5.0]);
/// let c = outer(&a, &b, &backend).expect("valid outer-product doctest inputs");
/// assert_eq!(c.shape(), &[3, 2]);
/// assert_eq!(c.as_slice(), &[4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
/// ```
#[inline]
pub fn outer<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    assert_eq!(a.ndim(), 1, "outer: a must be 1-D, got {}-D", a.ndim());
    assert_eq!(b.ndim(), 1, "outer: b must be 1-D, got {}-D", b.ndim());
    let m = a.shape()[0];
    let n = b.shape()[0];
    let a_col = a.clone().reshape([m, 1]);
    let b_row = b.clone().reshape([1, n]);
    kernel::matmul(&a_col, &b_row, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    #[test]
    fn bmm_matches_manual_batch_matmul() {
        let backend = SequentialBackend::new();
        let a = Tensor::<f32, SequentialBackend>::from_slice(
            [2, 2, 3],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 0.0, 2.0, 0.0, 1.0, 3.0],
        );
        let b = Tensor::<f32, SequentialBackend>::from_slice(
            [2, 3, 2],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 0.0, 1.0, 3.0, 4.0],
        );
        let out = bmm(&a, &b, &backend).expect("valid batched matmul test shapes");
        assert_eq!(out.shape(), &[2, 2, 2]);
        assert_eq!(
            out.as_slice(),
            &[10.0, 13.0, 28.0, 40.0, 7.0, 10.0, 9.0, 13.0]
        );
    }

    #[test]
    fn outer_matches_pairwise_products() {
        let backend = SequentialBackend::new();
        let a = Tensor::<f32, SequentialBackend>::from_slice([3], &[1.0, 2.0, 3.0]);
        let b = Tensor::<f32, SequentialBackend>::from_slice([2], &[4.0, 5.0]);
        let out = outer(&a, &b, &backend).expect("valid outer-product test shapes");
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(out.as_slice(), &[4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
    }
}
