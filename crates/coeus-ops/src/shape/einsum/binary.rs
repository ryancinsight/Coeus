//! Two-operand einsum patterns over the recognised ML contractions.
//!
//! Each arm validates its own rank and contracted extents before touching
//! data, and dispatches to the optimised kernel where one exists (`matmul`,
//! `bmm`) instead of re-deriving the same loop here.

use super::subscript::{check_dim, check_rank, unsupported};
use crate::backend_ops::BackendOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Evaluate a recognised two-operand pattern, or reject the subscript.
///
/// # Errors
///
/// Returns the caller's backend error for a rank or contraction-extent
/// mismatch ([`EinsumRank`](coeus_core::BackendError::EinsumRank),
/// [`EinsumDim`](coeus_core::BackendError::EinsumDim)) or an unsupported
/// subscript
/// ([`EinsumUnsupportedPattern`](coeus_core::BackendError::EinsumUnsupportedPattern)).
#[inline]
pub(super) fn eval<T: Scalar, B: BackendOps<T> + Default>(
    subscript: &str,
    lhs_parts: &[&str],
    rhs: &str,
    operands: &[&Tensor<T, B>],
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let a = operands[0];
    let b_t = operands[1];
    let a_lhs = lhs_parts[0];
    let b_lhs = lhs_parts[1];

    // "i,i->" — dot product
    if a_lhs == "i" && b_lhs == "i" && rhs.is_empty() {
        check_rank::<T, B>(subscript, 0, a.ndim(), 1)?;
        check_rank::<T, B>(subscript, 1, b_t.ndim(), 1)?;
        // Same label, one extent: zipping silently truncates, so mismatch
        // is a typed error, never a quiet short sum.
        check_dim::<T, B>(subscript, 1, b_t.shape()[0], a.shape()[0])?;
        let a_cont = a.to_contiguous()?;
        let b_cont = b_t.to_contiguous()?;
        let dot = a_cont
            .as_slice()
            .iter()
            .zip(b_cont.as_slice().iter())
            .map(|(&x, &y)| x * y)
            .fold(T::ZERO, |acc, v| acc + v);
        return Tensor::from_slice(vec![1], &[dot]);
    }

    // "i,j->ij" — outer product
    if a_lhs == "i" && b_lhs == "j" && rhs == "ij" {
        check_rank::<T, B>(subscript, 0, a.ndim(), 1)?;
        check_rank::<T, B>(subscript, 1, b_t.ndim(), 1)?;
        let m = a.shape()[0];
        let n = b_t.shape()[0];
        let a_cont = a.to_contiguous()?;
        let b_cont = b_t.to_contiguous()?;
        let a_s = a_cont.as_slice();
        let b_s = b_cont.as_slice();
        let mut data = Vec::with_capacity(m * n);
        for &ai in &a_s[..m] {
            for &bj in &b_s[..n] {
                data.push(ai * bj);
            }
        }
        return Tensor::from_slice(vec![m, n], &data);
    }

    // "ij,j->i" — matrix-vector multiply (right)
    if a_lhs == "ij" && b_lhs == "j" && rhs == "i" {
        check_rank::<T, B>(subscript, 0, a.ndim(), 2)?;
        check_rank::<T, B>(subscript, 1, b_t.ndim(), 1)?;
        let m = a.shape()[0];
        let k = a.shape()[1];
        check_dim::<T, B>(subscript, 1, b_t.shape()[0], k)?;
        let a_cont = a.to_contiguous()?;
        let b_cont = b_t.to_contiguous()?;
        let a_s = a_cont.as_slice();
        let b_s = b_cont.as_slice();
        let data: Vec<T> = (0..m)
            .map(|i| {
                (0..k)
                    .map(|j| a_s[i * k + j] * b_s[j])
                    .fold(T::ZERO, |acc, v| acc + v)
            })
            .collect();
        return Tensor::from_slice(vec![m], &data);
    }

    // "ij,kj->ik" — a @ b.T (inner dot on last dim)
    if a_lhs == "ij" && b_lhs == "kj" && rhs == "ik" {
        check_rank::<T, B>(subscript, 0, a.ndim(), 2)?;
        check_rank::<T, B>(subscript, 1, b_t.ndim(), 2)?;
        check_dim::<T, B>(subscript, 1, b_t.shape()[1], a.shape()[1])?;
        let b_transposed = b_t
            .to_contiguous()?
            .permute(&[1, 0])
            .to_contiguous_on(backend)?;
        return crate::matmul::matmul(a, &b_transposed, backend);
    }

    // "ij,jk->ik" — 2-D matrix multiply
    if a_lhs == "ij" && b_lhs == "jk" && rhs == "ik" {
        check_rank::<T, B>(subscript, 0, a.ndim(), 2)?;
        check_rank::<T, B>(subscript, 1, b_t.ndim(), 2)?;
        return crate::matmul::matmul(a, b_t, backend);
    }

    // "bij,bjk->bik" — batched 3-D matrix multiply
    if a_lhs == "bij" && b_lhs == "bjk" && rhs == "bik" {
        check_rank::<T, B>(subscript, 0, a.ndim(), 3)?;
        check_rank::<T, B>(subscript, 1, b_t.ndim(), 3)?;
        check_dim::<T, B>(subscript, 1, b_t.shape()[0], a.shape()[0])?;
        check_dim::<T, B>(subscript, 1, b_t.shape()[1], a.shape()[2])?;
        return crate::matmul::bmm(a, b_t, backend);
    }

    // "bik,bk->bi" — batched matrix-vector multiply
    if a_lhs == "bik" && b_lhs == "bk" && rhs == "bi" {
        check_rank::<T, B>(subscript, 0, a.ndim(), 3)?;
        check_rank::<T, B>(subscript, 1, b_t.ndim(), 2)?;
        let batch = a.shape()[0];
        let m = a.shape()[1];
        let k = a.shape()[2];
        check_dim::<T, B>(subscript, 1, b_t.shape()[0], batch)?;
        check_dim::<T, B>(subscript, 1, b_t.shape()[1], k)?;
        let a_cont = a.to_contiguous()?;
        let b_cont = b_t.to_contiguous()?;
        let a_s = a_cont.as_slice();
        let b_s = b_cont.as_slice();
        let data: Vec<T> = (0..batch)
            .flat_map(|bi| {
                (0..m).map(move |i| {
                    (0..k)
                        .map(|j| a_s[bi * m * k + i * k + j] * b_s[bi * k + j])
                        .fold(T::ZERO, |acc, v| acc + v)
                })
            })
            .collect();
        return Tensor::from_slice(vec![batch, m], &data);
    }

    // "bi,bj->bij" — batched outer product
    if a_lhs == "bi" && b_lhs == "bj" && rhs == "bij" {
        check_rank::<T, B>(subscript, 0, a.ndim(), 2)?;
        check_rank::<T, B>(subscript, 1, b_t.ndim(), 2)?;
        let batch = a.shape()[0];
        let m = a.shape()[1];
        let n = b_t.shape()[1];
        check_dim::<T, B>(subscript, 1, b_t.shape()[0], batch)?;
        let a_cont = a.to_contiguous()?;
        let b_cont = b_t.to_contiguous()?;
        let a_s = a_cont.as_slice();
        let b_s = b_cont.as_slice();
        let mut data = Vec::with_capacity(batch * m * n);
        for bi in 0..batch {
            // One slice per batch row: the inner loops then walk the tile
            // without re-deriving a batch offset per element.
            let a_row = &a_s[bi * m..bi * m + m];
            let b_row = &b_s[bi * n..bi * n + n];
            for &ai in a_row {
                for &bj in b_row {
                    data.push(ai * bj);
                }
            }
        }
        return Tensor::from_slice(vec![batch, m, n], &data);
    }

    Err(unsupported::<T, B>(subscript, 2))
}
