//! CSR sparse matrix multiplication
//!
//! This module provides matrix multiplication operations for CSR sparse matrices.

use crate::Result;
use storage::{CooStorage, CsrStorage, SparseFormat, StorageError, DataType, Storage, DenseStorage};
use alloc::{vec::Vec, format};
use backend::Backend;

/// Sparse matrix multiplication for CSR format
///
/// Multiplies two CSR sparse matrices and returns the result in COO format.
pub fn matmul_sparse<T, B>(
    lhs: &CsrStorage<T>,
    rhs: &CsrStorage<T>,
    _result_format: SparseFormat,
    _backend: &B,
) -> Result<CooStorage<T>>
where
    T: DataType + Default + 'static,
    B: Backend<Data = T>,
{
    // TODO: Implement efficient CSR x CSR multiplication
    Err(StorageError::BackendError(format!("Unsupported sparse matmul operation")))
}

/// Sparse matrix-vector multiplication for CSR format
///
/// Multiplies a CSR sparse matrix by a dense vector.
pub fn matvec_mul<T, B>(
    matrix: &CsrStorage<T>,
    vector: &[T],
    backend: &B,
) -> Result<Vec<T>>
where
    T: DataType + Default + 'static,
    B: Backend<Data = T>,
{
    backend.spmv_csr(
        matrix.as_slice(),
        matrix.indices(),
        matrix.indptr(),
        vector,
        matrix.shape().dims()[0],
        matrix.shape().dims()[1]
    ).map_err(|e| StorageError::BackendError(format!("{:?}", e)))
}

/// Sparse matrix-dense matrix multiplication for CSR format
///
/// Multiplies a CSR sparse matrix by a dense matrix.
pub fn matmul_dense<T, B>(
    sparse_matrix: &CsrStorage<T>,
    dense_matrix: &[T],
    dense_rows: usize,
    dense_cols: usize,
    backend: &B,
) -> Result<Vec<T>>
where
    T: DataType + Default + 'static,
    B: Backend<Data = T>,
{
    let dense_copy = dense_matrix.to_vec();
    let other_storage = DenseStorage::from_vec(dense_copy, &[dense_rows, dense_cols])
        .map_err(|e| e)?;

    backend.spmm_csr(
        sparse_matrix.as_slice(),
        sparse_matrix.indices(),
        sparse_matrix.indptr(),
        &other_storage,
        sparse_matrix.shape().dims()[0],
        dense_cols
    ).map_err(|e| StorageError::BackendError(format!("{:?}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_matmul_basic() {
        // TODO: Implement test
    }
}
