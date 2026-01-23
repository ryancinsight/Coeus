//! COO sparse matrix multiplication
//!
//! This module provides matrix multiplication operations for COO sparse matrices.

use crate::Result;
use storage::{CooStorage, SparseFormat, DataType};
use alloc::vec::Vec;
use backend::Backend;

/// Sparse matrix multiplication for COO format
///
/// Converts to CSR format and delegates to CSR multiplication.
pub fn matmul_sparse<T, B>(
    lhs: &CooStorage<T>,
    rhs: &CooStorage<T>,
    result_format: SparseFormat,
    backend: &B,
) -> Result<CooStorage<T>>
where
    T: DataType + Default + 'static,
    B: Backend<Data = T>,
{
    // Convert to CSR and delegate
    let lhs_csr = lhs.to_csr();
    let rhs_csr = rhs.to_csr();
    crate::formats::csr::arithmetic::matmul::matmul_sparse(&lhs_csr, &rhs_csr, result_format, backend)
}

/// Sparse matrix-vector multiplication for COO format
///
/// Converts to CSR format and delegates to CSR multiplication.
pub fn matvec_mul<T, B>(
    matrix: &CooStorage<T>,
    vector: &[T],
    backend: &B,
) -> Result<Vec<T>>
where
    T: DataType + Default + 'static,
    B: Backend<Data = T>,
{
    // Convert to CSR and delegate
    let matrix_csr = matrix.to_csr();
    crate::formats::csr::arithmetic::matmul::matvec_mul(&matrix_csr, vector, backend)
}

/// Sparse matrix-dense matrix multiplication for COO format
///
/// Converts to CSR format and delegates to CSR multiplication.
pub fn matmul_dense<T, B>(
    sparse_matrix: &CooStorage<T>,
    dense_matrix: &[T],
    dense_rows: usize,
    dense_cols: usize,
    backend: &B,
) -> Result<Vec<T>>
where
    T: DataType + Default + 'static,
    B: Backend<Data = T>,
{
    // Convert to CSR and delegate
    let sparse_csr = sparse_matrix.to_csr();
    crate::formats::csr::arithmetic::matmul::matmul_dense(&sparse_csr, dense_matrix, dense_rows, dense_cols, backend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coo_matmul_basic() {
        // TODO: Implement test
    }
}
