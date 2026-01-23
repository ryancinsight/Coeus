//! Sparse matrix multiplication trait and implementations
//!
//! All sparse operations use CSR format as the single optimized format.

use crate::Result;
use alloc::vec;
use alloc::vec::Vec;
use backend::Backend;
use storage::{CsrStorage, DataType, StorageError};


/// Sparse matrix multiplication trait
pub trait SparseMatMul<T: DataType> {
    /// Sparse matrix-vector multiplication: y = A * x
    fn matvec_mul<B: Backend<Data = T>>(&self, vector: &[T], backend: &B) -> Result<Vec<T>>;

    /// Sparse matrix × dense matrix multiplication
    fn matmul_dense<B: Backend<Data = T>>(
        &self,
        dense_matrix: &[T],
        dense_rows: usize,
        dense_cols: usize,
        backend: &B,
    ) -> Result<Vec<T>>;
}

/// CSR sparse matrix multiplication implementation
impl<T: DataType + Copy + num_traits::Zero + core::ops::Add<Output = T> + core::ops::Mul<Output = T>>
    SparseMatMul<T> for CsrStorage<T>
{
    fn matvec_mul<B: Backend<Data = T>>(&self, vector: &[T], _backend: &B) -> Result<Vec<T>> {
        let (rows, cols) = self.dims();

        if vector.len() != cols {
            return Err(StorageError::ShapeMismatch {
                expected: cols,
                actual: vector.len(),
            });
        }

        let mut result = vec![T::zero(); rows];

        for row in 0..rows {
            let start = self.indptr()[row];
            let end = self.indptr()[row + 1];

            for idx in start..end {
                let col = self.indices()[idx];
                let value = self.data()[idx];
                result[row] = result[row] + value * vector[col];
            }
        }

        Ok(result)
    }

    fn matmul_dense<B: Backend<Data = T>>(
        &self,
        dense_matrix: &[T],
        dense_rows: usize,
        dense_cols: usize,
        _backend: &B,
    ) -> Result<Vec<T>> {
        let (sparse_rows, sparse_cols) = self.dims();

        if sparse_cols != dense_rows {
            return Err(StorageError::ShapeMismatch {
                expected: sparse_cols,
                actual: dense_rows,
            });
        }

        // Result is sparse_rows × dense_cols
        let mut result = vec![T::zero(); sparse_rows * dense_cols];

        for row in 0..sparse_rows {
            let start = self.indptr()[row];
            let end = self.indptr()[row + 1];

            for idx in start..end {
                let k = self.indices()[idx];
                let sparse_val = self.data()[idx];

                // Multiply sparse value with row k of dense matrix
                for col in 0..dense_cols {
                    result[row * dense_cols + col] =
                        result[row * dense_cols + col] + sparse_val * dense_matrix[k * dense_cols + col];
                }
            }
        }

        Ok(result)
    }
}
