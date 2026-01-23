//! Sparse matrix transpose operations
//!
//! All sparse operations use CSR format as the single optimized format.

use crate::Result;
use alloc::vec;
use storage::{CsrStorage, DataType};

/// Sparse matrix transpose trait
pub trait SparseTranspose<T: DataType> {
    /// Transpose the sparse matrix
    fn transpose_sparse(&self) -> Result<CsrStorage<T>>;
}

/// CSR transpose implementation
///
/// Transposes a CSR matrix by converting to CSC structure and reinterpreting as CSR.
/// This is an O(nnz) operation.
impl<T: DataType + Default + Copy> SparseTranspose<T> for CsrStorage<T> {
    fn transpose_sparse(&self) -> Result<CsrStorage<T>> {
        let (rows, cols) = self.dims();

        // Count non-zeros per column
        let mut col_counts = vec![0usize; cols];
        for &col in self.indices() {
            col_counts[col] += 1;
        }

        // Build column pointers for transposed matrix
        let mut new_indptr = vec![0usize; cols + 1];
        for col in 0..cols {
            new_indptr[col + 1] = new_indptr[col] + col_counts[col];
        }

        // Fill transposed data
        let mut new_data = vec![T::default(); self.data().len()];
        let mut new_indices = vec![0usize; self.indices().len()];
        let mut col_positions = new_indptr[..cols].to_vec();

        for row in 0..rows {
            let start = self.indptr()[row];
            let end = self.indptr()[row + 1];

            for idx in start..end {
                let col = self.indices()[idx];
                let value = self.data()[idx];
                let pos = col_positions[col];

                new_data[pos] = value;
                new_indices[pos] = row;
                col_positions[col] += 1;
            }
        }

        CsrStorage::new(new_data, new_indices, new_indptr, &[cols, rows])
    }
}
