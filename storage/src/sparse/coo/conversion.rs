//! COO format conversion functions

use super::core::CooStorage;
use crate::{DataType, DenseStorage, Result};
use alloc::vec;

impl<T: DataType> CooStorage<T> {
    /// Convert to dense storage
    pub fn to_dense(&self) -> Result<DenseStorage<T>>
    where
        T: num_traits::Zero,
    {
        let (rows, cols) = self.dims();
        let mut dense_data = vec![T::zero(); rows * cols];

        for i in 0..self.nnz() {
            let row = self.row_indices[i];
            let col = self.col_indices[i];
            dense_data[row * cols + col] = self.data[i];
        }

        DenseStorage::from_vec(dense_data, self.shape.dims())
    }

    /// Convert to CSR format
    pub fn to_csr(&self) -> Result<crate::sparse::CsrStorage<T>>
    where
        T: Copy + Default,
    {
        let (rows, _cols) = self.dims();

        // Count non-zeros per row
        let mut row_counts = vec![0usize; rows];
        for &row in &self.row_indices {
            row_counts[row] += 1;
        }

        // Build indptr
        let mut indptr = vec![0usize; rows + 1];
        for i in 0..rows {
            indptr[i + 1] = indptr[i] + row_counts[i];
        }

        // Fill data and indices
        let mut csr_data = vec![T::default(); self.nnz()];
        let mut csr_indices = vec![0usize; self.nnz()];
        let mut row_positions = indptr[..rows].to_vec();

        for i in 0..self.nnz() {
            let row = self.row_indices[i];
            let col = self.col_indices[i];
            let value = self.data[i];
            let pos = row_positions[row];

            csr_data[pos] = value;
            csr_indices[pos] = col;
            row_positions[row] += 1;
        }

        // Sort each row by column index
        for row in 0..rows {
            let start = indptr[row];
            let end = indptr[row + 1];
            if end > start {
                for i in start..end {
                    for j in i + 1..end {
                        if csr_indices[j] < csr_indices[i] {
                            csr_indices.swap(i, j);
                            csr_data.swap(i, j);
                        }
                    }
                }
            }
        }

        crate::sparse::CsrStorage::new(csr_data, csr_indices, indptr, self.shape.dims())
    }

    /// Convert to CSC format
    pub fn to_csc(&self) -> Result<crate::sparse::CscStorage<T>>
    where
        T: Copy + Default,
    {
        // Go through CSR for simplicity
        self.to_csr()?.to_csc()
    }
}
