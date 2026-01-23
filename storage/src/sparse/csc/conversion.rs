//! CSC format conversion functions

use super::core::CscStorage;
use crate::{DataType, DenseStorage, Result};
use alloc::{vec, vec::Vec};

impl<T: DataType> CscStorage<T> {
    /// Convert to dense storage
    pub fn to_dense(&self) -> Result<DenseStorage<T>>
    where
        T: num_traits::Zero,
    {
        let (rows, cols) = self.dims();
        let mut dense_data = vec![T::zero(); rows * cols];

        for col in 0..cols {
            let start = self.indptr[col];
            let end = self.indptr[col + 1];

            for idx in start..end {
                let row = self.indices[idx];
                let value = self.data[idx];
                dense_data[row * cols + col] = value;
            }
        }

        DenseStorage::from_vec(dense_data, self.shape.dims())
    }

    /// Convert to CSR format
    pub fn to_csr(&self) -> Result<crate::sparse::CsrStorage<T>>
    where
        T: Copy + Default,
    {
        let (rows, cols) = self.dims();

        // Count non-zeros per row
        let mut row_counts = vec![0usize; rows];
        for &row in &self.indices {
            row_counts[row] += 1;
        }

        // Build CSR indptr
        let mut csr_indptr = vec![0usize; rows + 1];
        for i in 0..rows {
            csr_indptr[i + 1] = csr_indptr[i] + row_counts[i];
        }

        // Fill data and indices
        let mut csr_data = vec![T::default(); self.nnz()];
        let mut csr_indices = vec![0usize; self.nnz()];
        let mut row_positions = csr_indptr[..rows].to_vec();

        for col in 0..cols {
            let start = self.indptr[col];
            let end = self.indptr[col + 1];

            for idx in start..end {
                let row = self.indices[idx];
                let value = self.data[idx];
                let pos = row_positions[row];

                csr_data[pos] = value;
                csr_indices[pos] = col;
                row_positions[row] += 1;
            }
        }

        // Sort each row by column index
        for row in 0..rows {
            let start = csr_indptr[row];
            let end = csr_indptr[row + 1];
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

        crate::sparse::CsrStorage::new(csr_data, csr_indices, csr_indptr, self.shape.dims())
    }

    /// Convert to COO format
    pub fn to_coo(&self) -> Result<crate::sparse::CooStorage<T>>
    where
        T: Copy,
    {
        let (_rows, cols) = self.dims();

        let mut data = Vec::with_capacity(self.nnz());
        let mut row_indices = Vec::with_capacity(self.nnz());
        let mut col_indices = Vec::with_capacity(self.nnz());

        for col in 0..cols {
            let start = self.indptr[col];
            let end = self.indptr[col + 1];

            for idx in start..end {
                data.push(self.data[idx]);
                row_indices.push(self.indices[idx]);
                col_indices.push(col);
            }
        }

        crate::sparse::CooStorage::new(data, row_indices, col_indices, self.shape.dims())
    }
}
