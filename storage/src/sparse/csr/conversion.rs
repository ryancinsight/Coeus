//! CSR format conversion functions
//!
//! Contains methods for converting CSR to other formats.

use super::core::CsrStorage;
use crate::{DataType, DenseStorage, Result};
use alloc::{vec, vec::Vec};

impl<T: DataType> CsrStorage<T> {
    /// Convert to dense storage
    pub fn to_dense(&self) -> Result<DenseStorage<T>>
    where
        T: num_traits::Zero,
    {
        let (rows, cols) = self.dims();
        let mut dense_data = vec![T::zero(); rows * cols];

        for row in 0..rows {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];

            for idx in start..end {
                let col = self.indices[idx];
                let value = self.data[idx];
                dense_data[row * cols + col] = value;
            }
        }

        DenseStorage::from_vec(dense_data, self.shape.dims())
    }

    /// Convert to CSC format
    pub fn to_csc(&self) -> Result<crate::sparse::CscStorage<T>>
    where
        T: Copy + Default,
    {
        let (rows, cols) = self.dims();

        // Count non-zeros per column
        let mut col_counts = vec![0usize; cols];
        for &col in &self.indices {
            col_counts[col] += 1;
        }

        // Build CSC indptr
        let mut csc_indptr = vec![0usize; cols + 1];
        for i in 0..cols {
            csc_indptr[i + 1] = csc_indptr[i] + col_counts[i];
        }

        // Fill data and indices
        let mut csc_data = vec![T::default(); self.nnz()];
        let mut csc_indices = vec![0usize; self.nnz()];
        let mut col_positions = csc_indptr[..cols].to_vec();

        for row in 0..rows {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];

            for idx in start..end {
                let col = self.indices[idx];
                let value = self.data[idx];
                let pos = col_positions[col];

                csc_data[pos] = value;
                csc_indices[pos] = row;
                col_positions[col] += 1;
            }
        }

        // Sort each column by row index
        for col in 0..cols {
            let start = csc_indptr[col];
            let end = csc_indptr[col + 1];
            if end > start {
                for i in start..end {
                    for j in i + 1..end {
                        if csc_indices[j] < csc_indices[i] {
                            csc_indices.swap(i, j);
                            csc_data.swap(i, j);
                        }
                    }
                }
            }
        }

        crate::sparse::CscStorage::new(csc_data, csc_indices, csc_indptr, self.shape.dims())
    }

    /// Convert to COO format
    pub fn to_coo(&self) -> Result<crate::sparse::CooStorage<T>>
    where
        T: Copy,
    {
        let (rows, _cols) = self.dims();

        let mut data = Vec::with_capacity(self.nnz());
        let mut row_indices = Vec::with_capacity(self.nnz());
        let mut col_indices = Vec::with_capacity(self.nnz());

        for row in 0..rows {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];

            for idx in start..end {
                data.push(self.data[idx]);
                row_indices.push(row);
                col_indices.push(self.indices[idx]);
            }
        }

        crate::sparse::CooStorage::new(data, row_indices, col_indices, self.shape.dims())
    }
}
