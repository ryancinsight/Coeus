//! CSR (Compressed Sparse Row) specific operations
//!
//! Provides specialized operations and utilities for CSR format matrices.

use crate::{Result, CsrStorage, CooStorage, CscStorage, StorageError};
use alloc::{vec, vec::Vec};

impl<T: crate::DataType> CsrStorage<T> {
    /// Extract a specific row as a dense vector
    ///
    /// # Arguments
    /// * `row_idx` - Index of the row to extract
    ///
    /// # Returns
    /// Dense vector containing the row elements
    ///
    /// # Errors
    /// Returns error if row index is out of bounds
    pub fn row_as_dense(&self, row_idx: usize) -> Result<Vec<T>>
    where
        T: num_traits::Zero + Copy,
    {
        if row_idx >= self.shape().dims()[0] {
            return Err(StorageError::IndexOutOfBounds {
                index: row_idx,
                bound: self.shape().dims()[0],
            });
        }

        let cols = self.shape().dims()[1];
        let mut row = vec![T::zero(); cols];

        let row_start = self.indptr()[row_idx];
        let row_end = self.indptr()[row_idx + 1];

        for i in row_start..row_end {
            let col_idx = self.indices()[i];
            row[col_idx] = self.data()[i];
        }

        Ok(row)
    }

    /// Get row pointers (indptr) slice
    #[must_use]
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    /// Get column indices slice
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Get data values slice
    #[must_use]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Count non-zero elements (nnz)
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Calculate sparsity ratio (0.0 = dense, 1.0 = all zeros)
    #[must_use]
    pub fn sparsity(&self) -> f64 {
        let total_elements = self.shape().size();
        if total_elements == 0 {
            0.0
        } else {
            1.0 - (self.nnz() as f64 / total_elements as f64)
        }
    }

    /// Convert CSR to COO format
    #[must_use]
    pub fn to_coo(&self) -> CooStorage<T>
    where
        T: Copy,
    {
        let mut row_indices = Vec::with_capacity(self.nnz());
        let mut col_indices = Vec::with_capacity(self.nnz());

        for row in 0..self.shape().dims()[0] {
            let row_start = self.indptr()[row];
            let row_end = self.indptr()[row + 1];

            for _ in row_start..row_end {
                row_indices.push(row);
            }
        }

        col_indices.extend_from_slice(self.indices());

        CooStorage {
            data: self.data.clone(),
            row_indices,
            col_indices,
            shape: self.shape.clone(),
        }
    }

    /// Convert CSR to CSC format
    #[must_use]
    pub fn to_csc(&self) -> CscStorage<T>
    where
        T: Copy,
    {
        // Convert via COO for simplicity
        let coo = self.to_coo();
        coo.to_csc()
    }

    /// Transpose CSR matrix
    #[must_use]
    pub fn transpose(&self) -> CsrStorage<T>
    where
        T: Copy,
    {
        // Convert to COO and transpose
        let coo = self.to_coo();
        coo.transpose().to_csr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_csr_row_as_dense() {
        // Create 2x3 CSR matrix: [[1, 0, 2], [0, 3, 0]]
        let data = vec![1.0, 2.0, 3.0];
        let indices = vec![0, 2, 1];
        let indptr = vec![0, 2, 3];
        let csr = CsrStorage::new(data, indices, indptr, &[2, 3]).unwrap();

        let row0 = csr.row_as_dense(0).unwrap();
        assert_eq!(row0, vec![1.0, 0.0, 2.0]);

        let row1 = csr.row_as_dense(1).unwrap();
        assert_eq!(row1, vec![0.0, 3.0, 0.0]);
    }
}
