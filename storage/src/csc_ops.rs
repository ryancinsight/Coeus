//! CSC (Compressed Sparse Column) specific operations
//!
//! Provides specialized operations and utilities for CSC format matrices.

use crate::{Result, CsrStorage, CooStorage, CscStorage, StorageError};
use alloc::{vec, vec::Vec};

impl<T: crate::DataType> CscStorage<T> {
    /// Extract a specific column as a dense vector
    ///
    /// # Arguments
    /// * `col_idx` - Index of the column to extract
    ///
    /// # Returns
    /// Dense vector containing the column elements
    ///
    /// # Errors
    /// Returns error if column index is out of bounds
    pub fn col_as_dense(&self, col_idx: usize) -> Result<Vec<T>>
    where
        T: num_traits::Zero + Copy,
    {
        if col_idx >= self.shape().dims()[1] {
            return Err(StorageError::IndexOutOfBounds {
                index: col_idx,
                bound: self.shape().dims()[1],
            });
        }

        let rows = self.shape().dims()[0];
        let mut col = vec![T::zero(); rows];

        let col_start = self.indptr()[col_idx];
        let col_end = self.indptr()[col_idx + 1];

        for i in col_start..col_end {
            let row_idx = self.indices()[i];
            col[row_idx] = self.data()[i];
        }

        Ok(col)
    }

    /// Get column pointers (indptr) slice
    #[must_use]
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    /// Get row indices slice
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

    /// Convert CSC to COO format
    #[must_use]
    pub fn to_coo(&self) -> CooStorage<T>
    where
        T: Copy,
    {
        let mut row_indices = Vec::with_capacity(self.nnz());
        let mut col_indices = Vec::with_capacity(self.nnz());

        for col in 0..self.shape().dims()[1] {
            let col_start = self.indptr()[col];
            let col_end = self.indptr()[col + 1];

            for _ in col_start..col_end {
                col_indices.push(col);
            }
        }

        row_indices.extend_from_slice(self.indices());

        CooStorage {
            data: self.data.clone(),
            row_indices,
            col_indices,
            shape: self.shape.clone(),
        }
    }

    /// Convert CSC to CSR format
    #[must_use]
    pub fn to_csr(&self) -> CsrStorage<T>
    where
        T: Copy,
    {
        // Convert via COO for simplicity
        let coo = self.to_coo();
        coo.to_csr()
    }

    /// Transpose CSC matrix
    #[must_use]
    pub fn transpose(&self) -> CscStorage<T>
    where
        T: Copy,
    {
        // Convert to COO and transpose
        let coo = self.to_coo();
        coo.transpose().to_csc()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_csc_col_as_dense() {
        // Create 2x3 CSC matrix: [[1, 0, 2], [0, 3, 0]]
        let data = vec![1.0, 3.0, 2.0];
        let indices = vec![0, 1, 0];
        let indptr = vec![0, 1, 2, 3];
        let csc = CscStorage::new(data, indices, indptr, &[2, 3]).unwrap();

        let col0 = csc.col_as_dense(0).unwrap();
        assert_eq!(col0, vec![1.0, 0.0]);

        let col1 = csc.col_as_dense(1).unwrap();
        assert_eq!(col1, vec![0.0, 3.0]);

        let col2 = csc.col_as_dense(2).unwrap();
        assert_eq!(col2, vec![2.0, 0.0]);
    }
}
