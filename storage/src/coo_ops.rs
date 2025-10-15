//! COO (Coordinate) specific operations
//!
//! Provides specialized operations and utilities for COO format matrices.

use crate::{Result, CsrStorage, CooStorage, CscStorage, StorageError};
use alloc::{vec, vec::Vec};

impl<T: crate::DataType> CooStorage<T> {
    /// Get row indices slice
    #[must_use]
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// Get column indices slice
    #[must_use]
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
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

    /// Sort COO matrix in row-major order
    ///
    /// This can improve performance for certain operations
    pub fn sort(&mut self)
    where
        T: Copy,
    {
        // Create indices for sorting
        let mut indices: Vec<usize> = (0..self.nnz()).collect();

        // Sort by row, then by column
        indices.sort_by(|&a, &b| {
            let row_a = self.row_indices[a];
            let row_b = self.row_indices[b];
            let col_a = self.col_indices[a];
            let col_b = self.col_indices[b];

            match row_a.cmp(&row_b) {
                core::cmp::Ordering::Equal => col_a.cmp(&col_b),
                ord => ord,
            }
        });

        // Reorder data arrays
        let mut new_data = Vec::with_capacity(self.nnz());
        let mut new_row_indices = Vec::with_capacity(self.nnz());
        let mut new_col_indices = Vec::with_capacity(self.nnz());

        for &idx in &indices {
            new_data.push(self.data[idx]);
            new_row_indices.push(self.row_indices[idx]);
            new_col_indices.push(self.col_indices[idx]);
        }

        self.data = new_data;
        self.row_indices = new_row_indices;
        self.col_indices = new_col_indices;
    }

    /// Convert COO to CSR format
    #[must_use]
    pub fn to_csr(&self) -> CsrStorage<T>
    where
        T: Copy,
    {
        let rows = self.shape().dims()[0];
        let mut indptr = vec![0; rows + 1];
        let mut indices = Vec::with_capacity(self.nnz());
        let mut data = Vec::with_capacity(self.nnz());

        // Count elements per row
        for &row in self.row_indices() {
            indptr[row + 1] += 1;
        }

        // Compute cumulative sum for indptr
        for i in 1..=rows {
            indptr[i] += indptr[i - 1];
        }

        // Create temporary arrays to track insertion positions
        let mut positions = indptr.clone();
        positions.pop(); // Remove last element

        // Fill CSR arrays
        for i in 0..self.nnz() {
            let row = self.row_indices()[i];
            let pos = positions[row];
            indices.push(self.col_indices()[i]);
            data.push(self.data()[i]);
            positions[row] += 1;
        }

        CsrStorage {
            data,
            indices,
            indptr,
            shape: self.shape.clone(),
        }
    }

    /// Convert COO to CSC format
    #[must_use]
    pub fn to_csc(&self) -> CscStorage<T>
    where
        T: Copy,
    {
        let cols = self.shape().dims()[1];
        let mut indptr = vec![0; cols + 1];
        let mut indices = Vec::with_capacity(self.nnz());
        let mut data = Vec::with_capacity(self.nnz());

        // Count elements per column
        for &col in self.col_indices() {
            indptr[col + 1] += 1;
        }

        // Compute cumulative sum for indptr
        for i in 1..=cols {
            indptr[i] += indptr[i - 1];
        }

        // Create temporary arrays to track insertion positions
        let mut positions = indptr.clone();
        positions.pop(); // Remove last element

        // Fill CSC arrays
        for i in 0..self.nnz() {
            let col = self.col_indices()[i];
            let pos = positions[col];
            indices.push(self.row_indices()[i]);
            data.push(self.data()[i]);
            positions[col] += 1;
        }

        CscStorage {
            data,
            indices,
            indptr,
            shape: self.shape.clone(),
        }
    }

    /// Transpose COO matrix
    #[must_use]
    pub fn transpose(&self) -> CooStorage<T>
    where
        T: Copy,
    {
        CooStorage {
            data: self.data.clone(),
            row_indices: self.col_indices.clone(),
            col_indices: self.row_indices.clone(),
            shape: crate::Shape::new(&[self.shape().dims()[1], self.shape().dims()[0]]).unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_coo_sort() {
        // Create unsorted COO matrix
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let row_indices = vec![1, 0, 1, 0];
        let col_indices = vec![1, 0, 0, 1];
        let mut coo = CooStorage::new(data, row_indices, col_indices, &[2, 2]).unwrap();

        coo.sort();

        // Should be sorted by row, then column: (0,0), (0,1), (1,0), (1,1)
        assert_eq!(coo.row_indices(), &[0, 0, 1, 1]);
        assert_eq!(coo.col_indices(), &[0, 1, 0, 1]);
        assert_eq!(coo.data(), &[2.0, 4.0, 3.0, 1.0]);
    }

    #[test]
    fn test_coo_to_csr() {
        // Create COO matrix: [[1, 0, 2], [0, 3, 0]]
        let data = vec![1.0, 2.0, 3.0];
        let row_indices = vec![0, 0, 1];
        let col_indices = vec![0, 2, 1];
        let coo = CooStorage::new(data, row_indices, col_indices, &[2, 3]).unwrap();

        let csr = coo.to_csr();

        // Verify CSR structure
        assert_eq!(csr.data(), &[1.0, 2.0, 3.0]);
        assert_eq!(csr.indices(), &[0, 2, 1]);
        assert_eq!(csr.indptr(), &[0, 2, 3]);
    }
}
