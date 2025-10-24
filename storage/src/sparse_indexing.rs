//! Advanced indexing operations for sparse tensors
//!
//! Provides boolean indexing, fancy indexing, and slicing operations
//! optimized for sparse matrix formats.

use crate::{CooStorage, CscStorage, CsrStorage, Result, Storage, StorageError};
use alloc::vec::Vec;

/// Boolean indexing for sparse tensors
pub trait SparseBooleanIndex<T: crate::DataType> {
    /// Select elements where boolean mask is true
    ///
    /// # Arguments
    /// * `mask` - Boolean mask with same length as tensor
    ///
    /// # Returns
    /// COO matrix containing selected elements
    ///
    /// # Errors
    /// Returns error if mask length doesn't match tensor length
    fn boolean_index(&self, mask: &[bool]) -> Result<CooStorage<T>>;
}

/// Fancy indexing for sparse tensors
pub trait SparseFancyIndex<T: crate::DataType> {
    /// Select elements at specified indices
    ///
    /// # Arguments
    /// * `indices` - 1D array of integer indices
    ///
    /// # Returns
    /// COO matrix containing selected elements
    ///
    /// # Errors
    /// Returns error if any index is out of bounds
    fn fancy_index(&self, indices: &[i32]) -> Result<CooStorage<T>>;
}

// CSR Boolean Indexing
impl<T: crate::DataType + Copy> SparseBooleanIndex<T> for CsrStorage<T> {
    fn boolean_index(&self, mask: &[bool]) -> Result<CooStorage<T>> {
        let tensor_len = self.shape().size();

        if mask.len() != tensor_len {
            return Err(StorageError::ShapeMismatch {
                expected: tensor_len,
                actual: mask.len(),
            });
        }

        let mut result_data = Vec::new();
        let mut result_row_indices = Vec::new();
        let mut result_col_indices = Vec::new();

        let rows = self.shape().dims()[0];
        let cols = self.shape().dims()[1];

        // For each row in CSR
        for i in 0..rows {
            let row_start = self.indptr()[i];
            let row_end = self.indptr()[i + 1];

            // For each non-zero element in this row
            for j in row_start..row_end {
                let col_idx = self.indices()[j];
                let flat_idx = i * cols + col_idx;

                // Check if this position should be selected
                if mask[flat_idx] {
                    result_data.push(self.as_slice()[j]);
                    result_row_indices.push(i);
                    result_col_indices.push(col_idx);
                }
            }
        }

        CooStorage::new(
            result_data,
            result_row_indices,
            result_col_indices,
            self.shape().dims(),
        )
    }
}

// CSC Boolean Indexing
impl<T: crate::DataType + Copy> SparseBooleanIndex<T> for CscStorage<T> {
    fn boolean_index(&self, mask: &[bool]) -> Result<CooStorage<T>> {
        // Convert to CSR and use CSR implementation
        let csr = self.to_csr();
        csr.boolean_index(mask)
    }
}

// COO Boolean Indexing
impl<T: crate::DataType + Copy> SparseBooleanIndex<T> for CooStorage<T> {
    fn boolean_index(&self, mask: &[bool]) -> Result<CooStorage<T>> {
        let tensor_len = self.shape().size();

        if mask.len() != tensor_len {
            return Err(StorageError::ShapeMismatch {
                expected: tensor_len,
                actual: mask.len(),
            });
        }

        let mut result_data = Vec::new();
        let mut result_row_indices = Vec::new();
        let mut result_col_indices = Vec::new();

        let _rows = self.shape().dims()[0];
        let cols = self.shape().dims()[1];

        // Check each non-zero element
        for i in 0..self.nnz() {
            let row = self.row_indices()[i];
            let col = self.col_indices()[i];
            let flat_idx = row * cols + col;

            if mask[flat_idx] {
                result_data.push(self.as_slice()[i]);
                result_row_indices.push(row);
                result_col_indices.push(col);
            }
        }

        CooStorage::new(
            result_data,
            result_row_indices,
            result_col_indices,
            self.shape().dims(),
        )
    }
}

// CSR Fancy Indexing
impl<T: crate::DataType + Copy> SparseFancyIndex<T> for CsrStorage<T> {
    fn fancy_index(&self, indices: &[i32]) -> Result<CooStorage<T>> {
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let tensor_len = self.shape().size() as i32;

        // Validate indices
        for &idx in indices {
            if idx < 0 || idx >= tensor_len {
                return Err(StorageError::IndexOutOfBounds {
                    #[allow(clippy::cast_sign_loss)]
                    index: idx as usize,
                    #[allow(clippy::cast_sign_loss)]
                    bound: tensor_len as usize,
                });
            }
        }

        let mut result_data = Vec::new();
        let mut result_row_indices = Vec::new();
        let mut result_col_indices = Vec::new();

        let _rows = self.shape().dims()[0];
        let cols = self.shape().dims()[1];

        // For each requested index
        for &flat_idx in indices {
            #[allow(clippy::cast_sign_loss)]
            let flat_idx = flat_idx as usize;
            let row = flat_idx / cols;
            let col = flat_idx % cols;

            // Find if this position has a non-zero element
            let row_start = self.indptr()[row];
            let row_end = self.indptr()[row + 1];

            // Binary search for the column in this row
            if let Ok(pos) = self.indices()[row_start..row_end].binary_search(&col) {
                let data_idx = row_start + pos;
                result_data.push(self.as_slice()[data_idx]);
                result_row_indices.push(row);
                result_col_indices.push(col);
            }
            // If not found, it's zero - skip (sparse indexing only returns non-zeros)
        }

        CooStorage::new(
            result_data,
            result_row_indices,
            result_col_indices,
            self.shape().dims(),
        )
    }
}

// CSC Fancy Indexing
impl<T: crate::DataType + Copy> SparseFancyIndex<T> for CscStorage<T> {
    fn fancy_index(&self, indices: &[i32]) -> Result<CooStorage<T>> {
        let csr = self.to_csr();
        csr.fancy_index(indices)
    }
}

// COO Fancy Indexing
impl<T: crate::DataType + Copy> SparseFancyIndex<T> for CooStorage<T> {
    fn fancy_index(&self, indices: &[i32]) -> Result<CooStorage<T>> {
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let tensor_len = self.shape().size() as i32;

        // Validate indices
        for &idx in indices {
            if idx < 0 || idx >= tensor_len {
                return Err(StorageError::IndexOutOfBounds {
                    #[allow(clippy::cast_sign_loss)]
                    index: idx as usize,
                    #[allow(clippy::cast_sign_loss)]
                    bound: tensor_len as usize,
                });
            }
        }

        let mut result_data = Vec::new();
        let mut result_row_indices = Vec::new();
        let mut result_col_indices = Vec::new();

        let _rows = self.shape().dims()[0];
        let cols = self.shape().dims()[1];

        // Create a set of requested positions for efficient lookup
        let mut requested_positions = alloc::collections::BTreeSet::new();
        for &flat_idx in indices {
            #[allow(clippy::cast_sign_loss)]
            let flat_idx = flat_idx as usize;
            let row = flat_idx / cols;
            let col = flat_idx % cols;
            requested_positions.insert((row, col));
        }

        // Check each non-zero element
        for i in 0..self.nnz() {
            let row = self.row_indices()[i];
            let col = self.col_indices()[i];

            if requested_positions.contains(&(row, col)) {
                result_data.push(self.as_slice()[i]);
                result_row_indices.push(row);
                result_col_indices.push(col);
            }
        }

        CooStorage::new(
            result_data,
            result_row_indices,
            result_col_indices,
            self.shape().dims(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use coeus_dtype::float::F32;

    #[test]
    fn test_csr_boolean_index() {
        // Create 2x2 CSR matrix: [[1, 0], [2, 3]]
        let data = vec![F32::new(1.0), F32::new(2.0), F32::new(3.0)];
        let indices = vec![0, 0, 1];
        let indptr = vec![0, 1, 3];
        let csr = CsrStorage::new(data, indices, indptr, &[2, 2]).unwrap();

        // Select elements at positions [0, 2] (first and third elements)
        let mask = vec![true, false, true, false];
        let result = csr.boolean_index(&mask).unwrap();

        // Should have 2 elements: (0,0)=1 and (1,0)=2
        assert_eq!(result.nnz(), 2);
        assert_eq!(result.as_slice(), &[F32::new(1.0), F32::new(2.0)]);
        assert_eq!(result.row_indices(), &[0, 1]);
        assert_eq!(result.col_indices(), &[0, 0]);
    }

    #[test]
    fn test_coo_fancy_index() {
        // Create 2x2 COO matrix: [[1, 0], [0, 2]]
        let data = vec![F32::new(1.0), F32::new(2.0)];
        let row_indices = vec![0, 1];
        let col_indices = vec![0, 1];
        let coo = CooStorage::new(data, row_indices, col_indices, &[2, 2]).unwrap();

        // Select elements at indices [0, 3] (positions (0,0) and (1,1))
        let indices = [0i32, 3];
        let result = coo.fancy_index(&indices).unwrap();

        // Should have both elements
        assert_eq!(result.nnz(), 2);
        assert_eq!(result.as_slice(), &[F32::new(1.0), F32::new(2.0)]);
    }
}
