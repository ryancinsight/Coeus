//! Advanced indexing operations for sparse tensors
//!
//! Provides boolean indexing, fancy indexing, and slicing operations
//! optimized for the CSR sparse matrix format.

use crate::{CsrStorage, Result, StorageError};
use alloc::{vec, vec::Vec};

/// Boolean indexing for sparse tensors using CSR format
pub trait SparseBooleanIndex<T: crate::DataType> {
    /// Apply boolean mask to select elements
    ///
    /// Returns a new CSR storage containing only elements where mask is true.
    /// The mask is applied row-wise for 2D matrices.
    fn boolean_index(&self, mask: &[bool]) -> Result<CsrStorage<T>>;
}

/// Fancy indexing for sparse tensors using CSR format
///
/// Provides advanced indexing operations for selecting specific rows/columns
/// or arbitrary element combinations from sparse matrices.
pub trait SparseFancyIndex<T: crate::DataType> {
    /// Select specific indices from the sparse tensor
    ///
    /// For 2D matrices, this selects specific rows based on the indices array.
    /// Returns a new CSR storage with the selected rows.
    fn fancy_index(&self, indices: &[i32]) -> Result<CsrStorage<T>>;
}

// Implementation for CsrStorage - the optimal sparse format
impl<T: crate::DataType + Copy> SparseBooleanIndex<T> for CsrStorage<T> {
    fn boolean_index(&self, mask: &[bool]) -> Result<CsrStorage<T>> {
        let (rows, cols) = self.dims();
        
        if mask.len() != rows {
            return Err(StorageError::ShapeMismatch {
                expected: rows,
                actual: mask.len(),
            });
        }
        
        let mut new_data = Vec::new();
        let mut new_indices = Vec::new();
        let mut new_indptr = vec![0];
        let mut selected_rows = 0;
        
        for (row, &include) in mask.iter().enumerate() {
            if include {
                let start = self.indptr()[row];
                let end = self.indptr()[row + 1];
                
                // Copy this row's data
                for idx in start..end {
                    new_data.push(self.data()[idx]);
                    new_indices.push(self.indices()[idx]);
                }
                
                selected_rows += 1;
                new_indptr.push(new_data.len());
            }
        }
        
        CsrStorage::new(new_data, new_indices, new_indptr, &[selected_rows, cols])
    }
}

impl<T: crate::DataType + Copy> SparseFancyIndex<T> for CsrStorage<T> {
    fn fancy_index(&self, indices: &[i32]) -> Result<CsrStorage<T>> {
        let (rows, cols) = self.dims();
        
        // Validate indices
        for &idx in indices {
            if idx < 0 || idx as usize >= rows {
                return Err(StorageError::IndexOutOfBounds {
                    index: idx as usize,
                    bound: rows,
                });
            }
        }
        
        let mut new_data = Vec::new();
        let mut new_indices = Vec::new();
        let mut new_indptr = vec![0];
        
        for &row_idx in indices {
            let row = row_idx as usize;
            let start = self.indptr()[row];
            let end = self.indptr()[row + 1];
            
            // Copy this row's data
            for idx in start..end {
                new_data.push(self.data()[idx]);
                new_indices.push(self.indices()[idx]);
            }
            
            new_indptr.push(new_data.len());
        }
        
        CsrStorage::new(new_data, new_indices, new_indptr, &[indices.len(), cols])
    }
}