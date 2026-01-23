//! Sparse matrix reshape operations
//!
//! All sparse operations use CSR format as the single optimized format.

use crate::Result;
use alloc::vec;
use alloc::vec::Vec;
use storage::{CsrStorage, DataType, Storage, StorageError};


/// Sparse matrix reshape trait
pub trait SparseReshape<T: DataType> {
    /// Reshape sparse matrix to new dimensions
    fn reshape_sparse(&self, new_shape: &[usize]) -> Result<CsrStorage<T>>;
}

/// CSR reshape implementation
///
/// Reshapes a sparse matrix by remapping linear indices to new coordinates.
impl<T: DataType + Default + Copy + num_traits::Zero + PartialEq> SparseReshape<T> for CsrStorage<T> {
    fn reshape_sparse(&self, new_shape: &[usize]) -> Result<CsrStorage<T>> {
        // Validate total elements match
        let new_total_elements: usize = new_shape.iter().product();
        let current_total_elements = self.shape().size();

        if new_total_elements != current_total_elements {
            return Err(StorageError::ShapeMismatch {
                expected: current_total_elements,
                actual: new_total_elements,
            });
        }

        // Only support 2D reshape for now
        if new_shape.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "Sparse reshape only supports 2D output shapes",
            });
        }

        let old_shape = self.shape().dims();
        if old_shape.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "Sparse reshape only supports 2D input shapes",
            });
        }

        let old_cols = old_shape[1];
        let new_cols = new_shape[1];
        let new_rows = new_shape[0];

        // Collect all non-zero (row, col, value) tuples with new coordinates
        let mut entries: Vec<(usize, usize, T)> = Vec::new();

        for old_row in 0..old_shape[0] {
            let start = self.indptr()[old_row];
            let end = self.indptr()[old_row + 1];

            for idx in start..end {
                let old_col = self.indices()[idx];
                let value = self.data()[idx];

                // Convert to linear index, then to new coordinates
                let linear_idx = old_row * old_cols + old_col;
                let new_row = linear_idx / new_cols;
                let new_col = linear_idx % new_cols;

                entries.push((new_row, new_col, value));
            }
        }

        // Sort by (row, col) for CSR format
        entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // Build CSR arrays
        let mut data = Vec::with_capacity(entries.len());
        let mut indices = Vec::with_capacity(entries.len());
        let mut indptr = vec![0usize; new_rows + 1];

        for (new_row, new_col, value) in entries {
            data.push(value);
            indices.push(new_col);
            indptr[new_row + 1] += 1;
        }

        // Convert counts to cumulative pointers
        for i in 1..=new_rows {
            indptr[i] += indptr[i - 1];
        }

        CsrStorage::new(data, indices, indptr, new_shape)
    }
}
