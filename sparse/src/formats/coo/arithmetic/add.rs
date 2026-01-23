//! COO sparse matrix addition
//!
//! This module provides addition operations for COO sparse matrices.

use crate::Result;
use storage::{CooStorage, StorageError, DataType, Storage};
use alloc::vec::Vec;

/// Add two COO sparse matrices
///
/// This is the core implementation that other formats delegate to.
pub fn add<T: DataType + core::ops::Add<Output = T> + Copy>(
    lhs: &CooStorage<T>,
    rhs: &CooStorage<T>,
) -> Result<CooStorage<T>> {
    // Validate dimensions
    if lhs.shape().dims() != rhs.shape().dims() {
        return Err(StorageError::ShapeMismatch {
            expected: lhs.shape().size(),
            actual: rhs.shape().size(),
        });
    }

    let mut result_data = Vec::new();
    let mut result_row_indices = Vec::new();
    let mut result_col_indices = Vec::new();

    // Add all elements from lhs
    result_data.extend_from_slice(lhs.as_slice());
    result_row_indices.extend_from_slice(lhs.row_indices());
    result_col_indices.extend_from_slice(lhs.col_indices());

    // Add all elements from rhs
    result_data.extend_from_slice(rhs.as_slice());
    result_row_indices.extend_from_slice(rhs.row_indices());
    result_col_indices.extend_from_slice(rhs.col_indices());

    // Create result COO matrix
    CooStorage::new(
        result_data,
        result_row_indices,
        result_col_indices,
        lhs.shape().dims(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coo_add_basic() {
        // TODO: Implement test
    }
}
