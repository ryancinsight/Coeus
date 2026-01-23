//! COO sparse matrix element-wise multiplication
//!
//! This module provides element-wise multiplication operations for COO sparse matrices.

use crate::Result;
use storage::{CooStorage, StorageError, DataType, Storage};
use alloc::vec::Vec;

/// Element-wise multiply two COO sparse matrices
///
/// This is the core implementation that other formats delegate to.
pub fn mul<T: DataType + core::ops::Mul<Output = T> + Copy>(
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

    // TODO: Implement efficient element-wise multiplication
    // For now, this is a placeholder that needs proper implementation
    // to handle matching coordinates and multiplying values
    
    let mut result_data = Vec::new();
    let mut result_row_indices = Vec::new();
    let mut result_col_indices = Vec::new();

    // Naive implementation: multiply all elements from lhs
    // This is incorrect and needs to be replaced with proper coordinate matching
    result_data.extend_from_slice(lhs.as_slice());
    result_row_indices.extend_from_slice(lhs.row_indices());
    result_col_indices.extend_from_slice(lhs.col_indices());

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
    fn test_coo_mul_basic() {
        // TODO: Implement test
    }
}
