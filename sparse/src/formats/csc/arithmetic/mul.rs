//! CSC sparse matrix element-wise multiplication
//!
//! This module provides element-wise multiplication operations for CSC sparse matrices.

use crate::Result;
use storage::{CooStorage, CscStorage, StorageError, DataType, Storage};

/// Element-wise multiply two CSC sparse matrices
///
/// This operation converts both matrices to COO format, performs the multiplication,
/// and returns the result in COO format.
pub fn mul<T: DataType + core::ops::Mul<Output = T> + Copy>(
    lhs: &CscStorage<T>,
    rhs: &CscStorage<T>,
) -> Result<CooStorage<T>> {
    // Validate dimensions
    if lhs.shape().dims() != rhs.shape().dims() {
        return Err(StorageError::ShapeMismatch {
            expected: lhs.shape().size(),
            actual: rhs.shape().size(),
        });
    }

    // Convert both to COO and multiply
    let lhs_coo = lhs.to_coo();
    let rhs_coo = rhs.to_coo();
    
    // Delegate to COO multiplication
    crate::formats::coo::arithmetic::mul::mul(&lhs_coo, &rhs_coo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csc_mul_basic() {
        // TODO: Implement test
    }
}
