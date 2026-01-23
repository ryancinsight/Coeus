//! CSR sparse matrix element-wise multiplication
//!
//! This module provides element-wise multiplication operations for CSR sparse matrices.

use crate::Result;
use storage::{CooStorage, CsrStorage, StorageError, DataType, Storage};

/// Element-wise multiply two CSR sparse matrices
///
/// This operation converts both matrices to COO format, performs the multiplication,
/// and returns the result in COO format.
pub fn mul<T: DataType + core::ops::Mul<Output = T> + Copy>(
    lhs: &CsrStorage<T>,
    rhs: &CsrStorage<T>,
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
    fn test_csr_mul_basic() {
        // TODO: Implement test
    }
}
