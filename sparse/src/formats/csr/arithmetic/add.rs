//! CSR sparse matrix addition
//!
//! This module provides addition operations for CSR sparse matrices.

use crate::Result;
use storage::{CooStorage, CsrStorage, StorageError, DataType, Storage};

/// Add two CSR sparse matrices
///
/// This operation converts both matrices to COO format, performs the addition,
/// and returns the result in COO format.
pub fn add<T: DataType + core::ops::Add<Output = T> + Copy>(
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

    // Convert both to COO and add
    let lhs_coo = lhs.to_coo();
    let rhs_coo = rhs.to_coo();
    
    // Delegate to COO addition
    crate::formats::coo::arithmetic::add::add(&lhs_coo, &rhs_coo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_add_basic() {
        // TODO: Implement test
    }
}
