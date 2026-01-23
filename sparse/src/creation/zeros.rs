//! Creation operations for sparse storage
//!
//! All sparse operations use CSR format as the single optimized format.

use crate::Result;
use storage::CsrStorage;

/// Create an empty CSR sparse matrix (all zeros)
pub fn zeros<T: storage::DataType + Copy>(shape: &[usize]) -> Result<CsrStorage<T>> {
    CsrStorage::empty(shape)
}

/// Alias for zeros (CSR format)
pub fn zeros_csr<T: storage::DataType + Copy>(shape: &[usize]) -> Result<CsrStorage<T>> {
    zeros(shape)
}

/// Alias for zeros (returns CSR since all formats unified)
pub fn zeros_csc<T: storage::DataType + Copy>(shape: &[usize]) -> Result<CsrStorage<T>> {
    zeros(shape)
}
