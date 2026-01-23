//! Sparse matrix addition trait and implementations
//!
//! All sparse operations use CSR format as the single optimized format.
//! CscStorage and CooStorage are type aliases for CsrStorage.

use crate::Result;
use alloc::vec;
use alloc::vec::Vec;
use storage::{CsrStorage, DataType, Storage, StorageError};


/// Sparse matrix addition trait
///
/// Adds two sparse matrices element-wise. Since all sparse formats are
/// unified as CSR, operations work directly on CSR representation.
pub trait SparseAdd<T: DataType> {
    /// Add two sparse matrices, returning the result
    fn add_sparse(&self, other: &Self) -> Result<Self>
    where
        Self: Sized;
}

/// CSR sparse addition implementation
///
/// Uses optimized row-by-row merge algorithm for O(nnz) complexity.
impl<T: DataType + core::ops::Add<Output = T> + Copy + PartialEq + num_traits::Zero> SparseAdd<T> for CsrStorage<T> {
    fn add_sparse(&self, other: &Self) -> Result<Self> {
        // Validate dimensions
        if self.shape().dims() != other.shape().dims() {
            return Err(StorageError::ShapeMismatch {
                expected: self.shape().size(),
                actual: other.shape().size(),
            });
        }

        let (rows, _) = self.dims();
        let mut result_data = Vec::new();
        let mut result_indices = Vec::new();
        let mut result_indptr = vec![0];

        for row in 0..rows {
            let a_start = self.indptr()[row];
            let a_end = self.indptr()[row + 1];
            let b_start = other.indptr()[row];
            let b_end = other.indptr()[row + 1];

            let mut a_idx = a_start;
            let mut b_idx = b_start;

            // Merge sorted column indices
            while a_idx < a_end || b_idx < b_end {
                let (col, sum) = if a_idx < a_end && b_idx < b_end {
                    let a_col = self.indices()[a_idx];
                    let b_col = other.indices()[b_idx];

                    match a_col.cmp(&b_col) {
                        core::cmp::Ordering::Less => {
                            let val = self.data()[a_idx];
                            a_idx += 1;
                            (a_col, val)
                        }
                        core::cmp::Ordering::Greater => {
                            let val = other.data()[b_idx];
                            b_idx += 1;
                            (b_col, val)
                        }
                        core::cmp::Ordering::Equal => {
                            let val = self.data()[a_idx] + other.data()[b_idx];
                            a_idx += 1;
                            b_idx += 1;
                            (a_col, val)
                        }
                    }
                } else if a_idx < a_end {
                    let val = self.data()[a_idx];
                    let col = self.indices()[a_idx];
                    a_idx += 1;
                    (col, val)
                } else {
                    let val = other.data()[b_idx];
                    let col = other.indices()[b_idx];
                    b_idx += 1;
                    (col, val)
                };

                // Only store non-zero results
                if sum != T::zero() {
                    result_data.push(sum);
                    result_indices.push(col);
                }
            }

            result_indptr.push(result_data.len());
        }

        CsrStorage::new(result_data, result_indices, result_indptr, self.shape().dims())
    }
}

// Type aliases use the same implementation (they are the same type)
// CscStorage = CsrStorage, CooStorage = CsrStorage
