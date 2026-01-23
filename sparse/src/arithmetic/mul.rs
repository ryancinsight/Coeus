//! Sparse matrix element-wise multiplication trait and implementations
//!
//! All sparse operations use CSR format as the single optimized format.

use crate::Result;
use alloc::vec;
use alloc::vec::Vec;
use storage::{CsrStorage, DataType, Storage, StorageError};


/// Sparse matrix element-wise multiplication trait
pub trait SparseMul<T: DataType> {
    /// Element-wise multiply two sparse matrices
    fn mul_sparse(&self, other: &Self) -> Result<Self>
    where
        Self: Sized;
}

/// CSR sparse multiplication implementation
///
/// Element-wise multiplication of sparse matrices. Only positions where
/// both matrices have non-zero values produce non-zero results.
impl<T: DataType + core::ops::Mul<Output = T> + Copy + PartialEq + num_traits::Zero> SparseMul<T>
    for CsrStorage<T>
{
    fn mul_sparse(&self, other: &Self) -> Result<Self> {
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

            // Only matching column indices produce results
            while a_idx < a_end && b_idx < b_end {
                let a_col = self.indices()[a_idx];
                let b_col = other.indices()[b_idx];

                match a_col.cmp(&b_col) {
                    core::cmp::Ordering::Less => a_idx += 1,
                    core::cmp::Ordering::Greater => b_idx += 1,
                    core::cmp::Ordering::Equal => {
                        let val = self.data()[a_idx] * other.data()[b_idx];
                        if val != T::zero() {
                            result_data.push(val);
                            result_indices.push(a_col);
                        }
                        a_idx += 1;
                        b_idx += 1;
                    }
                }
            }

            result_indptr.push(result_data.len());
        }

        CsrStorage::new(result_data, result_indices, result_indptr, self.shape().dims())
    }
}
