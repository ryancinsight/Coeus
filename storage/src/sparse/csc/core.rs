//! CSC core struct and accessors

use crate::{DataType, Result, Shape, StorageError};
use alloc::vec::Vec;

/// Compressed Sparse Column (CSC) storage
///
/// Optimal for column-wise access and A^T x operations.
#[derive(Debug, Clone, PartialEq)]
pub struct CscStorage<T: DataType> {
    /// Non-zero values in column-major order
    pub(crate) data: Vec<T>,
    /// Row indices for each non-zero element
    pub(crate) indices: Vec<usize>,
    /// Column pointers (start index of each column in data/indices)
    pub(crate) indptr: Vec<usize>,
    /// Matrix shape [rows, cols]
    pub(crate) shape: Shape,
}

impl<T: DataType> CscStorage<T> {
    /// Create CSC storage from components with validation
    pub fn new(
        data: Vec<T>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        shape: &[usize],
    ) -> Result<Self> {
        if data.len() != indices.len() {
            return Err(StorageError::ShapeMismatch {
                expected: data.len(),
                actual: indices.len(),
            });
        }

        if shape.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "CSC storage requires 2D shape",
            });
        }

        let rows = shape[0];
        let cols = shape[1];

        if indptr.len() != cols + 1 {
            return Err(StorageError::ShapeMismatch {
                expected: cols + 1,
                actual: indptr.len(),
            });
        }

        // Validate indptr is non-decreasing
        for i in 1..indptr.len() {
            if indptr[i] < indptr[i - 1] {
                return Err(StorageError::InvalidShape {
                    reason: "indptr must be non-decreasing",
                });
            }
        }

        // Validate row indices
        for &row_idx in &indices {
            if row_idx >= rows {
                return Err(StorageError::IndexOutOfBounds {
                    index: row_idx,
                    bound: rows,
                });
            }
        }

        Ok(Self {
            data,
            indices,
            indptr,
            shape: Shape::new(shape)?,
        })
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Get matrix dimensions
    pub fn dims(&self) -> (usize, usize) {
        let dims = self.shape.dims();
        (dims[0], dims[1])
    }

    /// Get reference to non-zero values
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Get reference to row indices
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Get reference to column pointers
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    /// Get reference to shape
    pub fn shape_ref(&self) -> &Shape {
        &self.shape
    }

    /// Calculate sparsity ratio (nnz / total_elements)
    #[allow(clippy::cast_precision_loss)]
    pub fn sparsity(&self) -> f64 {
        let (rows, cols) = self.dims();
        let total = rows * cols;
        if total == 0 {
            0.0
        } else {
            self.nnz() as f64 / total as f64
        }
    }
}
