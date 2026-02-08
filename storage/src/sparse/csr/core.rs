//! CSR core struct and accessors
//!
//! Contains the CsrStorage struct definition and basic accessor methods.

use crate::{DataType, Result, Shape, StorageError};
use alloc::vec::Vec;

/// Compressed Sparse Row (CSR) storage
///
/// Optimal for row-wise access and Ax operations.
///
/// # Memory Layout
/// - `data`: Non-zero values in row-major order
/// - `indices`: Column indices for each non-zero
/// - `indptr`: Row pointers (length = rows + 1)
#[derive(Debug, Clone, PartialEq)]
pub struct CsrStorage<T: DataType> {
    /// Non-zero values in row-major order
    pub(crate) data: Vec<T>,
    /// Column indices for each non-zero element
    pub(crate) indices: Vec<usize>,
    /// Row pointers (start index of each row in data/indices)
    pub(crate) indptr: Vec<usize>,
    /// Matrix shape [rows, cols]
    pub(crate) shape: Shape,
}

impl<T: DataType> CsrStorage<T> {
    /// Create CSR storage from components with validation
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
                reason: "CSR storage requires 2D shape",
            });
        }

        let rows = shape[0];
        let cols = shape[1];

        if indptr.len() != rows + 1 {
            return Err(StorageError::ShapeMismatch {
                expected: rows + 1,
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

        // Validate column indices
        for &col_idx in &indices {
            if col_idx >= cols {
                return Err(StorageError::IndexOutOfBounds {
                    index: col_idx,
                    bound: cols,
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

    /// Get reference to column indices
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Get reference to row pointers
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    /// Get mutable reference to non-zero values
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Get value at (row, col), returns zero if not stored
    pub fn get(&self, row: usize, col: usize) -> T
    where
        T: num_traits::Zero,
    {
        let (rows, cols) = self.dims();
        if row >= rows || col >= cols {
            return T::zero();
        }

        let start = self.indptr[row];
        let end = self.indptr[row + 1];

        for idx in start..end {
            match self.indices[idx].cmp(&col) {
                core::cmp::Ordering::Equal => return self.data[idx],
                core::cmp::Ordering::Greater => break,
                core::cmp::Ordering::Less => continue,
            }
        }

        T::zero()
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

impl<T: DataType + core::ops::Neg<Output = T>> core::ops::Neg for CsrStorage<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut data = self.data;
        for val in &mut data {
            *val = -*val;
        }
        Self {
            data,
            indices: self.indices,
            indptr: self.indptr,
            shape: self.shape,
        }
    }
}
