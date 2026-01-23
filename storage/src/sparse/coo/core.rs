//! COO core struct and accessors

use crate::{DataType, Result, Shape, StorageError};
use alloc::vec::Vec;

/// Coordinate (COO) sparse storage
///
/// Optimal for matrix construction and modification.
///
/// # Memory Layout
/// - `data`: Non-zero values
/// - `row_indices`: Row index for each value
/// - `col_indices`: Column index for each value
#[derive(Debug, Clone, PartialEq)]
pub struct CooStorage<T: DataType> {
    /// Non-zero values
    pub(crate) data: Vec<T>,
    /// Row indices for each non-zero
    pub(crate) row_indices: Vec<usize>,
    /// Column indices for each non-zero
    pub(crate) col_indices: Vec<usize>,
    /// Matrix shape [rows, cols]
    pub(crate) shape: Shape,
}

impl<T: DataType> CooStorage<T> {
    /// Create COO storage from components with validation
    pub fn new(
        data: Vec<T>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        shape: &[usize],
    ) -> Result<Self> {
        if data.len() != row_indices.len() || data.len() != col_indices.len() {
            return Err(StorageError::ShapeMismatch {
                expected: data.len(),
                actual: row_indices.len().max(col_indices.len()),
            });
        }

        if shape.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "COO storage requires 2D shape",
            });
        }

        let rows = shape[0];
        let cols = shape[1];

        for &row in &row_indices {
            if row >= rows {
                return Err(StorageError::IndexOutOfBounds {
                    index: row,
                    bound: rows,
                });
            }
        }
        for &col in &col_indices {
            if col >= cols {
                return Err(StorageError::IndexOutOfBounds {
                    index: col,
                    bound: cols,
                });
            }
        }

        Ok(Self {
            data,
            row_indices,
            col_indices,
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
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// Get reference to column indices
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
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

    /// Sort COO entries by row index, then by column index within each row
    /// 
    /// This is useful for efficient conversion to CSR format.
    pub fn sort(&mut self) {
        // Create permutation indices
        let mut perm: Vec<usize> = (0..self.nnz()).collect();
        perm.sort_by(|&a, &b| {
            let row_cmp = self.row_indices[a].cmp(&self.row_indices[b]);
            if row_cmp == core::cmp::Ordering::Equal {
                self.col_indices[a].cmp(&self.col_indices[b])
            } else {
                row_cmp
            }
        });

        // Apply permutation to all arrays
        let new_data: Vec<T> = perm.iter().map(|&i| self.data[i]).collect();
        let new_rows: Vec<usize> = perm.iter().map(|&i| self.row_indices[i]).collect();
        let new_cols: Vec<usize> = perm.iter().map(|&i| self.col_indices[i]).collect();

        self.data = new_data;
        self.row_indices = new_rows;
        self.col_indices = new_cols;
    }
}
