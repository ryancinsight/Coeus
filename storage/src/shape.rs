//! Shape and dimensionality handling for tensors

use crate::{Result, StorageError};
use alloc::vec::Vec;
use core::fmt;

/// Multi-dimensional shape specification.
///
/// Represents the dimensions of a tensor in row-major (C-contiguous) order.
///
/// # Examples
///
/// ```
/// use storage::Shape;
///
/// // Scalar (0-D tensor)
/// let scalar = Shape::new(&[]).unwrap();
/// assert_eq!(scalar.ndim(), 0);
/// assert_eq!(scalar.size(), 1);
///
/// // Vector (1-D tensor)
/// let vec = Shape::new(&[5]).unwrap();
/// assert_eq!(vec.ndim(), 1);
/// assert_eq!(vec.size(), 5);
///
/// // Matrix (2-D tensor)
/// let mat = Shape::new(&[3, 4]).unwrap();
/// assert_eq!(mat.ndim(), 2);
/// assert_eq!(mat.size(), 12);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
    size: usize,
}

impl Shape {
    /// Creates a new shape from dimensions.
    ///
    /// # Errors
    ///
    /// Returns error if any dimension is zero (except for empty tensors).
    pub fn new(dims: &[usize]) -> Result<Self> {
        // Calculate total size
        let size = if dims.is_empty() {
            1 // Scalar has size 1
        } else {
            dims.iter().product()
        };

        Ok(Self {
            dims: dims.to_vec(),
            size,
        })
    }

    /// Returns the number of dimensions (rank).
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total number of elements.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Returns the dimensions as a slice.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the dimension at the given axis.
    ///
    /// # Errors
    ///
    /// Returns error if axis is out of bounds.
    pub fn dim(&self, axis: usize) -> Result<usize> {
        self.dims
            .get(axis)
            .copied()
            .ok_or(StorageError::IndexOutOfBounds {
                index: axis,
                bound: self.dims.len(),
            })
    }

    /// Computes row-major (C-contiguous) strides for this shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use storage::Shape;
    ///
    /// let shape = Shape::new(&[2, 3, 4]).unwrap();
    /// let strides = shape.row_major_strides();
    /// assert_eq!(strides, vec![12, 4, 1]);
    /// ```
    #[must_use]
    pub fn row_major_strides(&self) -> Vec<usize> {
        let mut strides = Vec::with_capacity(self.dims.len());
        let mut stride = 1;

        // Compute strides from right to left (row-major)
        for &dim in self.dims.iter().rev() {
            strides.push(stride);
            stride *= dim;
        }

        strides.reverse();
        strides
    }

    /// Computes column-major (Fortran-contiguous) strides for this shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use storage::Shape;
    ///
    /// let shape = Shape::new(&[2, 3, 4]).unwrap();
    /// let strides = shape.column_major_strides();
    /// assert_eq!(strides, vec![1, 2, 6]);
    /// ```
    #[must_use]
    pub fn column_major_strides(&self) -> Vec<usize> {
        let mut strides = Vec::with_capacity(self.dims.len());
        let mut stride = 1;

        // Compute strides from left to right (column-major)
        for &dim in &self.dims {
            strides.push(stride);
            stride *= dim;
        }

        strides
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, &dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{dim}")?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_scalar_shape() {
        let shape = Shape::new(&[]).unwrap();
        assert_eq!(shape.ndim(), 0);
        assert_eq!(shape.size(), 1);
        assert!(shape.dims().is_empty());
    }

    #[test]
    fn test_vector_shape() {
        let shape = Shape::new(&[5]).unwrap();
        assert_eq!(shape.ndim(), 1);
        assert_eq!(shape.size(), 5);
        assert_eq!(shape.dim(0).unwrap(), 5);
    }

    #[test]
    fn test_matrix_shape() {
        let shape = Shape::new(&[3, 4]).unwrap();
        assert_eq!(shape.ndim(), 2);
        assert_eq!(shape.size(), 12);
        assert_eq!(shape.dim(0).unwrap(), 3);
        assert_eq!(shape.dim(1).unwrap(), 4);
    }

    #[test]
    fn test_row_major_strides() {
        let shape = Shape::new(&[2, 3, 4]).unwrap();
        let strides = shape.row_major_strides();
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_column_major_strides() {
        let shape = Shape::new(&[2, 3, 4]).unwrap();
        let strides = shape.column_major_strides();
        assert_eq!(strides, vec![1, 2, 6]);
    }

    #[test]
    fn test_out_of_bounds_dim() {
        let shape = Shape::new(&[3, 4]).unwrap();
        assert!(shape.dim(2).is_err());
    }
}
