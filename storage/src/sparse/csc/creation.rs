//! CSC creation functions

use super::core::CscStorage;
use crate::{DataType, DenseStorage, Result, Storage, StorageError};
use alloc::{vec, vec::Vec};

impl<T: DataType> CscStorage<T> {
    /// Create empty CSC storage with given shape
    pub fn empty(shape: &[usize]) -> Result<Self> {
        if shape.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "CSC storage requires 2D shape",
            });
        }
        let cols = shape[1];
        Self::new(Vec::new(), Vec::new(), vec![0; cols + 1], shape)
    }

    /// Create from dense storage
    pub fn from_dense(dense: &DenseStorage<T>) -> Result<Self>
    where
        T: num_traits::Zero + PartialEq,
    {
        let shape_dims = dense.shape().dims();
        if shape_dims.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "Can only convert 2D dense storage to CSC",
            });
        }

        let rows = shape_dims[0];
        let cols = shape_dims[1];
        let dense_data = dense.as_slice();

        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];

        // Column-major iteration
        for col in 0..cols {
            for row in 0..rows {
                let value = dense_data[row * cols + col];
                if value != T::zero() {
                    data.push(value);
                    indices.push(row);
                }
            }
            indptr.push(data.len());
        }

        Self::new(data, indices, indptr, shape_dims)
    }

    /// Create filled with constant value
    pub fn full_value(dims: &[usize], value: T) -> Result<Self> {
        if dims.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "CSC storage requires 2D shape",
            });
        }

        let rows = dims[0];
        let cols = dims[1];

        if value == T::zero() {
            Self::empty(dims)
        } else {
            let total = rows * cols;
            let mut data = Vec::with_capacity(total);
            let mut indices = Vec::with_capacity(total);
            let mut indptr = vec![0];

            for _col in 0..cols {
                for row in 0..rows {
                    data.push(value);
                    indices.push(row);
                }
                indptr.push(data.len());
            }

            Self::new(data, indices, indptr, dims)
        }
    }
}
