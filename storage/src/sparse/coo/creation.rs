//! COO creation functions

use super::core::CooStorage;
use crate::{DataType, DenseStorage, Result, Storage, StorageError};
use alloc::vec::Vec;

impl<T: DataType> CooStorage<T> {
    /// Create empty COO storage with given shape
    pub fn empty(shape: &[usize]) -> Result<Self> {
        Self::new(Vec::new(), Vec::new(), Vec::new(), shape)
    }

    /// Create from dense storage
    pub fn from_dense(dense: &DenseStorage<T>) -> Result<Self>
    where
        T: num_traits::Zero + PartialEq,
    {
        let shape_dims = dense.shape().dims();
        if shape_dims.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "Can only convert 2D dense storage to COO",
            });
        }

        let rows = shape_dims[0];
        let cols = shape_dims[1];
        let dense_data = dense.as_slice();

        let mut data = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();

        for row in 0..rows {
            for col in 0..cols {
                let value = dense_data[row * cols + col];
                if value != T::zero() {
                    data.push(value);
                    row_indices.push(row);
                    col_indices.push(col);
                }
            }
        }

        Self::new(data, row_indices, col_indices, shape_dims)
    }

    /// Create filled with constant value
    pub fn full_value(dims: &[usize], value: T) -> Result<Self> {
        if dims.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "COO storage requires 2D shape",
            });
        }

        let rows = dims[0];
        let cols = dims[1];

        if value == T::zero() {
            Self::empty(dims)
        } else {
            let total = rows * cols;
            let mut data = Vec::with_capacity(total);
            let mut row_indices = Vec::with_capacity(total);
            let mut col_indices = Vec::with_capacity(total);

            for row in 0..rows {
                for col in 0..cols {
                    data.push(value);
                    row_indices.push(row);
                    col_indices.push(col);
                }
            }

            Self::new(data, row_indices, col_indices, dims)
        }
    }
}
