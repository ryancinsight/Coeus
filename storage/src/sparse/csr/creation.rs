//! CSR creation functions
//!
//! Contains constructors for creating CsrStorage instances.

use super::core::CsrStorage;
use crate::{DataType, DenseStorage, Result, Storage, StorageError};
use alloc::{vec, vec::Vec};

impl<T: DataType> CsrStorage<T> {
    /// Create empty CSR storage with given shape
    pub fn empty(shape: &[usize]) -> Result<Self> {
        if shape.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "CSR storage requires 2D shape",
            });
        }
        let rows = shape[0];
        Self::new(Vec::new(), Vec::new(), vec![0; rows + 1], shape)
    }

    /// Create identity matrix
    pub fn eye(size: usize) -> Result<Self>
    where
        T: num_traits::One,
    {
        let data = vec![T::one(); size];
        let indices: Vec<usize> = (0..size).collect();
        let indptr: Vec<usize> = (0..=size).collect();
        Self::new(data, indices, indptr, &[size, size])
    }

    /// Create from dense storage
    pub fn from_dense(dense: &DenseStorage<T>) -> Result<Self>
    where
        T: num_traits::Zero + PartialEq,
    {
        let shape_dims = dense.shape().dims();
        if shape_dims.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "Can only convert 2D dense storage to CSR",
            });
        }

        let rows = shape_dims[0];
        let cols = shape_dims[1];
        let dense_data = dense.as_slice();

        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];

        for row in 0..rows {
            for col in 0..cols {
                let value = dense_data[row * cols + col];
                if value != T::zero() {
                    data.push(value);
                    indices.push(col);
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
                reason: "CSR storage requires 2D shape",
            });
        }

        let rows = dims[0];
        let cols = dims[1];
        let total_elements = rows * cols;

        if value == T::zero() {
            Self::empty(dims)
        } else {
            let data = vec![value; total_elements];
            let indices: Vec<usize> = (0..cols).cycle().take(total_elements).collect();
            let indptr: Vec<usize> = (0..=rows).map(|i| i * cols).collect();
            Self::new(data, indices, indptr, dims)
        }
    }
}
