//! Dense creation functions

use super::core::DenseStorage;
use crate::{DataType, Result, Shape};
use alloc::vec;

impl<T: DataType> DenseStorage<T> {
    /// Creates dense storage from a vector with specified shape.
    pub fn from_vec(data: alloc::vec::Vec<T>, dims: &[usize]) -> Result<Self> {
        let shape = Shape::new(dims)?;

        if data.len() != shape.size() {
            return Err(crate::StorageError::ShapeMismatch {
                expected: shape.size(),
                actual: data.len(),
            });
        }

        let strides = shape.row_major_strides();

        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    /// Creates dense storage from a slice with specified shape.
    pub fn from_slice(data: &[T], dims: &[usize]) -> Result<Self> {
        Self::from_vec(data.to_vec(), dims)
    }

    /// Creates dense storage filled with zeros.
    pub fn zeros(dims: &[usize]) -> Result<Self>
    where
        T: num_traits::Zero,
    {
        let shape = Shape::new(dims)?;
        let data = vec![T::zero(); shape.size()];
        let strides = shape.row_major_strides();

        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    /// Creates dense storage filled with a constant value.
    pub fn full(dims: &[usize], value: T) -> Result<Self> {
        let shape = Shape::new(dims)?;
        let data = vec![value; shape.size()];
        let strides = shape.row_major_strides();

        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    /// Creates dense storage filled with ones.
    pub fn ones(dims: &[usize]) -> Result<Self>
    where
        T: num_traits::One,
    {
        let shape = Shape::new(dims)?;
        let data = vec![T::one(); shape.size()];
        let strides = shape.row_major_strides();

        Ok(Self {
            data,
            shape,
            strides,
        })
    }
}
