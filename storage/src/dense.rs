//! Dense storage implementation
//!
//! Provides contiguous memory storage for tensors with row-major (C-contiguous) layout.

use crate::{AsAny, DataType, Result, Shape, Storage, StorageError};
use alloc::vec;
use alloc::vec::Vec;

/// Dense contiguous storage with row-major layout.
///
/// Memory is allocated as a single contiguous block with elements ordered
/// in row-major (C-contiguous) format for cache-efficient access.
///
/// # Examples
///
/// ```
/// use storage::{DenseStorage, Storage};
/// use dtype::float::Float32;
///
/// // Create 2x3 matrix storage
/// let data = vec![
///     Float32::new(1.0), Float32::new(2.0), Float32::new(3.0),
///     Float32::new(4.0), Float32::new(5.0), Float32::new(6.0),
/// ];
/// let storage = DenseStorage::from_vec(data, &[2, 3]).unwrap();
/// assert_eq!(storage.shape().dims(), &[2, 3]);
/// assert_eq!(storage.len(), 6);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DenseStorage<T: DataType> {
    data: Vec<T>,
    shape: Shape,
    strides: Vec<usize>,
}

impl<T: DataType> Default for DenseStorage<T> {
    fn default() -> Self {
        let shape = Shape::new(&[]).expect("Scalar shape is valid");
        Self {
            data: vec![T::default()],
            shape,
            strides: vec![],
        }
    }
}

impl<T: DataType> DenseStorage<T> {
    /// Creates dense storage from a vector with specified shape.
    ///
    /// # Errors
    ///
    /// Returns error if data size doesn't match shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use storage::{DenseStorage, Storage};
    /// use dtype::float::Float32;
    ///
    /// let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
    /// let storage = DenseStorage::from_vec(data, &[3]).unwrap();
    /// assert_eq!(storage.len(), 3);
    /// ```
    pub fn from_vec(data: Vec<T>, dims: &[usize]) -> Result<Self> {
        let shape = Shape::new(dims)?;

        if data.len() != shape.size() {
            return Err(StorageError::ShapeMismatch {
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
    ///
    /// # Errors
    ///
    /// Returns error if slice size doesn't match shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use storage::{DenseStorage, Storage};
    /// use dtype::int::Int32;
    ///
    /// let data = [Int32::new(1), Int32::new(2), Int32::new(3), Int32::new(4)];
    /// let storage = DenseStorage::from_slice(&data, &[2, 2]).unwrap();
    /// assert_eq!(storage.shape().dims(), &[2, 2]);
    /// ```
    pub fn from_slice(data: &[T], dims: &[usize]) -> Result<Self> {
        Self::from_vec(data.to_vec(), dims)
    }

    /// Creates dense storage filled with zeros.
    ///
    /// # Errors
    ///
    /// Returns error if shape specification is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use storage::{DenseStorage, Storage};
    /// use dtype::float::Float64;
    /// use num_traits::Zero;
    ///
    /// let storage = DenseStorage::<Float64>::zeros(&[2, 3]).unwrap();
    /// assert_eq!(storage.len(), 6);
    /// assert!(storage.as_slice().iter().all(|&x| x.is_zero()));
    /// ```
    pub fn zeros(dims: &[usize]) -> Result<Self>
    where
        T: num_traits::Zero,
    {
        let shape = Shape::new(dims)?;
        let data = alloc::vec![T::zero(); shape.size()];
        let strides = shape.row_major_strides();

        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    /// Creates dense storage filled with a constant value.
    ///
    /// # Arguments
    /// * `dims` - Shape dimensions
    /// * `value` - Value to fill storage with
    ///
    /// # Errors
    ///
    /// Returns error if shape specification is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use storage::{DenseStorage, Storage};
    /// use dtype::float::Float32;
    ///
    /// let storage = DenseStorage::<Float32>::full(&[2, 3], Float32::new(5.0)).unwrap();
    /// assert_eq!(storage.len(), 6);
    /// assert!(storage.as_slice().iter().all(|&x| x == Float32::new(5.0)));
    /// ```
    pub fn full(dims: &[usize], value: T) -> Result<Self> {
        let shape = Shape::new(dims)?;
        let data = alloc::vec![value; shape.size()];
        let strides = shape.row_major_strides();

        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    /// Creates dense storage filled with ones.
    ///
    /// # Errors
    ///
    /// Returns error if shape specification is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use storage::{DenseStorage, Storage};
    /// use dtype::float::Float32;
    /// use num_traits::One;
    ///
    /// let storage = DenseStorage::<Float32>::ones(&[3]).unwrap();
    /// assert!(storage.as_slice().iter().all(|&x| x.is_one()));
    /// ```
    pub fn ones(dims: &[usize]) -> Result<Self>
    where
        T: num_traits::One,
    {
        let shape = Shape::new(dims)?;
        let data = alloc::vec![T::one(); shape.size()];
        let strides = shape.row_major_strides();

        Ok(Self {
            data,
            shape,
            strides,
        })
    }
}

impl<T: DataType> AsAny for DenseStorage<T> {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<T: DataType> Storage<T> for DenseStorage<T> {
    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn strides(&self) -> &[usize] {
        &self.strides
    }

    fn is_contiguous(&self) -> bool {
        // Dense storage is always contiguous
        true
    }

    fn as_storage_ref(&self) -> &Self {
        self
    }

    fn full(dims: &[usize], value: T) -> Result<Self> {
        Self::full(dims, value)
    }
}

impl<T: DataType> crate::StorageFromVec<T> for DenseStorage<T> {
    fn from_vec(data: Vec<T>, dims: &[usize]) -> crate::Result<Self> {
        Self::from_vec(data, dims)
    }

    fn zeros(dims: &[usize]) -> crate::Result<Self>
    where
        T: num_traits::Zero,
    {
        Self::zeros(dims)
    }

    fn ones(dims: &[usize]) -> crate::Result<Self>
    where
        T: num_traits::One,
    {
        Self::ones(dims)
    }
}

impl<T: DataType> crate::StorageToDense<T> for DenseStorage<T> {
    fn to_dense(&self) -> crate::Result<DenseStorage<T>> {
        // Dense storage is already dense - return clone
        Ok(self.clone())
    }
}

impl<T: DataType> crate::MatMulStorage<T> for DenseStorage<T>
where
    T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + num_traits::Zero + Copy,
{
    fn matmul_storage(&self, other: &Self) -> crate::Result<Self> {
        // Validate dimensions
        let self_shape = self.shape().dims();
        let other_shape = other.shape().dims();

        if self_shape.len() != 2 || other_shape.len() != 2 {
            return Err(crate::StorageError::ShapeMismatch {
                expected: 2,
                actual: self_shape.len().max(other_shape.len()),
            });
        }

        if self_shape[1] != other_shape[0] {
            return Err(crate::StorageError::ShapeMismatch {
                expected: self_shape[1],
                actual: other_shape[0],
            });
        }

        let m = self_shape[0];
        let n = self_shape[1];
        let p = other_shape[1];

        // Perform dense matrix multiplication
        let mut result_data = vec![T::zero(); m * p];

        for i in 0..m {
            for j in 0..p {
                let mut sum = T::zero();
                for k in 0..n {
                    let self_idx = i * n + k;
                    let other_idx = k * p + j;
                    sum = sum + self.data[self_idx] * other.data[other_idx];
                }
                result_data[i * p + j] = sum;
            }
        }

        DenseStorage::from_vec(result_data, &[m, p])
    }
}

impl<T: DataType> crate::TransposeStorage<T> for DenseStorage<T> {
    fn transpose_storage(&self, dim0: usize, dim1: usize) -> crate::Result<Self> {
        let shape = self.shape().dims();

        if dim0 >= shape.len() || dim1 >= shape.len() {
            return Err(crate::StorageError::ShapeMismatch {
                expected: shape.len(),
                actual: dim0.max(dim1),
            });
        }

        if shape.len() != 2 {
            // For now, only support 2D transpose
            // Future enhancement: Implement general ND transpose for tensors > 2D
            return Err(crate::StorageError::ShapeMismatch {
                expected: 2,
                actual: shape.len(),
            });
        }

        let mut new_shape = shape.to_vec();
        new_shape.swap(dim0, dim1);

        let mut result_data = vec![T::zero(); self.data.len()];

        if dim0 == 0 && dim1 == 1 {
            // Standard matrix transpose
            let rows = shape[0];
            let cols = shape[1];

            for i in 0..rows {
                for j in 0..cols {
                    let src_idx = i * cols + j;
                    let dst_idx = j * rows + i;
                    result_data[dst_idx] = self.data[src_idx];
                }
            }
        } else {
            // General case (for future ND support)
            return Err(crate::StorageError::ShapeMismatch {
                expected: 0,
                actual: 1, // Placeholder error
            });
        }

        DenseStorage::from_vec(result_data, &new_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use dtype::float::{Float32, Float64};
    use dtype::int::Int32;

    #[test]
    fn test_from_vec_correct_shape() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let storage = DenseStorage::from_vec(data, &[3]).unwrap();
        assert_eq!(storage.len(), 3);
        assert_eq!(storage.shape().dims(), &[3]);
    }

    #[test]
    fn test_from_vec_shape_mismatch() {
        let data = vec![Float32::new(1.0), Float32::new(2.0)];
        let result = DenseStorage::from_vec(data, &[3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_slice_2d() {
        let data = [Int32::new(1), Int32::new(2), Int32::new(3), Int32::new(4)];
        let storage = DenseStorage::from_slice(&data, &[2, 2]).unwrap();
        assert_eq!(storage.shape().dims(), &[2, 2]);
        assert_eq!(storage.len(), 4);
    }

    #[test]
    fn test_zeros() {
        let storage = DenseStorage::<Float64>::zeros(&[2, 3]).unwrap();
        assert_eq!(storage.len(), 6);
        assert!(storage.as_slice().iter().all(num_traits::Zero::is_zero));
    }

    #[test]
    fn test_ones() {
        let storage = DenseStorage::<Float32>::ones(&[3]).unwrap();
        assert_eq!(storage.len(), 3);
        assert!(storage.as_slice().iter().all(num_traits::One::is_one));
    }

    #[test]
    fn test_is_contiguous() {
        let storage = DenseStorage::<Int32>::zeros(&[2, 3]).unwrap();
        assert!(storage.is_contiguous());
    }

    #[test]
    fn test_strides_row_major() {
        let storage = DenseStorage::<Float32>::zeros(&[2, 3, 4]).unwrap();
        assert_eq!(storage.strides(), &[12, 4, 1]);
    }

    #[test]
    fn test_mut_slice() {
        let mut storage = DenseStorage::<Int32>::zeros(&[3]).unwrap();
        let slice = storage.as_mut_slice();
        slice[0] = Int32::new(42);
        assert_eq!(storage.as_slice()[0], Int32::new(42));
    }
}
