//! Strided storage for tensor views and advanced indexing
//!
//! Provides memory-efficient tensor views with custom strides for operations
//! like transpose, slicing, and broadcasting without copying data.

use crate::{AsAny, Result, Shape, Storage, StorageError};
use alloc::{vec, vec::Vec};

/// Strided storage with custom memory layout
///
/// Enables zero-copy tensor operations like transpose, slicing, and views
/// by storing stride information alongside the underlying data buffer.
#[derive(Debug, Clone)]
pub struct StridedStorage<T: crate::DataType> {
    /// Reference to the underlying contiguous data buffer
    data: Vec<T>,
    /// Shape of this strided view
    shape: Shape,
    /// Strides for each dimension (elements to skip to move along each axis)
    strides: Vec<usize>,
    /// Offset into the underlying data buffer (for sliced views)
    offset: usize,
}

impl<T: crate::DataType> StridedStorage<T> {
    /// Creates a new strided storage from contiguous data
    ///
    /// # Arguments
    /// * `data` - Contiguous data buffer
    /// * `shape` - Shape of the strided view
    ///
    /// # Errors
    /// Returns error if shape dimensions don't match data size
    pub fn new(data: Vec<T>, shape: &[usize]) -> Result<Self> {
        let total_size = shape.iter().product();
        if total_size != data.len() {
            return Err(StorageError::ShapeMismatch {
                expected: total_size,
                actual: data.len(),
            });
        }

        let strides = shape
            .iter()
            .rev()
            .scan(1, |stride, &dim| {
                let current_stride = *stride;
                *stride *= dim;
                Some(current_stride)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        Ok(Self {
            data,
            shape: Shape::new(shape)?,
            strides,
            offset: 0,
        })
    }

    /// Creates a strided view from existing data with custom strides
    ///
    /// # Arguments
    /// * `data` - Reference to underlying data buffer
    /// * `shape` - Shape of the strided view
    /// * `strides` - Strides for each dimension
    /// * `offset` - Offset into the data buffer
    ///
    /// # Errors
    /// Returns error if dimensions are incompatible
    pub fn view(data: Vec<T>, shape: &[usize], strides: &[usize], offset: usize) -> Result<Self> {
        if shape.len() != strides.len() {
            return Err(StorageError::InvalidShape {
                reason: "Shape and strides must have same number of dimensions",
            });
        }

        // Validate that offset + computed size doesn't exceed data buffer
        let max_index = offset
            + shape
                .iter()
                .zip(strides.iter())
                .fold(0, |acc, (&dim, &stride)| {
                    acc + (dim.saturating_sub(1)) * stride
                });

        if max_index >= data.len() {
            return Err(StorageError::IndexOutOfBounds {
                index: max_index,
                bound: data.len(),
            });
        }

        Ok(Self {
            data,
            shape: Shape::new(shape)?,
            strides: strides.to_vec(),
            offset,
        })
    }

    /// Returns the strides for each dimension
    #[must_use]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the offset into the underlying data buffer
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Creates a transposed view of this strided storage
    ///
    /// # Arguments
    /// * `axes` - Permutation of dimensions for transpose
    ///
    /// # Returns
    /// New strided storage with transposed layout
    ///
    /// # Errors
    /// Returns error if axes are invalid
    pub fn transpose(&self, axes: Option<&[usize]>) -> Result<Self> {
        let ndim = self.shape.ndim();

        let axes = match axes {
            Some(axes) => {
                if axes.len() != ndim {
                    return Err(StorageError::InvalidShape {
                        reason: "Transpose axes must match number of dimensions",
                    });
                }

                // Validate that axes are a valid permutation
                let mut seen = vec![false; ndim];
                for &axis in axes {
                    if axis >= ndim || seen[axis] {
                        return Err(StorageError::InvalidShape {
                            reason: "Invalid transpose axes permutation",
                        });
                    }
                    seen[axis] = true;
                }

                axes.to_vec()
            }
            None => {
                // Default: reverse all dimensions
                (0..ndim).rev().collect()
            }
        };

        // Compute new shape and strides
        let mut new_shape = vec![0; ndim];
        let mut new_strides = vec![0; ndim];

        for (i, &axis) in axes.iter().enumerate() {
            new_shape[i] = self.shape.dims()[axis];
            new_strides[i] = self.strides[axis];
        }

        Self::view(self.data.clone(), &new_shape, &new_strides, self.offset)
    }

    /// Creates a sliced view of this strided storage
    ///
    /// # Arguments
    /// * `slices` - Slice specifications for each dimension as (start, end, step)
    ///
    /// # Returns
    /// New strided storage with sliced view
    ///
    /// # Errors
    /// Returns error if slice specifications are invalid
    pub fn slice(&self, slices: &[(Option<i32>, Option<i32>, i32)]) -> Result<Self> {
        if slices.len() != self.shape.ndim() {
            return Err(StorageError::ShapeMismatch {
                expected: self.shape.ndim(),
                actual: slices.len(),
            });
        }

        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();
        let mut new_offset = self.offset;

        for (dim_idx, &(start_opt, end_opt, step)) in slices.iter().enumerate() {
            let dim_size = i32::try_from(self.shape.dims()[dim_idx]).map_err(|_| {
                StorageError::InvalidShape {
                    reason: "Dimension too large",
                }
            })?;
            let stride = self.strides[dim_idx];

            // Resolve start and end bounds
            let start = start_opt.unwrap_or(if step >= 0 { 0 } else { dim_size - 1 });
            let end = end_opt.unwrap_or(if step >= 0 { dim_size } else { -1 });

            // Handle negative indices
            let start_idx = if start < 0 { dim_size + start } else { start };
            let end_idx = if end < 0 { dim_size + end } else { end };

            // Validate bounds
            if start_idx < 0 || start_idx >= dim_size || end_idx < -1 || end_idx > dim_size {
                let index = if start_idx < 0 {
                    usize::try_from(-start_idx).unwrap_or(0)
                } else {
                    usize::try_from(end_idx).unwrap_or(0)
                };
                let bound = usize::try_from(dim_size).unwrap_or(0);
                return Err(StorageError::IndexOutOfBounds { index, bound });
            }

            // Calculate slice length
            let slice_len = match step.cmp(&0) {
                std::cmp::Ordering::Greater => {
                    if end_idx > start_idx {
                        ((end_idx - start_idx - 1) / step) + 1
                    } else {
                        0
                    }
                }
                std::cmp::Ordering::Less => {
                    if start_idx > end_idx {
                        ((start_idx - end_idx - 1) / (-step)) + 1
                    } else {
                        0
                    }
                }
                std::cmp::Ordering::Equal => {
                    return Err(StorageError::InvalidShape {
                        reason: "Slice step cannot be zero",
                    });
                }
            };

            new_shape.push(usize::try_from(slice_len).unwrap_or(0));
            new_strides
                .push(usize::try_from(i32::try_from(stride).unwrap_or(0) * step).unwrap_or(0));

            // Update offset
            new_offset += usize::try_from(start_idx).unwrap_or(0) * stride;
        }

        Self::view(self.data.clone(), &new_shape, &new_strides, new_offset)
    }

    /// Converts this strided storage to contiguous dense storage
    ///
    /// This copies the data to ensure contiguous memory layout.
    ///
    /// # Panics
    /// Panics if the strided data cannot be reshaped into a dense storage.
    #[must_use]
    pub fn to_dense(&self) -> crate::DenseStorage<T>
    where
        T: Copy,
    {
        let len = self.shape.size();
        let mut dense_data = vec![self.data[0]; len]; // Initialize with default

        // Copy data according to strides
        self.copy_to_contiguous(&mut dense_data);

        crate::DenseStorage::from_vec(dense_data, self.shape.dims()).unwrap()
    }

    /// Internal helper to copy strided data to contiguous buffer
    fn copy_to_contiguous(&self, output: &mut [T])
    where
        T: Copy,
    {
        let mut output_idx = 0;
        self.copy_strided_recursive(&mut output_idx, output, &mut vec![0; self.shape.ndim()], 0);
    }

    /// Recursive helper for copying strided data
    fn copy_strided_recursive(
        &self,
        output_idx: &mut usize,
        output: &mut [T],
        indices: &mut [usize],
        dim: usize,
    ) where
        T: Copy,
    {
        if dim == self.shape.ndim() {
            // Compute flat index in strided storage
            let flat_idx = self.offset
                + indices
                    .iter()
                    .zip(&self.strides)
                    .map(|(&idx, &stride)| idx * stride)
                    .sum::<usize>();

            output[*output_idx] = self.data[flat_idx];
            *output_idx += 1;
            return;
        }

        for i in 0..self.shape.dims()[dim] {
            indices[dim] = i;
            self.copy_strided_recursive(output_idx, output, indices, dim + 1);
        }
    }
}

impl<T: crate::DataType> AsAny for StridedStorage<T> {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<T: crate::DataType> Storage<T> for StridedStorage<T> {
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
        // Check if strides match row-major order
        let row_major_strides = self.shape.row_major_strides();
        self.strides == row_major_strides && self.offset == 0
    }

    fn as_storage_ref(&self) -> &Self {
        self
    }

    fn full(dims: &[usize], value: T) -> crate::Result<Self> {
        let size = dims.iter().product();
        let data = vec![value; size];
        Self::new(data, dims)
    }
}

impl<T: crate::DataType> crate::StorageFromVec<T> for StridedStorage<T> {
    fn from_vec(data: Vec<T>, dims: &[usize]) -> crate::Result<Self> {
        Self::new(data, dims)
    }

    fn zeros(dims: &[usize]) -> crate::Result<Self>
    where
        T: num_traits::Zero,
    {
        let size = dims.iter().product();
        let data = vec![T::zero(); size];
        Self::new(data, dims)
    }

    fn ones(dims: &[usize]) -> crate::Result<Self>
    where
        T: num_traits::One,
    {
        let size = dims.iter().product();
        let data = vec![T::one(); size];
        Self::new(data, dims)
    }
}

impl<T: crate::DataType> crate::StorageToDense<T> for StridedStorage<T> {
    fn to_dense(&self) -> crate::Result<crate::DenseStorage<T>> {
        // Convert strided storage to dense by copying data in row-major order
        let data = self.as_slice().to_vec();
        crate::DenseStorage::from_vec(data, self.shape().dims())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use dtype::float::F32;

    #[test]
    fn test_strided_storage_creation() {
        let data = vec![
            F32::new(1.0),
            F32::new(2.0),
            F32::new(3.0),
            F32::new(4.0),
            F32::new(5.0),
            F32::new(6.0),
        ];
        let storage = StridedStorage::new(data, &[2, 3]).unwrap();

        assert_eq!(storage.shape().dims(), &[2, 3]);
        assert_eq!(storage.strides(), &[3, 1]);
        assert_eq!(storage.offset(), 0);
        assert!(storage.is_contiguous());
    }

    #[test]
    fn test_strided_transpose() {
        let data = vec![
            F32::new(1.0),
            F32::new(2.0),
            F32::new(3.0),
            F32::new(4.0),
            F32::new(5.0),
            F32::new(6.0),
        ];
        let storage = StridedStorage::new(data, &[2, 3]).unwrap();

        // Transpose 2x3 -> 3x2
        let transposed = storage.transpose(None).unwrap();
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        assert_eq!(transposed.strides(), &[1, 3]);
        assert!(!transposed.is_contiguous());
    }

    #[test]
    fn test_strided_slice() {
        let data = vec![
            F32::new(0.0),
            F32::new(1.0),
            F32::new(2.0),
            F32::new(3.0),
            F32::new(4.0),
            F32::new(5.0),
        ];
        let storage = StridedStorage::new(data, &[6]).unwrap();

        // Slice [1:5:2] -> elements at indices 1, 3
        let sliced = storage.slice(&[(Some(1), Some(5), 2)]).unwrap();
        assert_eq!(sliced.shape().dims(), &[2]);

        // Convert to dense to check values
        let dense = sliced.to_dense();
        assert_eq!(dense.as_slice(), &[F32::new(1.0), F32::new(3.0)]);
    }

    #[test]
    fn test_strided_to_dense() {
        let data = vec![
            F32::new(1.0),
            F32::new(2.0),
            F32::new(3.0),
            F32::new(4.0),
            F32::new(5.0),
            F32::new(6.0),
        ];
        let storage = StridedStorage::new(data, &[2, 3]).unwrap();

        // Transpose and convert to dense
        let transposed = storage.transpose(None).unwrap();
        let dense = transposed.to_dense();

        // Transposed 2x3 matrix: [[1, 4], [2, 5], [3, 6]]
        assert_eq!(
            dense.as_slice(),
            &[
                F32::new(1.0),
                F32::new(4.0),
                F32::new(2.0),
                F32::new(5.0),
                F32::new(3.0),
                F32::new(6.0)
            ]
        );
    }
}
