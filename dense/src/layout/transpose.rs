//! Transpose operations for dense storage

use storage::{DenseStorage, Result, StorageError, Storage};
use dtype::DataType;
use alloc::{vec, vec::Vec};

/// Transpose dense storage (general N-dimensional)
///
/// # Arguments
/// * `storage` - Input storage
/// * `axes` - Optional axis permutation (if None, reverses all axes)
///
/// # Returns
/// New dense storage with transposed dimensions
///
/// # Examples
/// ```
/// use dense::layout::transpose;
/// use dense::creation::from_vec;
/// use dtype::float::Float32;
///
/// let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
/// let storage = from_vec(data, &[2, 2]).unwrap();
/// let transposed = transpose(&storage, None).unwrap();
/// // Transposes 2x2 matrix
/// ```
pub fn transpose<T: DataType>(
    storage: &DenseStorage<T>,
    axes: Option<&[usize]>,
) -> Result<DenseStorage<T>>
where
    T: Clone,
{
    let shape = storage.shape().dims();
    let ndim = shape.len();
    
    if ndim == 0 {
        // Scalar case - return copy
        return Ok(storage.clone());
    }
    
    if ndim == 1 {
        // 1D case - return copy
        return Ok(storage.clone());
    }
    
    // Determine axis permutation
    let perm: Vec<usize> = match axes {
        Some(axes) => {
            if axes.len() != ndim {
                return Err(StorageError::InvalidShape {
                    reason: "axes length must match number of dimensions",
                });
            }
            axes.to_vec()
        }
        None => (0..ndim).rev().collect(), // Reverse all axes
    };
    
    // Validate permutation
    let mut sorted_perm = perm.clone();
    sorted_perm.sort();
    if sorted_perm != (0..ndim).collect::<Vec<_>>() {
        return Err(StorageError::InvalidShape {
            reason: "axes must be a valid permutation",
        });
    }
    
    // Compute new shape
    let new_shape: Vec<usize> = perm.iter().map(|&i| shape[i]).collect();
    
    // Compute strides for original and new layouts
    let old_strides = compute_strides(shape);
    let new_strides: Vec<usize> = perm.iter().map(|&i| old_strides[i]).collect();
    
    // Create output data
    let total_size = shape.iter().product();
    let mut result_data = vec![storage.as_slice()[0].clone(); total_size];
    
    // Copy data with transposed indexing
    transpose_copy(
        storage.as_slice(),
        &mut result_data,
        shape,
        &new_shape,
        &old_strides,
        &compute_strides(&new_shape), &perm,
    );
    
    DenseStorage::from_vec(result_data, &new_shape)
}

/// Transpose 2D dense storage (matrix transpose)
///
/// # Arguments
/// * `storage` - Input 2D storage
///
/// # Returns
/// New dense storage with transposed matrix
///
/// # Examples
/// ```
/// use dense::layout::transpose_2d;
/// use dense::creation::from_vec;
/// use dtype::float::Float32;
///
/// let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
/// let storage = from_vec(data, &[2, 2]).unwrap();
/// let transposed = transpose_2d(&storage).unwrap();
/// // Transposes 2x2 matrix
/// ```
pub fn transpose_2d<T: DataType>(storage: &DenseStorage<T>) -> Result<DenseStorage<T>>
where
    T: Clone,
{
    let shape = storage.shape().dims();
    
    if shape.len() != 2 {
        return Err(StorageError::InvalidShape {
            reason: "transpose_2d requires 2D storage",
        });
    }
    
    transpose(storage, Some(&[1, 0]))
}

/// Swap two axes in dense storage
///
/// # Arguments
/// * `storage` - Input storage
/// * `axis1` - First axis to swap
/// * `axis2` - Second axis to swap
///
/// # Returns
/// New dense storage with swapped axes
///
/// # Examples
/// ```
/// use dense::layout::swap_axes;
/// use dense::creation::from_vec;
/// use dtype::float::Float32;
///
/// let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
/// let storage = from_vec(data, &[2, 2]).unwrap();
/// let swapped = swap_axes(&storage, 0, 1).unwrap();
/// // Swaps axes 0 and 1
/// ```
pub fn swap_axes<T: DataType>(
    storage: &DenseStorage<T>,
    axis1: usize,
    axis2: usize,
) -> Result<DenseStorage<T>>
where
    T: Clone,
{
    let shape = storage.shape().dims();
    let ndim = shape.len();
    
    if axis1 >= ndim || axis2 >= ndim {
        return Err(StorageError::IndexOutOfBounds {
            index: axis1.max(axis2),
            bound: ndim,
        });
    }
    
    if axis1 == axis2 {
        return Ok(storage.clone());
    }
    
    // Create permutation that swaps the two axes
    let mut perm: Vec<usize> = (0..ndim).collect();
    perm.swap(axis1, axis2);
    
    transpose(storage, Some(&perm))
}

/// Helper function to compute strides for a given shape
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    if !shape.is_empty() {
        strides[shape.len() - 1] = 1;
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    strides
}

/// Helper function to copy data with transposed indexing
fn transpose_copy<T: Clone>(
    src: &[T],
    dst: &mut [T],
    old_shape: &[usize],
    _new_shape: &[usize],
    old_strides: &[usize],
    new_strides: &[usize],
    perm: &[usize],
) {
    let total_size = old_shape.iter().product();

    for flat_idx in 0..total_size {
        // Convert flat index to multi-dimensional indices in old layout
        let mut old_indices = vec![0; old_shape.len()];
        let mut remaining = flat_idx;
        for (i, &stride) in old_strides.iter().enumerate() {
            old_indices[i] = remaining / stride;
            remaining %= stride;
        }

        // Permute indices according to transpose
        let mut new_indices = vec![0; perm.len()];
        for (new_i, &old_i) in perm.iter().enumerate() {
            new_indices[new_i] = old_indices[old_i];
        }

        // Convert new multi-dimensional indices to flat index in new layout
        let mut new_flat_idx = 0;
        for (i, &idx) in new_indices.iter().enumerate() {
            new_flat_idx += idx * new_strides[i];
        }

        dst[new_flat_idx] = src[flat_idx].clone();
    }
}


#[cfg(test)]
mod tests {
    use alloc::vec;
    use super::*;
    use crate::creation::from_vec;
    use dtype::float::Float32;

    #[test]
    fn test_transpose_2d() {
        let data = vec![
            Float32::new(1.0), Float32::new(2.0),
            Float32::new(3.0), Float32::new(4.0),
        ];
        let storage = from_vec(data, &[2, 2]).unwrap();
        let transposed = transpose_2d(&storage).unwrap();
        
        let expected = vec![
            Float32::new(1.0), Float32::new(3.0),
            Float32::new(2.0), Float32::new(4.0),
        ];
        
        assert_eq!(transposed.as_slice(), expected.as_slice());
        assert_eq!(transposed.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_transpose_rectangular() {
        let data = vec![
            Float32::new(1.0), Float32::new(2.0), Float32::new(3.0),
            Float32::new(4.0), Float32::new(5.0), Float32::new(6.0),
        ];
        let storage = from_vec(data, &[2, 3]).unwrap();
        let transposed = transpose_2d(&storage).unwrap();
        
        let expected = vec![
            Float32::new(1.0), Float32::new(4.0),
            Float32::new(2.0), Float32::new(5.0),
            Float32::new(3.0), Float32::new(6.0),
        ];
        
        assert_eq!(transposed.as_slice(), expected.as_slice());
        assert_eq!(transposed.shape().dims(), &[3, 2]);
    }

    #[test]
    fn test_swap_axes() {
        let data = vec![
            Float32::new(1.0), Float32::new(2.0),
            Float32::new(3.0), Float32::new(4.0),
        ];
        let storage = from_vec(data, &[2, 2]).unwrap();
        let swapped = swap_axes(&storage, 0, 1).unwrap();
        
        let expected = vec![
            Float32::new(1.0), Float32::new(3.0),
            Float32::new(2.0), Float32::new(4.0),
        ];
        
        assert_eq!(swapped.as_slice(), expected.as_slice());
        assert_eq!(swapped.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_transpose_1d() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let storage = from_vec(data.clone(), &[3]).unwrap();
        let transposed = transpose(&storage, None).unwrap();
        
        assert_eq!(transposed.as_slice(), data.as_slice());
        assert_eq!(transposed.shape().dims(), &[3]);
    }

    #[test]
    fn test_transpose_scalar() {
        let data = vec![Float32::new(42.0)];
        let storage = from_vec(data.clone(), &[]).unwrap();
        let transposed = transpose(&storage, None).unwrap();
        
        assert_eq!(transposed.as_slice(), data.as_slice());
        assert_eq!(transposed.shape().dims(), &[] as &[usize]);
    }
}