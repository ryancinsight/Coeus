//! Stride operations for dense storage

use storage::{DenseStorage, Result, StorageError, Storage};
use dtype::DataType;
use alloc::{vec, vec::Vec};

/// Create strided view of dense storage
///
/// # Arguments
/// * `storage` - Input storage
/// * `shape` - New shape dimensions
/// * `strides` - Stride for each dimension
/// * `offset` - Starting offset in the data
///
/// # Returns
/// New dense storage with strided view
///
/// # Examples
/// ```
/// use dense::layout::as_strided;
/// use dense::creation::from_vec;
/// use dtype::float::Float32;
///
/// let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
/// let storage = from_vec(data, &[4]).unwrap();
/// let strided = as_strided(&storage, &[2], &[2], 0).unwrap();
/// // Creates strided view with shape [2] and stride [2]
/// ```
pub fn as_strided<T: DataType>(
    storage: &DenseStorage<T>,
    shape: &[usize],
    strides: &[usize],
    offset: usize,
) -> Result<DenseStorage<T>>
where
    T: Clone,
{
    if shape.len() != strides.len() {
        return Err(StorageError::InvalidStride {
            reason: "shape and strides must have same length",
        });
    }
    
    let total_elements = shape.iter().product::<usize>();
    if total_elements == 0 {
        return DenseStorage::from_vec(Vec::new(), shape);
    }
    
    // Validate that strided access doesn't go out of bounds
    let max_index = compute_max_strided_index(shape, strides, offset);
    if max_index >= storage.len() {
        return Err(StorageError::IndexOutOfBounds {
            index: max_index,
            bound: storage.len(),
        });
    }
    
    // Extract strided data
    let mut result_data = Vec::with_capacity(total_elements);
    let src_data = storage.as_slice();
    
    // Generate all multi-dimensional indices and convert to flat indices
    let mut indices = vec![0; shape.len()];
    for _ in 0..total_elements {
        // Compute flat index using strides
        let flat_idx = offset + indices.iter().zip(strides.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum::<usize>();
        
        result_data.push(src_data[flat_idx].clone());
        
        // Increment multi-dimensional indices
        increment_indices(&mut indices, shape);
    }
    
    DenseStorage::from_vec(result_data, shape)
}

/// Compute strides for contiguous memory layout
///
/// # Arguments
/// * `shape` - Shape dimensions
///
/// # Returns
/// Vector of strides for each dimension
///
/// # Examples
/// ```
/// use dense::layout::compute_strides;
///
/// let strides = compute_strides(&[2, 3, 4]);
/// assert_eq!(strides, vec![12, 4, 1]);
/// ```
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    if !shape.is_empty() {
        strides[shape.len() - 1] = 1;
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    strides
}

/// Check if dense storage has contiguous memory layout
///
/// # Arguments
/// * `storage` - Input storage
///
/// # Returns
/// True if storage is contiguous, false otherwise
///
/// # Examples
/// ```
/// use dense::layout::is_contiguous;
/// use dense::creation::from_vec;
/// use dtype::float::Float32;
///
/// let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
/// let storage = from_vec(data, &[2, 2]).unwrap();
/// assert!(is_contiguous(&storage));
/// ```
pub fn is_contiguous<T: DataType>(storage: &DenseStorage<T>) -> bool {
    let shape = storage.shape().dims();
    let actual_strides = storage.strides();
    let expected_strides = compute_strides(shape);
    
    actual_strides == expected_strides
}

/// Helper function to compute maximum index accessed by strided operation
fn compute_max_strided_index(shape: &[usize], strides: &[usize], offset: usize) -> usize {
    let mut max_index = offset;
    
    for (&dim_size, &stride) in shape.iter().zip(strides.iter()) {
        if dim_size > 0 {
            max_index += (dim_size - 1) * stride;
        }
    }
    
    max_index
}

/// Helper function to increment multi-dimensional indices
fn increment_indices(indices: &mut [usize], shape: &[usize]) {
    for i in (0..indices.len()).rev() {
        indices[i] += 1;
        if indices[i] < shape[i] {
            break;
        }
        indices[i] = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::from_vec;
    use dtype::float::Float32;

    #[test]
    fn test_compute_strides() {
        assert_eq!(compute_strides(&[]), vec![]);
        assert_eq!(compute_strides(&[5]), vec![1]);
        assert_eq!(compute_strides(&[2, 3]), vec![3, 1]);
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn test_is_contiguous() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
        let storage = from_vec(data, &[2, 2]).unwrap();
        assert!(is_contiguous(&storage));
    }

    #[test]
    fn test_as_strided_simple() {
        let data = vec![
            Float32::new(1.0), Float32::new(2.0), 
            Float32::new(3.0), Float32::new(4.0),
        ];
        let storage = from_vec(data, &[4]).unwrap();
        
        // Extract every other element
        let strided = as_strided(&storage, &[2], &[2], 0).unwrap();
        let expected = vec![Float32::new(1.0), Float32::new(3.0)];
        
        assert_eq!(strided.as_slice(), expected.as_slice());
        assert_eq!(strided.shape().dims(), &[2]);
    }

    #[test]
    fn test_as_strided_2d() {
        let data = vec![
            Float32::new(1.0), Float32::new(2.0), Float32::new(3.0),
            Float32::new(4.0), Float32::new(5.0), Float32::new(6.0),
        ];
        let storage = from_vec(data, &[6]).unwrap();
        
        // Create 2x2 view with stride [3, 1]
        let strided = as_strided(&storage, &[2, 2], &[3, 1], 0).unwrap();
        let expected = vec![
            Float32::new(1.0), Float32::new(2.0),
            Float32::new(4.0), Float32::new(5.0),
        ];
        
        assert_eq!(strided.as_slice(), expected.as_slice());
        assert_eq!(strided.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_as_strided_with_offset() {
        let data = vec![
            Float32::new(1.0), Float32::new(2.0), 
            Float32::new(3.0), Float32::new(4.0),
        ];
        let storage = from_vec(data, &[4]).unwrap();
        
        // Extract 2 elements starting from index 1
        let strided = as_strided(&storage, &[2], &[1], 1).unwrap();
        let expected = vec![Float32::new(2.0), Float32::new(3.0)];
        
        assert_eq!(strided.as_slice(), expected.as_slice());
        assert_eq!(strided.shape().dims(), &[2]);
    }

    #[test]
    fn test_as_strided_out_of_bounds() {
        let data = vec![Float32::new(1.0), Float32::new(2.0)];
        let storage = from_vec(data, &[2]).unwrap();
        
        // Try to access beyond bounds
        let result = as_strided(&storage, &[2], &[2], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_increment_indices() {
        let mut indices = vec![0, 0];
        let shape = vec![2, 3];
        
        increment_indices(&mut indices, &shape);
        assert_eq!(indices, vec![0, 1]);
        
        increment_indices(&mut indices, &shape);
        assert_eq!(indices, vec![0, 2]);
        
        increment_indices(&mut indices, &shape);
        assert_eq!(indices, vec![1, 0]);
    }
}