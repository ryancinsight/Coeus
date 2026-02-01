//! Reshape operations for dense storage

use storage::{DenseStorage, Result, StorageError, Storage};
use dtype::DataType;

/// Reshape dense storage to new dimensions
///
/// # Arguments
/// * `storage` - Input storage
/// * `new_shape` - New shape dimensions
///
/// # Returns
/// New dense storage with reshaped dimensions
///
/// # Errors
/// Returns error if total size doesn't match
///
/// # Examples
/// ```
/// use dense::layout::reshape;
/// use dense::creation::from_vec;
/// use dtype::float::Float32;
///
/// let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
/// let storage = from_vec(data, &[4]).unwrap();
/// let reshaped = reshape(&storage, &[2, 2]).unwrap();
/// // Reshapes from [4] to [2, 2]
/// ```
pub fn reshape<T: DataType>(
    storage: &DenseStorage<T>,
    new_shape: &[usize],
) -> Result<DenseStorage<T>>
where
    T: Clone,
{
    let old_size = storage.len();
    let new_size = new_shape.iter().product();
    
    if old_size != new_size {
        return Err(StorageError::ShapeMismatch {
            expected: new_size,
            actual: old_size,
        });
    }

    // Create new storage with same data but different shape
    let data = storage.as_slice().to_vec();
    DenseStorage::from_vec(data, new_shape)
}

/// Reshape dense storage to match another storage's shape
///
/// # Arguments
/// * `storage` - Input storage to reshape
/// * `like` - Storage to match shape from
///
/// # Returns
/// New dense storage with same shape as `like`
///
/// # Examples
/// ```
/// use dense::layout::{reshape_like};
/// use dense::creation::{from_vec, zeros};
/// use dtype::float::Float32;
///
/// let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
/// let storage = from_vec(data, &[4]).unwrap();
/// let target = zeros::<Float32>(&[2, 2]).unwrap();
/// let reshaped = reshape_like(&storage, &target).unwrap();
/// // Reshapes to match target's [2, 2] shape
/// ```
pub fn reshape_like<T: DataType>(
    storage: &DenseStorage<T>,
    like: &DenseStorage<T>,
) -> Result<DenseStorage<T>>
where
    T: Clone,
{
    reshape(storage, like.shape().dims())
}

/// Flatten dense storage to 1D
///
/// # Arguments
/// * `storage` - Input storage
///
/// # Returns
/// New dense storage flattened to 1D
///
/// # Examples
/// ```
/// use dense::layout::flatten;
/// use dense::creation::from_vec;
/// use dtype::float::Float32;
///
/// let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
/// let storage = from_vec(data, &[2, 2]).unwrap();
/// let flattened = flatten(&storage).unwrap();
/// // Flattens from [2, 2] to [4]
/// ```
pub fn flatten<T: DataType>(storage: &DenseStorage<T>) -> Result<DenseStorage<T>>
where
    T: Clone,
{
    reshape(storage, &[storage.len()])
}

/// Unflatten 1D storage to specified shape
///
/// # Arguments
/// * `storage` - Input 1D storage
/// * `shape` - Target shape dimensions
///
/// # Returns
/// New dense storage with specified shape
///
/// # Examples
/// ```
/// use dense::layout::unflatten;
/// use dense::creation::from_vec;
/// use dtype::float::Float32;
///
/// let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
/// let storage = from_vec(data, &[4]).unwrap();
/// let unflattened = unflatten(&storage, &[2, 2]).unwrap();
/// // Unflattens from [4] to [2, 2]
/// ```
pub fn unflatten<T: DataType>(
    storage: &DenseStorage<T>,
    shape: &[usize],
) -> Result<DenseStorage<T>>
where
    T: Clone,
{
    // Check if storage is 1D
    if storage.shape().dims().len() != 1 {
        return Err(StorageError::InvalidShape {
            reason: "unflatten requires 1D input storage",
        });
    }
    
    reshape(storage, shape)
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use super::*;
    use crate::creation::from_vec;
    use dtype::float::Float32;

    #[test]
    fn test_reshape_2d_to_1d() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
        let storage = from_vec(data.clone(), &[2, 2]).unwrap();
        let reshaped = reshape(&storage, &[4]).unwrap();
        
        assert_eq!(reshaped.as_slice(), data.as_slice());
        assert_eq!(reshaped.shape().dims(), &[4]);
    }

    #[test]
    fn test_reshape_1d_to_2d() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
        let storage = from_vec(data.clone(), &[4]).unwrap();
        let reshaped = reshape(&storage, &[2, 2]).unwrap();
        
        assert_eq!(reshaped.as_slice(), data.as_slice());
        assert_eq!(reshaped.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_reshape_size_mismatch() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let storage = from_vec(data, &[3]).unwrap();
        let result = reshape(&storage, &[2, 2]);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_flatten() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
        let storage = from_vec(data.clone(), &[2, 2]).unwrap();
        let flattened = flatten(&storage).unwrap();
        
        assert_eq!(flattened.as_slice(), data.as_slice());
        assert_eq!(flattened.shape().dims(), &[4]);
    }

    #[test]
    fn test_unflatten() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
        let storage = from_vec(data.clone(), &[4]).unwrap();
        let unflattened = unflatten(&storage, &[2, 2]).unwrap();
        
        assert_eq!(unflattened.as_slice(), data.as_slice());
        assert_eq!(unflattened.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_unflatten_non_1d() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
        let storage = from_vec(data, &[2, 2]).unwrap();
        let result = unflatten(&storage, &[4]);
        
        assert!(result.is_err());
    }
}