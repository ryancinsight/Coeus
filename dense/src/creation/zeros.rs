//! Zero initialization operations for dense storage

use storage::{DenseStorage, Result, Storage};
use dtype::DataType;

/// Create dense storage filled with zeros
///
/// Allocates new dense storage with all elements initialized to zero.
/// Uses the zero value from the `num_traits::Zero` trait.
///
/// # Arguments
/// * `shape` - Dimensions of the storage to create
///
/// # Returns
/// New dense storage filled with zeros
///
/// # Errors
/// Returns error if shape is invalid (contains zero dimensions)
///
/// # Examples
/// ```
/// use dense::creation::zeros;
/// use dtype::float::Float32;
///
/// let storage = zeros::<Float32>(&[2, 3]).unwrap();
/// assert_eq!(storage.shape().dims(), &[2, 3]);
/// assert_eq!(storage.len(), 6);
/// // All elements are zero
/// ```
pub fn zeros<T: DataType>(shape: &[usize]) -> Result<DenseStorage<T>>
where
    T: num_traits::Zero,
{
    DenseStorage::zeros(shape)
}

/// Create dense storage filled with zeros matching another storage's shape
///
/// Creates a new zero-filled storage with the same dimensions as the input storage.
/// Useful for creating temporary buffers or initializing gradients.
///
/// # Arguments
/// * `like` - Storage whose shape to match
///
/// # Returns
/// New dense storage filled with zeros
///
/// # Examples
/// ```
/// use dense::creation::zeros_like;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let original = DenseStorage::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();
/// let zeros_storage = zeros_like(&original).unwrap();
/// assert_eq!(zeros_storage.shape().dims(), original.shape().dims());
/// ```
pub fn zeros_like<T: DataType>(like: &DenseStorage<T>) -> Result<DenseStorage<T>>
where
    T: num_traits::Zero,
{
    zeros(like.shape().dims())
}

/// Create scalar (0-dimensional) dense storage with zero value
///
/// Creates a scalar storage containing a single zero element.
/// Useful for scalar operations and reductions.
///
/// # Returns
/// New scalar dense storage containing zero
///
/// # Examples
/// ```
/// use dense::creation::scalar_zero;
/// use dtype::float::Float32;
///
/// let scalar = scalar_zero::<Float32>().unwrap();
/// assert_eq!(scalar.shape().dims(), &[1]);
/// assert_eq!(scalar.len(), 1);
/// ```
pub fn scalar_zero<T: DataType>() -> Result<DenseStorage<T>>
where
    T: num_traits::Zero,
{
    zeros(&[1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::{Float32, Float64};
    use dtype::int::Int32;

    #[test]
    fn test_zeros_1d() {
        let storage = zeros::<Float32>(&[5]).unwrap();
        assert_eq!(storage.shape().dims(), &[5]);
        assert_eq!(storage.len(), 5);
        
        // Check all elements are zero
        for &val in storage.as_slice() {
            assert!(val.is_zero());
        }
    }

    #[test]
    fn test_zeros_2d() {
        let storage = zeros::<Float64>(&[3, 4]).unwrap();
        assert_eq!(storage.shape().dims(), &[3, 4]);
        assert_eq!(storage.len(), 12);
        
        // Check all elements are zero
        for &val in storage.as_slice() {
            assert!(val.is_zero());
        }
    }

    #[test]
    fn test_zeros_like() {
        let original = DenseStorage::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)],
            &[2, 2],
        ).unwrap();

        let zeros_storage = zeros_like(&original).unwrap();
        assert_eq!(zeros_storage.shape().dims(), &[2, 2]);
        assert_eq!(zeros_storage.len(), 4);
        
        // Check all elements are zero
        for &val in zeros_storage.as_slice() {
            assert!(val.is_zero());
        }
    }

    #[test]
    fn test_scalar_zero() {
        let scalar = scalar_zero::<Float32>().unwrap();
        assert_eq!(scalar.shape().dims(), &[1]);
        assert_eq!(scalar.len(), 1);
        assert!(scalar.as_slice()[0].is_zero());
    }
}