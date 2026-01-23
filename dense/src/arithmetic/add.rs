//! Element-wise addition operations for dense storage

use storage::{DenseStorage, Result, StorageError, Storage};
use dtype::DataType;
use backend::Backend;
use alloc::{format, vec::Vec};

/// Element-wise addition of two dense storages
///
/// Performs element-wise addition: result[i] = lhs[i] + rhs[i]
/// Both storages must have the same shape.
/// 
/// This operation delegates to the backend for hardware-optimized execution.
///
/// # Arguments
/// * `lhs` - Left-hand side storage
/// * `rhs` - Right-hand side storage
///
/// # Returns
/// New dense storage containing the element-wise sum
///
/// # Errors
/// Returns error if shapes don't match or backend operation fails
///
/// # Examples
/// ```
/// use dense::arithmetic::add;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let a = DenseStorage::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();
/// let b = DenseStorage::from_vec(vec![Float32::new(3.0), Float32::new(4.0)], &[2]).unwrap();
/// let result = add(&a, &b).unwrap();
/// // result contains [4.0, 6.0]
/// ```
pub fn add<T: DataType, B: Backend<Data = T>>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
    backend: &B,
) -> Result<DenseStorage<T>>
where
    T: core::ops::Add<Output = T> + Clone,
{
    // Validate shapes match
    if lhs.shape().dims() != rhs.shape().dims() {
        return Err(StorageError::ShapeMismatch {
            expected: lhs.len(),
            actual: rhs.len(),
        });
    }

    // Delegate to backend for hardware-optimized execution
    backend.add_dense(lhs, rhs).map_err(|backend_err| {
        StorageError::BackendError(format!("Backend addition failed: {:?}", backend_err))
    })
}

/// Element-wise addition with scalar
///
/// Performs element-wise addition with scalar: result[i] = storage[i] + scalar
/// 
/// This operation uses direct computation as backends typically don't have 
/// specialized scalar addition primitives.
///
/// # Arguments
/// * `storage` - Input storage
/// * `scalar` - Scalar value to add
///
/// # Returns
/// New dense storage with scalar added to each element
///
/// # Examples
/// ```
/// use dense::arithmetic::add_scalar;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let a = DenseStorage::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();
/// let result = add_scalar(&a, Float32::new(5.0)).unwrap();
/// // result contains [6.0, 7.0]
/// ```
pub fn add_scalar<T: DataType>(
    storage: &DenseStorage<T>,
    scalar: T,
) -> Result<DenseStorage<T>>
where
    T: core::ops::Add<Output = T> + Clone,
{
    let data = storage.as_slice();
    let result_data: Vec<T> = data
        .iter()
        .map(|x| x.clone() + scalar.clone())
        .collect();

    DenseStorage::from_vec(result_data, storage.shape().dims())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_add_same_shape() {
        let a = DenseStorage::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        ).unwrap();
        let b = DenseStorage::from_vec(
            vec![Float32::new(4.0), Float32::new(5.0), Float32::new(6.0)],
            &[3],
        ).unwrap();

        let result = add(&a, &b).unwrap();
        let expected = vec![Float32::new(5.0), Float32::new(7.0), Float32::new(9.0)];
        
        assert_eq!(result.as_slice(), expected.as_slice());
        assert_eq!(result.shape().dims(), &[3]);
    }

    #[test]
    fn test_add_different_shapes() {
        let a = DenseStorage::from_vec(vec![Float32::new(1.0)], &[1]).unwrap();
        let b = DenseStorage::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();

        let result = add(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_scalar() {
        let a = DenseStorage::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        ).unwrap();

        let result = add_scalar(&a, Float32::new(10.0)).unwrap();
        let expected = vec![Float32::new(11.0), Float32::new(12.0)];
        
        assert_eq!(result.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_add_2d() {
        let a = DenseStorage::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)],
            &[2, 2],
        ).unwrap();
        let b = DenseStorage::from_vec(
            vec![Float32::new(5.0), Float32::new(6.0), Float32::new(7.0), Float32::new(8.0)],
            &[2, 2],
        ).unwrap();

        let result = add(&a, &b).unwrap();
        let expected = vec![Float32::new(6.0), Float32::new(8.0), Float32::new(10.0), Float32::new(12.0)];
        
        assert_eq!(result.as_slice(), expected.as_slice());
        assert_eq!(result.shape().dims(), &[2, 2]);
    }
}