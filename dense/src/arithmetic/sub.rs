//! Element-wise subtraction operations for dense storage

use storage::{DenseStorage, Result, StorageError, Storage};
use dtype::DataType;
use backend::Backend;
use alloc::{format, vec::Vec};

/// Element-wise subtraction of two dense storages
///
/// Performs element-wise subtraction: result[i] = lhs[i] - rhs[i]
/// Both storages must have the same shape.
/// 
/// This operation delegates to the backend for hardware-optimized execution.
///
/// # Arguments
/// * `lhs` - Left-hand side storage
/// * `rhs` - Right-hand side storage
///
/// # Returns
/// New dense storage containing the element-wise difference
///
/// # Errors
/// Returns error if shapes don't match or backend operation fails
///
/// # Examples
/// ```
/// use dense::arithmetic::sub;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let a = DenseStorage::from_vec(vec![Float32::new(5.0), Float32::new(3.0)], &[2]).unwrap();
/// let b = DenseStorage::from_vec(vec![Float32::new(2.0), Float32::new(1.0)], &[2]).unwrap();
/// let result = sub(&a, &b).unwrap();
/// // result contains [3.0, 2.0]
/// ```
pub fn sub<T: DataType, B: Backend<Data = T>>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
    backend: &B,
) -> Result<DenseStorage<T>>
where
    T: core::ops::Sub<Output = T> + Clone,
{
    // Validate shapes match
    if lhs.shape().dims() != rhs.shape().dims() {
        return Err(StorageError::ShapeMismatch {
            expected: lhs.len(),
            actual: rhs.len(),
        });
    }

    // Delegate to backend for hardware-optimized execution
    backend.sub_dense(lhs, rhs).map_err(|backend_err| {
        StorageError::BackendError(format!("Backend subtraction failed: {:?}", backend_err))
    })
}

/// Element-wise subtraction with scalar
///
/// Performs element-wise subtraction with scalar: result[i] = storage[i] - scalar
/// 
/// This operation uses direct computation as backends typically don't have 
/// specialized scalar subtraction primitives.
///
/// # Arguments
/// * `storage` - Input storage
/// * `scalar` - Scalar value to subtract
///
/// # Returns
/// New dense storage with scalar subtracted from each element
///
/// # Examples
/// ```
/// use dense::arithmetic::sub_scalar;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let a = DenseStorage::from_vec(vec![Float32::new(5.0), Float32::new(3.0)], &[2]).unwrap();
/// let result = sub_scalar(&a, Float32::new(1.0)).unwrap();
/// // result contains [4.0, 2.0]
/// ```
pub fn sub_scalar<T: DataType>(
    storage: &DenseStorage<T>,
    scalar: T,
) -> Result<DenseStorage<T>>
where
    T: core::ops::Sub<Output = T> + Clone,
{
    let data = storage.as_slice();
    let result_data: Vec<T> = data
        .iter()
        .map(|x| x.clone() - scalar.clone())
        .collect();

    DenseStorage::from_vec(result_data, storage.shape().dims())
}

/// Scalar subtraction from storage elements
///
/// Performs scalar subtraction: result[i] = scalar - storage[i]
/// 
/// This operation uses direct computation as backends typically don't have 
/// specialized scalar subtraction primitives.
///
/// # Arguments
/// * `scalar` - Scalar value
/// * `storage` - Input storage
///
/// # Returns
/// New dense storage with each element subtracted from scalar
///
/// # Examples
/// ```
/// use dense::arithmetic::scalar_sub;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let a = DenseStorage::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();
/// let result = scalar_sub(Float32::new(5.0), &a).unwrap();
/// // result contains [4.0, 3.0]
/// ```
pub fn scalar_sub<T: DataType>(
    scalar: T,
    storage: &DenseStorage<T>,
) -> Result<DenseStorage<T>>
where
    T: core::ops::Sub<Output = T> + Clone,
{
    let data = storage.as_slice();
    let result_data: Vec<T> = data
        .iter()
        .map(|x| scalar.clone() - x.clone())
        .collect();

    DenseStorage::from_vec(result_data, storage.shape().dims())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_sub_same_shape() {
        let a = DenseStorage::from_vec(
            vec![Float32::new(5.0), Float32::new(3.0), Float32::new(7.0)],
            &[3],
        ).unwrap();
        let b = DenseStorage::from_vec(
            vec![Float32::new(2.0), Float32::new(1.0), Float32::new(3.0)],
            &[3],
        ).unwrap();

        let result = sub(&a, &b).unwrap();
        let expected = vec![Float32::new(3.0), Float32::new(2.0), Float32::new(4.0)];
        
        assert_eq!(result.as_slice(), expected.as_slice());
        assert_eq!(result.shape().dims(), &[3]);
    }

    #[test]
    fn test_sub_different_shapes() {
        let a = DenseStorage::from_vec(vec![Float32::new(1.0)], &[1]).unwrap();
        let b = DenseStorage::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();

        let result = sub(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_sub_scalar() {
        let a = DenseStorage::from_vec(
            vec![Float32::new(5.0), Float32::new(3.0)],
            &[2],
        ).unwrap();

        let result = sub_scalar(&a, Float32::new(1.0)).unwrap();
        let expected = vec![Float32::new(4.0), Float32::new(2.0)];
        
        assert_eq!(result.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_scalar_sub() {
        let a = DenseStorage::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        ).unwrap();

        let result = scalar_sub(Float32::new(5.0), &a).unwrap();
        let expected = vec![Float32::new(4.0), Float32::new(3.0)];
        
        assert_eq!(result.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_sub_2d() {
        let a = DenseStorage::from_vec(
            vec![Float32::new(5.0), Float32::new(6.0), Float32::new(7.0), Float32::new(8.0)],
            &[2, 2],
        ).unwrap();
        let b = DenseStorage::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)],
            &[2, 2],
        ).unwrap();

        let result = sub(&a, &b).unwrap();
        let expected = vec![Float32::new(4.0), Float32::new(4.0), Float32::new(4.0), Float32::new(4.0)];
        
        assert_eq!(result.as_slice(), expected.as_slice());
        assert_eq!(result.shape().dims(), &[2, 2]);
    }
}