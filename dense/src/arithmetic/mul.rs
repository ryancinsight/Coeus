//! Element-wise multiplication operations for dense storage

use storage::{DenseStorage, Result, StorageError, Storage};
use dtype::DataType;
use backend::Backend;
use alloc::{format, vec::Vec};

/// Element-wise multiplication of two dense storages
///
/// Performs element-wise multiplication: result[i] = lhs[i] * rhs[i]
/// Both storages must have the same shape.
///
/// This operation delegates to the backend for hardware-optimized execution.
///
/// # Arguments
/// * `lhs` - Left-hand side storage
/// * `rhs` - Right-hand side storage
///
/// # Returns
/// New dense storage containing the element-wise product
///
/// # Errors
/// Returns error if shapes don't match or backend operation fails
///
/// # Examples
/// ```
/// use dense::arithmetic::mul;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let a = DenseStorage::from_vec(vec![Float32::new(2.0), Float32::new(3.0)], &[2]).unwrap();
/// let b = DenseStorage::from_vec(vec![Float32::new(4.0), Float32::new(5.0)], &[2]).unwrap();
/// let result = mul(&a, &b).unwrap();
/// // result contains [8.0, 15.0]
/// ```
pub fn mul<T: DataType, B: Backend<Data = T>>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
    backend: &B,
) -> Result<DenseStorage<T>>
where
    T: core::ops::Mul<Output = T> + Clone,
{
    // Validate shapes match
    if lhs.shape().dims() != rhs.shape().dims() {
        return Err(StorageError::ShapeMismatch {
            expected: lhs.len(),
            actual: rhs.len(),
        });
    }

    // Delegate to backend for hardware-optimized execution
    backend.mul_dense(lhs, rhs).map_err(|backend_err| {
        StorageError::BackendError(format!("Backend multiplication failed: {:?}", backend_err))
    })
}

/// Element-wise multiplication with scalar
///
/// Performs element-wise multiplication with scalar: result[i] = storage[i] * scalar
///
/// This operation uses direct computation as backends typically don't have
/// specialized scalar multiplication primitives.
///
/// # Arguments
/// * `storage` - Input storage
/// * `scalar` - Scalar value to multiply
///
/// # Returns
/// New dense storage with scalar multiplied to each element
///
/// # Examples
/// ```
/// use dense::arithmetic::mul_scalar;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let a = DenseStorage::from_vec(vec![Float32::new(2.0), Float32::new(3.0)], &[2]).unwrap();
/// let result = mul_scalar(&a, Float32::new(5.0)).unwrap();
/// // result contains [10.0, 15.0]
/// ```
pub fn mul_scalar<T: DataType>(
    storage: &DenseStorage<T>,
    scalar: T,
) -> Result<DenseStorage<T>>
where
    T: core::ops::Mul<Output = T> + Clone,
{
    let data = storage.as_slice();
    let result_data: Vec<T> = data
        .iter()
        .map(|x| x.clone() * scalar.clone())
        .collect();

    DenseStorage::from_vec(result_data, storage.shape().dims())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;
    use alloc::vec;
    use backend::cpu::CpuBackend;

    #[test]
    fn test_mul_same_shape() {
        let backend = CpuBackend::<Float32>::default();
        let a = DenseStorage::from_vec(
            vec![Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)],
            &[3],
        ).unwrap();
        let b = DenseStorage::from_vec(
            vec![Float32::new(5.0), Float32::new(6.0), Float32::new(7.0)],
            &[3],
        ).unwrap();

        let result = mul(&a, &b, &backend).unwrap();
        let expected = vec![Float32::new(10.0), Float32::new(18.0), Float32::new(28.0)];

        assert_eq!(result.as_slice(), expected.as_slice());
        assert_eq!(result.shape().dims(), &[3]);
    }

    #[test]
    fn test_mul_different_shapes() {
        let backend = CpuBackend::<Float32>::default();
        let a = DenseStorage::from_vec(vec![Float32::new(1.0)], &[1]).unwrap();
        let b = DenseStorage::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();

        let result = mul(&a, &b, &backend);
        assert!(result.is_err());
    }

    #[test]
    fn test_mul_scalar() {
        let a = DenseStorage::from_vec(
            vec![Float32::new(2.0), Float32::new(3.0)],
            &[2],
        ).unwrap();

        let result = mul_scalar(&a, Float32::new(5.0)).unwrap();
        let expected = vec![Float32::new(10.0), Float32::new(15.0)];

        assert_eq!(result.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_mul_2d() {
        let backend = CpuBackend::<Float32>::default();
        let a = DenseStorage::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)],
            &[2, 2],
        ).unwrap();
        let b = DenseStorage::from_vec(
            vec![Float32::new(5.0), Float32::new(6.0), Float32::new(7.0), Float32::new(8.0)],
            &[2, 2],
        ).unwrap();

        let result = mul(&a, &b, &backend).unwrap();
        let expected = vec![Float32::new(5.0), Float32::new(12.0), Float32::new(21.0), Float32::new(32.0)];

        assert_eq!(result.as_slice(), expected.as_slice());
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_mul_zero() {
        let backend = CpuBackend::<Float32>::default();
        let a = DenseStorage::from_vec(
            vec![Float32::new(5.0), Float32::new(3.0)],
            &[2],
        ).unwrap();
        let b = DenseStorage::from_vec(
            vec![Float32::new(0.0), Float32::new(0.0)],
            &[2],
        ).unwrap();

        let result = mul(&a, &b, &backend).unwrap();
        let expected = vec![Float32::new(0.0), Float32::new(0.0)];

        assert_eq!(result.as_slice(), expected.as_slice());
    }
}
