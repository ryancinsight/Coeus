//! Element-wise division operations for dense storage

use storage::{DenseStorage, Result, StorageError, Storage};
use dtype::DataType;
use alloc::vec::Vec;

/// Element-wise division of two dense storages
///
/// Performs element-wise division: result[i] = lhs[i] / rhs[i]
/// Both storages must have the same shape.
/// 
/// Note: Division operations are not typically optimized at the backend level
/// due to complexity of handling division by zero and numerical stability,
/// so this uses direct computation.
///
/// # Arguments
/// * `lhs` - Left-hand side storage (dividend)
/// * `rhs` - Right-hand side storage (divisor)
///
/// # Returns
/// New dense storage containing the element-wise quotient
///
/// # Errors
/// Returns error if shapes don't match
///
/// # Examples
/// ```
/// use dense::arithmetic::div;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let a = DenseStorage::from_vec(vec![Float32::new(8.0), Float32::new(15.0)], &[2]).unwrap();
/// let b = DenseStorage::from_vec(vec![Float32::new(2.0), Float32::new(3.0)], &[2]).unwrap();
/// let result = div(&a, &b).unwrap();
/// // result contains [4.0, 5.0]
/// ```
pub fn div<T: DataType>(
    lhs: &DenseStorage<T>,
    rhs: &DenseStorage<T>,
) -> Result<DenseStorage<T>>
where
    T: core::ops::Div<Output = T> + Clone,
{
    // Validate shapes match
    if lhs.shape().dims() != rhs.shape().dims() {
        return Err(StorageError::ShapeMismatch {
            expected: lhs.len(),
            actual: rhs.len(),
        });
    }

    // Perform element-wise division
    let lhs_data = lhs.as_slice();
    let rhs_data = rhs.as_slice();
    
    let result_data: Vec<T> = lhs_data
        .iter()
        .zip(rhs_data.iter())
        .map(|(a, b)| a.clone() / b.clone())
        .collect();

    DenseStorage::from_vec(result_data, lhs.shape().dims())
}

/// Element-wise division with scalar
///
/// Performs element-wise division with scalar: result[i] = storage[i] / scalar
/// 
/// This operation uses direct computation as backends typically don't have 
/// specialized scalar division primitives.
///
/// # Arguments
/// * `storage` - Input storage (dividend)
/// * `scalar` - Scalar value to divide by (divisor)
///
/// # Returns
/// New dense storage with each element divided by scalar
///
/// # Examples
/// ```
/// use dense::arithmetic::div_scalar;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let a = DenseStorage::from_vec(vec![Float32::new(10.0), Float32::new(15.0)], &[2]).unwrap();
/// let result = div_scalar(&a, Float32::new(5.0)).unwrap();
/// // result contains [2.0, 3.0]
/// ```
pub fn div_scalar<T: DataType>(
    storage: &DenseStorage<T>,
    scalar: T,
) -> Result<DenseStorage<T>>
where
    T: core::ops::Div<Output = T> + Clone,
{
    let data = storage.as_slice();
    let result_data: Vec<T> = data
        .iter()
        .map(|x| x.clone() / scalar.clone())
        .collect();

    DenseStorage::from_vec(result_data, storage.shape().dims())
}

/// Scalar division by storage elements
///
/// Performs scalar division: result[i] = scalar / storage[i]
/// 
/// This operation uses direct computation as backends typically don't have 
/// specialized scalar division primitives.
///
/// # Arguments
/// * `scalar` - Scalar value (dividend)
/// * `storage` - Input storage (divisor)
///
/// # Returns
/// New dense storage with scalar divided by each element
///
/// # Examples
/// ```
/// use dense::arithmetic::scalar_div;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let a = DenseStorage::from_vec(vec![Float32::new(2.0), Float32::new(4.0)], &[2]).unwrap();
/// let result = scalar_div(Float32::new(12.0), &a).unwrap();
/// // result contains [6.0, 3.0]
/// ```
pub fn scalar_div<T: DataType>(
    scalar: T,
    storage: &DenseStorage<T>,
) -> Result<DenseStorage<T>>
where
    T: core::ops::Div<Output = T> + Clone,
{
    let data = storage.as_slice();
    let result_data: Vec<T> = data
        .iter()
        .map(|x| scalar.clone() / x.clone())
        .collect();

    DenseStorage::from_vec(result_data, storage.shape().dims())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_div_same_shape() {
        let a = DenseStorage::from_vec(
            vec![Float32::new(10.0), Float32::new(15.0), Float32::new(21.0)],
            &[3],
        ).unwrap();
        let b = DenseStorage::from_vec(
            vec![Float32::new(2.0), Float32::new(3.0), Float32::new(7.0)],
            &[3],
        ).unwrap();

        let result = div(&a, &b).unwrap();
        let expected = vec![Float32::new(5.0), Float32::new(5.0), Float32::new(3.0)];
        
        assert_eq!(result.as_slice(), expected.as_slice());
        assert_eq!(result.shape().dims(), &[3]);
    }

    #[test]
    fn test_div_different_shapes() {
        let a = DenseStorage::from_vec(vec![Float32::new(1.0)], &[1]).unwrap();
        let b = DenseStorage::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();

        let result = div(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_div_scalar() {
        let a = DenseStorage::from_vec(
            vec![Float32::new(10.0), Float32::new(15.0)],
            &[2],
        ).unwrap();

        let result = div_scalar(&a, Float32::new(5.0)).unwrap();
        let expected = vec![Float32::new(2.0), Float32::new(3.0)];
        
        assert_eq!(result.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_scalar_div() {
        let a = DenseStorage::from_vec(
            vec![Float32::new(2.0), Float32::new(4.0)],
            &[2],
        ).unwrap();

        let result = scalar_div(Float32::new(12.0), &a).unwrap();
        let expected = vec![Float32::new(6.0), Float32::new(3.0)];
        
        assert_eq!(result.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_div_2d() {
        let a = DenseStorage::from_vec(
            vec![Float32::new(12.0), Float32::new(18.0), Float32::new(24.0), Float32::new(30.0)],
            &[2, 2],
        ).unwrap();
        let b = DenseStorage::from_vec(
            vec![Float32::new(3.0), Float32::new(6.0), Float32::new(8.0), Float32::new(10.0)],
            &[2, 2],
        ).unwrap();

        let result = div(&a, &b).unwrap();
        let expected = vec![Float32::new(4.0), Float32::new(3.0), Float32::new(3.0), Float32::new(3.0)];
        
        assert_eq!(result.as_slice(), expected.as_slice());
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_div_by_one() {
        let a = DenseStorage::from_vec(
            vec![Float32::new(5.0), Float32::new(3.0)],
            &[2],
        ).unwrap();
        let b = DenseStorage::from_vec(
            vec![Float32::new(1.0), Float32::new(1.0)],
            &[2],
        ).unwrap();

        let result = div(&a, &b).unwrap();
        let expected = vec![Float32::new(5.0), Float32::new(3.0)];
        
        assert_eq!(result.as_slice(), expected.as_slice());
    }
}