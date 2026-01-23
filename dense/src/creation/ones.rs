//! Dense storage creation with ones

use storage::{DenseStorage, Result, Storage};
use dtype::DataType;
use num_traits::One;
use alloc::vec;

/// Create dense storage filled with ones
///
/// # Arguments
/// * `shape` - Shape dimensions
///
/// # Returns
/// Dense storage filled with ones
///
/// # Examples
/// ```
/// use dense::creation::ones;
/// use dtype::float::Float32;
///
/// let storage = ones::<Float32>(&[2, 3]).unwrap();
/// // Creates 2x3 storage filled with 1.0
/// ```
pub fn ones<T: DataType>(shape: &[usize]) -> Result<DenseStorage<T>>
where
    T: One + Clone,
{
    let size = shape.iter().product();
    let data = vec![T::one(); size];
    DenseStorage::from_vec(data, shape)
}

/// Create dense storage filled with ones, matching another storage's shape
///
/// # Arguments
/// * `like` - Storage to match shape from
///
/// # Returns
/// Dense storage filled with ones with same shape as input
///
/// # Examples
/// ```
/// use dense::creation::{zeros, ones_like};
/// use dtype::float::Float32;
///
/// let original = zeros::<Float32>(&[2, 3]).unwrap();
/// let ones_storage = ones_like(&original).unwrap();
/// // Creates 2x3 storage filled with 1.0
/// ```
pub fn ones_like<T: DataType>(like: &DenseStorage<T>) -> Result<DenseStorage<T>>
where
    T: One + Clone,
{
    ones(like.shape().dims())
}

/// Create scalar storage with value one
///
/// # Returns
/// Scalar dense storage containing one
///
/// # Examples
/// ```
/// use dense::creation::scalar_one;
/// use dtype::float::Float32;
///
/// let scalar = scalar_one::<Float32>().unwrap();
/// // Creates scalar storage with value 1.0
/// ```
pub fn scalar_one<T: DataType>() -> Result<DenseStorage<T>>
where
    T: One + Clone,
{
    ones(&[])
}

/// Create identity matrix (square)
///
/// # Arguments
/// * `n` - Size of square identity matrix
///
/// # Returns
/// Square identity matrix of size n×n
///
/// # Examples
/// ```
/// use dense::creation::eye;
/// use dtype::float::Float32;
///
/// let identity = eye::<Float32>(3).unwrap();
/// // Creates 3x3 identity matrix
/// ```
pub fn eye<T: DataType>(n: usize) -> Result<DenseStorage<T>>
where
    T: One + num_traits::Zero + Clone,
{
    eye_rectangular(n, n)
}

/// Create rectangular identity matrix
///
/// # Arguments
/// * `rows` - Number of rows
/// * `cols` - Number of columns
///
/// # Returns
/// Rectangular identity matrix of size rows×cols
///
/// # Examples
/// ```
/// use dense::creation::eye_rectangular;
/// use dtype::float::Float32;
///
/// let identity = eye_rectangular::<Float32>(3, 4).unwrap();
/// // Creates 3x4 identity matrix
/// ```
pub fn eye_rectangular<T: DataType>(rows: usize, cols: usize) -> Result<DenseStorage<T>>
where
    T: One + num_traits::Zero + Clone,
{
    let mut data = vec![T::zero(); rows * cols];
    let min_dim = rows.min(cols);
    
    for i in 0..min_dim {
        data[i * cols + i] = T::one();
    }
    
    DenseStorage::from_vec(data, &[rows, cols])
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_ones_1d() {
        let storage = ones::<Float32>(&[3]).unwrap();
        let expected = vec![Float32::new(1.0); 3];
        assert_eq!(storage.as_slice(), expected.as_slice());
        assert_eq!(storage.shape().dims(), &[3]);
    }

    #[test]
    fn test_ones_2d() {
        let storage = ones::<Float32>(&[2, 3]).unwrap();
        let expected = vec![Float32::new(1.0); 6];
        assert_eq!(storage.as_slice(), expected.as_slice());
        assert_eq!(storage.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_ones_like() {
        use crate::creation::zeros;
        let original = zeros::<Float32>(&[2, 2]).unwrap();
        let ones_storage = ones_like(&original).unwrap();
        let expected = vec![Float32::new(1.0); 4];
        assert_eq!(ones_storage.as_slice(), expected.as_slice());
        assert_eq!(ones_storage.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_scalar_one() {
        let scalar = scalar_one::<Float32>().unwrap();
        assert_eq!(scalar.as_slice(), &[Float32::new(1.0)]);
        assert_eq!(scalar.shape().dims(), &[]);
    }

    #[test]
    fn test_eye_square() {
        let identity = eye::<Float32>(3).unwrap();
        let expected = vec![
            Float32::new(1.0), Float32::new(0.0), Float32::new(0.0),
            Float32::new(0.0), Float32::new(1.0), Float32::new(0.0),
            Float32::new(0.0), Float32::new(0.0), Float32::new(1.0),
        ];
        assert_eq!(identity.as_slice(), expected.as_slice());
        assert_eq!(identity.shape().dims(), &[3, 3]);
    }

    #[test]
    fn test_eye_rectangular() {
        let identity = eye_rectangular::<Float32>(2, 3).unwrap();
        let expected = vec![
            Float32::new(1.0), Float32::new(0.0), Float32::new(0.0),
            Float32::new(0.0), Float32::new(1.0), Float32::new(0.0),
        ];
        assert_eq!(identity.as_slice(), expected.as_slice());
        assert_eq!(identity.shape().dims(), &[2, 3]);
    }
}