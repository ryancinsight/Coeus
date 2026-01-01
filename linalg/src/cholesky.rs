//! Cholesky Decomposition.
//!
//! Computes the Cholesky decomposition of a symmetric positive-definite matrix.

use crate::error::{dimension_mismatch, not_square, singular_matrix};
use backend::Backend;
use dtype::DataType;
use num_traits::{Float, One, Zero};
use storage::DenseStorage;
use tensor::Tensor;

/// Trait for Cholesky decomposition.
pub trait Cholesky<B, T> {
    /// Computes the Cholesky decomposition of a symmetric positive-definite matrix.
    ///
    /// # Returns
    /// Lower triangular matrix L such that A = LL^T.
    ///
    /// # Errors
    /// - `DimensionMismatch` if matrix is not 2D
    /// - `NotSquare` if matrix is not square
    /// - `SingularMatrix` if matrix is not positive definite
    fn cholesky(&self) -> coeus_error::Result<Self>
    where
        Self: Sized;
}

impl<B, T> Cholesky<B, T> for Tensor<B, DenseStorage<T>, T>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType
        + Float
        + Zero
        + One
        + std::ops::SubAssign
        + std::ops::DivAssign
        + std::ops::MulAssign
        + std::ops::AddAssign
        + std::fmt::Debug,
{
    fn cholesky(&self) -> coeus_error::Result<Self> {
        let dims = self.shape().dims();
        if dims.len() != 2 {
            return Err(dimension_mismatch("Matrix must be 2D"));
        }
        let n = dims[0];
        if n != dims[1] {
            return Err(not_square(n, dims[1]));
        }

        let a_data = self.as_slice();
        let mut l_data = vec![T::zero(); n * n];

        // Cholesky-Crout algorithm
        for i in 0..n {
            for j in 0..=i {
                let mut sum = T::zero();
                for k in 0..j {
                    sum += l_data[i * n + k] * l_data[j * n + k];
                }

                if i == j {
                    let val = a_data[i * n + i] - sum;
                    if val <= T::zero() {
                        return Err(singular_matrix()); // Not positive definite
                    }
                    l_data[i * n + j] = val.sqrt();
                } else {
                    let val = a_data[i * n + j] - sum;
                    l_data[i * n + j] = val / l_data[j * n + j];
                }
            }
        }

        let storage = DenseStorage::from_vec(l_data, &[n, n]).map_err(|e| {
            coeus_error::Error::Storage(coeus_error::StorageError::InvalidShape(format!("{e}")))
        })?;

        Ok(Tensor::from_storage(storage, self.backend().clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor::Float32;

    fn create_matrix(
        data: Vec<f32>,
        shape: &[usize],
    ) -> Tensor<backend::CpuBackend<Float32>, DenseStorage<Float32>, Float32> {
        let data = data.into_iter().map(Float32).collect();
        let storage = DenseStorage::from_vec(data, shape).unwrap();
        Tensor::from_storage(storage, backend::CpuBackend::default())
    }

    #[test]
    fn test_cholesky_simple() {
        // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
        // L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
        let data = vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
        let a = create_matrix(data, &[3, 3]);
        let l = a.cholesky().unwrap();

        let l_data = l.as_slice();
        // Check diagonal
        assert!((l_data[0].0 - 2.0).abs() < 1e-5);
        assert!((l_data[4].0 - 1.0).abs() < 1e-5); // 1*3+1 = 4
        assert!((l_data[8].0 - 3.0).abs() < 1e-5); // 2*3+2 = 8

        // Check lower triangle
        assert!((l_data[3].0 - 6.0).abs() < 1e-5);
        assert!((l_data[6].0 - -8.0).abs() < 1e-5);
    }

    #[test]
    fn test_cholesky_fail_not_spd() {
        // [[-1, 0], [0, 1]]
        let a = create_matrix(vec![-1.0, 0.0, 0.0, 1.0], &[2, 2]);
        assert!(a.cholesky().is_err());
    }
}
