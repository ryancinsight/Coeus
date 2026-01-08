//! Matrix determinant computation.
//!
//! Computes determinant using LU decomposition with partial pivoting.

use crate::error::{dimension_mismatch, not_square};
use backend::Backend;
use dtype::DataType;
use num_traits::{Float, One, Zero};
use storage::DenseStorage;
use tensor::Tensor;

/// Trait for computing matrix determinant.
pub trait Det<B, T> {
    /// Computes the determinant of a square matrix.
    ///
    /// # Returns
    /// The determinant as a scalar value.
    ///
    /// # Errors
    /// - `DimensionMismatch` if matrix is not 2D
    /// - `NotSquare` if matrix is not square
    fn det(&self) -> coeus_error::Result<T>
    where
        Self: Sized;
}

impl<B, T> Det<B, T> for Tensor<B, DenseStorage<T>, T>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType
        + Float
        + Zero
        + One
        + std::ops::SubAssign
        + std::ops::DivAssign
        + std::ops::MulAssign
        + std::fmt::Debug,
{
    fn det(&self) -> coeus_error::Result<T> {
        let dims = self.shape().dims();
        if dims.len() != 2 {
            return Err(dimension_mismatch("Matrix must be 2D for determinant"));
        }
        let n = dims[0];
        if n != dims[1] {
            return Err(not_square(n, dims[1]));
        }

        if n == 0 {
            return Ok(T::one()); // Empty matrix has det = 1
        }

        // Clone data for LU decomposition
        let mut data = self.as_slice().to_vec();
        let mut sign = T::one();
        let neg_one = T::zero() - T::one();

        // LU decomposition with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut pivot_idx = i;
            let mut max_val = data[i * n + i].abs();

            for k in (i + 1)..n {
                let val = data[k * n + i].abs();
                if val > max_val {
                    max_val = val;
                    pivot_idx = k;
                }
            }

            // Check for singularity
            if max_val == T::zero() {
                return Ok(T::zero()); // Singular matrix has det = 0
            }

            // Swap rows if needed
            if pivot_idx != i {
                for j in 0..n {
                    data.swap(i * n + j, pivot_idx * n + j);
                }
                sign *= neg_one; // Each swap flips sign
            }

            // Eliminate below pivot
            let pivot = data[i * n + i];
            for k in (i + 1)..n {
                let factor = data[k * n + i] / pivot;
                data[k * n + i] = T::zero();
                for j in (i + 1)..n {
                    let val = data[i * n + j] * factor;
                    data[k * n + j] -= val;
                }
            }
        }

        // Determinant is product of diagonal elements times sign
        let mut det = sign;
        for i in 0..n {
            det *= data[i * n + i];
        }

        Ok(det)
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
    fn test_det_identity() {
        let eye = create_matrix(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let d = eye.det().unwrap();
        assert!((d.0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_det_2x2() {
        // [[4, 7], [2, 6]] -> det = 24 - 14 = 10
        let m = create_matrix(vec![4.0, 7.0, 2.0, 6.0], &[2, 2]);
        let d = m.det().unwrap();
        assert!((d.0 - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_det_singular() {
        // [[1, 2], [2, 4]] -> det = 0
        let m = create_matrix(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]);
        let d = m.det().unwrap();
        assert!(d.0.abs() < 1e-6);
    }

    #[test]
    fn test_det_3x3() {
        // [[1, 2, 3], [4, 5, 6], [7, 8, 10]] -> det = -3
        let m = create_matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0], &[3, 3]);
        let d = m.det().unwrap();
        assert!((d.0 - (-3.0)).abs() < 1e-5);
    }
}
