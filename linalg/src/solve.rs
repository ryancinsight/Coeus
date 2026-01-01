//! Linear system solver.
//!
//! Solves Ax = b using Gaussian elimination with partial pivoting.

use crate::error::{dimension_mismatch, not_square, singular_matrix};
use backend::Backend;
use dtype::DataType;
use num_traits::{Float, One, Zero};
use storage::DenseStorage;
use tensor::Tensor;

/// Trait for solving linear systems.
pub trait Solve<B, T> {
    /// Solves the linear system Ax = b.
    ///
    /// # Arguments
    /// * `b` - The right-hand side tensor/matrix
    ///
    /// # Returns
    /// The solution tensor x such that Ax = b.
    ///
    /// # Errors
    /// - `DimensionMismatch` if dimensions are incompatible
    /// - `NotSquare` if A is not square
    /// - `SingularMatrix` if A is singular
    fn solve(&self, b: &Self) -> coeus_error::Result<Self>
    where
        Self: Sized;
}

impl<B, T> Solve<B, T> for Tensor<B, DenseStorage<T>, T>
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
    fn solve(&self, b: &Self) -> coeus_error::Result<Self> {
        let a_dims = self.shape().dims();
        let b_dims = b.shape().dims();

        if a_dims.len() != 2 {
            return Err(dimension_mismatch("Matrix A must be 2D"));
        }
        let n = a_dims[0];
        if n != a_dims[1] {
            return Err(not_square(n, a_dims[1]));
        }

        if b_dims.is_empty() {
            return Err(dimension_mismatch("B must be at least 1D"));
        }
        if b_dims[0] != n {
            return Err(dimension_mismatch(format!(
                "Dimension mismatch: A is {}x{}, B is {}...",
                n, n, b_dims[0]
            )));
        }

        // Determine columns in B
        let n_rhs = if b_dims.len() == 1 {
            1
        } else {
            b_dims.iter().skip(1).product()
        };

        // Flatten B logically to [N, N_RHS] for computation
        // Note: We need to handle strided B if we support it, but for DenseStorage,
        // it's contiguous. We just treat it as row-major.

        // Clone data for Gaussian elimination
        let mut a_data = self.as_slice().to_vec();
        let mut b_data = b.as_slice().to_vec();

        // Gaussian elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut pivot_idx = i;
            let mut max_val = a_data[i * n + i].abs();

            for k in (i + 1)..n {
                let val = a_data[k * n + i].abs();
                if val > max_val {
                    max_val = val;
                    pivot_idx = k;
                }
            }

            if max_val == T::zero() {
                return Err(singular_matrix());
            }

            // Swap rows in A and b
            if pivot_idx != i {
                // Swap row i and pivot_idx in A
                for j in 0..n {
                    a_data.swap(i * n + j, pivot_idx * n + j);
                }
                // Swap row i and pivot_idx in b
                for j in 0..n_rhs {
                    b_data.swap(i * n_rhs + j, pivot_idx * n_rhs + j);
                }
            }

            // Scale pivot row
            let pivot = a_data[i * n + i];
            // No need to scale row i of A for elimination, we can just use the factor.
            // But we do need to normalize the pivot row eventually or during backsub.
            // Let's normalize now to make diagonal 1.

            // Optimization: Usually we divide by pivot at the end or use it in the factor.
            // Let's divide row i by pivot now.
            for j in i..n {
                // Start from i, as columns < i are 0
                a_data[i * n + j] /= pivot;
            }
            for j in 0..n_rhs {
                b_data[i * n_rhs + j] /= pivot;
            }

            // Eliminate other rows
            for k in 0..n {
                if k != i {
                    let factor = a_data[k * n + i];
                    if factor != T::zero() {
                        // Row k = Row k - factor * Row i
                        // Optimize: Start from i because A[i, :i] are 0/processed
                        // But for A[k, :], the left part might be relevant if k < i (already processed)
                        // Actually in Gauss-Jordan we process all other rows.
                        // But we normalized row i so A[i,i] is 1.

                        // Update A
                        for j in i..n {
                            let val = a_data[i * n + j] * factor; // Read before write?
                                                                  // No, distinct indices.
                            a_data[k * n + j] -= val;
                        }

                        // Update b
                        for j in 0..n_rhs {
                            let val = b_data[i * n_rhs + j] * factor;
                            b_data[k * n_rhs + j] -= val;
                        }
                    }
                }
            }
        }

        // Result is in b_data
        let storage = DenseStorage::from_vec(b_data, b.shape().dims()).map_err(|e| {
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
    fn test_solve_simple() {
        // A = [[2, 1], [3, 2]]
        // b = [3, 5]
        // x = [1, 1]
        let a = create_matrix(vec![2.0, 1.0, 3.0, 2.0], &[2, 2]); // Det = 1
        let b = create_matrix(vec![3.0, 5.0], &[2]);

        let x = a.solve(&b).unwrap();

        assert_eq!(x.shape().dims(), &[2]);
        let x_data = x.as_slice();
        assert!((x_data[0].0 - 1.0).abs() < 1e-5);
        assert!((x_data[1].0 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_solve_identity() {
        let eye = create_matrix(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let b = create_matrix(vec![10.0, 20.0], &[2]);
        let x = eye.solve(&b).unwrap();
        let x_data = x.as_slice();
        assert!((x_data[0].0 - 10.0).abs() < 1e-6);
        assert!((x_data[1].0 - 20.0).abs() < 1e-6);
    }
}
