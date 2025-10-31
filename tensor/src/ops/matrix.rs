//! Tensor matrix operations.
//!
//! This module provides matrix operations such as matrix multiplication (matmul).

use std::vec::Vec;
use tracing::instrument;

/// Matrix operations for tensors with dense storage.
///
/// This trait provides methods for matrix algebra operations
/// on 2D tensors.
impl<B, T> crate::Tensor<B, storage::DenseStorage<T>, T>
where
    B: crate::Backend<Data = T> + Clone + Default,
    T: crate::DataType + Clone + Copy + num_traits::Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    /// Compute matrix multiplication with another tensor.
    ///
    /// # Mathematical Definition
    ///
    /// For matrices A (m×n) and B (n×p):
    /// ```text
    /// C = A @ B
    /// C[i,j] = Σₖ A[i,k] * B[k,j]
    /// ```
    ///
    /// # Arguments
    /// * `other` - The right-hand side matrix tensor
    ///
    /// # Returns
    /// A new tensor containing the matrix product
    ///
    /// # Errors
    /// Returns `TensorError::ShapeError` if:
    /// - Either tensor is not 2D
    /// - Matrix dimensions are incompatible (A.cols ≠ B.rows)
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::float::Float32;
    ///
    /// // Create 2x3 matrix A
    /// let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
    ///     vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0),
    ///          Float32::new(4.0), Float32::new(5.0), Float32::new(6.0)],
    ///     &[2, 3]
    /// ).unwrap();
    ///
    /// // Create 3x2 matrix B
    /// let b = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
    ///     vec![Float32::new(7.0), Float32::new(8.0),
    ///          Float32::new(9.0), Float32::new(10.0),
    ///          Float32::new(11.0), Float32::new(12.0)],
    ///     &[3, 2]
    /// ).unwrap();
    ///
    /// // Compute A @ B (2x3 @ 3x2 = 2x2)
    /// let c = a.matmul(&b).unwrap();
    /// assert_eq!(c.shape().dims(), &[2, 2]);
    /// ```
    #[instrument(level = "trace", skip(self, other))]
    pub fn matmul(&self, other: &Self) -> crate::Result<Self> {
        let lhs_shape = <storage::DenseStorage<T> as storage::Storage<T>>::shape(&self.storage).dims();
        let rhs_shape = <storage::DenseStorage<T> as storage::Storage<T>>::shape(&other.storage).dims();

        // Validate 2D matrices
        if lhs_shape.len() != 2 {
            return Err(crate::TensorError::ShapeError {
                expected: 2,
                actual: lhs_shape.len(),
                message: format!("Left matrix must be 2D, got shape {lhs_shape:?}"),
            });
        }
        if rhs_shape.len() != 2 {
            return Err(crate::TensorError::ShapeError {
                expected: 2,
                actual: rhs_shape.len(),
                message: format!("Right matrix must be 2D, got shape {rhs_shape:?}"),
            });
        }

        // Validate compatible dimensions
        let m = lhs_shape[0];
        let n = lhs_shape[1];
        let p = rhs_shape[1];

        if n != rhs_shape[0] {
            return Err(crate::TensorError::ShapeError {
                expected: n,
                actual: rhs_shape[0],
                message: format!(
                    "Matrix dimension mismatch: {}×{} @ {}×{} (inner dimensions {} ≠ {})",
                    m, n, rhs_shape[0], p, n, rhs_shape[0]
                ),
            });
        }

        // Use backend's matmul implementation (CPU by default, GPU if available)
        self.matmul_backend(other, m, n, p)
    }

    /// CPU implementation of matrix multiplication
    #[instrument(level = "trace", skip(self, other))]
    fn matmul_cpu(&self, m: usize, n: usize, p: usize, other: &Self) -> crate::Result<Self> {
        // Perform matrix multiplication using iterator-based approach
        let lhs_data = <storage::DenseStorage<T> as storage::Storage<T>>::as_slice(&self.storage);
        let rhs_data = <storage::DenseStorage<T> as storage::Storage<T>>::as_slice(&other.storage);

        let result_data: Vec<T> = (0..m)
            .flat_map(|i| {
                (0..p).map(move |j| {
                    (0..n)
                        .map(|k| {
                            let lhs_idx = i * n + k;
                            let rhs_idx = k * p + j;
                            lhs_data[lhs_idx] * rhs_data[rhs_idx]
                        })
                        .fold(<T as num_traits::Zero>::zero(), |acc, x| acc + x)
                })
            })
            .collect();

        {
            let storage = storage::DenseStorage::from_vec(result_data, &[m, p])?;
            Ok(<crate::Tensor<B, storage::DenseStorage<T>, T>>::from_storage(storage, self.backend.clone()))
        }
    }

    /// Backend-agnostic matrix multiplication implementation
    #[instrument(level = "trace", skip(self, other))]
    fn matmul_backend(&self, other: &Self, m: usize, n: usize, p: usize) -> crate::Result<Self> {
        // For now, delegate to CPU implementation
        // Future enhancement: Add GPU support via backend trait extension
        self.matmul_cpu(m, n, p, other)
    }
}
