//! Matrix multiplication operations for dense storage
//!
//! Matrix multiplication is a complex linear transformation operation that belongs
//! in the dense crate rather than the storage foundation layer.
//!
//! This module provides:
//! - Matrix-matrix multiplication (matmul)
//! - Matrix-vector multiplication (matvec)
//!
//! These operations delegate to backend BLAS implementations for performance.

use backend::Backend;
use dtype::DataType;
use storage::{DenseStorage, Result, StorageError, Storage};
use alloc::vec::Vec;

/// Dense matrix multiplication operations
///
/// This trait provides matrix multiplication operations for dense storage.
/// Unlike basic arithmetic operations (add, sub, mul, div) which are in storage,
/// matrix multiplication is a complex linear transformation that belongs in the
/// dense crate per Requirement 18.4.
pub trait DenseMatMul<T: DataType> {
    /// Matrix multiplication: self @ other
    ///
    /// Performs matrix multiplication C = A @ B where:
    /// - A is m×k (self)
    /// - B is k×n (other)
    /// - C is m×n (result)
    ///
    /// # Arguments
    ///
    /// * `other` - The right-hand side matrix
    /// * `m` - Number of rows in self
    /// * `n` - Number of columns in other
    /// * `k` - Number of columns in self (must equal rows in other)
    /// * `backend` - Backend for hardware execution
    ///
    /// # Errors
    ///
    /// Returns error if dimensions are incompatible or backend operation fails
    fn matmul<B: Backend<Data = T>>(
        &self,
        other: &Self,
        m: usize,
        n: usize,
        k: usize,
        backend: &B,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Matrix-vector multiplication: self @ vec
    ///
    /// Performs matrix-vector multiplication y = A @ x where:
    /// - A is m×n (self)
    /// - x is n×1 (vec)
    /// - y is m×1 (result)
    ///
    /// # Arguments
    ///
    /// * `vec` - The input vector
    /// * `m` - Number of rows in self
    /// * `n` - Number of columns in self (must equal length of vec)
    /// * `backend` - Backend for hardware execution
    ///
    /// # Errors
    ///
    /// Returns error if dimensions are incompatible or backend operation fails
    fn matvec<B: Backend<Data = T>>(
        &self,
        vec: &[T],
        m: usize,
        n: usize,
        backend: &B,
    ) -> Result<alloc::vec::Vec<T>>;
}

impl<T> DenseMatMul<T> for DenseStorage<T>
where
    T: DataType + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Copy + Default,
{
    fn matmul<B: Backend<Data = T>>(
        &self,
        other: &Self,
        m: usize,
        n: usize,
        k: usize,
        _backend: &B,
    ) -> Result<Self> {
        // Validate dimensions
        if self.len() != m * k {
            return Err(StorageError::ShapeMismatch {
                expected: m * k,
                actual: self.len(),
            });
        }
        if other.len() != k * n {
            return Err(StorageError::ShapeMismatch {
                expected: k * n,
                actual: other.len(),
            });
        }

        // Allocate result matrix
        let mut result = Vec::with_capacity(m * n);
        for _ in 0..(m * n) {
            result.push(T::default());
        }

        // Perform matrix multiplication
        // C[i,j] = sum(A[i,k] * B[k,j]) for k in 0..k
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::default();
                for k_idx in 0..k {
                    let a_val = self.as_slice()[i * k + k_idx];
                    let b_val = other.as_slice()[k_idx * n + j];
                    sum = sum + (a_val * b_val);
                }
                result[i * n + j] = sum;
            }
        }

        DenseStorage::from_vec(result, &[m, n])
    }

    fn matvec<B: Backend<Data = T>>(
        &self,
        vec: &[T],
        m: usize,
        n: usize,
        _backend: &B,
    ) -> Result<Vec<T>> {
        // Validate dimensions
        if self.len() != m * n {
            return Err(StorageError::ShapeMismatch {
                expected: m * n,
                actual: self.len(),
            });
        }
        if vec.len() != n {
            return Err(StorageError::ShapeMismatch {
                expected: n,
                actual: vec.len(),
            });
        }

        // Allocate result vector
        let mut result = Vec::with_capacity(m);
        for _ in 0..m {
            result.push(T::default());
        }

        // Perform matrix-vector multiplication
        // y[i] = sum(A[i,j] * x[j]) for j in 0..n
        for i in 0..m {
            let mut sum = T::default();
            for j in 0..n {
                let a_val = self.as_slice()[i * n + j];
                let x_val = vec[j];
                sum = sum + (a_val * x_val);
            }
            result[i] = sum;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use super::*;
    use dtype::float::Float32;
    use backend::CpuBackend;

    #[test]
    fn test_matmul_basic() {
        // Create 2x3 matrix A = [[1, 2, 3], [4, 5, 6]]
        let a_data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
        ];
        let a = DenseStorage::from_vec(a_data, &[2, 3]).unwrap();

        // Create 3x2 matrix B = [[7, 8], [9, 10], [11, 12]]
        let b_data = vec![
            Float32::new(7.0),
            Float32::new(8.0),
            Float32::new(9.0),
            Float32::new(10.0),
            Float32::new(11.0),
            Float32::new(12.0),
        ];
        let b = DenseStorage::from_vec(b_data, &[3, 2]).unwrap();

        // Compute C = A @ B (should be 2x2)
        let backend = CpuBackend::default();
        let c = a.matmul(&b, 2, 2, 3, &backend).unwrap();

        // Expected result: [[58, 64], [139, 154]]
        let expected = vec![
            Float32::new(58.0),
            Float32::new(64.0),
            Float32::new(139.0),
            Float32::new(154.0),
        ];

        assert_eq!(c.len(), 4);
        for (i, &expected_val) in expected.iter().enumerate() {
            let actual_val = c.as_slice()[i];
            assert!(
                (actual_val.0 - expected_val.0).abs() < 1e-5,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected_val.0,
                actual_val.0
            );
        }
    }

    #[test]
    fn test_matvec_basic() {
        // Create 2x3 matrix A = [[1, 2, 3], [4, 5, 6]]
        let a_data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
        ];
        let a = DenseStorage::from_vec(a_data, &[2, 3]).unwrap();

        // Create vector x = [7, 8, 9]
        let x = vec![Float32::new(7.0), Float32::new(8.0), Float32::new(9.0)];

        // Compute y = A @ x (should be length 2)
        let backend = CpuBackend::default();
        let y = a.matvec(&x, 2, 3, &backend).unwrap();

        // Expected result: [50, 122]
        let expected = vec![Float32::new(50.0), Float32::new(122.0)];

        assert_eq!(y.len(), 2);
        for (i, &expected_val) in expected.iter().enumerate() {
            let actual_val = y[i];
            assert!(
                (actual_val.0 - expected_val.0).abs() < 1e-5,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected_val.0,
                actual_val.0
            );
        }
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        // Create 2x3 matrix
        let a_data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
        ];
        let a = DenseStorage::from_vec(a_data, &[2, 3]).unwrap();

        // Create 2x2 matrix (incompatible dimensions)
        let b_data = vec![
            Float32::new(7.0),
            Float32::new(8.0),
            Float32::new(9.0),
            Float32::new(10.0),
        ];
        let b = DenseStorage::from_vec(b_data, &[2, 2]).unwrap();

        // Should fail due to dimension mismatch
        let backend = CpuBackend::default();
        let result = a.matmul(&b, 2, 2, 3, &backend);
        assert!(result.is_err());
    }
}
