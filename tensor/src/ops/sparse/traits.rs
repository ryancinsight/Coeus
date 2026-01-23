//! Sparse tensor operation traits
//!
//! This module defines traits for sparse tensor operations, providing a unified
//! interface for sparse matrix computations using the optimal CSR format.

use crate::{Result, TensorError};
use alloc::vec::Vec;
use backend::Backend;
use coeus_sparse::{SparseAdd, SparseTranspose};
use dtype::DataType;
use storage::{CsrStorage, DenseStorage};


/// Trait for sparse storage operations using CSR format
///
/// All sparse operations in Coeus use the CSR (Compressed Sparse Row) format
/// for optimal performance and memory efficiency.
pub trait TensorSparseOps<T: DataType>: Sized {
    /// Sparse matrix-vector multiplication: y = A * x
    fn sparse_matvec_mul<B: Backend<Data = T>>(
        &self,
        vector: &DenseStorage<T>,
        backend: &B,
    ) -> Result<DenseStorage<T>>
    where
        T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + num_traits::Zero + Copy;

    /// Sparse matrix-matrix multiplication: C = A * B
    fn sparse_matmul<B: Backend<Data = T>>(
        &self,
        other: &Self,
        backend: &B,
    ) -> Result<Self>
    where
        T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + num_traits::Zero + Copy;

    /// Element-wise sparse addition
    fn sparse_add(&self, other: &Self) -> Result<Self>
    where
        T: core::ops::Add<Output = T> + num_traits::Zero + PartialEq + Copy;

    /// Element-wise sparse multiplication
    fn sparse_mul(&self, other: &Self) -> Result<Self>
    where
        T: core::ops::Mul<Output = T> + num_traits::Zero + PartialEq + Copy;
}

// Implementation for CsrStorage - the optimal sparse format
impl<T: DataType + Default + 'static> TensorSparseOps<T> for CsrStorage<T> {
    fn sparse_matvec_mul<B: Backend<Data = T>>(
        &self,
        vector: &DenseStorage<T>,
        _backend: &B,
    ) -> Result<DenseStorage<T>>
    where
        T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + num_traits::Zero + Copy,
    {
        use storage::Storage;
        let (rows, cols) = self.dims();
        let vector_data = vector.as_slice();
        
        if vector_data.len() != cols {
            return Err(TensorError::ShapeMismatch {
                expected: vec![cols],
                actual: vec![vector_data.len()],
                operation: "sparse_matvec_mul",
            });
        }
        
        // Inline SpMV: y = A * x
        let mut result = vec![T::zero(); rows];
        for row in 0..rows {
            let start = self.indptr()[row];
            let end = self.indptr()[row + 1];
            let mut sum = T::zero();
            for idx in start..end {
                let col = self.indices()[idx];
                sum = sum + self.data()[idx] * vector_data[col];
            }
            result[row] = sum;
        }
        
        DenseStorage::from_vec(result, &[rows])
            .map_err(|e| TensorError::StorageError(e))
    }

    fn sparse_matmul<B: Backend<Data = T>>(
        &self,
        other: &Self,
        _backend: &B,
    ) -> Result<Self>
    where
        T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + num_traits::Zero + Copy,
    {
        let (self_rows, self_cols) = self.dims();
        let (other_rows, other_cols) = other.dims();
        
        if self_cols != other_rows {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self_cols],
                actual: vec![other_rows],
                operation: "sparse_matmul",
            });
        }
        
        // Transpose other matrix for efficient column access
        let other_t = SparseTranspose::transpose_sparse(other).map_err(|e| TensorError::StorageError(e))?;
        
        let mut result_data = Vec::new();
        let mut result_indices = Vec::new();
        let mut result_indptr = vec![0];
        
        for row in 0..self_rows {
            let self_start = self.indptr()[row];
            let self_end = self.indptr()[row + 1];
            
            for col in 0..other_cols {
                let other_start = other_t.indptr()[col];
                let other_end = other_t.indptr()[col + 1];
                
                let mut dot_product = T::zero();
                let mut self_idx = self_start;
                let mut other_idx = other_start;
                
                // Compute dot product of sparse vectors
                while self_idx < self_end && other_idx < other_end {
                    let self_col = self.indices()[self_idx];
                    let other_row = other_t.indices()[other_idx];
                    
                    match self_col.cmp(&other_row) {
                        core::cmp::Ordering::Equal => {
                            dot_product = dot_product + self.data()[self_idx] * other_t.data()[other_idx];
                            self_idx += 1;
                            other_idx += 1;
                        }
                        core::cmp::Ordering::Less => self_idx += 1,
                        core::cmp::Ordering::Greater => other_idx += 1,
                    }
                }
                
                if dot_product != T::zero() {
                    result_data.push(dot_product);
                    result_indices.push(col);
                }
            }
            
            result_indptr.push(result_data.len());
        }
        
        CsrStorage::new(result_data, result_indices, result_indptr, &[self_rows, other_cols])
            .map_err(|e| TensorError::StorageError(e))
    }

    fn sparse_add(&self, other: &Self) -> Result<Self>
    where
        T: core::ops::Add<Output = T> + num_traits::Zero + PartialEq + Copy,
    {
        SparseAdd::add_sparse(self, other).map_err(|e| TensorError::StorageError(e))
    }

    fn sparse_mul(&self, other: &Self) -> Result<Self>
    where
        T: core::ops::Mul<Output = T> + num_traits::Zero + PartialEq + Copy,
    {
        use storage::Storage;
        // Element-wise multiplication of sparse matrices
        // For now, convert to dense, multiply, then back to sparse
        let self_dense = self.to_dense().map_err(|e| TensorError::StorageError(e))?;
        let other_dense = other.to_dense().map_err(|e| TensorError::StorageError(e))?;
        
        let self_data = self_dense.as_slice();
        let other_data = other_dense.as_slice();
        
        if self_data.len() != other_data.len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self_data.len()],
                actual: vec![other_data.len()],
                operation: "sparse_mul",
            });
        }
        
        let dims = self.shape_ref().dims();
        let result_data: Vec<T> = self_data.iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        
        let result_dense = DenseStorage::from_vec(result_data, dims)
            .map_err(|e| TensorError::StorageError(e))?;
        
        CsrStorage::from_dense(&result_dense).map_err(|e| TensorError::StorageError(e))
    }
}