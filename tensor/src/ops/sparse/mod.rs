//! Sparse tensor operations
//!
//! This module provides operations specifically optimized for sparse tensors
//! using the CSR (Compressed Sparse Row) format for maximum efficiency.

pub mod traits;

use crate::{Result, Tensor};
use storage::CsrStorage;
pub use traits::TensorSparseOps;

/// Sparse matrix operations for CSR tensors (the optimal sparse format)
impl<B, T> Tensor<B, CsrStorage<T>, T>
where
    B: crate::Backend<Data = T> + Clone + Default,
    T: crate::DataType
        + Default
        + Copy
        + 'static
        + num_traits::Zero
        + PartialEq
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>,
{
    /// Matrix multiplication with another CSR tensor
    pub fn matmul(&self, other: &Self) -> Result<Tensor<B, CsrStorage<T>, T>> {
        self.sparse_matmul(other)
    }

    /// Sparse matrix multiplication using optimized CSR algorithms
    pub fn sparse_matmul(&self, other: &Self) -> Result<Tensor<B, CsrStorage<T>, T>> {
        let result_storage = self.storage.sparse_matmul(&other.storage, &self.backend)?;

        Ok(Tensor::from_storage(result_storage, self.backend.clone()))
    }

    /// Element-wise addition with another CSR tensor
    pub fn sparse_add(&self, other: &Self) -> Result<Tensor<B, CsrStorage<T>, T>> {
        let result_storage = self.storage.sparse_add(&other.storage)?;

        Ok(Tensor::from_storage(result_storage, self.backend.clone()))
    }

    /// Element-wise multiplication with another CSR tensor
    pub fn sparse_mul(&self, other: &Self) -> Result<Tensor<B, CsrStorage<T>, T>> {
        let result_storage = self.storage.sparse_mul(&other.storage)?;

        Ok(Tensor::from_storage(result_storage, self.backend.clone()))
    }

    /// Matrix-vector multiplication: y = A * x
    pub fn matvec(
        &self,
        vector: &Tensor<B, storage::DenseStorage<T>, T>,
    ) -> Result<Tensor<B, storage::DenseStorage<T>, T>>
    where
        T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + num_traits::Zero + Copy,
    {
        let result_storage = self
            .storage
            .sparse_matvec_mul(&vector.storage, &self.backend)?;

        Ok(Tensor::from_storage(result_storage, self.backend.clone()))
    }

    // Note: nnz, matrix_dims, to_dense, and transpose are defined in tensor/sparse/csr.rs to avoid duplication
}
