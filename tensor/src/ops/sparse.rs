//! Tensor-level sparse operations.
//!
//! Provides a high-level API for sparse tensor operations, dispatching to
//! optimized kernels in `coeus-sparse`.

use crate::{Result, Tensor, TensorError};
use coeus_sparse::cpu::arithmetic::*;
use storage::{CooStorage, CscStorage, CsrStorage, SparseFormat};

/// Sparse matrix operations for tensors.
impl<B, T> Tensor<B, CsrStorage<T>, T>
where
    B: crate::Backend<Data = T> + Clone + Default,
    T: storage::DataType
        + Clone
        + Copy
        + num_traits::Zero
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>,
{
    pub fn matmul(&self, other: &Self) -> Result<Tensor<B, CooStorage<T>, T>> {
        self.sparse_matmul(other)
    }

    /// Multiply CSR tensor with another CSR tensor.
    pub fn sparse_matmul(&self, other: &Self) -> Result<Tensor<B, CooStorage<T>, T>> {
        let result_storage = self
            .storage
            .matmul_sparse(&other.storage, SparseFormat::Csr)
            .map_err(TensorError::StorageError)?;

        Ok(Tensor::from_storage(result_storage, self.backend.clone()))
    }

    /// Multiply CSR tensor with a dense vector.
    pub fn sparse_matvec_mul(&self, vector: &[T]) -> Result<Vec<T>> {
        self.storage
            .matvec_mul(vector)
            .map_err(TensorError::StorageError)
    }
}

impl<B, T> Tensor<B, CscStorage<T>, T>
where
    B: crate::Backend<Data = T> + Clone + Default,
    T: storage::DataType
        + Clone
        + Copy
        + num_traits::Zero
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>,
{
    pub fn matmul(&self, other: &Self) -> Result<Tensor<B, CooStorage<T>, T>> {
        self.sparse_matmul(other)
    }

    pub fn sparse_matmul(&self, other: &Self) -> Result<Tensor<B, CooStorage<T>, T>> {
        let result_storage = self
            .storage
            .matmul_sparse(&other.storage, SparseFormat::Csc)
            .map_err(TensorError::StorageError)?;

        Ok(Tensor::from_storage(result_storage, self.backend.clone()))
    }

    pub fn sparse_matvec_mul(&self, vector: &[T]) -> Result<Vec<T>> {
        self.storage
            .matvec_mul(vector)
            .map_err(TensorError::StorageError)
    }
}

impl<B, T> Tensor<B, CooStorage<T>, T>
where
    B: crate::Backend<Data = T> + Clone + Default,
    T: storage::DataType
        + Clone
        + Copy
        + num_traits::Zero
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>,
{
    pub fn matmul(&self, other: &Self) -> Result<Tensor<B, CooStorage<T>, T>> {
        self.sparse_matmul(other)
    }

    pub fn sparse_matmul(&self, other: &Self) -> Result<Tensor<B, CooStorage<T>, T>> {
        let result_storage = self
            .storage
            .matmul_sparse(&other.storage, SparseFormat::Coo)
            .map_err(TensorError::StorageError)?;

        Ok(Tensor::from_storage(result_storage, self.backend.clone()))
    }
}

impl<B, T> Tensor<B, CooStorage<T>, T>
where
    B: crate::Backend<Data = T> + Clone + Default,
    T: storage::DataType + Clone + Copy + num_traits::Zero + core::ops::Add<Output = T>,
{
    /// Add two COO tensors.
    pub fn sparse_add(&self, other: &Self) -> Result<Tensor<B, CooStorage<T>, T>> {
        let result_storage = self
            .storage
            .add_sparse(&other.storage)
            .map_err(TensorError::StorageError)?;

        Ok(Tensor::from_storage(result_storage, self.backend.clone()))
    }
}

// Additional sparse operations can be added here mirroring PyTorch API
