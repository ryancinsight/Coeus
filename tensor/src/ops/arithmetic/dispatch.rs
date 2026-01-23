//! Arithmetic dispatch implementations for different storage types
//!
//! This module provides storage-specific implementations of arithmetic operations,
//! enabling efficient dispatch based on the underlying storage format.
//!
//! ## Dense Dispatch
//!
//! `DenseStorage` operations delegate to `dense::DenseArithmetic` for optimized
//! contiguous memory operations with SIMD vectorization support.
//!
//! ## Sparse Dispatch
//!
//! `CsrStorage` (the optimal sparse format) delegates to enhanced sparse operations.
//! All sparse operations use CSR format for optimal performance and memory efficiency.
//!
//! ## Backend Parameter
//!
//! All methods accept `backend: &B` where `B: Backend<Data = T>`. This enables:
//! - CPU execution via `CpuBackend`
//! - GPU execution via `GpuBackend` (for dense operations)

use crate::ops::arithmetic::traits::TensorStorageArithmetic;
use crate::{Result, TensorError};

use backend::Backend;
use dtype::DataType;
use storage::{DenseStorage, CsrStorage, Storage};

use dense::DenseArithmetic;
use coeus_sparse::{SparseAdd, SparseSub, SparseMul, SparseDiv};

// ================== DenseStorage Implementation ==================

impl<T: DataType> TensorStorageArithmetic<T> for DenseStorage<T>
where
    T: core::ops::Add<Output = T> + core::ops::Sub<Output = T> + core::ops::Mul<Output = T> + core::ops::Div<Output = T> + core::ops::Neg<Output=T> + Copy + Clone + Default + 'static,
{
    fn tensor_add<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        DenseArithmetic::add(self, other, backend).map_err(|e| TensorError::StorageError(e))
    }

    fn tensor_sub<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        DenseArithmetic::sub(self, other, backend).map_err(|e| TensorError::StorageError(e))
    }

    fn tensor_mul<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        DenseArithmetic::mul(self, other, backend).map_err(|e| TensorError::StorageError(e))
    }

    fn tensor_div<B: Backend<Data = T>>(&self, other: &Self, _backend: &B) -> Result<Self> {
        DenseArithmetic::div(self, other).map_err(|e| TensorError::StorageError(e))
    }

    fn tensor_neg<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self> {
        // Negate all elements inline
        let negated: Vec<T> = self.as_slice().iter().map(|&x| -x).collect();
        DenseStorage::from_vec(negated, self.shape().dims()).map_err(TensorError::StorageError)
    }
}

// ================== CsrStorage Implementation ==================

impl<T: DataType> TensorStorageArithmetic<T> for CsrStorage<T>
where
    T: core::ops::Add<Output = T> + core::ops::Sub<Output = T> + core::ops::Mul<Output = T> + core::ops::Div<Output = T> + core::ops::Neg<Output=T> + Copy + Default + num_traits::Zero + PartialEq + 'static,
{
    fn tensor_add<B: Backend<Data = T>>(&self, other: &Self, _backend: &B) -> Result<Self> {
        SparseAdd::add_sparse(self, other).map_err(|e| TensorError::StorageError(e))
    }

    fn tensor_sub<B: Backend<Data = T>>(&self, other: &Self, _backend: &B) -> Result<Self> {
        SparseSub::sub_sparse(self, other).map_err(|e| TensorError::StorageError(e))
    }

    fn tensor_mul<B: Backend<Data = T>>(&self, other: &Self, _backend: &B) -> Result<Self> {
        SparseMul::mul_sparse(self, other).map_err(|e| TensorError::StorageError(e))
    }

    fn tensor_div<B: Backend<Data = T>>(&self, other: &Self, _backend: &B) -> Result<Self> {
        SparseDiv::div_sparse(self, other).map_err(|e| TensorError::StorageError(e))
    }

    fn tensor_neg<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self> {
        // Negate all non-zero values in place
        let mut new_data = self.data().to_vec();
        for value in &mut new_data {
            *value = -*value;
        }
        
        CsrStorage::new(
            new_data,
            self.indices().to_vec(),
            self.indptr().to_vec(),
            self.shape().dims(),
        ).map_err(|e| TensorError::StorageError(e))
    }
}