//! Sparse storage implementations of TensorStorageOps
//!
//! Implements TensorStorageOps for CsrStorage using optimized sparse algorithms.
//! All sparse operations use CSR format for maximum performance and efficiency.

use crate::ops::dispatch::traits::TensorStorageOps;
use crate::{Result, TensorError};
use backend::Backend;
use coeus_sparse::{
    SparseAdd, SparseDiv, SparseMul, SparseSub, SparseActivation, SparseTranspose, SparseElementWise
};
use dtype::DataType;
use storage::{CsrStorage, Storage};

// ================== CsrStorage Implementation ==================

impl<T: DataType + Default + 'static> TensorStorageOps<T> for CsrStorage<T>
where
    T: core::ops::Add<Output = T>
        + core::ops::Sub<Output = T>
        + core::ops::Mul<Output = T>
        + core::ops::Div<Output = T>
        + core::ops::Neg<Output = T>
        + num_traits::Zero
        + num_traits::One
        + num_traits::FromPrimitive
        + PartialEq
        + PartialOrd
        + Copy
        + Default,
{
    // ========== Arithmetic Operations ==========

    fn storage_add<B: Backend<Data = T>>(&self, other: &Self, _backend: &B) -> Result<Self> {
        SparseAdd::add_sparse(self, other).map_err(TensorError::StorageError)
    }

    fn storage_sub<B: Backend<Data = T>>(&self, other: &Self, _backend: &B) -> Result<Self> {
        SparseSub::sub_sparse(self, other).map_err(TensorError::StorageError)
    }

    fn storage_mul<B: Backend<Data = T>>(&self, other: &Self, _backend: &B) -> Result<Self> {
        SparseMul::mul_sparse(self, other).map_err(TensorError::StorageError)
    }

    fn storage_div<B: Backend<Data = T>>(&self, other: &Self, _backend: &B) -> Result<Self> {
        SparseDiv::div_sparse(self, other).map_err(TensorError::StorageError)
    }

    fn storage_neg<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self> {
        // Negate all non-zero values directly in CSR format
        let mut new_data = self.data().to_vec();
        for value in &mut new_data {
            *value = -*value;
        }

        Self::new(
            new_data,
            self.indices().to_vec(),
            self.indptr().to_vec(),
            self.shape().dims(),
        )
        .map_err(TensorError::StorageError)
    }

    // ========== Matrix Operations ==========

    fn storage_matmul<B: Backend<Data = T>>(&self, other: &Self, _backend: &B) -> Result<Self> {
        // Use optimized sparse matrix multiplication
        let (self_rows, self_cols) = self.dims();
        let (other_rows, other_cols) = other.dims();

        if self_cols != other_rows {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self_cols],
                actual: vec![other_rows],
                operation: "matmul",
            });
        }

        // Transpose other matrix for efficient access
        let other_t = SparseTranspose::transpose_sparse(other)
            .map_err(TensorError::StorageError)?;

        let mut result_data = alloc::vec::Vec::new();
        let mut result_indices = alloc::vec::Vec::new();
        let mut result_indptr = alloc::vec![0];

        for row in 0..self_rows {
            let self_start = self.indptr()[row];
            let self_end = self.indptr()[row + 1];

            for col in 0..other_cols {
                let other_start = other_t.indptr()[col];
                let other_end = other_t.indptr()[col + 1];

                let mut dot_product = T::zero();
                let mut self_idx = self_start;
                let mut other_idx = other_start;

                while self_idx < self_end && other_idx < other_end {
                    let self_col = self.indices()[self_idx];
                    let other_row = other_t.indices()[other_idx];

                    match self_col.cmp(&other_row) {
                        core::cmp::Ordering::Equal => {
                            dot_product =
                                dot_product + self.data()[self_idx] * other_t.data()[other_idx];
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

        Self::new(result_data, result_indices, result_indptr, &[self_rows, other_cols])
            .map_err(TensorError::StorageError)
    }

    fn storage_transpose<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self> {
        SparseTranspose::transpose_sparse(self).map_err(TensorError::StorageError)
    }

    // ========== Activation Functions (Native Sparse) ==========

    fn storage_relu<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self> {
        self.activation_relu(backend).map_err(TensorError::StorageError)
    }

    fn storage_sigmoid<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        // Sigmoid(0) = 0.5, so this operation results in a dense tensor.
        // We follow absolute mathematical correctness by converting to dense.
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let result_dense = backend.sigmoid_dense(&dense).map_err(|e| TensorError::from(e))?;
        // Special case: we must return Self (CsrStorage). If the caller expects sparse,
        // we convert back, though it will likely be very dense.
        Self::from_dense(&result_dense).map_err(TensorError::StorageError)
    }

    fn storage_tanh<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        self.tanh_sparse().map_err(TensorError::StorageError)
    }

    fn storage_gelu<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        self.activation_gelu(_backend).map_err(TensorError::StorageError)
    }

    // ========== Transcendental Operations (via dense conversion) ==========

    fn storage_exp<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let result_dense = backend.exp_dense(&dense).map_err(|e| TensorError::from(e))?;
        Self::from_dense(&result_dense).map_err(TensorError::StorageError)
    }

    fn storage_log<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        // Log(0) is undefined/-inf, but torch behavior for Sparse Log(x) is to apply to non-zeros.
        // If we want total correctness (dense), we should convert.
        // For PyTorch parity, we apply to non-zeros if possible, but Log(0) usually means dense.
        // torch.log(sparse) -> dense.
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let result_dense = backend.log_dense(&dense).map_err(|e| TensorError::from(e))?;
        Self::from_dense(&result_dense).map_err(TensorError::StorageError)
    }

    fn storage_sin<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        self.sin_sparse().map_err(TensorError::StorageError)
    }

    fn storage_cos<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        // Cos(0) = 1, MUST be dense.
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let result_dense = backend.cos_dense(&dense).map_err(|e| TensorError::from(e))?;
        Self::from_dense(&result_dense).map_err(TensorError::StorageError)
    }

    fn storage_abs<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self>
    where
        T: num_traits::Signed,
    {
        self.abs_sparse().map_err(TensorError::StorageError)
    }

    fn storage_ceil<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        self.ceil_sparse().map_err(TensorError::StorageError)
    }

    fn storage_floor<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        self.floor_sparse().map_err(TensorError::StorageError)
    }

    fn storage_round<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        self.round_sparse().map_err(TensorError::StorageError)
    }

    // ========== Reduction Operations ==========

    fn storage_sum<B: Backend<Data = T>>(&self, backend: &B) -> Result<T> {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        backend.sum_dense(&dense).map_err(|e| TensorError::from(e))
    }

    fn storage_mean<B: Backend<Data = T>>(&self, backend: &B) -> Result<T>
    where
        T: num_traits::FromPrimitive,
    {
        let sum = self.storage_sum(backend)?;
        let total_elements = T::from_usize(self.len()).unwrap_or(T::one());
        Ok(sum / total_elements)
    }

    fn storage_max<B: Backend<Data = T>>(&self, backend: &B) -> Result<T> {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        backend.max_dense(&dense).map_err(Into::into)
    }

    fn storage_min<B: Backend<Data = T>>(&self, backend: &B) -> Result<T> {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        backend.min_dense(&dense).map_err(Into::into)
    }

    // ========== Conversion Operations ==========

    fn storage_to_dense(&self) -> Result<storage::DenseStorage<T>>
    where
        T: num_traits::Zero + Clone,
    {
        self.to_dense().map_err(TensorError::StorageError)
    }
}