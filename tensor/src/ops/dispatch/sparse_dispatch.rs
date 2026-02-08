//! Sparse storage implementations of TensorStorageOps
//!
//! Implements TensorStorageOps for CsrStorage using optimized sparse algorithms.
//! All sparse operations use CSR format for maximum performance and efficiency.

use crate::ops::dispatch::traits::TensorStorageOps;
use crate::{Result, TensorError};
use backend::Backend;
use coeus_sparse::{
    SparseActivation, SparseAdd, SparseDiv, SparseElementWise, SparseMul, SparseSub,
    SparseTranspose,
};
use dtype::DataType;
use storage::{CooStorage, CsrStorage, Storage};

// ================== CsrStorage Implementation ==================

impl<T: DataType + Default + 'static> TensorStorageOps<T> for CsrStorage<T>
where
    T: core::ops::Add<Output = T>
        + core::ops::Sub<Output = T>
        + core::ops::Mul<Output = T>
        + core::ops::Div<Output = T>
        + num_traits::Zero
        + num_traits::One
        + num_traits::FromPrimitive
        + PartialEq
        + PartialOrd
        + Copy
        + Default,
{
    // ========== Arithmetic Operations ==========

    fn storage_add<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        backend.add_csr(self, other).map_err(TensorError::from)
    }

    fn storage_sub<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        backend.sub_csr(self, other).map_err(TensorError::from)
    }

    fn storage_mul<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        backend.mul_csr(self, other).map_err(TensorError::from)
    }

    fn storage_div<B: Backend<Data = T>>(&self, other: &Self, _backend: &B) -> Result<Self> {
        // Fallback for now if not in backend, but div is tricky for sparse (0/0 etc)
        SparseDiv::div_sparse(self, other).map_err(TensorError::StorageError)
    }

    fn storage_neg<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: core::ops::Neg<Output = T> {
        backend.neg_csr(self).map_err(Into::into)
    }

    // ========== Comparison Operations (via dense) ==========

    fn storage_eq<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        let dense_self = self.storage_to_dense()?;
        let dense_other = other.storage_to_dense()?;
        let res = dense_self.storage_eq(&dense_other, backend)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_ne<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        let dense_self = self.storage_to_dense()?;
        let dense_other = other.storage_to_dense()?;
        let res = dense_self.storage_ne(&dense_other, backend)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_gt<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + num_traits::One + num_traits::Zero,
    {
        let dense_self = self.storage_to_dense()?;
        let dense_other = other.storage_to_dense()?;
        let res = dense_self.storage_gt(&dense_other, backend)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_ge<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + num_traits::One + num_traits::Zero,
    {
        let dense_self = self.storage_to_dense()?;
        let dense_other = other.storage_to_dense()?;
        let res = dense_self.storage_ge(&dense_other, backend)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_lt<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + num_traits::One + num_traits::Zero,
    {
        let dense_self = self.storage_to_dense()?;
        let dense_other = other.storage_to_dense()?;
        let res = dense_self.storage_lt(&dense_other, backend)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_le<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + num_traits::One + num_traits::Zero,
    {
        let dense_self = self.storage_to_dense()?;
        let dense_other = other.storage_to_dense()?;
        let res = dense_self.storage_le(&dense_other, backend)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    // ========== Matrix Operations ==========

    fn storage_matmul<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        backend.matmul_csr(self, other).map_err(TensorError::from)
    }

    fn storage_transpose<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self> {
        SparseTranspose::transpose_sparse(self).map_err(TensorError::StorageError)
    }

    fn storage_addmm<B: Backend<Data = T>>(
        &self,
        mat1: &Self,
        mat2: &Self,
        beta: T,
        alpha: T,
        backend: &B,
    ) -> Result<Self> {
        backend.addmm_csr(self, mat1, mat2, beta, alpha).map_err(TensorError::from)
    }

    fn storage_addmv<B: Backend<Data = T>>(
        &self,
        mat: &Self,
        vec: &Self,
        beta: T,
        alpha: T,
        backend: &B,
    ) -> Result<Self> {
        backend.addmv_csr(self, mat, vec, beta, alpha).map_err(TensorError::from)
    }

    // ========== Activation Functions (Native Sparse) ==========

    fn storage_relu<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + Default,
    {
        backend.relu_csr(self).map_err(Into::into)
    }

    fn storage_sigmoid<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        // Sigmoid(0) = 0.5, so this operation results in a dense tensor.
        // We follow absolute mathematical correctness by converting to dense.
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let result_dense = backend
            .sigmoid_dense(&dense)
            .map_err(|e| TensorError::from(e))?;
        // Special case: we must return Self (CsrStorage). If the caller expects sparse,
        // we convert back, though it will likely be very dense.
        Self::from_dense(&result_dense).map_err(TensorError::StorageError)
    }

    fn storage_tanh<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.tanh_csr(self).map_err(Into::into)
    }

    fn storage_gelu<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        self.activation_gelu(_backend)
            .map_err(TensorError::StorageError)
    }

    // ========== Transcendental Operations (via dense conversion) ==========

    fn storage_exp<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let result_dense = backend
            .exp_dense(&dense)
            .map_err(|e| TensorError::from(e))?;
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
        let result_dense = backend
            .log_dense(&dense)
            .map_err(|e| TensorError::from(e))?;
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
        let result_dense = backend
            .cos_dense(&dense)
            .map_err(|e| TensorError::from(e))?;
        Self::from_dense(&result_dense).map_err(TensorError::StorageError)
    }

    fn storage_abs<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Signed,
    {
        backend.abs_csr(self).map_err(Into::into)
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

    fn storage_sum<B: Backend<Data = T>>(&self, _backend: &B) -> Result<T> {
        // Optimized sum: sum the values directly.
        // For CsrStorage, checking zeros isn't strictly necessary if we assume 
        // sum(zeros) = 0. However, if T is special (e.g. reduction implies something else),
        // we might care. But standard sum is sum of all elements.
        // sum = sum(values) + sum(zeros).
        // sum(values) is sum of self.data().
        // sum(zeros) is 0 * num_zeros = 0.
        // So just returning sum(values) is correct for addition.
        
        let values = self.data();
        let mut sum = T::zero();
        for &val in values {
            sum = sum + val;
        }
        Ok(sum)
    }

    fn storage_mean<B: Backend<Data = T>>(&self, backend: &B) -> Result<T>
    where
        T: num_traits::FromPrimitive,
    {
        let sum = self.storage_sum(backend)?;
        let total_elements = T::from_usize(self.len()).unwrap_or(T::one());
        Ok(sum / total_elements)
    }

    fn storage_max<B: Backend<Data = T>>(&self, _backend: &B) -> Result<T> {
        let values = self.data();
        if values.is_empty() {
             return Ok(T::zero());
        }

        let mut max_val = values[0];
        for &val in values.iter().skip(1) {
            if val > max_val {
                max_val = val;
            }
        }

        // Implicit zeros check: if nnz < total elements, then 0 is present.
        if values.len() < self.len() {
             let zero = T::zero();
             if zero > max_val {
                 max_val = zero;
             }
        }
        Ok(max_val)
    }

    fn storage_min<B: Backend<Data = T>>(&self, _backend: &B) -> Result<T> {
        let values = self.data();
        if values.is_empty() {
             return Ok(T::zero());
        }

        let mut min_val = values[0];
        for &val in values.iter().skip(1) {
            if val < min_val {
                min_val = val;
            }
        }

        // Implicit zeros check
        if values.len() < self.len() {
             let zero = T::zero();
             if zero < min_val {
                 min_val = zero;
             }
        }
        Ok(min_val)
    }

    // ========== Conversion Operations ==========

    fn storage_to_dense(&self) -> Result<storage::DenseStorage<T>>
    where
        T: num_traits::Zero + Clone,
    {
        self.to_dense().map_err(TensorError::StorageError)
    }

    fn storage_sqrt<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.sqrt_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_rsqrt<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.rsqrt_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_erf<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.erf_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_erfc<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.erfc_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_erfinv<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.erfinv_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_atan2<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let dense_other = other.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.atan2_dense(&dense_self, &dense_other).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_log1p<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.log1p_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_expm1<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.expm1_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_reciprocal<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.reciprocal_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_isnan<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float + num_traits::One + num_traits::Zero,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.isnan_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_isinf<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float + num_traits::One + num_traits::Zero,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.isinf_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_isfinite<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float + num_traits::One + num_traits::Zero,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.isfinite_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_logical_and<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let dense_other = other.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.logical_and_dense(&dense_self, &dense_other).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_logical_or<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let dense_other = other.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.logical_or_dense(&dense_self, &dense_other).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_logical_xor<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let dense_other = other.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.logical_xor_dense(&dense_self, &dense_other).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_logical_not<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.logical_not_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }
}

impl<T: DataType + Default + 'static> super::traits::StorageBinaryOps<storage::DenseStorage<T>, T> for CsrStorage<T>
where
    T: core::ops::Add<Output = T>
        + core::ops::Sub<Output = T>
        + core::ops::Mul<Output = T>
        + core::ops::Div<Output = T>
        + num_traits::Zero
        + num_traits::One
        + num_traits::FromPrimitive
        + core::ops::Neg<Output = T>
        + PartialEq
        + PartialOrd
        + Copy
        + Default,
{
    type Output = storage::DenseStorage<T>;

    fn storage_add_mixed<B: Backend<Data = T>>(&self, other: &storage::DenseStorage<T>, backend: &B) -> Result<Self::Output> {
        backend.add_csr_dense(self, other).map_err(Into::into)
    }

    fn storage_sub_mixed<B: Backend<Data = T>>(&self, other: &storage::DenseStorage<T>, backend: &B) -> Result<Self::Output> {
        // self - other = (-other) + self
        let mut neg_other = other.clone();
        for val in neg_other.as_mut_slice() {
            *val = T::zero() - *val;
        }
        backend.add_dense_csr(&neg_other, self).map_err(Into::into)
    }

    fn storage_mul_mixed<B: Backend<Data = T>>(&self, other: &storage::DenseStorage<T>, backend: &B) -> Result<Self::Output> {
        backend.mul_csr_dense(self, other).map_err(Into::into)
    }

    fn storage_div_mixed<B: Backend<Data = T>>(&self, other: &storage::DenseStorage<T>, backend: &B) -> Result<Self::Output> {
        let dense_self = self.storage_to_dense()?;
        dense_self.storage_div(other, backend)
    }

    fn storage_matmul_mixed<B: Backend<Data = T>>(&self, other: &storage::DenseStorage<T>, backend: &B) -> Result<Self::Output> {
        backend.matmul_csr_dense(self, other).map_err(Into::into)
    }
}

// ================== CooStorage Implementation ==================

impl<T: DataType + Default + 'static> TensorStorageOps<T> for CooStorage<T>
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
    // For COO, we delegate most operations to Dense storage to ensure correctness and reused logic.
    // Operations that are O(nnz) natively (transpose, sum, neg) are implemented directly.

    fn storage_add<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        let res_csr = backend.coo_add_sparse(
            self.data(), self.row_indices(), self.col_indices(),
            other.data(), other.row_indices(), other.col_indices(),
            self.shape().dims()[0], self.shape().dims()[1]
        ).map_err(TensorError::from)?;
        res_csr.to_coo().map_err(TensorError::StorageError)
    }

    fn storage_sub<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        let res_csr = backend.coo_sub_sparse(
            self.data(), self.row_indices(), self.col_indices(),
            other.data(), other.row_indices(), other.col_indices(),
            self.shape().dims()[0], self.shape().dims()[1]
        ).map_err(TensorError::from)?;
        res_csr.to_coo().map_err(TensorError::StorageError)
    }

    fn storage_mul<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        let res_csr = backend.coo_mul_sparse(
            self.data(), self.row_indices(), self.col_indices(),
            other.data(), other.row_indices(), other.col_indices(),
            self.shape().dims()[0], self.shape().dims()[1]
        ).map_err(TensorError::from)?;
        res_csr.to_coo().map_err(TensorError::StorageError)
    }

    fn storage_div<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let dense_other = other.to_dense().map_err(TensorError::StorageError)?;
        let res_dense = dense_self.storage_div(&dense_other, backend)?;
        Self::from_dense(&res_dense).map_err(TensorError::StorageError)
    }

    fn storage_neg<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self> {
        // Direct O(nnz) implementation
        let mut new_data = self.data().to_vec();
        for x in &mut new_data {
            *x = -*x;
        }
        Self::new(new_data, self.row_indices().to_vec(), self.col_indices().to_vec(), self.shape().dims())
            .map_err(TensorError::StorageError)
    }

    // ========== Comparison Operations (via dense) ==========

    fn storage_eq<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let dense_other = other.to_dense().map_err(TensorError::StorageError)?;
        let res = dense_self.storage_eq(&dense_other, backend)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_ne<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let dense_other = other.to_dense().map_err(TensorError::StorageError)?;
        let res = dense_self.storage_ne(&dense_other, backend)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_gt<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + num_traits::One + num_traits::Zero,
    {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let dense_other = other.to_dense().map_err(TensorError::StorageError)?;
        let res = dense_self.storage_gt(&dense_other, backend)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_ge<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + num_traits::One + num_traits::Zero,
    {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let dense_other = other.to_dense().map_err(TensorError::StorageError)?;
        let res = dense_self.storage_ge(&dense_other, backend)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_lt<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + num_traits::One + num_traits::Zero,
    {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let dense_other = other.to_dense().map_err(TensorError::StorageError)?;
        let res = dense_self.storage_lt(&dense_other, backend)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_le<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + num_traits::One + num_traits::Zero,
    {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let dense_other = other.to_dense().map_err(TensorError::StorageError)?;
        let res = dense_self.storage_le(&dense_other, backend)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_matmul<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let dense_other = other.to_dense().map_err(TensorError::StorageError)?;
        let res_dense = dense_self.storage_matmul(&dense_other, backend)?;
        Self::from_dense(&res_dense).map_err(TensorError::StorageError)
    }

    fn storage_addmm<B: Backend<Data = T>>(
        &self,
        mat1: &Self,
        mat2: &Self,
        beta: T,
        alpha: T,
        backend: &B,
    ) -> Result<Self> {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let dense_mat1 = mat1.to_dense().map_err(TensorError::StorageError)?;
        let dense_mat2 = mat2.to_dense().map_err(TensorError::StorageError)?;
        
        let res_dense = dense_self.storage_addmm(&dense_mat1, &dense_mat2, beta, alpha, backend)?;
        Self::from_dense(&res_dense).map_err(TensorError::StorageError)
    }

    fn storage_addmv<B: Backend<Data = T>>(
        &self,
        mat: &Self,
        vec: &Self,
        beta: T,
        alpha: T,
        backend: &B,
    ) -> Result<Self> {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let dense_mat = mat.to_dense().map_err(TensorError::StorageError)?;
        let dense_vec = vec.to_dense().map_err(TensorError::StorageError)?;
        
        let res_dense = dense_self.storage_addmv(&dense_mat, &dense_vec, beta, alpha, backend)?;
        Self::from_dense(&res_dense).map_err(TensorError::StorageError)
    }

    fn storage_transpose<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self> {
        // Direct O(nnz) implementation
        Self::new(
            self.data().to_vec(),
            self.col_indices().to_vec(), 
            self.row_indices().to_vec(), 
            &[self.shape().dims()[1], self.shape().dims()[0]]
        ).map_err(TensorError::StorageError)
    }

    fn storage_relu<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self> where T: PartialOrd + Default {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let res_dense = dense_self.storage_relu(backend)?;
        Self::from_dense(&res_dense).map_err(TensorError::StorageError)
    }

    fn storage_sigmoid<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self> where T: num_traits::Float {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let res_dense = dense_self.storage_sigmoid(backend)?;
        Self::from_dense(&res_dense).map_err(TensorError::StorageError)
    }

    fn storage_tanh<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self> where T: num_traits::Float {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let res_dense = dense_self.storage_tanh(backend)?;
        Self::from_dense(&res_dense).map_err(TensorError::StorageError)
    }

    fn storage_gelu<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self> where T: num_traits::Float {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let res_dense = dense_self.storage_gelu(backend)?;
        Self::from_dense(&res_dense).map_err(TensorError::StorageError)
    }

    fn storage_sum<B: Backend<Data = T>>(&self, _backend: &B) -> Result<T> {
        // Sum of nnz is sum of all (since others are 0)
        let mut sum = T::zero();
        for x in self.data() { sum = sum + *x; }
        Ok(sum)
    }

    fn storage_mean<B: Backend<Data = T>>(&self, backend: &B) -> Result<T> where T: num_traits::FromPrimitive {
        let sum = self.storage_sum(backend)?;
        let count = T::from_usize(self.shape().size()).unwrap();
        Ok(sum / count)
    }

    fn storage_max<B: Backend<Data = T>>(&self, _backend: &B) -> Result<T> where T: PartialOrd {
        let values = self.data();
        if values.is_empty() {
             return Ok(T::zero());
        }

        let mut max_val = values[0];
        for &val in values.iter().skip(1) {
            if val > max_val {
                max_val = val;
            }
        }

        if values.len() < self.len() {
             let zero = T::zero();
             if zero > max_val {
                 max_val = zero;
             }
        }
        Ok(max_val)
    }

    fn storage_min<B: Backend<Data = T>>(&self, _backend: &B) -> Result<T> where T: PartialOrd {
         let values = self.data();
        if values.is_empty() {
             return Ok(T::zero());
        }

        let mut min_val = values[0];
        for &val in values.iter().skip(1) {
            if val < min_val {
                min_val = val;
            }
        }

        if values.len() < self.len() {
             let zero = T::zero();
             if zero < min_val {
                 min_val = zero;
             }
        }
        Ok(min_val)
    }

    fn storage_exp<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self> where T: num_traits::Float {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let res_dense = dense_self.storage_exp(backend)?;
        Self::from_dense(&res_dense).map_err(TensorError::StorageError)
    }

    fn storage_log<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self> where T: num_traits::Float {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let res_dense = dense_self.storage_log(backend)?;
        Self::from_dense(&res_dense).map_err(TensorError::StorageError)
    }

    fn storage_sin<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self> where T: num_traits::Float {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let res_dense = dense_self.storage_sin(backend)?;
        Self::from_dense(&res_dense).map_err(TensorError::StorageError)
    }

    fn storage_cos<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self> where T: num_traits::Float {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let res_dense = dense_self.storage_cos(backend)?;
        Self::from_dense(&res_dense).map_err(TensorError::StorageError)
    }

    fn storage_abs<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self> where T: num_traits::Signed {
         // Direct O(nnz) implementation
        let mut new_data = self.data().to_vec();
        for x in &mut new_data { *x = x.abs(); }
        Self::new(new_data, self.row_indices().to_vec(), self.col_indices().to_vec(), self.shape().dims())
            .map_err(TensorError::StorageError)
    }

    fn storage_ceil<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self> where T: num_traits::Float {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let res_dense = dense_self.storage_ceil(backend)?;
        Self::from_dense(&res_dense).map_err(TensorError::StorageError)
    }

    fn storage_floor<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self> where T: num_traits::Float {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let res_dense = dense_self.storage_floor(backend)?;
        Self::from_dense(&res_dense).map_err(TensorError::StorageError)
    }

    fn storage_round<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self> where T: num_traits::Float {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let res_dense = dense_self.storage_round(backend)?;
        Self::from_dense(&res_dense).map_err(TensorError::StorageError)
    }

    fn storage_to_dense(&self) -> Result<storage::DenseStorage<T>> where T: num_traits::Zero + Clone {
        self.to_dense().map_err(TensorError::StorageError)
    }

    fn storage_sqrt<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.sqrt_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_rsqrt<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.rsqrt_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_erf<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.erf_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_erfc<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.erfc_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_erfinv<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.erfinv_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_atan2<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let dense_other = other.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.atan2_dense(&dense_self, &dense_other).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_log1p<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.log1p_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_expm1<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.expm1_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_reciprocal<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.reciprocal_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_isnan<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float + num_traits::One + num_traits::Zero,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.isnan_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_isinf<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float + num_traits::One + num_traits::Zero,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.isinf_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_isfinite<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float + num_traits::One + num_traits::Zero,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.isfinite_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_logical_and<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let dense_other = other.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.logical_and_dense(&dense_self, &dense_other).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_logical_or<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let dense_other = other.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.logical_or_dense(&dense_self, &dense_other).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_logical_xor<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        let dense_self = self.to_dense().map_err(TensorError::StorageError)?;
        let dense_other = other.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.logical_xor_dense(&dense_self, &dense_other).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }

    fn storage_logical_not<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        let dense = self.to_dense().map_err(TensorError::StorageError)?;
        let res = backend.logical_not_dense(&dense).map_err(TensorError::from)?;
        Self::from_dense(&res).map_err(TensorError::StorageError)
    }
}

