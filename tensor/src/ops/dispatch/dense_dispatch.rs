//! DenseStorage implementation of TensorStorageOps
//!
//! Delegates all operations to the `dense` crate and `Backend` trait,
//! enabling GPU acceleration when GpuBackend is used.

use super::traits::TensorStorageOps;
use crate::{Result, TensorError};
use backend::Backend;
use dense::DenseArithmetic;
use dtype::DataType;
use storage::{DenseStorage, Storage};

impl<T: DataType> TensorStorageOps<T> for DenseStorage<T>
where
    T: core::ops::Add<Output = T>
        + core::ops::Sub<Output = T>
        + core::ops::Mul<Output = T>
        + core::ops::Div<Output = T>
        + num_traits::Zero
        + num_traits::One
        + Clone,
{
    // ================== Arithmetic ==================

    fn storage_add<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        self.add(other, backend).map_err(TensorError::StorageError)
    }

    fn storage_sub<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        self.sub(other, backend).map_err(TensorError::StorageError)
    }

    fn storage_mul<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        self.mul(other, backend).map_err(TensorError::StorageError)
    }

    fn storage_div<B: Backend<Data = T>>(&self, other: &Self, _backend: &B) -> Result<Self> {
        self.div(other).map_err(TensorError::StorageError)
    }

    fn storage_neg<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self>
    where
        T: core::ops::Neg<Output = T> {
        let mut result = self.clone();
        for value in result.as_mut_slice() {
            *value = -*value;
        }
        Ok(result)
    }

    // ================== Comparison ==================

    fn storage_eq<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        backend.eq_dense(self, other).map_err(Into::into)
    }

    fn storage_ne<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        backend.ne_dense(self, other).map_err(Into::into)
    }

    fn storage_gt<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + num_traits::One + num_traits::Zero,
    {
        backend.gt_dense(self, other).map_err(Into::into)
    }

    fn storage_ge<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + num_traits::One + num_traits::Zero,
    {
        backend.ge_dense(self, other).map_err(Into::into)
    }

    fn storage_lt<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + num_traits::One + num_traits::Zero,
    {
        backend.lt_dense(self, other).map_err(Into::into)
    }

    fn storage_le<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + num_traits::One + num_traits::Zero,
    {
        backend.le_dense(self, other).map_err(Into::into)
    }

    // ================== Linear Algebra ==================

    fn storage_matmul<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        backend.matmul_dense(self, other).map_err(Into::into)
    }

    fn storage_transpose<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self> {
        use storage::Storage;
        let shape = self.shape().dims();
        if shape.len() != 2 {
            return Err(TensorError::ShapeError {
                expected: 2,
                actual: shape.len(),
                message: "transpose requires 2D tensor".to_string(),
            });
        }
        let (rows, cols) = (shape[0], shape[1]);
        let src = self.as_slice();
        let mut dst = alloc::vec![T::default(); rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
        DenseStorage::from_vec(dst, &[cols, rows]).map_err(TensorError::StorageError)
    }

    fn storage_addmm<B: Backend<Data = T>>(
        &self,
        mat1: &Self,
        mat2: &Self,
        beta: T,
        alpha: T,
        backend: &B,
    ) -> Result<Self> {
        match backend.addmm_dense(self, mat1, mat2, beta.clone(), alpha.clone()) {
             Ok(res) => Ok(res),
             Err(backend::BackendError::UnsupportedOperation { .. }) => {
                 Err(TensorError::UnsupportedOperation {
                     operation: "storage_addmm".to_string(),
                     storage_type: "DenseStorage".to_string(),
                 })
             }
             Err(e) => Err(e.into()),
        }
    }

    fn storage_addmv<B: Backend<Data = T>>(
        &self,
        mat: &Self,
        vec: &Self,
        beta: T,
        alpha: T,
        backend: &B,
    ) -> Result<Self> {
        match backend.addmv_dense(self, mat, vec, beta.clone(), alpha.clone()) {
             Ok(res) => Ok(res),
             Err(backend::BackendError::UnsupportedOperation { .. }) => {
                 Err(TensorError::UnsupportedOperation {
                     operation: "storage_addmv".to_string(),
                     storage_type: "DenseStorage".to_string(),
                 })
             }
             Err(e) => Err(e.into()),
        }
    }

    // ================== Activation Functions ==================

    fn storage_relu<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + Default,
    {
        backend.relu_dense(self).map_err(Into::into)
    }

    fn storage_sigmoid<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.sigmoid_dense(self).map_err(Into::into)
    }

    fn storage_tanh<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.tanh_dense(self).map_err(Into::into)
    }

    fn storage_gelu<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.gelu_dense(self).map_err(Into::into)
    }

    // ================== Reductions ==================

    fn storage_sum<B: Backend<Data = T>>(&self, backend: &B) -> Result<T> {
        backend.sum_dense(self).map_err(Into::into)
    }

    fn storage_mean<B: Backend<Data = T>>(&self, backend: &B) -> Result<T>
    where
        T: num_traits::FromPrimitive,
    {
        let sum = self.storage_sum(backend)?;
        let n = T::from_usize(self.len()).unwrap_or(T::one());
        Ok(sum / n)
    }

    fn storage_max<B: Backend<Data = T>>(&self, backend: &B) -> Result<T>
    where
        T: PartialOrd,
    {
        backend.max_dense(self).map_err(Into::into)
    }

    fn storage_min<B: Backend<Data = T>>(&self, backend: &B) -> Result<T>
    where
        T: PartialOrd,
    {
        backend.min_dense(self).map_err(Into::into)
    }

    // ================== Transcendentals ==================

    fn storage_exp<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.exp_dense(self).map_err(Into::into)
    }

    fn storage_log<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.log_dense(self).map_err(Into::into)
    }

    fn storage_sin<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.sin_dense(self).map_err(Into::into)
    }

    fn storage_cos<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.cos_dense(self).map_err(Into::into)
    }

    fn storage_abs<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Signed,
    {
        backend.abs_dense(self).map_err(Into::into)
    }

    fn storage_ceil<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.ceil_dense(self).map_err(Into::into)
    }

    fn storage_floor<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.floor_dense(self).map_err(Into::into)
    }

    fn storage_round<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.round_dense(self).map_err(Into::into)
    }

    fn storage_sqrt<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.sqrt_dense(self).map_err(Into::into)
    }

    fn storage_rsqrt<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.rsqrt_dense(self).map_err(Into::into)
    }

    fn storage_erf<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.erf_dense(self).map_err(Into::into)
    }

    fn storage_erfc<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.erfc_dense(self).map_err(Into::into)
    }

    fn storage_erfinv<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.erfinv_dense(self).map_err(Into::into)
    }

    fn storage_atan2<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.atan2_dense(self, other).map_err(Into::into)
    }

    fn storage_log1p<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.log1p_dense(self).map_err(Into::into)
    }

    fn storage_expm1<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.expm1_dense(self).map_err(Into::into)
    }

    fn storage_reciprocal<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.reciprocal_dense(self).map_err(Into::into)
    }

    // ================== Comparison/Status Operations ==================

    fn storage_isnan<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float + num_traits::One + num_traits::Zero,
    {
        backend.isnan_dense(self).map_err(Into::into)
    }

    fn storage_isinf<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float + num_traits::One + num_traits::Zero,
    {
        backend.isinf_dense(self).map_err(Into::into)
    }

    fn storage_isfinite<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float + num_traits::One + num_traits::Zero,
    {
        backend.isfinite_dense(self).map_err(Into::into)
    }

    // ================== Logical Operations ==================

    fn storage_logical_and<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        backend.logical_and_dense(self, other).map_err(Into::into)
    }

    fn storage_logical_or<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        backend.logical_or_dense(self, other).map_err(Into::into)
    }

    fn storage_logical_xor<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        backend.logical_xor_dense(self, other).map_err(Into::into)
    }

    fn storage_logical_not<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        backend.logical_not_dense(self).map_err(Into::into)
    }

    // ================== Conversion ==================

    fn storage_to_dense(&self) -> Result<DenseStorage<T>>
    where
        T: num_traits::Zero + Clone,
    {
        Ok(self.clone())
    }
}

impl<T: DataType> super::traits::StorageBinaryOps<storage::CsrStorage<T>, T> for DenseStorage<T>
where
    T: core::ops::Add<Output = T>
        + core::ops::Sub<Output = T>
        + core::ops::Mul<Output = T>
        + core::ops::Div<Output = T>
        + num_traits::Zero
        + num_traits::One
        + Clone,
{
    type Output = DenseStorage<T>;

    fn storage_add_mixed<B: Backend<Data = T>>(&self, other: &storage::CsrStorage<T>, backend: &B) -> Result<Self::Output> {
        backend.add_dense_csr(self, other).map_err(Into::into)
    }

    fn storage_sub_mixed<B: Backend<Data = T>>(&self, other: &storage::CsrStorage<T>, backend: &B) -> Result<Self::Output> {
        // a - b = a + (-b)
        let mut neg_b = other.clone();
        for val in neg_b.data_mut() {
            *val = T::zero() - *val;
        }
        backend.add_dense_csr(self, &neg_b).map_err(Into::into)
    }

    fn storage_mul_mixed<B: Backend<Data = T>>(&self, other: &storage::CsrStorage<T>, backend: &B) -> Result<Self::Output> {
        backend.mul_dense_csr(self, other).map_err(Into::into)
    }

    fn storage_div_mixed<B: Backend<Data = T>>(&self, _other: &storage::CsrStorage<T>, _backend: &B) -> Result<Self::Output> {
        Err(TensorError::UnsupportedOperation {
            operation: "dense_div_sparse".to_string(),
            storage_type: "CsrStorage".to_string(),
        })
    }

    fn storage_matmul_mixed<B: Backend<Data = T>>(&self, other: &storage::CsrStorage<T>, backend: &B) -> Result<Self::Output> {
        backend.matmul_dense_csr(self, other).map_err(Into::into)
    }
}
