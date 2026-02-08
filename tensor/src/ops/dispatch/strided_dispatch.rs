//! StridedStorage implementation of TensorStorageOps
//!
//! Delegates operations to the `Backend` trait, utilizing zero-copy
//! strided primitives when available.

use super::traits::TensorStorageOps;
use crate::{Result, TensorError};
use backend::Backend;
use dtype::DataType;
use storage::{DenseStorage, Storage, StridedStorage};

impl<T: DataType> TensorStorageOps<T> for StridedStorage<T>
where
    T: core::ops::Add<Output = T>
        + core::ops::Sub<Output = T>
        + core::ops::Mul<Output = T>
        + core::ops::Div<Output = T>
        + num_traits::Zero
        + num_traits::One
        + Clone
        + Default,
{
    // ================== Arithmetic ==================

    fn storage_add<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        backend.add_strided(self, other).map_err(Into::into)
    }

    fn storage_sub<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        backend.sub_strided(self, other).map_err(Into::into)
    }

    fn storage_mul<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        backend.mul_strided(self, other).map_err(Into::into)
    }

    fn storage_div<B: Backend<Data = T>>(&self, _other: &Self, _backend: &B) -> Result<Self> {
        // Fallback to dense for now if not implemented in backend
        let dense_self = self.storage_to_dense()?;
        let dense_other = _other.storage_to_dense()?;
        let res = dense_self.storage_div(&dense_other, _backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_neg<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: core::ops::Neg<Output = T> {
        backend.neg_strided(self).map_err(Into::into)
    }

    // ================== Comparison ==================

    fn storage_eq<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        backend.eq_strided(self, other).map_err(Into::into)
    }

    fn storage_ne<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        backend.ne_strided(self, other).map_err(Into::into)
    }

    fn storage_gt<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + num_traits::One + num_traits::Zero,
    {
        backend.gt_strided(self, other).map_err(Into::into)
    }

    fn storage_ge<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + num_traits::One + num_traits::Zero,
    {
        backend.ge_strided(self, other).map_err(Into::into)
    }

    fn storage_lt<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + num_traits::One + num_traits::Zero,
    {
        backend.lt_strided(self, other).map_err(Into::into)
    }

    fn storage_le<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + num_traits::One + num_traits::Zero,
    {
        backend.le_strided(self, other).map_err(Into::into)
    }

    // ================== Linear Algebra ==================

    fn storage_matmul<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        let dense_self = self.storage_to_dense()?;
        let dense_other = other.storage_to_dense()?;
        let res = dense_self.storage_matmul(&dense_other, backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_transpose<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self> {
        self.transpose(None).map_err(TensorError::StorageError)
    }

    // ================== Activation Functions ==================

    fn storage_relu<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + Default,
    {
        backend.relu_strided(self).map_err(Into::into)
    }

    fn storage_sigmoid<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.sigmoid_strided(self).map_err(Into::into)
    }

    fn storage_tanh<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.tanh_strided(self).map_err(Into::into)
    }

    fn storage_gelu<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.gelu_strided(self).map_err(Into::into)
    }

    // ================== Reductions ==================

    fn storage_sum<B: Backend<Data = T>>(&self, backend: &B) -> Result<T> {
        backend.sum_strided(self).map_err(Into::into)
    }

    fn storage_mean<B: Backend<Data = T>>(&self, backend: &B) -> Result<T>
    where
        T: num_traits::FromPrimitive,
    {
        let dense = self.storage_to_dense()?;
        dense.storage_mean(backend)
    }

    fn storage_max<B: Backend<Data = T>>(&self, backend: &B) -> Result<T>
    where
        T: PartialOrd,
    {
        backend.max_strided(self).map_err(Into::into)
    }

    fn storage_min<B: Backend<Data = T>>(&self, backend: &B) -> Result<T>
    where
        T: PartialOrd,
    {
        backend.min_strided(self).map_err(Into::into)
    }

    // ================== Transcendentals ==================

    fn storage_exp<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.exp_strided(self).map_err(Into::into)
    }

    fn storage_log<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.log_strided(self).map_err(Into::into)
    }

    fn storage_sin<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_sin(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_cos<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_cos(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_abs<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Signed,
    {
        backend.abs_strided(self).map_err(Into::into)
    }

    fn storage_ceil<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_ceil(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_floor<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_floor(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_round<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_round(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_sqrt<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend.sqrt_strided(self).map_err(Into::into)
    }

    fn storage_rsqrt<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_rsqrt(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_erf<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_erf(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_erfc<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_erfc(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_erfinv<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_erfinv(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_atan2<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense_self = self.storage_to_dense()?;
        let dense_other = other.storage_to_dense()?;
        let res = dense_self.storage_atan2(&dense_other, backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_log1p<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_log1p(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_expm1<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_expm1(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_reciprocal<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_reciprocal(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    // ================== Comparison/Status ==================

    fn storage_isnan<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float + num_traits::One + num_traits::Zero,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_isnan(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_isinf<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float + num_traits::One + num_traits::Zero,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_isinf(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_isfinite<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float + num_traits::One + num_traits::Zero,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_isfinite(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    // ================== Logical ==================

    fn storage_logical_and<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        let dense_self = self.storage_to_dense()?;
        let dense_other = other.storage_to_dense()?;
        let res = dense_self.storage_logical_and(&dense_other, backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_logical_or<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        let dense_self = self.storage_to_dense()?;
        let dense_other = other.storage_to_dense()?;
        let res = dense_self.storage_logical_or(&dense_other, backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_logical_xor<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        let dense_self = self.storage_to_dense()?;
        let dense_other = other.storage_to_dense()?;
        let res = dense_self.storage_logical_xor(&dense_other, backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_logical_not<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::One + num_traits::Zero,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_logical_not(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    // ================== Conversion ==================

    fn storage_to_dense(&self) -> Result<DenseStorage<T>>
    where
        T: num_traits::Zero + Clone,
    {
        // For now using the existing to_dense which copies data
        // This is safe even if not perfectly optimized for all views yet
        Ok(self.to_dense())
    }
}
