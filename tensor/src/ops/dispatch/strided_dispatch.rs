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
        + core::ops::Neg<Output = T>
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

    fn storage_neg<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self> {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_neg(_backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
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
        let dense = self.storage_to_dense()?;
        let res = dense.storage_relu(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_sigmoid<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_sigmoid(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_tanh<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_tanh(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_gelu<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_gelu(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    // ================== Reductions ==================

    fn storage_sum<B: Backend<Data = T>>(&self, backend: &B) -> Result<T> {
        let dense = self.storage_to_dense()?;
        dense.storage_sum(backend)
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
        let dense = self.storage_to_dense()?;
        dense.storage_max(backend)
    }

    fn storage_min<B: Backend<Data = T>>(&self, backend: &B) -> Result<T>
    where
        T: PartialOrd,
    {
        let dense = self.storage_to_dense()?;
        dense.storage_min(backend)
    }

    // ================== Transcendentals ==================

    fn storage_exp<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_exp(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
    }

    fn storage_log<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        let dense = self.storage_to_dense()?;
        let res = dense.storage_log(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
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
        let dense = self.storage_to_dense()?;
        let res = dense.storage_abs(backend)?;
        Ok(StridedStorage::new(res.as_slice().to_vec(), res.shape().dims())
            .map_err(TensorError::StorageError)?)
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
