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
        + core::ops::Neg<Output = T>
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

    fn storage_neg<B: Backend<Data = T>>(&self, _backend: &B) -> Result<Self> {
        let mut result = self.clone();
        for value in result.as_mut_slice() {
            *value = -*value;
        }
        Ok(result)
    }

    // ================== Linear Algebra ==================

    fn storage_matmul<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self> {
        backend
            .matmul_dense(self, other)
            .map_err(Into::into)
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

    // ================== Activation Functions ==================

    fn storage_relu<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: PartialOrd + Default,
    {
        backend
            .relu_dense(self)
            .map_err(Into::into)
    }

    fn storage_sigmoid<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend
            .sigmoid_dense(self)
            .map_err(Into::into)
    }

    fn storage_tanh<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend
            .tanh_dense(self)
            .map_err(Into::into)
    }

    fn storage_gelu<B: Backend<Data = T>>(&self, backend: &B) -> Result<Self>
    where
        T: num_traits::Float,
    {
        backend
            .gelu_dense(self)
            .map_err(Into::into)
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

    // ================== Conversion ==================

    fn storage_to_dense(&self) -> Result<DenseStorage<T>>
    where
        T: num_traits::Zero + Clone,
    {
        Ok(self.clone())
    }
}
