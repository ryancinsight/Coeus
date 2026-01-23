//! Creation methods for Tensor
//!
//! This module provides functions for creating tensors from various data sources
//! and generating tensors with specific fill patterns.

use crate::{Backend, DataType, Storage, Tensor, TensorError};
use std::vec::Vec;
use storage;

/// Creation operations for tensors with any storage type.
impl<B, S, T> Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + storage::StorageFromVec<T>,
    T: DataType,
{
    /// Creates a tensor from a vector with specified shape.
    pub fn from_vec_with_backend(data: Vec<T>, dims: &[usize], backend: B) -> crate::Result<Self>
    where
        S: storage::StorageFromVec<T>,
    {
        let storage = S::from_vec(data, dims).map_err(TensorError::StorageError)?;
        Ok(Self::from_storage(storage, backend))
    }



    /// Creates a tensor from a vector with specified shape using default backend.
    pub fn from_vec(data: Vec<T>, dims: &[usize]) -> crate::Result<Self>
    where
        B: Default,
        S: storage::StorageFromVec<T>,
    {
        let storage = S::from_vec(data, dims).map_err(TensorError::StorageError)?;
        Ok(Self::from_storage(storage, B::default()))
    }

    /// Creates a tensor filled with zeros using any Storage implementation.
    pub fn zeros(dims: &[usize]) -> crate::Result<Self>
    where
        B: Default,
        S: storage::StorageFromVec<T>,
    {
        let size = dims.iter().product();
        let data = vec![T::zero(); size];
        let storage = S::from_vec(data, dims).map_err(TensorError::StorageError)?;
        Ok(Self::from_storage(storage, B::default()))
    }
}

/// Creation operations specifically for floating point tensors.
impl<B, S, T> Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + storage::StorageFromVec<T>,
    T: DataType + crate::FloatExt,
{
    /// Creates a tensor with random numbers from a standard normal distribution (mean=0, std=1).
    pub fn randn(dims: &[usize]) -> crate::Result<Self>
    where
        B: Default,
        S: storage::StorageFromVec<T>,
    {
        use rand::Rng;
        use rand_distr::StandardNormal;

        let mut rng = rand::thread_rng();
        let size: usize = dims.iter().product();
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            let sample: f64 = rng.sample(StandardNormal);
            let val = num_traits::NumCast::from(sample).ok_or_else(|| {
                TensorError::BackendError(format!(
                    "Failed to convert random sample {} to type {}",
                    sample,
                    T::name()
                ))
            })?;
            data.push(val);
        }

        Self::from_vec(data, dims)
    }

    /// Creates a tensor with random numbers from a uniform distribution [0, 1).
    pub fn rand(dims: &[usize]) -> crate::Result<Self>
    where
        B: Default,
        S: storage::StorageFromVec<T>,
    {
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let size: usize = dims.iter().product();
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            let sample: f64 = rng.gen_range(0.0..1.0);
            let val = num_traits::NumCast::from(sample).ok_or_else(|| {
                TensorError::BackendError(format!(
                    "Failed to convert random sample {} to type {}",
                    sample,
                    T::name()
                ))
            })?;
            data.push(val);
        }

        Self::from_vec(data, dims)
    }

    /// Creates a tensor with random integers from a discrete uniform distribution [low, high).
    pub fn randint(low: i64, high: i64, dims: &[usize]) -> crate::Result<Self>
    where
        B: Default,
        S: storage::StorageFromVec<T>,
    {
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let size: usize = dims.iter().product();
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            let sample: i64 = rng.gen_range(low..high);
            let val = num_traits::NumCast::from(sample).ok_or_else(|| {
                TensorError::BackendError(format!(
                    "Failed to convert random sample {} to type {}",
                    sample,
                    T::name()
                ))
            })?;
            data.push(val);
        }

        Self::from_vec(data, dims)
    }
}

impl<B, S, T> Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + storage::StorageFromVec<T>,
    T: DataType,
{
    /// Creates a tensor filled with a constant value.
    pub fn full(dims: &[usize], value: T) -> crate::Result<Self>
    where
        B: Default,
    {
        let size = dims.iter().product();
        let data = vec![value; size];
        let storage = S::from_vec(data, dims).map_err(TensorError::StorageError)?;
        Ok(Self::from_storage(storage, B::default()))
    }

    /// Creates an identity matrix of size n x m.
    pub fn eye(n: usize, m: usize) -> crate::Result<Self>
    where
        B: Default,
    {
        let mut data = vec![T::zero(); n * m];
        let diag_len = n.min(m);
        for i in 0..diag_len {
            data[i * m + i] = T::one();
        }
        let storage = S::from_vec(data, &[n, m]).map_err(TensorError::StorageError)?;
        Ok(Self::from_storage(storage, B::default()))
    }

    /// Creates a 1D tensor with values from start to end (exclusive) with given step.
    pub fn arange(start: T, end: T, step: T) -> crate::Result<Self>
    where
        B: Default,
    {
        if step.is_zero() {
            return Err(TensorError::InvalidOperation {
                operation: "arange",
                dtype: T::dtype(),
                reason: "step cannot be zero",
            });
        }

        let mut values = Vec::new();
        let mut current = start;

        let end_f = end.to_f64().unwrap();
        let step_f = step.to_f64().unwrap();

        if step_f > 0.0 {
            while current.to_f64().unwrap() < end_f {
                values.push(current);
                current = current + step;
            }
        } else {
            while current.to_f64().unwrap() > end_f {
                values.push(current);
                current = current + step;
            }
        }

        let len = values.len();
        let storage = S::from_vec(values, &[len]).map_err(TensorError::StorageError)?;
        Ok(Self::from_storage(storage, B::default()))
    }

    /// Creates a 1D tensor with linearly spaced values.
    pub fn linspace(start: T, end: T, steps: usize) -> crate::Result<Self>
    where
        B: Default,
    {
        if steps == 0 {
            return Err(TensorError::InvalidOperation {
                operation: "linspace",
                dtype: T::dtype(),
                reason: "steps must be positive",
            });
        }

        let mut values = Vec::with_capacity(steps);

        if steps == 1 {
            values.push(start);
        } else {
            let start_f = start.to_f64().unwrap();
            let end_f = end.to_f64().unwrap();
            let step_f = (end_f - start_f) / (steps - 1) as f64;

            for i in 0..steps {
                let val = start_f + step_f * i as f64;
                values.push(T::from(val).unwrap());
            }
        }

        let storage = S::from_vec(values, &[steps]).map_err(TensorError::StorageError)?;
        Ok(Self::from_storage(storage, B::default()))
    }

    /// Creates a tensor filled with ones using any Storage implementation.
    pub fn ones(dims: &[usize]) -> crate::Result<Self>
    where
        B: Default,
        T: num_traits::One,
    {
        let storage = S::ones(dims).map_err(TensorError::StorageError)?;
        Ok(Self::from_storage(storage, B::default()))
    }

    pub fn ones_with_backend(dims: &[usize], backend: B) -> crate::Result<Self>
    where
        T: num_traits::One,
    {
        let storage = S::ones(dims).map_err(TensorError::StorageError)?;
        Ok(Self::from_storage(storage, backend))
    }

    /// Creates a tensor filled with zeros using a specific backend.
    pub fn zeros_with_backend(dims: &[usize], backend: B) -> crate::Result<Self>
    where
        T: num_traits::Zero,
    {
        let storage = S::zeros(dims).map_err(TensorError::StorageError)?;
        Ok(Self::from_storage(storage, backend))
    }

    /// Creates a tensor filled with zeros with the same shape as the input tensor.
    pub fn zeros_like(tensor: &Self) -> crate::Result<Self>
    where
        S: Clone,
        T: num_traits::Zero,
    {
        let storage = S::zeros(tensor.shape().dims()).map_err(TensorError::StorageError)?;
        Ok(Self::from_storage(storage, tensor.backend.clone()))
    }

    /// Creates a tensor filled with ones with the same shape as the input tensor.
    pub fn ones_like(tensor: &Self) -> crate::Result<Self>
    where
        S: Clone,
        T: num_traits::One,
    {
        let storage = S::ones(tensor.shape().dims()).map_err(TensorError::StorageError)?;
        Ok(Self::from_storage(storage, tensor.backend.clone()))
    }

    /// Creates a tensor filled with a constant value with the same shape as the input tensor.
    pub fn full_like(tensor: &Self, value: T) -> crate::Result<Self>
    where
        B: Clone + Default,
        S: Clone,
    {
        let storage = S::full(tensor.shape().dims(), value).map_err(TensorError::StorageError)?;
        Ok(Self::from_storage(storage, tensor.backend.clone()))
    }
}

// Separate impl for DenseStorage to provide from_slice
impl<B, T> Tensor<B, storage::DenseStorage<T>, T>
where
    B: Backend<Data = T>,
    T: DataType,
{
    /// Creates a tensor from a slice with specified shape.
    pub fn from_slice(data: &[T], dims: &[usize]) -> crate::Result<Self>
    where
        B: Default,
    {
        let storage =
            storage::DenseStorage::from_slice(data, dims).map_err(TensorError::StorageError)?;
        Ok(Self::from_storage(storage, B::default()))
    }

    /// Creates a tensor from a slice with specified shape and backend.
    pub fn from_slice_with_backend(data: &[T], dims: &[usize], backend: B) -> crate::Result<Self> {
        let storage =
            storage::DenseStorage::from_slice(data, dims).map_err(TensorError::StorageError)?;
        Ok(Self::from_storage(storage, backend))
    }
}

// Sparse creation methods - using optimal CSR format
impl<B, T> Tensor<B, crate::CsrStorage<T>, T>
where
    B: Backend<Data = T>,
    T: DataType,
{
    /// Creates a CSR tensor from raw components.
    pub fn from_csr(
        data: Vec<T>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        dims: &[usize],
        backend: B,
    ) -> crate::Result<Self> {
        let storage = crate::CsrStorage::new(data, indices, indptr, dims)
            .map_err(TensorError::StorageError)?;
        Ok(Self::from_storage(storage, backend))
    }

    /// Creates a sparse identity matrix in CSR format
    pub fn sparse_eye(size: usize, backend: B) -> crate::Result<Self>
    where
        T: num_traits::One,
    {
        let storage = crate::CsrStorage::eye(size)
            .map_err(TensorError::StorageError)?;
        Ok(Self::from_storage(storage, backend))
    }

    /// Creates an empty sparse matrix in CSR format
    pub fn sparse_empty(dims: &[usize], backend: B) -> crate::Result<Self> {
        let storage = crate::CsrStorage::empty(dims)
            .map_err(TensorError::StorageError)?;
        Ok(Self::from_storage(storage, backend))
    }

    /// Creates a CSR tensor from a dense tensor, keeping only non-zero elements
    pub fn from_dense_sparse(dense_tensor: &Tensor<B, crate::DenseStorage<T>, T>) -> crate::Result<Self>
    where
        T: num_traits::Zero + PartialEq,
        B: Clone,
    {
        let storage = crate::CsrStorage::from_dense(&dense_tensor.storage)
            .map_err(TensorError::StorageError)?;
        Ok(Self::from_storage(storage, dense_tensor.backend.clone()))
    }
}
