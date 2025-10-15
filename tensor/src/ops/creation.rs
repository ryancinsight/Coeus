//! Tensor creation operations.
//!
//! This module provides functions for creating tensors from various data sources
//! and generating tensors with specific fill patterns.

use std::vec::Vec;

/// Creation operations for tensors with any storage type.
///
/// This trait provides methods for creating tensors from vectors, slices,
/// and generating tensors filled with zeros or ones.
impl<B, S, T> crate::Tensor<B, S, T>
where
    B: crate::Backend + Default,
    S: crate::Storage<T> + crate::StorageFromVec<T> + 'static,
    T: crate::DataType,
{
    /// Creates a tensor from a vector with specified shape.
    ///
    /// Uses default backend instance.
    ///
    /// # Errors
    ///
    /// Returns error if data size doesn't match shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::int::Int32;
    ///
    /// let data = vec![Int32::new(1), Int32::new(2), Int32::new(3)];
    /// let tensor = Tensor::<CpuBackend, DenseStorage<Int32>, Int32>::from_vec(data, &[3]).unwrap();
    /// assert_eq!(tensor.len(), 3);
    /// ```
    pub fn from_vec(data: Vec<T>, dims: &[usize]) -> crate::Result<Self>
    where
        S: coeus_storage::StorageFromVec<T>,
    {
        let storage = S::from_vec(data, dims)
            .map_err(crate::TensorError::StorageError)?;
        Ok(Self::from_storage(storage, B::default()))
    }

    /// Creates a tensor filled with zeros using any Storage implementation.
    ///
    /// This provides a generic way to create zero-filled tensors that works
    /// with any storage type that implements the Storage trait.
    ///
    /// # Errors
    ///
    /// Returns error if shape specification is invalid or storage creation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::float::Float32;
    /// use num_traits::Zero;
    ///
    /// let tensor = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[2, 3]).unwrap();
    /// assert_eq!(tensor.len(), 6);
    /// assert!(tensor.as_slice().iter().all(|&x| x.is_zero()));
    /// ```
    pub fn zeros(dims: &[usize]) -> crate::Result<Self>
    where
        T: num_traits::Zero,
    {
        let storage = S::zeros(dims)
            .map_err(crate::TensorError::StorageError)?;
        Ok(Self::from_storage(storage, B::default()))
    }

    /// Creates a tensor filled with ones using any Storage implementation.
    ///
    /// This provides a generic way to create one-filled tensors that works
    /// with any storage type that implements the Storage trait.
    ///
    /// # Errors
    ///
    /// Returns error if shape specification is invalid or storage creation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::int::Int64;
    /// use num_traits::One;
    ///
    /// let tensor = Tensor::<CpuBackend, DenseStorage<Int64>, Int64>::ones(&[4]).unwrap();
    /// assert_eq!(tensor.len(), 4);
    /// assert!(tensor.as_slice().iter().all(|&x| x.is_one()));
    /// ```
    pub fn ones(dims: &[usize]) -> crate::Result<Self>
    where
        T: num_traits::One,
    {
        let storage = S::ones(dims)
            .map_err(crate::TensorError::StorageError)?;
        Ok(Self::from_storage(storage, B::default()))
    }

}
// Separate impl for DenseStorage to provide from_slice
impl<B, T> crate::Tensor<B, coeus_storage::DenseStorage<T>, T>
where
    B: crate::Backend + Default,
    T: crate::DataType,
{
    /// Creates a tensor from a slice with specified shape.
    ///
    /// # Errors
    ///
    /// Returns error if slice size doesn't match shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::float::Float64;
    ///
    /// let data = [Float64::new(1.0), Float64::new(2.0), Float64::new(3.0), Float64::new(4.0)];
    /// let tensor = Tensor::<CpuBackend, DenseStorage<Float64>, Float64>::from_slice(&data, &[2, 2]).unwrap();
    /// assert_eq!(tensor.shape().dims(), &[2, 2]);
    /// ```
    pub fn from_slice(data: &[T], dims: &[usize]) -> crate::Result<Self> {
        let storage = coeus_storage::DenseStorage::from_slice(data, dims)
            .map_err(crate::TensorError::StorageError)?;
        Ok(Self::from_storage(storage, B::default()))
    }

}

/// Generic creation operations for tensors with any StorageFromVec storage.
///
///
impl<B, S, T> crate::Tensor<B, S, T>
where
    B: crate::Backend + Default,
    S: crate::Storage<T> + crate::StorageFromVec<T> + 'static,
    T: crate::DataType,
{
    /// Creates a tensor filled with zeros using any StorageFromVec implementation.
    ///
    /// This provides a generic way to create zero-filled tensors that works
    /// with any storage type that implements StorageFromVec.
    ///
    /// # Errors
    ///
    /// Returns error if shape specification is invalid or storage creation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::float::Float32;
    /// use num_traits::Zero;
    ///
    /// let tensor = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros_generic(&[2, 3]).unwrap();
    /// assert_eq!(tensor.len(), 6);
    /// assert!(tensor.as_slice().iter().all(|&x| x.is_zero()));
    /// ```
    pub fn zeros_generic(dims: &[usize]) -> crate::Result<Self>
    where
        T: num_traits::Zero,
    {
        let storage = S::zeros(dims)
            .map_err(crate::TensorError::StorageError)?;
        Ok(Self::from_storage(storage, B::default()))
    }

    /// Creates a tensor filled with ones using any StorageFromVec implementation.
    ///
    /// This provides a generic way to create one-filled tensors that works
    /// with any storage type that implements StorageFromVec.
    ///
    /// # Errors
    ///
    /// Returns error if shape specification is invalid or storage creation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::int::Int64;
    /// use num_traits::One;
    ///
    /// let tensor = Tensor::<CpuBackend, DenseStorage<Int64>, Int64>::ones_generic(&[4]).unwrap();
    /// assert_eq!(tensor.len(), 4);
    /// assert!(tensor.as_slice().iter().all(|&x| x.is_one()));
    /// ```
    pub fn ones_generic(dims: &[usize]) -> crate::Result<Self>
    where
        T: num_traits::One,
    {
        let storage = S::ones(dims)
            .map_err(crate::TensorError::StorageError)?;
        Ok(Self::from_storage(storage, B::default()))
    }
}
