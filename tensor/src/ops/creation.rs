//! Tensor creation operations.
//!
//! This module provides functions for creating tensors from various data sources
//! and generating tensors with specific fill patterns.

use std::vec::Vec;

/// Creation operations for tensors with any storage type.
///
/// This trait provides methods for creating tensors from vectors, slices,
/// and generating tensors with specific fill patterns.
impl<B, S, T> crate::Tensor<B, S, T>
where
    B: crate::Backend<Data = T> + Clone,
    S: crate::Storage<T> + Clone + crate::StorageFromVec<T>,
    T: crate::DataType,
{
    /// Creates a tensor from a vector with specified shape.
    ///
    /// # Arguments
    /// * `data` - Vector of tensor data
    /// * `dims` - Shape dimensions
    /// * `backend` - Backend instance to use
    ///
    /// # Errors
    ///
    /// Returns error if data size doesn't match shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::int::Int32;
    ///
    /// let data = vec![Int32::new(1), Int32::new(2), Int32::new(3)];
    /// let backend = CpuBackend::new();
    /// let tensor = Tensor::<CpuBackend<Int32>, DenseStorage<Int32>, Int32>::from_vec_with_backend(data, &[3], backend).unwrap();
    /// assert_eq!(tensor.len(), 3);
    /// ```
    pub fn from_vec_with_backend(data: Vec<T>, dims: &[usize], backend: B) -> crate::Result<Self>
    where
        S: storage::StorageFromVec<T>,
    {
        let storage = S::from_vec(data, dims).map_err(crate::TensorError::StorageError)?;
        Ok(Self::from_storage(storage, backend))
    }

    /// Creates a tensor from a vector with specified shape using default backend.
    ///
    /// # Errors
    ///
    /// Returns error if data size doesn't match shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::int::Int32;
    ///
    /// let data = vec![Int32::new(1), Int32::new(2), Int32::new(3)];
    /// let tensor = Tensor::<CpuBackend<Int32>, DenseStorage<Int32>, Int32>::from_vec(data, &[3]).unwrap();
    /// assert_eq!(tensor.len(), 3);
    /// ```
    pub fn from_vec(data: Vec<T>, dims: &[usize]) -> crate::Result<Self>
    where
        B: Default,
        S: storage::StorageFromVec<T>,
    {
        let storage = S::from_vec(data, dims).map_err(crate::TensorError::StorageError)?;
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
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::float::Float32;
    /// use num_traits::Zero;
    ///
    /// let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2, 3]).unwrap();
    /// assert_eq!(tensor.len(), 6);
    /// assert!(tensor.as_slice().iter().all(|&x| x.is_zero()));
    /// ```
    pub fn zeros(dims: &[usize]) -> crate::Result<Self>
    where
        B: Default,
        S: storage::StorageFromVec<T>,
    {
        let size = dims.iter().product();
        let data = vec![T::zero(); size];
        let storage = S::from_vec(data, dims).map_err(crate::TensorError::StorageError)?;
        Ok(Self::from_storage(storage, B::default()))
    }
}

/// Creation operations specifically for floating point tensors.
impl<B, S, T> crate::Tensor<B, S, T>
where
    B: crate::Backend<Data = T> + Clone,
    S: crate::Storage<T> + Clone + crate::StorageFromVec<T>,
    T: crate::DataType + dtype::traits::FloatExt,
{
    /// Creates a tensor with random numbers from a standard normal distribution (mean=0, std=1).
    ///
    /// # Arguments
    /// * `dims` - Shape dimensions
    ///
    /// # Errors
    ///
    /// Returns error if storage creation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::float::Float32;
    ///
    /// let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::randn(&[2, 3]).unwrap();
    /// assert_eq!(tensor.len(), 6);
    /// ```
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
            // Convert f64 to T. Since T is FloatExt, this should be safe for representable values.
            let val = num_traits::NumCast::from(sample).ok_or_else(|| {
                crate::TensorError::BackendError(format!(
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
                crate::TensorError::BackendError(format!(
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
                crate::TensorError::BackendError(format!(
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

impl<B, S, T> crate::Tensor<B, S, T>
where
    B: crate::Backend<Data = T> + Clone,
    S: crate::Storage<T> + Clone + crate::StorageFromVec<T>,
    T: crate::DataType,
{
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
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::int::Int64;
    /// use num_traits::One;
    ///
    /// let tensor = Tensor::<CpuBackend<Int64>, DenseStorage<Int64>, Int64>::ones(&[4]).unwrap();
    /// assert_eq!(tensor.len(), 4);
    /// assert!(tensor.as_slice().iter().all(|&x| x.is_one()));
    /// ```
    pub fn ones(dims: &[usize]) -> crate::Result<Self>
    where
        B: Default,
        T: num_traits::One,
    {
        let storage = S::ones(dims).map_err(crate::TensorError::StorageError)?;
        Ok(Self::from_storage(storage, B::default()))
    }

    pub fn ones_with_backend(dims: &[usize], backend: B) -> crate::Result<Self>
    where
        T: num_traits::One,
    {
        let storage = S::ones(dims).map_err(crate::TensorError::StorageError)?;
        Ok(Self::from_storage(storage, backend))
    }

    /// Creates a tensor filled with zeros with the same shape as the input tensor.
    ///
    /// # Arguments
    /// * `tensor` - Reference tensor to copy shape from
    ///
    /// # Errors
    ///
    /// Returns error if storage creation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::float::Float32;
    ///
    /// let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_slice(&[Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();
    /// let zeros = Tensor::zeros_like(&a).unwrap();
    /// assert_eq!(zeros.shape().dims(), a.shape().dims());
    /// ```
    pub fn zeros_like(tensor: &Self) -> crate::Result<Self>
    where
        S: Clone,
        T: num_traits::Zero,
    {
        let storage = S::zeros(tensor.shape().dims()).map_err(crate::TensorError::StorageError)?;
        Ok(Self::from_storage(storage, tensor.backend().clone()))
    }

    /// Creates a tensor filled with ones with the same shape as the input tensor.
    ///
    /// # Arguments
    /// * `tensor` - Reference tensor to copy shape from
    ///
    /// # Errors
    ///
    /// Returns error if storage creation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::float::Float32;
    ///
    /// let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_slice(&[Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();
    /// let ones = Tensor::ones_like(&a).unwrap();
    /// assert_eq!(ones.shape().dims(), a.shape().dims());
    /// ```
    pub fn ones_like(tensor: &Self) -> crate::Result<Self>
    where
        S: Clone,
        T: num_traits::One,
    {
        let storage = S::ones(tensor.shape().dims()).map_err(crate::TensorError::StorageError)?;
        Ok(Self::from_storage(storage, tensor.backend().clone()))
    }

    /// Creates a tensor filled with a constant value with the same shape as the input tensor.
    ///
    /// # Arguments
    /// * `tensor` - Reference tensor to copy shape from
    /// * `value` - Value to fill the tensor with
    ///
    /// # Errors
    ///
    /// Returns error if storage creation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::float::Float32;
    ///
    /// let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_slice(&[Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();
    /// let full = Tensor::full_like(&a, Float32::new(5.0)).unwrap();
    /// assert_eq!(full.shape().dims(), a.shape().dims());
    /// ```
    pub fn full_like(tensor: &Self, value: T) -> crate::Result<Self>
    where
        B: Clone + Default,
        S: Clone,
    {
        let storage =
            S::full(tensor.shape().dims(), value).map_err(crate::TensorError::StorageError)?;
        Ok(Self::from_storage(storage, tensor.backend().clone()))
    }
}
// Separate impl for DenseStorage to provide from_slice
impl<B, T> crate::Tensor<B, storage::DenseStorage<T>, T>
where
    B: crate::Backend<Data = T>,
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
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::float::Float64;
    ///
    /// let data = [Float64::new(1.0), Float64::new(2.0), Float64::new(3.0), Float64::new(4.0)];
    /// let tensor = Tensor::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::from_slice(&data, &[2, 2]).unwrap();
    /// assert_eq!(tensor.shape().dims(), &[2, 2]);
    /// ```
    pub fn from_slice(data: &[T], dims: &[usize]) -> crate::Result<Self>
    where
        B: Default,
    {
        let storage = storage::DenseStorage::from_slice(data, dims)
            .map_err(crate::TensorError::StorageError)?;
        Ok(Self::from_storage(storage, B::default()))
    }

    /// Creates a tensor from a slice with specified shape and backend.
    ///
    /// # Arguments
    /// * `data` - Slice of tensor data
    /// * `dims` - Shape dimensions
    /// * `backend` - Backend instance to use
    ///
    /// # Errors
    ///
    /// Returns error if slice size doesn't match shape.
    pub fn from_slice_with_backend(data: &[T], dims: &[usize], backend: B) -> crate::Result<Self> {
        let storage = storage::DenseStorage::from_slice(data, dims)
            .map_err(crate::TensorError::StorageError)?;
        Ok(Self::from_storage(storage, backend))
    }
}

/// Convenience operations for common tensor creation.
/// These are not generic and return concrete tensor types for simplified usage.
pub use tensor_creation_convenience::*;

#[allow(missing_docs)]
mod tensor_creation_convenience {
    use super::*;
    use rand::prelude::*;
    use std::sync::Mutex;

    /// Type alias for the most common CPU float32 tensor type.
    pub type CpuF32Tensor = crate::Tensor<
        crate::CpuBackend<dtype::float::Float32>,
        crate::DenseStorage<dtype::float::Float32>,
        dtype::float::Float32,
    >;

    /// Creates a tensor filled with random values from a normal distribution.
    ///
    /// This is a convenience method for CPU + DenseStorage + Float32 tensors.
    ///
    /// # Arguments
    /// * `shape` - Shape dimensions for the tensor
    ///
    /// # Returns
    /// A CPU Float32 tensor with normally distributed random values
    ///
    /// # Errors
    /// Returns error if tensor creation fails
    ///
    /// # Note
    /// This method is a convenience function that uses CPU backend with Float32 data type.
    /// For full control over backend/storage types, use the generic constructor methods.
    pub fn randn(shape: &[usize]) -> crate::Result<CpuF32Tensor> {
        // Global RNG for deterministic random number generation
        static RNG: Mutex<Option<rand::rngs::StdRng>> = Mutex::new(None);

        let mut rng_lock = RNG.lock().unwrap();
        let rng = rng_lock.get_or_insert_with(rand::rngs::StdRng::from_entropy);

        let total_elements: usize = shape.iter().product();
        let mut data = Vec::with_capacity(total_elements);

        // Generate random values from standard normal distribution
        for _ in 0..total_elements {
            let value: f32 = rng.sample(rand::distributions::Standard);
            data.push(dtype::float::Float32::new(value));
        }

        CpuF32Tensor::from_vec(data, shape)
    }

    /// Concatenates tensors along a specified dimension.
    ///
    /// # Arguments
    /// * `tensors` - Slice of tensors to concatenate
    /// * `dim` - Dimension along which to concatenate
    ///
    /// # Returns
    /// A new tensor with the concatenated result
    ///
    /// # Errors
    /// Returns error if concatenation fails
    ///
    /// # Note
    /// All input tensors must have the same shape except for the concatenation dimension.
    pub fn cat(tensors: &[CpuF32Tensor], dim: usize) -> crate::Result<CpuF32Tensor> {
        if tensors.is_empty() {
            return Err(crate::TensorError::ShapeError {
                expected: 0,
                actual: 0,
                message: "Cannot concatenate empty tensor list".to_string(),
            });
        }

        // Check all tensors have compatible shapes
        let first_shape = tensors[0].shape().dims();
        if dim >= first_shape.len() {
            return Err(crate::TensorError::ShapeError {
                expected: 0,
                actual: dim,
                message: format!(
                    "Dimension {} out of bounds for tensor with {} dimensions",
                    dim,
                    first_shape.len()
                ),
            });
        }

        // Verify all tensors have compatible shapes (same size in all dimensions except dim)
        for (i, tensor) in tensors.iter().enumerate() {
            let shape = tensor.shape().dims();
            if shape.len() != first_shape.len() {
                return Err(crate::TensorError::ShapeError {
                    expected: first_shape.len(),
                    actual: shape.len(),
                    message: format!(
                        "Tensor {} has {} dimensions, expected {}",
                        i,
                        shape.len(),
                        first_shape.len()
                    ),
                });
            }

            for (j, (&actual, &expected)) in shape.iter().zip(first_shape).enumerate() {
                if j != dim && actual != expected {
                    return Err(crate::TensorError::ShapeError {
                        expected,
                        actual,
                        message: format!(
                            "Tensor {} dimension {} has size {}, expected {}",
                            i, j, actual, expected
                        ),
                    });
                }
            }
        }

        // Calculate output shape
        let mut output_shape = first_shape.to_vec();
        let total_dim_size: usize = tensors.iter().map(|t| t.shape().dims()[dim]).sum();
        output_shape[dim] = total_dim_size;

        // Calculate total number of elements
        let total_elements: usize = output_shape.iter().product();

        // Concatenate the data
        let mut concatenated_data = vec![dtype::float::Float32::default(); total_elements];

        let mut offsets = vec![0; output_shape.len()];
        for tensor in tensors {
            let tensor_shape = tensor.shape().dims();
            let tensor_size = tensor_shape.iter().product::<usize>();

            // Copy this tensor's data with proper index calculation
            for linear_idx in 0..tensor_size {
                // Convert linear index to multi-dimensional coordinates
                let mut coords = vec![0; tensor_shape.len()];
                let mut remaining = linear_idx;
                for (i, &dim_size) in tensor_shape.iter().enumerate().rev() {
                    coords[i] = remaining % dim_size;
                    remaining /= dim_size;
                }

                // Apply offset for concatenation dimension
                coords[dim] += offsets[dim];

                // Convert back to linear index in output tensor
                let mut output_linear_idx = 0;
                let mut multiplier = 1;
                for (i, &coord) in coords.iter().enumerate().rev() {
                    output_linear_idx += coord * multiplier;
                    multiplier *= output_shape[i];
                }

                // Copy the element
                concatenated_data[output_linear_idx] = tensor.as_slice()[linear_idx];
            }

            // Update offset for next tensor
            offsets[dim] += tensor_shape[dim];
        }

        CpuF32Tensor::from_vec(concatenated_data, &output_shape)
    }
}

/// Generic creation operations for tensors with any StorageFromVec storage.
///
///
impl<B, S, T> crate::Tensor<B, S, T>
where
    B: crate::Backend<Data = T>,
    S: crate::Storage<T> + Clone + crate::StorageFromVec<T> + 'static,
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
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::float::Float32;
    /// use num_traits::Zero;
    ///
    /// let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros_generic(&[2, 3]).unwrap();
    /// assert_eq!(tensor.len(), 6);
    /// assert!(tensor.as_slice().iter().all(|&x| x.is_zero()));
    /// ```
    pub fn zeros_generic(dims: &[usize]) -> crate::Result<Self>
    where
        T: num_traits::Zero,
    {
        let storage = S::zeros(dims).map_err(crate::TensorError::StorageError)?;
        Ok(Self::from_storage(storage, B::default()))
    }

    pub fn zeros_generic_with_backend(dims: &[usize], backend: B) -> crate::Result<Self>
    where
        T: num_traits::Zero,
    {
        let storage = S::zeros(dims).map_err(crate::TensorError::StorageError)?;
        Ok(Self::from_storage(storage, backend))
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
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::int::Int64;
    /// use num_traits::One;
    ///
    /// let tensor = Tensor::<CpuBackend<Int64>, DenseStorage<Int64>, Int64>::ones_generic(&[4]).unwrap();
    /// assert_eq!(tensor.len(), 4);
    /// assert!(tensor.as_slice().iter().all(|&x| x.is_one()));
    /// ```
    pub fn ones_generic(dims: &[usize]) -> crate::Result<Self>
    where
        T: num_traits::One,
    {
        let storage = S::ones(dims).map_err(crate::TensorError::StorageError)?;
        Ok(Self::from_storage(storage, B::default()))
    }
}
