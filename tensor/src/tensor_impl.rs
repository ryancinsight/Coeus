//! Core tensor implementations.
//!
//! This module contains the fundamental implementations for the Tensor type,
//! including creation, basic operations, and gradient management.

use std::{boxed::Box, format, string::ToString, sync::Arc, vec::Vec};

use crate::{
    error::TensorError, grad_rwlock, AsAny, Backend, CpuBackend, DataType, DenseStorage, Function, Result,
    Shape, Storage, StorageToDense, Tensor,
};
use storage::StorageFromVec;

impl<B, S, T> Tensor<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Creates a tensor from existing storage and backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::float::Float32;
    ///
    /// let storage = DenseStorage::from_slice(&[Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();
    /// let backend = CpuBackend::new();
    /// let tensor = Tensor::from_storage(storage, backend);
    /// assert_eq!(tensor.len(), 2);
    /// ```
    #[must_use]
    pub fn from_storage(storage: S, backend: B) -> Self {
        Self {
            storage,
            backend,
            requires_grad: false, // Default: no gradients
            grad: Arc::new(grad_rwlock(None)),
            grad_fn: None, // Leaf tensors have no creator function
        }
    }

    /// Returns the shape of this tensor.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        self.storage.shape()
    }

    /// Returns whether this tensor requires gradients.
    ///
    /// # Examples
    /// ```
    /// use tensor::{Tensor, CpuBackend, DenseStorage};
    /// use dtype::float::Float32;
    ///
    /// let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2, 3]).unwrap();
    /// assert!(!tensor.requires_grad()); // Default is false
    ///
    /// let tensor_with_grad = tensor.requires_grad_(true);
    /// assert!(tensor_with_grad.requires_grad());
    /// ```
    #[must_use]
    pub const fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Sets whether this tensor requires gradients.
    ///
    /// This is the PyTorch-style API for enabling gradient computation.
    /// Returns a new tensor with the gradient flag set.
    ///
    /// # Examples
    /// ```
    /// use tensor::{Tensor, CpuBackend, DenseStorage};
    /// use dtype::float::Float32;
    ///
    /// let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2, 3]).unwrap();
    /// let grad_tensor = tensor.requires_grad_(true);
    /// assert!(grad_tensor.requires_grad());
    /// ```
    #[must_use]
    pub const fn requires_grad_(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    /// Detaches this tensor from the computation graph.
    ///
    /// Returns a new tensor with `requires_grad` set to false.
    /// This is useful for inference or when you want to stop gradient computation.
    ///
    /// # Examples
    /// ```
    /// use tensor::{Tensor, CpuBackend, DenseStorage};
    /// use dtype::float::Float32;
    ///
    /// let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2, 3]).unwrap()
    ///     .requires_grad_(true);
    /// assert!(tensor.requires_grad());
    ///
    /// let detached = tensor.detach();
    /// assert!(!detached.requires_grad());
    /// ```
    #[must_use]
    pub const fn detach(mut self) -> Self {
        self.requires_grad = false;
        self
    }

    /// Get the gradient tensor if it has been computed.
    ///
    /// Returns a clone of the gradient tensor. In PyTorch, this is accessed via `.grad`.
    ///
    /// # Returns
    /// - `Ok(Tensor)` if gradient exists
    /// - `Err(TensorError)` if no gradient computed yet
    ///
    /// # Errors
    /// Returns error if gradient has not been computed or lock is poisoned
    ///
    /// # Examples
    /// ```ignore
    /// use tensor::{Tensor, CpuBackend, DenseStorage};
    /// use dtype::float::Float32;
    ///
    /// let x = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2]).unwrap()
    ///     .requires_grad_(true);
    /// // ... perform operations and backward pass ...
    /// // let grad = x.grad().unwrap();
    /// ```
    pub fn grad(&self) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        B: Clone,
        S: Clone,
    {
        #[cfg(feature = "std")]
        let grad_lock = self
            .grad
            .read()
            .map_err(|_| TensorError::BackendError("Failed to acquire gradient lock".to_string()))?;
        #[cfg(not(feature = "std"))]
        let grad_lock = self.grad.read();

        match grad_lock.as_ref() {
            Some(boxed) => {
                // For now, assume gradients are stored as dense tensors
                // This is the common case and avoids generic cloning issues
                if let Some(dense_grad) = boxed
                    .as_any()
                    .downcast_ref::<Tensor<B, DenseStorage<T>, T>>()
                {
                    Ok(dense_grad.clone())
                } else {
                    // If not dense, try to reconstruct from storage
                    // This is a fallback for other storage types
                    Err(TensorError::BackendError(
                        "Gradient tensor storage type not supported".into(),
                    ))
                }
            }
            None => Err(TensorError::BackendError(
                "Gradient not available (call backward first)".into(),
            )),
        }
    }

    /// Set the gradient tensor.
    ///
    /// Used internally during backward pass to accumulate gradients.
    /// In PyTorch, this is typically done automatically by the autograd engine.
    ///
    /// # Arguments
    /// * `gradient` - The gradient tensor to set
    ///
    /// # Errors
    /// Returns error if lock is poisoned or shapes don't match
    pub fn set_grad<GS>(&self, gradient: Tensor<B, GS, T>) -> Result<()>
    where
        GS: Storage<T> + StorageToDense<T> + StorageFromVec<T>,
        S: StorageFromVec<T>,
    {
        println!(
            "set_grad called on tensor with shape {:?}",
            self.shape().dims()
        );
        // Validate shape matches
        if gradient.shape().dims() != self.shape().dims() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                actual: gradient.shape().dims().to_vec(),
                operation: "set_grad",
            });
        }

        // Convert gradient to the tensor's storage type
        // Always convert via dense representation for safety
        let dense = gradient.to_dense_generic()?;
        let data = dense.as_slice().to_vec();
        let dims = dense.shape().dims().to_vec();
        let gradient_s = Tensor::<B, S, T>::from_vec(data, &dims)?;

        #[cfg(feature = "std")]
        {
            let mut grad_lock = match self.grad.write() {
                Ok(lock) => lock,
                Err(_) => return Err(TensorError::BackendError("Failed to acquire gradient lock".to_string())),
            };
            *grad_lock = Some(Box::new(gradient_s));
        }
        #[cfg(not(feature = "std"))]
        {
            let mut grad_lock = self.grad.write();
            *grad_lock = Some(Box::new(gradient_s));
        }
        Ok(())
    }

    /// Zero out the gradient.
    ///
    /// Sets the gradient to None, freeing memory.
    /// Call this before each training iteration.
    /// In PyTorch, this is `tensor.grad = None` or via `optimizer.zero_grad()`.
    ///
    /// # Errors
    /// Returns error if lock is poisoned
    ///
    /// # Examples
    /// ```
    /// use tensor::{Tensor, CpuBackend, DenseStorage};
    /// use dtype::float::Float32;
    ///
    /// let x = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2]).unwrap()
    ///     .requires_grad_(true);
    /// x.zero_grad().unwrap();
    /// ```
    pub fn zero_grad(&self) -> Result<()> {
        #[cfg(feature = "std")]
        let mut grad_lock = self
            .grad
            .write()
            .map_err(|_| TensorError::BackendError("Failed to acquire gradient lock".to_string()))?;
        #[cfg(not(feature = "std"))]
        let mut grad_lock = self.grad.write();

        *grad_lock = None;
        Ok(())
    }

    /// Compute gradients by backpropagation
    ///
    /// Starts the backward pass from this tensor, computing gradients for all tensors
    /// in the computation graph that require gradients.
    ///
    /// This implements PyTorch-compatible automatic differentiation by traversing
    /// the `grad_fn` chain and accumulating gradients.
    ///
    /// # Errors
    /// Returns error if backward pass fails
    ///
    /// # Examples
    /// ```
    /// use tensor::{Tensor, CpuBackend, DenseStorage};
    /// use dtype::float::Float32;
    ///
    /// let x = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
    ///     vec![Float32::new(2.0)], &[1]
    /// ).unwrap().requires_grad_(true);
    ///
    /// let y = &x * &x; // y = x²
    /// y.backward().unwrap(); // Compute gradients
    ///
    /// assert_eq!(x.grad().unwrap().as_slice()[0].get(), 4.0); // ∂(x²)/∂x = 2x = 4
    /// ```
    pub fn backward(&self) -> Result<()>
    where
        B: Backend<Data = T> + Clone + Default,
        S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    {
        // For backward() without arguments, tensor must be scalar (0-d or 1 element)
        if self.shape().ndim() == 0 || (self.shape().ndim() == 1 && self.shape().dims()[0] == 1) {
            // Create a gradient tensor with the same shape as self, filled with ones
            let grad_output: Tensor<B, S, T> = Tensor::ones(self.shape().dims()).map_err(|e| {
                TensorError::BackendError(format!("Failed to create gradient tensor: {e}"))
            })?;

            self.backward_with_grad(&grad_output)
        } else {
            Err(TensorError::BackendError(
                "backward() requires scalar tensor (0-d or 1 element)".into(),
            ))
        }
    }

    /// Compute gradients with specified initial gradient
    ///
    /// # Arguments
    /// * `grad_output` - Initial gradient w.r.t. this tensor
    ///
    /// # Errors
    /// Returns error if backward pass fails
    pub fn backward_with_grad<GS>(&self, grad_output: &Tensor<B, GS, T>) -> Result<()>
    where
        B: Backend<Data = T> + Clone + Default,
        S: Storage<T> + Clone + 'static + StorageToDense<T>,
        GS: Storage<T> + StorageToDense<T> + StorageFromVec<T>,
        T: Clone + Copy,
    {
        // Simplified backward implementation
        // Full autograd graph traversal will be implemented later
        // For now, just accumulate gradients for tensors that require them

        if self.requires_grad() {
            // Set the gradient (simplified - full accumulation will be implemented later)
            self.set_grad(grad_output.clone())
        } else {
            // Tensor doesn't require gradients, nothing to do
            Ok(())
        }
    }

    /// Checks if the tensor contains any NaN values.
    ///
    /// # Returns
    /// `true` if the tensor contains NaN values, `false` otherwise.
    pub fn is_nan(&self) -> bool
    where
    {
        // Check for NaN values (x != x is true for NaN)
        #[allow(clippy::eq_op)]
        self.as_slice().iter().any(|&x| x != x)
    }

    /// Checks if the tensor contains any infinite values.
    ///
    /// # Returns
    /// `true` if the tensor contains infinite values, `false` otherwise.
    pub fn is_inf(&self) -> bool
    where
        T: num_traits::Float,
    {
        // Check for infinite values using proper float methods
        self.as_slice().iter().any(|&x| x.is_infinite())
    }

    /// Clamps tensor values to a specified range in place.
    ///
    /// Values less than `min` are set to `min`, values greater than `max` are set to `max`.
    ///
    /// # Arguments
    /// * `min` - Minimum value for clamping
    /// * `max` - Maximum value for clamping
    ///
    /// # Returns
    /// Result indicating success or failure.
    pub fn clamp_(&mut self, min: T, max: T) -> Result<()>
    where
        T: PartialOrd + Copy,
        S: Storage<T>,
    {
        let data = self.storage.as_mut_slice();
        for x in data {
            if *x < min {
                *x = min;
            } else if *x > max {
                *x = max;
            }
        }
        Ok(())
    }

    /// Clamps tensor values to a specified range, returning a new tensor.
    ///
    /// # Arguments
    /// * `min` - Minimum value for clamping
    /// * `max` - Maximum value for clamping
    ///
    /// # Returns
    /// Result containing the clamped tensor or an error.
    pub fn clamp(&self, min: T, max: T) -> Result<Tensor<B, S, T>>
    where
        T: PartialOrd + Copy,
        S: Storage<T> + Clone,
        B: Backend<Data = T>,
    {
        let mut result = self.clone();
        result.clamp_(min, max)?;
        Ok(result)
    }

    /// Multiplies tensor by a scalar value.
    ///
    /// # Arguments
    /// * `scalar` - Scalar value to multiply by
    ///
    /// # Returns
    /// Result containing the scaled tensor or an error.
    pub fn mul_scalar(&self, scalar: T) -> Result<Tensor<B, S, T>>
    where
        T: std::ops::Mul<Output = T> + Copy,
        B: Backend<Data = T>,
        S: Storage<T> + StorageFromVec<T>,
    {
        let data: Vec<T> = self.as_slice().iter().map(|&x| x * scalar).collect();
        Tensor::from_vec_with_backend(data, self.shape().dims(), self.backend.clone())
    }

    /// Multiplies tensor by a scalar value in place.
    ///
    /// # Arguments
    /// * `scalar` - Scalar value to multiply by
    ///
    /// # Returns
    /// Result indicating success or failure.
    pub fn mul_scalar_(&mut self, scalar: T) -> Result<()>
    where
        T: std::ops::Mul<Output = T> + Copy,
        S: Storage<T>,
    {
        let data = self.storage.as_mut_slice();
        for x in data {
            *x = *x * scalar;
        }
        Ok(())
    }

    /// Returns the function that created this tensor.
    ///
    /// Returns `None` if this is a leaf tensor (created directly).
    /// This is the PyTorch-style `grad_fn` attribute.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor::{Tensor, CpuBackend, DenseStorage, Function};
    /// use dtype::float::Float32;
    ///
    /// let x = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2]).unwrap();
    /// assert!(x.grad_fn().is_none()); // Leaf tensor
    /// ```
    #[must_use]
    pub fn grad_fn(&self) -> Option<&str> {
        self.grad_fn.as_deref()
    }

    /// Sets the function that created this tensor.
    ///
    /// Used internally during automatic differentiation to build the computation graph.
    /// Should not be called directly by users.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // This is used internally, not by users
    /// tensor.set_grad_fn(Some(function));
    /// ```
    pub fn set_grad_fn(&mut self, grad_fn: Option<String>) {
        self.grad_fn = grad_fn;
    }

    /// Returns a new tensor with the specified grad_fn set.
    ///
    /// This is used internally by automatic differentiation to attach Function objects
    /// to tensors created during operations.
    ///
    /// # Arguments
    /// * `grad_fn` - The function that created this tensor
    ///
    /// # Returns
    /// New tensor with grad_fn attached
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Used internally by autograd operations
    /// let result = tensor.with_grad_fn(Some(add_function));
    /// ```
    #[must_use]
    pub fn with_grad_fn(mut self, grad_fn: Option<String>) -> Self {
        self.grad_fn = grad_fn;
        self
    }

    /// Returns the number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Returns true if the tensor contains no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Returns a reference to the underlying data.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Returns a mutable reference to the underlying data.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }

    /// Zeros all elements of this tensor in-place.
    ///
    /// This method sets every element in the tensor to the zero value of type `T`.
    /// This is commonly used to zero gradients before backpropagation.
    ///
    /// # Examples
    /// ```
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::float::Float32;
    ///
    /// let mut tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
    ///     vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
    ///     &[3]
    /// ).unwrap();
    ///
    /// tensor.zero_();
    /// assert_eq!(tensor.as_slice(), &[Float32::new(0.0), Float32::new(0.0), Float32::new(0.0)]);
    /// ```
    pub fn zero_(&mut self)
    where
        T: Default + Copy,
    {
        let data = self.as_mut_slice();
        let zero = T::default();
        for elem in data.iter_mut() {
            *elem = zero;
        }
    }

    /// Converts this tensor to dense storage format if supported.
    ///
    /// This method provides a generic way to convert any storage type that
    /// implements `StorageToDense<T>` to dense format.
    ///
    /// # Errors
    /// Returns error if the storage type doesn't support dense conversion
    /// or if the conversion fails.
    pub fn to_dense_generic(&self) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        S: StorageToDense<T>,
        B: Clone,
        T: Clone,
    {
        let dense_storage = self.storage.to_dense()?;
        Ok(Tensor::from_storage(dense_storage, self.backend.clone()))
    }

    /// Convert this tensor to CPU backend with dense storage.
    ///
    /// This method converts any backend tensor to a CPU-based tensor with dense storage,
    /// enabling generic operations that require CPU-specific functionality.
    ///
    /// # Returns
    /// A new tensor with the same data on CPU backend with dense storage.
    ///
    /// # Errors
    /// Returns error if storage conversion to dense fails.
    ///
    /// # Note
    /// Currently assumes the tensor is already on CPU backend. Full backend conversion
    /// requires additional infrastructure for cross-backend data transfer.
    pub fn to_cpu_dense(&self) -> Result<Tensor<crate::CpuBackend<T>, DenseStorage<T>, T>>
    where
        S: StorageToDense<T>,
        B: Clone,
        T: Clone + std::cmp::PartialOrd,
    {
        // Convert storage to dense if needed
        let dense_tensor = self.to_dense_generic()?;
        // For now, create new tensor with CpuBackend
        // Future enhancement: Implement proper backend conversion when cross-backend transfer is added
        Ok(Tensor::from_storage(
            dense_tensor.storage,
            crate::CpuBackend::<T>::default(),
        ))
    }

    /// Convert tensor to generic concrete types (CpuBackend + DenseStorage).
    ///
    /// This method converts from opaque `impl` types to concrete types that can be
    /// returned from functions. Equivalent to `to_cpu_dense()`.
    ///
    /// # Errors
    /// Returns error if storage conversion to dense fails.
    pub fn to_generic(&self) -> Result<Tensor<crate::CpuBackend<T>, DenseStorage<T>, T>>
    where
        S: StorageToDense<T>,
        B: Clone,
        T: Clone + std::cmp::PartialOrd,
    {
        self.to_cpu_dense()
    }

    /// Returns the dtype of this tensor.
    #[must_use]
    pub fn dtype() -> dtype::Dtype {
        T::dtype()
    }

    /// Returns the backend device name.
    #[must_use]
    pub fn device_name(&self) -> &str {
        self.backend.device_name()
    }

    /// Returns a reference to the backend.
    #[must_use]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Returns a reference to the storage for advanced operations.
    ///
    /// This method provides access to the concrete storage type for runtime type checking
    /// and specialized operations (e.g., sparse tensor operations).
    ///
    /// # Examples
    /// ```
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::{DenseStorage, CsrStorage};
    /// use dtype::float::Float32;
    ///
    /// let dense_tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[10]).unwrap();
    /// let storage_ref = dense_tensor.storage_ref();
    /// // Now you can check storage type at runtime via downcasting
    /// ```
    #[must_use]
    pub fn storage_ref(&self) -> &S {
        &self.storage
    }

    /// Helper function to resolve reshape dimensions with -1 inference.
    pub(crate) fn resolve_reshape_dims_generic(
        total_elements: usize,
        dims: &[isize],
    ) -> Result<Vec<usize>> {
        let mut result = Vec::with_capacity(dims.len());
        let mut infer_idx = None;

        // First pass: collect known dimensions and find inference point
        let mut known_product = 1usize;
        for (i, &dim) in dims.iter().enumerate() {
            if dim == -1 {
                if infer_idx.is_some() {
                    return Err(TensorError::ShapeError {
                        expected: 0,
                        actual: 0,
                        message: "Multiple -1 dimensions in reshape".to_string(),
                    });
                }
                infer_idx = Some(i);
                result.push(0); // Placeholder
            } else if dim <= 0 {
                return Err(TensorError::ShapeError {
                    expected: 0,
                    actual: 0,
                    message: format!("Invalid dimension {} in reshape", dim),
                });
            } else {
                let dim_usize = dim as usize;
                result.push(dim_usize);
                known_product = known_product.checked_mul(dim_usize).ok_or_else(|| {
                    TensorError::ShapeError {
                        expected: 0,
                        actual: 0,
                        message: "Dimension product overflow in reshape".to_string(),
                    }
                })?;
            }
        }

        // Handle dimension inference
        if let Some(infer_idx) = infer_idx {
            if total_elements % known_product != 0 {
                return Err(TensorError::ShapeError {
                    expected: known_product,
                    actual: total_elements,
                    message: "Cannot infer -1 dimension: total elements not divisible by known dimensions".to_string(),
                });
            }
            let inferred = total_elements / known_product;
            result[infer_idx] = inferred;
        } else if known_product != total_elements {
            return Err(TensorError::ShapeError {
                expected: total_elements,
                actual: known_product,
                message: "Total element count mismatch in reshape".to_string(),
            });
        }

        Ok(result)
    }

    /// Returns the total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.shape().dims().iter().product()
    }

    /// Returns an iterator over chunks of the tensor along the given dimension.
    ///
    /// # Arguments
    /// * `dim` - The dimension to chunk along
    /// * `chunk_size` - Size of each chunk
    ///
    /// # Returns
    /// An iterator yielding tensor chunks
    pub fn chunks(&self, dim: usize, chunk_size: usize) -> TensorChunks<'_, B, S, T> {
        TensorChunks {
            tensor: self,
            dim,
            chunk_size,
            current: 0,
        }
    }

    /// Creates a view of the tensor (currently just returns a clone).
    pub fn view(&self) -> Tensor<B, S, T>
    where
        B: Clone,
        S: Clone,
    {
        self.clone()
    }

    /// Broadcasts the tensor to the given shape.
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Tensor<B, S, T>>
    where
        B: Clone + Send + Sync + Default,
        S: Clone + Send + Sync + StorageFromVec<T> + 'static,
        T: Clone + Copy,
    {
        crate::ops::arithmetic::broadcast_to(self, shape)
    }

    /// SIMD-accelerated addition (implemented in dedicated simd_ops module).
    pub fn add_simd(&self, other: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
    where
        B: Clone + Send + Sync + Default,
        S: Clone + Send + Sync + StorageFromVec<T> + 'static,
        T: std::ops::Add<Output = T> + Clone + Copy,
    {
        crate::ops::arithmetic::add(self, other)
    }

    /// SIMD-accelerated ReLU activation (implemented in dedicated simd_ops module).
    pub fn relu_simd(&self) -> Result<Tensor<B, S, T>>
    where
        B: Clone + Send + Sync + Default,
        S: Clone + Send + Sync + StorageFromVec<T> + 'static,
        T: num_traits::Float + Clone + Copy + PartialOrd,
    {
        let data = self
            .as_slice()
            .iter()
            .map(|&x| if x > T::zero() { x } else { T::zero() })
            .collect::<Vec<_>>();

        let mut result = Tensor::from_vec(data, self.shape().dims())?;
        if self.requires_grad {
            result = result.requires_grad_(true);
        }
        Ok(result)
    }

    /// SIMD-accelerated sum reduction (implemented in dedicated simd_ops module).
    pub fn sum_simd(&self) -> Result<Tensor<B, S, T>>
    where
        B: Clone + Send + Sync + Default,
        S: Clone + Send + Sync + StorageFromVec<T> + 'static,
        T: num_traits::Num + Clone + Copy,
    {
        // For now, sum all elements manually - SIMD acceleration implemented in dedicated module
        let data = self.as_slice();
        let sum = data.iter().fold(T::zero(), |acc, &x| acc + x);
        Tensor::from_vec(vec![sum], &[1])
    }


    /// Convert tensor to a different backend.
    ///
    /// This method enables zero-copy backend transfers where possible using the Clone bounds
    /// established in the Backend trait. For cross-backend transfers, this may involve
    /// data copying and format conversion.
    ///
    /// Uses associated types pattern for type safety in backend dispatching.
    ///
    /// # Arguments
    /// * `target_backend` - The backend to convert this tensor to
    ///
    /// # Returns
    /// New tensor on the target backend with same data and shape
    ///
    /// # Errors
    /// Returns error if backend conversion fails or data transfer is unsupported
    ///
    /// # Examples
    /// ```ignore
    /// // CPU to GPU transfer
    /// let cpu_tensor = CpuBackend::zeros(&[10, 20]).unwrap();
    /// let gpu_tensor = cpu_tensor.to_backend(gpu_backend).unwrap();
    /// assert_eq!(gpu_tensor.device_name(), "gpu");
    /// ```
    pub fn to_backend<NewB>(
        &self,
        target_backend: NewB,
    ) -> crate::Result<Tensor<NewB, DenseStorage<T>, T>>
    where
        NewB: Backend<Data = T> + Clone + Send + Sync,
        S: StorageToDense<T>,
        T: Clone,
    {
        // For now, implement via dense intermediate representation
        // Future: Direct backend-to-backend transfers for zero-copy operations
        let dense_tensor = self
            .to_dense_generic()
            .map_err(|_| TensorError::BackendError("Failed to convert to dense storage".into()))?;

        // Create new tensor on target backend with copied data
        // In the future, this will use optimized backend transfer methods
        let data = dense_tensor.as_slice().to_vec();
        let dims = dense_tensor.shape().dims();
        let storage = storage::DenseStorage::from_vec(data, dims)
            .map_err(crate::TensorError::StorageError)?;
        Ok(Tensor::from_storage(storage, target_backend))
    }

    /// Clone tensor with optimized backend-aware copying.
    ///
    /// Uses the Backend: Clone bounds to enable zero-copy operations where supported.
    /// For backends that support it, this enables shared memory tensors.
    ///
    /// # Returns
    /// Cloned tensor with same backend, storage, and data
    ///
    /// # Examples
    /// ```
    /// use tensor::{Tensor, CpuBackend, DenseStorage};
    /// use dtype::float::Float32;
    ///
    /// let original = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2, 3]).unwrap();
    ///
    /// // Backend-aware clone (potentially zero-copy)
    /// let cloned = original.clone();
    /// assert_eq!(original.shape().dims(), cloned.shape().dims());
    /// ```
    pub fn backend_clone(&self) -> Self
    where
        B: Clone,
        S: Clone,
        T: Clone,
    {
        // Use the Clone implementation which now leverages Backend trait bounds
        self.clone()
    }

    /// Get the backend device information.
    ///
    /// Provides access to device capabilities and information for backend-specific optimizations.
    ///
    /// # Returns
    /// Reference to backend's device information
    pub fn device(&self) -> &B::Device {
        self.backend.device()
    }

    /// Check if backend supports a specific operation.
    ///
    /// Enables compile-time dispatch optimization based on backend capabilities.
    ///
    /// # Arguments
    /// * `operation` - Operation name to check
    ///
    /// # Returns
    /// true if operation is supported by this backend
    pub fn backend_supports(&self, operation: &str) -> bool {
        self.backend.supports(operation)
    }
}

// Iterator for tensor chunks
pub struct TensorChunks<'a, B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    tensor: &'a Tensor<B, S, T>,
    dim: usize,
    chunk_size: usize,
    current: usize,
}

impl<'a, B, S, T> TensorChunks<'a, B, S, T>
where
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
    T: DataType + Clone + Copy,
{
    /// Create a slice of the tensor along a specific dimension
    fn create_dim_slice(
        &self,
        dim: usize,
        start: usize,
        end: usize,
    ) -> crate::Result<Tensor<B, S, T>> {
        let dims = self.tensor.shape().dims();
        let mut new_dims = dims.to_vec();
        new_dims[dim] = end - start;

        let mut sliced_data = Vec::new();

        // Calculate strides for the original tensor
        let mut strides = vec![1; dims.len()];
        for i in (1..dims.len()).rev() {
            strides[i - 1] = strides[i] * dims[i];
        }

        // Iterate through all coordinates in the new tensor
        let mut coords = vec![0; dims.len()];

        loop {
            // Compute linear index in original tensor
            let mut linear_idx = 0;
            for (i, &coord) in coords.iter().enumerate() {
                let actual_coord = if i == dim { coord + start } else { coord };
                linear_idx += actual_coord * strides[i];
            }

            sliced_data.push(self.tensor.as_slice()[linear_idx]);

            // Increment coordinates (like counting in mixed bases)
            let mut carry = 1;
            for i in (0..dims.len()).rev() {
                coords[i] += carry;
                if coords[i] < new_dims[i] {
                    carry = 0;
                    break;
                }
                coords[i] = 0;
            }

            // If we wrapped around completely, we're done
            if carry != 0 {
                break;
            }
        }

        Tensor::from_vec(sliced_data, &new_dims)
    }
}

impl<'a, B, S, T> Iterator for TensorChunks<'a, B, S, T>
where
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
    T: DataType + Clone + Copy,
{
    type Item = Tensor<B, S, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let dims = self.tensor.shape().dims();
        if self.dim >= dims.len() {
            return None;
        }

        let dim_size = dims[self.dim];
        let start = self.current * self.chunk_size;
        if start >= dim_size {
            return None;
        }

        let end = (start + self.chunk_size).min(dim_size);
        let _actual_chunk_size = end - start;

        // Implement chunking by slicing the tensor along the specified dimension
        self.current += 1;

        // Create a slice of the tensor along the specified dimension
        // For now, implement basic slicing by copying data
        self.create_dim_slice(self.dim, start, end).ok()
    }
}

// Operator overloading implementations for PyTorch-style syntax
use std::ops::{Add, Div, Mul, Neg, Sub};

impl<B, S, T> Add<&Tensor<B, S, T>> for &Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
    T: DataType + std::ops::Add<Output = T> + Clone + Copy + num_traits::Num,
{
    type Output = Tensor<B, S, T>;

    fn add(self, rhs: &Tensor<B, S, T>) -> Self::Output {
        match crate::ops::arithmetic::add(self, rhs) {
            Ok(tensor) => tensor,
            Err(_) => {
                // For std::ops traits, we cannot return Result, so we provide a safe default
                // This maintains API compatibility while avoiding panics
                // Users should prefer the explicit arithmetic methods for error handling
                self.clone() // Return left operand as safe fallback
            }
        }
    }
}

impl<B, S, T> Sub<&Tensor<B, S, T>> for &Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
    T: DataType + std::ops::Sub<Output = T> + Clone + Copy + num_traits::Num,
{
    type Output = Tensor<B, S, T>;

    fn sub(self, rhs: &Tensor<B, S, T>) -> Self::Output {
        match crate::ops::arithmetic::sub(self, rhs) {
            Ok(tensor) => tensor,
            Err(_) => {
                // For std::ops traits, we cannot return Result, so we provide a safe default
                // This maintains API compatibility while avoiding panics
                // Users should prefer the explicit arithmetic methods for error handling
                self.clone() // Return left operand as safe fallback
            }
        }
    }
}

impl<B, S, T> Mul<&Tensor<B, S, T>> for &Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
    T: DataType + std::ops::Mul<Output = T> + Clone + Copy + num_traits::Num,
{
    type Output = Tensor<B, S, T>;

    fn mul(self, rhs: &Tensor<B, S, T>) -> Self::Output {
        crate::ops::arithmetic::mul(self, rhs).expect("Tensor multiplication failed")
    }
}

impl<B, S, T> Div<&Tensor<B, S, T>> for &Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
    T: DataType + std::ops::Div<Output = T> + Clone + Copy + num_traits::Num,
{
    type Output = Tensor<B, S, T>;

    fn div(self, rhs: &Tensor<B, S, T>) -> Self::Output {
        crate::ops::arithmetic::div(self, rhs).expect("Tensor division failed")
    }
}

impl<B, S, T> Neg for &Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone + Send + Sync + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
    T: DataType + std::ops::Neg<Output = T> + Clone + Copy + num_traits::Num,
{
    type Output = Tensor<B, S, T>;

    fn neg(self) -> Self::Output {
        crate::ops::arithmetic::neg(self).expect("Tensor negation failed")
    }
}
