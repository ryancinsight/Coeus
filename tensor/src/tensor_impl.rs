//! Core tensor implementations.
//!
//! This module contains the fundamental implementations for the Tensor type,
//! including creation, basic operations, and gradient management.

use std::{boxed::Box, format, string::ToString, sync::Arc, vec::Vec};
use core::marker::PhantomData;

use crate::{
    error::TensorError,
    grad_rwlock,
    AsAny,
    Backend, DataType, DenseStorage, Function, Result, Shape, Storage, StorageToDense, Tensor,
};
use coeus_storage::StorageFromVec;

impl<B, S, T> Tensor<B, S, T>
where
    B: Backend,
    S: Storage<T> + 'static,
    T: DataType,
{
    /// Creates a tensor from existing storage and backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::float::Float32;
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
            _phantom: PhantomData,
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
    /// use coeus_tensor::{Tensor, CpuBackend, DenseStorage};
    /// use coeus_dtype::float::Float32;
    ///
    /// let tensor = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[2, 3]).unwrap();
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
    /// use coeus_tensor::{Tensor, CpuBackend, DenseStorage};
    /// use coeus_dtype::float::Float32;
    ///
    /// let tensor = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[2, 3]).unwrap();
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
    /// use coeus_tensor::{Tensor, CpuBackend, DenseStorage};
    /// use coeus_dtype::float::Float32;
    ///
    /// let tensor = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[2, 3]).unwrap()
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
    /// use coeus_tensor::{Tensor, CpuBackend, DenseStorage};
    /// use coeus_dtype::float::Float32;
    ///
    /// let x = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[2]).unwrap()
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
        let grad_lock = self.grad.read().map_err(|_| {
            TensorError::BackendError("Failed to acquire gradient lock".into())
        })?;
        #[cfg(not(feature = "std"))]
        let grad_lock = self.grad.read();

        match grad_lock.as_ref() {
            Some(boxed) => {
                // For now, assume gradients are stored as dense tensors
                // This is the common case and avoids generic cloning issues
                if let Some(dense_grad) =
                    boxed.as_any().downcast_ref::<Tensor<B, DenseStorage<T>, T>>()
                {
                    Ok(dense_grad.clone())
                } else {
                    // If not dense, try to reconstruct from storage
                    // This is a fallback for other storage types
                    Err(TensorError::BackendError("Gradient tensor storage type not supported".into()))
                }
            }
            None => Err(TensorError::BackendError("Gradient not available (call backward first)".into())),
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
    pub fn set_grad(&self, gradient: Tensor<B, S, T>) -> Result<()> {
        // Validate shape matches
        if gradient.shape().dims() != self.shape().dims() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                actual: gradient.shape().dims().to_vec(),
                operation: "set_grad",
            });
        }

        #[cfg(feature = "std")]
        let mut grad_lock = self.grad.write().map_err(|_| {
            TensorError::BackendError("Failed to acquire gradient lock".into())
        })?;
        #[cfg(not(feature = "std"))]
        let mut grad_lock = self.grad.write();

        *grad_lock = Some(Box::new(gradient));
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
    /// use coeus_tensor::{Tensor, CpuBackend, DenseStorage};
    /// use coeus_dtype::float::Float32;
    ///
    /// let x = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[2]).unwrap()
    ///     .requires_grad_(true);
    /// x.zero_grad().unwrap();
    /// ```
    pub fn zero_grad(&self) -> Result<()> {
        #[cfg(feature = "std")]
        let mut grad_lock = self.grad.write().map_err(|_| {
            TensorError::BackendError("Failed to acquire gradient lock".into())
        })?;
        #[cfg(not(feature = "std"))]
        let mut grad_lock = self.grad.write();

        *grad_lock = None;
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
    /// use coeus_tensor::{Tensor, CpuBackend, DenseStorage};
    /// use coeus_dtype::float::Float32;
    ///
    /// let x = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[2]).unwrap();
    /// assert!(x.grad_fn().is_none()); // Leaf tensor
    /// ```
    #[must_use]
    pub fn grad_fn(&self) -> Option<&Arc<dyn Function<B, S, T>>> {
        self.grad_fn.as_ref()
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
    pub fn set_grad_fn(&mut self, grad_fn: Option<Arc<dyn Function<B, S, T>>>) {
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
    pub fn with_grad_fn(mut self, grad_fn: Option<Arc<dyn Function<B, S, T>>>) -> Self {
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
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::float::Float32;
    ///
    /// let mut tensor = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
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
    pub fn to_cpu_dense(&self) -> Result<Tensor<crate::CpuBackend, DenseStorage<T>, T>>
    where
        S: StorageToDense<T>,
        B: Clone,
        T: Clone,
    {
        // Convert storage to dense if needed
        let dense_tensor = self.to_dense_generic()?;
        // For now, create new tensor with CpuBackend
        // TODO: Implement proper backend conversion when cross-backend transfer is added
        Ok(Tensor::from_storage(
            dense_tensor.storage,
            crate::CpuBackend::default(),
        ))
    }

    /// Returns the dtype of this tensor.
    #[must_use]
    pub fn dtype() -> coeus_dtype::Dtype {
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
    /// This method provides access to the storage for runtime type checking
    /// and specialized operations (e.g., sparse tensor operations).
    ///
    /// # Examples
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::{DenseStorage, CsrStorage};
    /// use coeus_dtype::float::Float32;
    ///
    /// let dense_tensor = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[10]).unwrap();
    /// let storage_ref = dense_tensor.storage_ref();
    /// // Now you can check storage type at runtime
    /// ```
    #[must_use]
    pub fn storage_ref(&self) -> &dyn Storage<T> {
        self.storage.as_storage_ref()
    }

    /// Helper function to resolve reshape dimensions with -1 inference.
    pub(crate) fn resolve_reshape_dims_generic(total_elements: usize, dims: &[isize]) -> Result<Vec<usize>> {
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
    pub fn chunks(&self, dim: usize, chunk_size: usize) -> TensorChunks<B, S, T> {
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

    /// SIMD-accelerated addition (placeholder - currently calls regular add).
    pub fn add_simd(&self, other: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
    where
        B: Clone + Send + Sync + Default,
        S: Clone + Send + Sync + StorageFromVec<T> + 'static,
        T: std::ops::Add<Output = T> + Clone + Copy,
    {
        crate::ops::arithmetic::add(self, other)
    }

    /// SIMD-accelerated ReLU activation (placeholder).
    pub fn relu_simd(&self) -> Result<Tensor<B, S, T>>
    where
        B: Clone + Send + Sync + Default,
        S: Clone + Send + Sync + StorageFromVec<T> + 'static,
        T: num_traits::Float + Clone + Copy + PartialOrd,
    {
        let data = self.as_slice()
            .iter()
            .map(|&x| if x > T::zero() { x } else { T::zero() })
            .collect::<Vec<_>>();

        let mut result = Tensor::from_vec(data, self.shape().dims())?;
        if self.requires_grad {
            result = result.requires_grad_(true);
        }
        Ok(result)
    }

    /// SIMD-accelerated sum reduction (placeholder - currently calls regular sum_dims).
    pub fn sum_simd(&self) -> Result<Tensor<B, S, T>>
    where
        B: Clone + Send + Sync + Default,
        S: Clone + Send + Sync + StorageFromVec<T> + 'static,
        T: num_traits::Num + Clone + Copy,
    {
        // For now, sum all elements manually - this would be SIMD accelerated in a real implementation
        let data = self.as_slice();
        let sum = data.iter().fold(T::zero(), |acc, &x| acc + x);
        Tensor::from_vec(vec![sum], &[1])
    }
}

// Iterator for tensor chunks
pub struct TensorChunks<'a, B, S, T>
where
    B: Backend,
    S: Storage<T>,
    T: DataType,
{
    tensor: &'a Tensor<B, S, T>,
    dim: usize,
    chunk_size: usize,
    current: usize,
}

impl<'a, B, S, T> Iterator for TensorChunks<'a, B, S, T>
where
    B: Backend + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
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
        let actual_chunk_size = end - start;

        // For now, return a simple slice - this is a placeholder implementation
        // A full implementation would need proper indexing operations
        self.current += 1;
        if self.current == 1 {
            Some(self.tensor.clone()) // Placeholder - return full tensor once
        } else {
            None // Placeholder - only return one chunk
        }
    }
}

// Operator overloading implementations for PyTorch-style syntax
use std::ops::{Add, Sub, Mul, Div, Neg};

impl<B, S, T> Add<&Tensor<B, S, T>> for &Tensor<B, S, T>
where
    B: Backend + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
    T: DataType + std::ops::Add<Output = T> + Clone + Copy + num_traits::Num,
{
    type Output = Tensor<B, S, T>;

    fn add(self, rhs: &Tensor<B, S, T>) -> Self::Output {
        match crate::ops::arithmetic::add(self, rhs) {
            Ok(tensor) => tensor,
            Err(_) => panic!("Incompatible shapes for broadcasting"),
        }
    }
}

impl<B, S, T> Sub<&Tensor<B, S, T>> for &Tensor<B, S, T>
where
    B: Backend + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
    T: DataType + std::ops::Sub<Output = T> + Clone + Copy + num_traits::Num,
{
    type Output = Tensor<B, S, T>;

    fn sub(self, rhs: &Tensor<B, S, T>) -> Self::Output {
        match crate::ops::arithmetic::sub(self, rhs) {
            Ok(tensor) => tensor,
            Err(_) => panic!("Incompatible shapes for broadcasting"),
        }
    }
}

impl<B, S, T> Mul<&Tensor<B, S, T>> for &Tensor<B, S, T>
where
    B: Backend + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
    T: DataType + std::ops::Mul<Output = T> + Clone + Copy + num_traits::Num,
{
    type Output = Tensor<B, S, T>;

    fn mul(self, rhs: &Tensor<B, S, T>) -> Self::Output {
        crate::ops::arithmetic::mul(self, rhs).expect("Tensor multiplication failed")
    }
}

impl<B, S, T> Div<&Tensor<B, S, T>> for &Tensor<B, S, T>
where
    B: Backend + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
    T: DataType + std::ops::Div<Output = T> + Clone + Copy + num_traits::Num,
{
    type Output = Tensor<B, S, T>;

    fn div(self, rhs: &Tensor<B, S, T>) -> Self::Output {
        crate::ops::arithmetic::div(self, rhs).expect("Tensor division failed")
    }
}

impl<B, S, T> Neg for &Tensor<B, S, T>
where
    B: Backend + Clone + Send + Sync + Default,
    S: Storage<T> + Clone + Send + Sync + StorageFromVec<T> + 'static,
    T: DataType + std::ops::Neg<Output = T> + Clone + Copy + num_traits::Num,
{
    type Output = Tensor<B, S, T>;

    fn neg(self) -> Self::Output {
        crate::ops::arithmetic::neg(self).expect("Tensor negation failed")
    }
}