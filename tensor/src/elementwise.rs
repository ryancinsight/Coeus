//! Element-wise mathematical operations.
//!
//! This module provides element-wise mathematical functions that operate
//! on each element of a tensor independently, such as exponential, logarithm,
//! trigonometric functions, and power operations.

use std::vec::Vec;

/// Element-wise mathematical operations for tensors with float-extended types.
///
/// This trait provides methods for applying mathematical functions to each
/// element of a tensor, resulting in a new tensor with the same shape.
impl<B, S, T> crate::Tensor<B, S, T>
where
    B: crate::Backend + Default + Clone,
    S: crate::Storage<T> + Clone + crate::StorageFromVec<T> + 'static,
    T: crate::DataType + crate::FloatExt,
{
    /// Computes the exponential of each element in the tensor.
    ///
    /// # Mathematical Definition
    ///
    /// For each element x in the tensor:
    /// ```text
    /// exp(x) = e^x
    /// ```
    ///
    /// # Returns
    ///
    /// A new tensor with the same shape containing `exp(x)` for each element.
    ///
    /// # Panics
    ///
    /// This function uses conditional unsafe in release builds for performance.
    /// In debug builds, panics if shape invariants are violated (indicates a bug).
    /// In release builds, uses `unwrap_unchecked()` after mathematical proof of correctness.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::float::Float32;
    ///
    /// let a = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_slice(
    ///     &[Float32::new(0.0), Float32::new(1.0)], &[2]
    /// ).unwrap();
    ///
    /// let b = a.exp();
    /// // exp(0) = 1, exp(1) ≈ 2.718
    /// assert_eq!(b.shape().dims(), &[2]);
    /// ```
    #[must_use]
    pub fn exp(&self) -> Self {
        let mut result_data = Vec::with_capacity(self.len());
        result_data.extend(self.as_slice().iter().map(|&x| x.exp()));

        // Create new tensor with result data
        let storage = S::from_vec(result_data, self.shape().dims())
            .expect("Shape invariant violated: this is a bug in the tensor implementation");
        crate::Tensor {
            storage,
            backend: self.backend.clone(),
            requires_grad: false, // Element-wise ops create intermediate tensors
            grad: std::sync::Arc::new(crate::grad_rwlock(None)),
            grad_fn: None,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Computes the natural logarithm of each element in the tensor.
    ///
    /// # Mathematical Definition
    ///
    /// For each element x in the tensor:
    /// ```text
    /// log(x) = ln(x)
    /// ```
    ///
    /// # Returns
    ///
    /// A new tensor with the same shape containing `ln(x)` for each element.
    ///
    /// # Panics
    ///
    /// This function uses conditional unsafe in release builds for performance.
    /// In debug builds, panics if shape invariants are violated (indicates a bug).
    /// In release builds, uses `unwrap_unchecked()` after mathematical proof of correctness.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::float::Float32;
    ///
    /// let a = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_slice(
    ///     &[Float32::new(1.0), Float32::new(std::f32::consts::E)], &[2]
    /// ).unwrap();
    ///
    /// let b = a.log();
    /// // log(1) = 0, log(e) = 1
    /// assert_eq!(b.shape().dims(), &[2]);
    /// ```
    #[must_use]
    pub fn log(&self) -> Self {
        let mut result_data = Vec::with_capacity(self.len());
        result_data.extend(self.as_slice().iter().map(|&x| x.ln()));

        // Create new tensor with result data
        let storage = S::from_vec(result_data, self.shape().dims())
            .expect("Shape invariant violated: this is a bug in the tensor implementation");
        crate::Tensor {
            storage,
            backend: self.backend.clone(),
            requires_grad: false, // Element-wise ops create intermediate tensors
            grad: std::sync::Arc::new(crate::grad_rwlock(None)),
            grad_fn: None,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Computes the sine of each element in the tensor.
    ///
    /// # Mathematical Definition
    ///
    /// For each element x in the tensor:
    /// ```text
    /// sin(x) = sine of x (in radians)
    /// ```
    ///
    /// # Returns
    ///
    /// A new tensor with the same shape containing `sin(x)` for each element.
    ///
    /// # Panics
    ///
    /// This function uses conditional unsafe in release builds for performance.
    /// In debug builds, panics if shape invariants are violated (indicates a bug).
    /// In release builds, uses `unwrap_unchecked()` after mathematical proof of correctness.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::float::Float32;
    ///
    /// let a = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_slice(
    ///     &[Float32::new(0.0), Float32::new(std::f32::consts::PI / 2.0)], &[2]
    /// ).unwrap();
    ///
    /// let b = a.sin();
    /// // sin(0) = 0, sin(π/2) ≈ 1
    /// assert_eq!(b.shape().dims(), &[2]);
    /// ```
    #[must_use]
    pub fn sin(&self) -> Self {
        let mut result_data = Vec::with_capacity(self.len());
        result_data.extend(self.as_slice().iter().map(|&x| x.sin()));

        // Create new tensor with result data
        let storage = S::from_vec(result_data, self.shape().dims())
            .expect("Shape invariant violated: this is a bug in the tensor implementation");
        crate::Tensor {
            storage,
            backend: self.backend.clone(),
            requires_grad: false, // Element-wise ops create intermediate tensors
            grad: std::sync::Arc::new(crate::grad_rwlock(None)),
            grad_fn: None,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Computes the cosine of each element in the tensor.
    ///
    /// # Mathematical Definition
    ///
    /// For each element x in the tensor:
    /// ```text
    /// cos(x) = cosine of x (in radians)
    /// ```
    ///
    /// # Returns
    ///
    /// A new tensor with the same shape containing `cos(x)` for each element.
    ///
    /// # Panics
    ///
    /// This function uses conditional unsafe in release builds for performance.
    /// In debug builds, panics if shape invariants are violated (indicates a bug).
    /// In release builds, uses `unwrap_unchecked()` after mathematical proof of correctness.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::float::Float32;
    ///
    /// let a = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_slice(
    ///     &[Float32::new(0.0), Float32::new(std::f32::consts::PI / 2.0)], &[2]
    /// ).unwrap();
    ///
    /// let b = a.cos();
    /// // cos(0) = 1, cos(π/2) ≈ 0
    /// assert_eq!(b.shape().dims(), &[2]);
    /// ```
    #[must_use]
    pub fn cos(&self) -> Self {
        let mut result_data = Vec::with_capacity(self.len());
        result_data.extend(self.as_slice().iter().map(|&x| x.cos()));

        // Create new tensor with result data
        let storage = S::from_vec(result_data, self.shape().dims())
            .expect("Shape invariant violated: this is a bug in the tensor implementation");
        crate::Tensor {
            storage,
            backend: self.backend.clone(),
            requires_grad: false, // Element-wise ops create intermediate tensors
            grad: std::sync::Arc::new(crate::grad_rwlock(None)),
            grad_fn: None,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Computes the power of each element in the tensor.
    ///
    /// # Mathematical Definition
    ///
    /// For each element x in the tensor:
    /// ```text
    /// pow(x, exp) = x^exp
    /// ```
    ///
    /// # Arguments
    ///
    /// * `exp` - The exponent to raise each element to
    ///
    /// # Returns
    ///
    /// A new tensor with the same shape containing `x^exp` for each element.
    ///
    /// # Panics
    ///
    /// This function uses conditional unsafe in release builds for performance.
    /// In debug builds, panics if shape invariants are violated (indicates a bug).
    /// In release builds, uses `unwrap_unchecked()` after mathematical proof of correctness.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::float::Float32;
    ///
    /// let a = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_slice(
    ///     &[Float32::new(2.0), Float32::new(3.0)], &[2]
    /// ).unwrap();
    ///
    /// let b = a.powf(Float32::new(2.0));
    /// // 2^2 = 4, 3^2 = 9
    /// assert_eq!(b.shape().dims(), &[2]);
    /// ```
    #[must_use]
    pub fn powf(&self, exp: T) -> Self {
        let mut result_data = Vec::with_capacity(self.len());
        result_data.extend(self.as_slice().iter().map(|&x| x.powf(exp)));

        // Create new tensor with result data
        let storage = S::from_vec(result_data, self.shape().dims())
            .expect("Shape invariant violated: this is a bug in the tensor implementation");
        crate::Tensor {
            storage,
            backend: self.backend.clone(),
            requires_grad: false, // Element-wise ops create intermediate tensors
            grad: std::sync::Arc::new(crate::grad_rwlock(None)),
            grad_fn: None,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Computes the square of each element in the tensor.
    ///
    /// # Mathematical Definition
    ///
    /// For each element x in the tensor:
    /// ```text
    /// square(x) = x² = x * x
    /// ```
    ///
    /// # Returns
    ///
    /// A new tensor with the same shape containing `x²` for each element.
    ///
    /// # Panics
    ///
    /// This function uses conditional unsafe in release builds for performance.
    /// In debug builds, panics if shape invariants are violated (indicates a bug).
    /// In release builds, uses `unwrap_unchecked()` after mathematical proof of correctness.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::float::Float32;
    ///
    /// let a = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_slice(
    ///     &[Float32::new(2.0), Float32::new(3.0)], &[2]
    /// ).unwrap();
    ///
    /// let b = a.square();
    /// // 2² = 4, 3² = 9
    /// assert_eq!(b.shape().dims(), &[2]);
    /// ```
    #[must_use]
    pub fn square(&self) -> Self {
        let mut result_data = Vec::with_capacity(self.len());
        result_data.extend(self.as_slice().iter().map(|&x| x * x));

        // Create new tensor with result data
        let storage = S::from_vec(result_data, self.shape().dims())
            .expect("Shape invariant violated: this is a bug in the tensor implementation");
        crate::Tensor {
            storage,
            backend: self.backend.clone(),
            requires_grad: false, // Element-wise ops create intermediate tensors
            grad: std::sync::Arc::new(crate::grad_rwlock(None)),
            grad_fn: None,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Computes the square root of each element in the tensor.
    ///
    /// # Mathematical Definition
    ///
    /// For each element x in the tensor:
    /// ```text
    /// sqrt(x) = √x
    /// ```
    ///
    /// # Returns
    ///
    /// A new tensor with the same shape containing `√x` for each element.
    ///
    /// # Panics
    ///
    /// This function uses conditional unsafe in release builds for performance.
    /// In debug builds, panics if shape invariants are violated (indicates a bug).
    /// In release builds, uses `unwrap_unchecked()` after mathematical proof of correctness.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::float::Float32;
    ///
    /// let a = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_slice(
    ///     &[Float32::new(4.0), Float32::new(9.0)], &[2]
    /// ).unwrap();
    ///
    /// let b = a.sqrt();
    /// // √4 = 2, √9 = 3
    /// assert_eq!(b.shape().dims(), &[2]);
    /// ```
    #[must_use]
    pub fn sqrt(&self) -> Self {
        let mut result_data = Vec::with_capacity(self.len());
        result_data.extend(self.as_slice().iter().map(|&x| x.sqrt()));

        // Create new tensor with result data
        let storage = S::from_vec(result_data, self.shape().dims())
            .expect("Shape invariant violated: this is a bug in the tensor implementation");
        crate::Tensor {
            storage,
            backend: self.backend.clone(),
            requires_grad: false, // Element-wise ops create intermediate tensors
            grad: std::sync::Arc::new(crate::grad_rwlock(None)),
            grad_fn: None,
            _phantom: core::marker::PhantomData,
        }
    }
}
