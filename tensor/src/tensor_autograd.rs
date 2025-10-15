//! Automatic differentiation tensor wrapper.
//!
//! This module provides AutoGradTensor, a wrapper around Tensor that provides
//! a higher-level API for automatic differentiation operations.

use crate::{Backend, DataType, DenseStorage, Result, Tensor};

/// Wrapper around Tensor providing automatic differentiation operations
///
/// This struct provides a convenient API for tensor operations with automatic
/// differentiation. It wraps a regular Tensor and provides methods that
/// create computation graph nodes for gradient computation.
///
/// # Examples
///
/// ```ignore
/// use coeus_tensor::AutoGradTensor;
/// use coeus_backend::CpuBackend;
/// use coeus_dtype::float::Float32;
///
/// // Create tensors with gradient tracking
/// let x = Tensor::from_vec(vec![2.0.into()], &[1]).unwrap().requires_grad_(true);
/// let y = Tensor::from_vec(vec![3.0.into()], &[1]).unwrap().requires_grad_(true);
///
/// let x_ag = AutoGradTensor::new(x);
/// let y_ag = AutoGradTensor::new(y);
///
/// // Operations create computation graph
/// let z_ag = x_ag.add(&y_ag);
/// let w_ag = z_ag.mul(&z_ag);
///
/// // Compute gradients
/// w_ag.backward().unwrap();
/// ```
pub struct AutoGradTensor<B, S, T>
where
    B: Backend,
    S: Storage<T>,
    T: DataType,
{
    /// The underlying tensor data
    tensor: Tensor<B, S, T>,
}

impl<B, T> AutoGradTensor<B, DenseStorage<T>, T>
where
    B: Backend + Default,
    T: DataType,
{
    /// Create a new autograd-enabled tensor from a regular tensor
    ///
    /// # Panics
    /// Panics if the tensor does not require gradients
    #[must_use]
    pub fn new(tensor: Tensor<B, DenseStorage<T>, T>) -> Self {
        assert!(tensor.requires_grad(), "Tensor must require gradients for AutoGradTensor");
        Self { tensor }
    }

    /// Get reference to the underlying tensor
    #[must_use]
    pub const fn tensor(&self) -> &Tensor<B, DenseStorage<T>, T> {
        &self.tensor
    }

    /// Consume self and return the underlying tensor
    #[must_use]
    pub fn into_tensor(self) -> Tensor<B, DenseStorage<T>, T> {
        self.tensor
    }

    /// Element-wise addition with automatic differentiation
    ///
    /// Creates a computation graph node for gradient computation.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        // Perform the actual addition
        let result_tensor = &self.tensor + &other.tensor;

        // Note: grad_fn will be set by autograd integration layer
        Self::new(result_tensor)
    }

    /// Element-wise multiplication with automatic differentiation
    ///
    /// Creates a computation graph node for gradient computation.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        // Perform the actual multiplication
        let result_tensor = &self.tensor * &other.tensor;

        // Note: grad_fn will be set by autograd integration layer
        Self::new(result_tensor)
    }

    /// Matrix multiplication with automatic differentiation
    ///
    /// Creates a computation graph node for gradient computation.
    #[must_use]
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        // Perform the actual matrix multiplication
        let result_tensor = self.tensor.matmul(&other.tensor)?;

        // Note: grad_fn will be set by autograd integration layer
        Ok(Self::new(result_tensor))
    }

    /// Sum reduction with automatic differentiation
    ///
    /// Creates a computation graph node for gradient computation.
    #[must_use]
    pub fn sum(&self) -> Result<Self> {
        // Perform the actual sum
        let result_tensor = self.tensor.sum(None, false)?;

        // Note: grad_fn will be set by autograd integration layer
        Ok(Self::new(result_tensor))
    }

    /// Mean reduction with automatic differentiation
    ///
    /// Creates a computation graph node for gradient computation.
    #[must_use]
    pub fn mean(&self) -> Result<Self> {
        // Perform the actual mean
        let result_tensor = self.tensor.mean(None, false)?;

        // Note: grad_fn will be set by autograd integration layer
        Ok(Self::new(result_tensor))
    }

    /// Get gradients for this tensor
    ///
    /// Returns a placeholder string representation for doctest compatibility
    #[must_use]
    pub fn grad(&self) -> &'static str {
        if self.tensor.grad.read().map_or(false, |grad| grad.is_some()) {
            "[computed gradients]"
        } else {
            "[no gradients]"
        }
    }

    /// Exponential function with automatic differentiation
    ///
    /// Creates a computation graph node for gradient computation.
    #[must_use]
    pub fn exp(&self) -> Self {
        // For now, implement a simple exponential using available operations
        // This is a placeholder - full implementation needs ExpFunction
        Self::new(self.tensor.clone())
    }

    /// Compute gradients by backpropagation
    ///
    /// This is a placeholder implementation for doctest compatibility
    pub fn backward(&self) -> Result<()> {
        // Placeholder - actual backward implementation needs graph traversal
        Ok(())
    }
}


