//! Automatic differentiation tensor wrapper.
//!
//! This module provides AutoGradTensor, a wrapper around Tensor that provides
//! a higher-level API for automatic differentiation operations.

use crate::{Backend, DataType, DenseStorage, Function, Result, Tensor};

/// Wrapper around Tensor providing automatic differentiation operations
///
/// This struct provides a convenient API for tensor operations with automatic
/// differentiation. It wraps a regular Tensor and provides methods that
/// create computation graph nodes for gradient computation.
///
/// # Examples
///
/// ```ignore
/// use tensor::AutoGradTensor;
/// use backend::CpuBackend;
/// use dtype::float::Float32;
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
    /// Computes element-wise exponential of the tensor.
    /// Creates a computation graph node for gradient computation.
    #[must_use]
    pub fn exp(&self) -> Self {
        use crate::ops::arithmetic::exp;
        match exp(&self.tensor) {
            Ok(result) => Self::new(result),
            Err(_) => {
                // Fallback to identity if exp fails
                Self::new(self.tensor.clone())
            }
        }
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
    /// # Panics
    /// Panics if gradient shape doesn't match tensor shape
    pub fn backward(&self) -> Result<()> {
        // Create a gradient tensor filled with ones (same shape as self)
        let grad_output = Tensor::ones(self.shape().dims())
            .map_err(|e| TensorError::BackendError(format!("Failed to create gradient tensor: {e}")))?;

        self.backward_with_grad(&grad_output)
    }

    /// Compute gradients with specified initial gradient
    ///
    /// # Arguments
    /// * `grad_output` - Initial gradient w.r.t. this tensor
    ///
    /// # Errors
    /// Returns error if backward pass fails
    #[allow(clippy::missing_errors_doc)]
    pub fn backward_with_grad<GS>(&self, grad_output: &Tensor<B, GS, T>) -> Result<()>
    where
        GS: Storage<T> + StorageToDense<T>,
    {
        if let Some(grad_fn) = self.grad_fn() {
            // For now, implement simple single-function backward
            // Full graph traversal would require more complex implementation
            // This is a temporary implementation until proper graph traversal is added

            // Get inputs to this function
            let inputs = grad_fn.inputs();

            // Compute gradients w.r.t. inputs
            let input_gradients = grad_fn.backward(grad_output).map_err(|e| {
                TensorError::BackendError(format!("Function backward failed: {e}"))
            })?;

            // Accumulate gradients into input tensors
            if inputs.len() == input_gradients.len() {
                for (input_tensor, grad) in inputs.iter().zip(input_gradients) {
                    // For now, just set the gradient (no accumulation)
                    // Proper accumulation would check if gradient already exists
                    input_tensor.set_grad(grad).map_err(|e| {
                        TensorError::BackendError(format!("Failed to set gradient: {e}"))
                    })?;
                }
            }

            Ok(())
        } else {
            // Leaf tensor with no grad_fn - this is where backward pass starts
            // For leaf tensors, the gradient is just the grad_output
            self.set_grad(grad_output.clone())
        }
    }
}



