//! Activation functions and elementary operations for tensors
//!
//! This module contains mathematical activation functions and elementary operations
//! with automatic differentiation support.

use crate::{with_autograd_context, Dtype, Tensor};
use coeus_autograd::context::Operation;
use num_traits::{Float, Signed};

/// Activation functions and elementary operations for tensors
impl<T: Dtype + Float> Tensor<T> {
    /// Compute the absolute value of each element
    ///
    /// # Returns
    /// New tensor with absolute values
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![-1.0, 2.0, -3.0], vec![3]);
    /// let abs_tensor = tensor.abs();
    /// assert_eq!(abs_tensor.data(), &[1.0, 2.0, 3.0]);
    /// ```
    pub fn abs(&self) -> Tensor<T>
    where
        T: Dtype + Signed,
    {
        let data = self.data.iter().map(|x| x.abs()).collect();
        let mut result = Tensor::from_vec(data, self.shape.clone());

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        // Create computational graph node if autograd is enabled
        if self.requires_grad() {
            if let Some(node_id) = self.node {
                result.set_requires_grad(true);
                with_autograd_context(|context| {
                    let inputs = vec![node_id];
                    let node_id = context.create_node(Operation::Abs, inputs);

                    // Register tensor data for gradient computation
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|x| Dtype::to_f64(x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node_id, data_f64, self.shape.clone());

                    result.node = Some(node_id);
                });
            }
        }

        result
    }

    /// Compute the hyperbolic tangent of each element
    ///
    /// # Returns
    /// New tensor with tanh values
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::scalar(0.0);
    /// let tanh_tensor = tensor.tanh();
    /// assert_eq!(tanh_tensor.item(), 0.0);
    /// ```
    pub fn tanh(&self) -> Tensor<T> {
        let data = self.data.iter().map(|x| x.tanh()).collect();
        let mut result = Tensor::from_vec(data, self.shape.clone());

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        // Create computational graph node if autograd is enabled
        if self.requires_grad() {
            if let Some(node_id) = self.node {
                result.set_requires_grad(true);
                with_autograd_context(|context| {
                    let inputs = vec![node_id];
                    let node_id = context.create_node(Operation::Tanh, inputs);

                    // Register tensor data for gradient computation
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|x| Dtype::to_f64(x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node_id, data_f64, self.shape.clone());

                    result.node = Some(node_id);
                });
            }
        }

        result
    }

    /// Compute the sigmoid of each element
    ///
    /// # Returns
    /// New tensor with sigmoid values
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::scalar(0.0);
    /// let sigmoid_tensor = tensor.sigmoid();
    /// assert_eq!(sigmoid_tensor.item(), 0.5);
    /// ```
    pub fn sigmoid(&self) -> Tensor<T> {
        let data = self
            .data
            .iter()
            .map(|x| T::one() / (T::one() + (-*x).exp()))
            .collect();
        let mut result = Tensor::from_vec(data, self.shape.clone());

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        // Create computational graph node if autograd is enabled
        if self.requires_grad() {
            if let Some(node_id) = self.node {
                result.set_requires_grad(true);
                with_autograd_context(|context| {
                    let inputs = vec![node_id];
                    let node_id = context.create_node(Operation::Sigmoid, inputs);

                    // Register tensor data for gradient computation
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|x| Dtype::to_f64(x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node_id, data_f64, self.shape.clone());

                    result.node = Some(node_id);
                });
            }
        }

        result
    }

    /// Compute the exponential of each element
    ///
    /// # Returns
    /// New tensor with exponential values
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::scalar(0.0);
    /// let exp_tensor = tensor.exp();
    /// assert_eq!(exp_tensor.item(), 1.0);
    /// ```
    pub fn exp(&self) -> Tensor<T> {
        let data = self.data.iter().map(|x| x.exp()).collect();
        let mut result = Tensor::from_vec(data, self.shape.clone());

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        // Create computational graph node if autograd is enabled
        if self.requires_grad() {
            if let Some(node_id) = self.node {
                result.set_requires_grad(true);
                with_autograd_context(|context| {
                    let inputs = vec![node_id];
                    let operation_node = context.create_node(Operation::Exp, inputs);

                    // Register result tensor data for gradient computation (not operation data)
                    let result_data_f64: Vec<f64> = result
                        .data
                        .iter()
                        .map(|x| Dtype::to_f64(x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(operation_node, result_data_f64, result.shape.clone());

                    result.node = Some(operation_node);
                });
            }
        }

        result
    }

    /// Compute the natural logarithm of each element
    ///
    /// # Returns
    /// New tensor with logarithm values
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::scalar(1.0);
    /// let log_tensor = tensor.log();
    /// assert_eq!(log_tensor.item(), 0.0);
    /// ```
    pub fn log(&self) -> Tensor<T> {
        let data = self.data.iter().map(|x| x.ln()).collect();
        let mut result = Tensor::from_vec(data, self.shape.clone());

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        // Create computational graph node if autograd is enabled
        if self.requires_grad() {
            if let Some(node_id) = self.node {
                result.set_requires_grad(true);
                with_autograd_context(|context| {
                    let inputs = vec![node_id];
                    let node_id = context.create_node(Operation::Log, inputs);

                    // Register tensor data for gradient computation
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|x| Dtype::to_f64(x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node_id, data_f64, self.shape.clone());

                    result.node = Some(node_id);
                });
            }
        }

        result
    }

    /// Compute the sine of each element
    ///
    /// # Returns
    /// New tensor with sine values
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::scalar(0.0);
    /// let sin_tensor = tensor.sin();
    /// assert_eq!(sin_tensor.item(), 0.0);
    /// ```
    pub fn sin(&self) -> Tensor<T> {
        let data = self.data.iter().map(|x| x.sin()).collect();
        let mut result = Tensor::from_vec(data, self.shape.clone());

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        // Create computational graph node if autograd is enabled
        if self.requires_grad() {
            if let Some(node_id) = self.node {
                result.set_requires_grad(true);
                with_autograd_context(|context| {
                    let inputs = vec![node_id];
                    let node_id = context.create_node(Operation::Sin, inputs);

                    // Register tensor data for gradient computation
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|x| Dtype::to_f64(x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node_id, data_f64, self.shape.clone());

                    result.node = Some(node_id);
                });
            }
        }

        result
    }

    /// Compute the cosine of each element
    ///
    /// # Returns
    /// New tensor with cosine values
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::scalar(0.0);
    /// let cos_tensor = tensor.cos();
    /// assert_eq!(cos_tensor.item(), 1.0);
    /// ```
    pub fn cos(&self) -> Tensor<T> {
        let data = self.data.iter().map(|x| x.cos()).collect();
        let mut result = Tensor::from_vec(data, self.shape.clone());

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        // Create computational graph node if autograd is enabled
        if self.requires_grad() {
            if let Some(node_id) = self.node {
                result.set_requires_grad(true);
                with_autograd_context(|context| {
                    let inputs = vec![node_id];
                    let node_id = context.create_node(Operation::Cos, inputs);

                    // Register tensor data for gradient computation
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|x| Dtype::to_f64(x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node_id, data_f64, self.shape.clone());

                    result.node = Some(node_id);
                });
            }
        }

        result
    }

    /// Compute the square root of each element
    ///
    /// # Returns
    /// New tensor with square root values
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::scalar(4.0);
    /// let sqrt_tensor = tensor.sqrt();
    /// assert_eq!(sqrt_tensor.item(), 2.0);
    /// ```
    pub fn sqrt(&self) -> Tensor<T> {
        let data = self.data.iter().map(|x| x.sqrt()).collect();
        let mut result = Tensor::from_vec(data, self.shape.clone());

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        // Create computational graph node if autograd is enabled
        if self.requires_grad() {
            if let Some(node_id) = self.node {
                result.set_requires_grad(true);
                with_autograd_context(|context| {
                    let inputs = vec![node_id];
                    let node_id = context.create_node(Operation::Sqrt, inputs);

                    // Register tensor data for gradient computation
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|x| Dtype::to_f64(x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node_id, data_f64, self.shape.clone());

                    result.node = Some(node_id);
                });
            }
        }

        result
    }

    /// Apply the Rectified Linear Unit activation function
    ///
    /// # Returns
    /// New tensor with ReLU applied element-wise
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![-1.0, 0.0, 1.0], vec![3]);
    /// let relu_tensor = tensor.relu();
    /// assert_eq!(relu_tensor.data(), &[0.0, 0.0, 1.0]);
    /// ```
    pub fn relu(&self) -> Tensor<T> {
        let data = self
            .data
            .iter()
            .map(|x| if *x > T::zero() { *x } else { T::zero() })
            .collect();
        let mut result = Tensor::from_vec(data, self.shape.clone());

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        // Create computational graph node if autograd is enabled
        if self.requires_grad() {
            if let Some(node_id) = self.node {
                result.set_requires_grad(true);
                with_autograd_context(|context| {
                    let inputs = vec![node_id];
                    let node_id = context.create_node(Operation::Relu, inputs);

                    // Register tensor data for gradient computation
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|x| Dtype::to_f64(x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node_id, data_f64, self.shape.clone());

                    result.node = Some(node_id);
                });
            }
        }

        result
    }
}
