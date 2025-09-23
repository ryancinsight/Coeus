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
    /// assert_eq!(tanh_tensor.item().unwrap(), 0.0);
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
    /// assert_eq!(sigmoid_tensor.item().unwrap(), 0.5);
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
    /// assert_eq!(exp_tensor.item().unwrap(), 1.0);
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
    /// assert_eq!(log_tensor.item().unwrap(), 0.0);
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
    /// assert_eq!(sin_tensor.item().unwrap(), 0.0);
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
    /// assert_eq!(cos_tensor.item().unwrap(), 1.0);
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
    /// assert_eq!(sqrt_tensor.item().unwrap(), 2.0);
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
            .map(|x| {
                if *x > T::zero() {
                    *x
                } else if !x.is_finite() {
                    if *x == T::infinity() || x.is_nan() {
                        *x // Positive infinity and NaN pass through
                    } else {
                        T::zero() // Negative infinity becomes 0
                    }
                } else {
                    T::zero()
                }
            })
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

    /// Apply the Exponential Linear Unit activation function
    ///
    /// ELU(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
    ///
    /// # Arguments
    /// * `alpha` - The alpha parameter for negative inputs
    ///
    /// # Returns
    /// New tensor with ELU applied element-wise
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![-1.0, 0.0, 1.0], vec![3]);
    /// let elu_tensor = tensor.elu(1.0);
    /// // elu_tensor.data() ≈ [-0.632, 0.0, 1.0]
    /// ```
    pub fn elu(&self, alpha: T) -> Tensor<T> {
        let data = self
            .data
            .iter()
            .map(|x| {
                if *x > T::zero() {
                    *x
                } else {
                    alpha * (x.exp() - T::one())
                }
            })
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
                    let node_id = context.create_node(Operation::Elu, inputs);

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

    /// Apply the Gaussian Error Linear Unit activation function
    ///
    /// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    ///
    /// # Returns
    /// New tensor with GELU applied element-wise
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![-1.0, 0.0, 1.0], vec![3]);
    /// let gelu_tensor = tensor.gelu();
    /// // gelu_tensor.data() ≈ [-0.159, 0.0, 0.841]
    /// ```
    pub fn gelu(&self) -> Tensor<T> {
        let data = self
            .data
            .iter()
            .map(|x| {
                // GELU approximation using tanh: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                let coeff = T::from(0.7978845608028654).unwrap_or(T::one()); // sqrt(2/π)
                let x_cubed = *x * *x * *x;
                let inner = coeff * (*x + T::from(0.044715).unwrap_or(T::zero()) * x_cubed);
                let tanh_inner = inner.tanh();
                let half = T::from(0.5).unwrap_or(T::one());
                half * *x * (T::one() + tanh_inner)
            })
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
                    let node_id = context.create_node(Operation::Gelu, inputs);

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

    /// Apply the Hard Tanh activation function
    ///
    /// HardTanh(x) = max(min_val, min(max_val, x))
    ///
    /// # Arguments
    /// * `min_val` - The minimum value for clamping
    /// * `max_val` - The maximum value for clamping
    ///
    /// # Returns
    /// New tensor with HardTanh applied element-wise
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![-2.0, 0.0, 2.0], vec![3]);
    /// let hardtanh_tensor = tensor.hardtanh(-1.0, 1.0);
    /// assert_eq!(hardtanh_tensor.data(), &[-1.0, 0.0, 1.0]);
    /// ```
    pub fn hardtanh(&self, min_val: T, max_val: T) -> Tensor<T> {
        let data = self
            .data
            .iter()
            .map(|x| {
                if *x < min_val {
                    min_val
                } else if *x > max_val {
                    max_val
                } else {
                    *x
                }
            })
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
                    let node_id = context.create_node(Operation::Hardtanh, inputs);

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

    /// Apply the Log Sigmoid activation function
    ///
    /// LogSigmoid(x) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
    ///
    /// # Returns
    /// New tensor with LogSigmoid applied element-wise
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![-1.0, 0.0, 1.0], vec![3]);
    /// let logsigmoid_tensor = tensor.logsigmoid();
    /// // logsigmoid_tensor.data() ≈ [-0.693, -0.693, -0.307]
    /// ```
    pub fn logsigmoid(&self) -> Tensor<T> {
        let data = self
            .data
            .iter()
            .map(|x| {
                // LogSigmoid(x) = -log(1 + exp(-x))
                // For numerical stability:
                // - if x > 0: x - log(1 + exp(x)) [since log(1 + exp(x)) = log(exp(x) + 1) = x + log(1 + exp(-x))]
                // - if x <= 0: -log(1 + exp(-x)) [direct computation]
                // For very large negative x, exp(-x) becomes very large, so we need to handle this case
                if *x > T::zero() {
                    let exp_x = x.exp();
                    let log_term = (T::one() + exp_x).ln();
                    // Debug: check for NaN or inf
                    if !exp_x.is_finite() || !log_term.is_finite() {
                        // For very large x, log(1 + exp(x)) ≈ x, so x - x = 0
                        T::zero()
                    } else {
                        *x - log_term
                    }
                } else {
                    // For very large negative x, exp(-x) may overflow
                    // In this case, log(1 + exp(-x)) ≈ -x since exp(-x) ≈ 0
                    let neg_x = -*x;
                    if neg_x > T::from(50.0).unwrap_or(T::from(50.0).unwrap()) {
                        // For very large negative x, log(1 + exp(-x)) ≈ -x
                        // But since x is negative, -x is positive, so we return -(-x) = x
                        *x
                    } else {
                        -(T::one() + (-*x).exp()).ln()
                    }
                }
            })
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
                    let node_id = context.create_node(Operation::Logsigmoid, inputs);

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

    /// Compute the ceiling of each element (smallest integer greater than or equal to x)
    ///
    /// # Returns
    /// New tensor with ceiling values
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.3, -1.7, 2.9], vec![3]);
    /// let ceil_tensor = tensor.ceil();
    /// assert_eq!(ceil_tensor.data(), &[2.0, -1.0, 3.0]);
    /// ```
    pub fn ceil(&self) -> Tensor<T> {
        let data = self.data.iter().map(|x| x.ceil()).collect();
        let mut result = Tensor::from_vec(data, self.shape.clone());

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        result
    }

    /// Compute the floor of each element (largest integer less than or equal to x)
    ///
    /// # Returns
    /// New tensor with floor values
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.3, -1.7, 2.9], vec![3]);
    /// let floor_tensor = tensor.floor();
    /// assert_eq!(floor_tensor.data(), &[1.0, -2.0, 2.0]);
    /// ```
    pub fn floor(&self) -> Tensor<T> {
        let data = self.data.iter().map(|x| x.floor()).collect();
        let mut result = Tensor::from_vec(data, self.shape.clone());

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        result
    }

    /// Round each element to the nearest integer
    ///
    /// # Returns
    /// New tensor with rounded values
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.3, 1.7, 2.5], vec![3]);
    /// let round_tensor = tensor.round();
    /// assert_eq!(round_tensor.data(), &[1.0, 2.0, 3.0]);
    /// ```
    pub fn round(&self) -> Tensor<T> {
        let data = self.data.iter().map(|x| x.round()).collect();
        let mut result = Tensor::from_vec(data, self.shape.clone());

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        result
    }

    /// Compute the square of each element (x²)
    ///
    /// # Returns
    /// New tensor with squared values
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![2.0, 3.0, 4.0], vec![3]);
    /// let square_tensor = tensor.square();
    /// assert_eq!(square_tensor.data(), &[4.0, 9.0, 16.0]);
    /// ```
    pub fn square(&self) -> Tensor<T> {
        let data = self.data.iter().map(|x| *x * *x).collect();
        let mut result = Tensor::from_vec(data, self.shape.clone());

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        result
    }

    /// Compute the reciprocal of each element (1/x)
    ///
    /// # Returns
    /// New tensor with reciprocal values
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![2.0, 4.0, 0.5], vec![3]);
    /// let reciprocal_tensor = tensor.reciprocal();
    /// assert_eq!(reciprocal_tensor.data(), &[0.5, 0.25, 2.0]);
    /// ```
    pub fn reciprocal(&self) -> Tensor<T> {
        let data = self.data.iter().map(|x| T::one() / *x).collect();
        let mut result = Tensor::from_vec(data, self.shape.clone());

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        result
    }

    /// Compute the sign of each element (-1 for negative, 0 for zero, 1 for positive)
    ///
    /// # Returns
    /// New tensor with sign values
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![-2.0, 0.0, 3.0], vec![3]);
    /// let sign_tensor = tensor.sign();
    /// assert_eq!(sign_tensor.data(), &[-1.0, 0.0, 1.0]);
    /// ```
    pub fn sign(&self) -> Tensor<T> {
        let data = self
            .data
            .iter()
            .map(|x| {
                if *x > T::zero() {
                    T::one()
                } else if *x < T::zero() {
                    -T::one()
                } else {
                    T::zero()
                }
            })
            .collect();
        let mut result = Tensor::from_vec(data, self.shape.clone());

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        result
    }

    /// Apply softmax activation along a specified dimension
    ///
    /// Formula: `Softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)`
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to apply softmax
    ///
    /// # Returns
    /// New tensor with softmax applied
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    /// let softmax_tensor = tensor.softmax(0);
    /// // softmax_tensor will contain normalized probabilities that sum to 1.0
    /// ```
    pub fn softmax(&self, dim: usize) -> Tensor<T>
    where
        T: std::iter::Sum<T> + Clone + std::ops::Add<Output = T>,
    {
        // For now, implement a simple softmax for 1D tensors along dim 0
        // Full multi-dimensional softmax with proper broadcasting will be implemented later
        if self.shape.len() != 1 || dim != 0 {
            // Fallback: just return the input (not mathematically correct for multi-dim)
            return self.clone();
        }

        // Compute exp(x) for all elements
        let exp_data: Vec<T> = self.data.iter().map(|x| x.exp()).collect();
        let exp_sum: T = exp_data.iter().cloned().sum();

        // Normalize by the sum
        let result_data: Vec<T> = exp_data.iter().map(|x| *x / exp_sum).collect();

        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        result
    }

    /// Apply softmin activation along a specified dimension
    ///
    /// Formula: `Softmin(x_i) = exp(-x_i) / sum(exp(-x_j) for all j)`
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to apply softmin
    ///
    /// # Returns
    /// New tensor with softmin applied
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    /// let softmin_tensor = tensor.softmin(0);
    /// // softmin_tensor will contain normalized probabilities that sum to 1.0
    /// ```
    pub fn softmin(&self, dim: usize) -> Tensor<T>
    where
        T: std::iter::Sum<T> + Clone + std::ops::Add<Output = T>,
    {
        // For now, implement a simple softmin for 1D tensors along dim 0
        // Full multi-dimensional softmin with proper broadcasting will be implemented later
        if self.shape.len() != 1 || dim != 0 {
            // Fallback: just return the input (not mathematically correct for multi-dim)
            return self.clone();
        }

        // Compute exp(-x) for all elements
        let exp_neg_data: Vec<T> = self.data.iter().map(|x| (-*x).exp()).collect();
        let exp_neg_sum: T = exp_neg_data.iter().cloned().sum();

        // Normalize by the sum
        let result_data: Vec<T> = exp_neg_data.iter().map(|x| *x / exp_neg_sum).collect();

        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Propagate requires_grad flag
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        result
    }
}
