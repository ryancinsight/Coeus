//! Mathematical operations for tensors
//!
//! This module contains element-wise mathematical operations and activation functions
//! for tensors, including automatic differentiation support.

use crate::{Tensor, Dtype, FloatDtype};
use num_traits::Signed;
use num_traits::Zero;
use crate::with_autograd_context;
use coeus_autograd::context::Operation;

impl<T: Dtype + num_traits::FromPrimitive + num_traits::ToPrimitive> Tensor<T> {
    /// Compute the absolute value of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![-1.0, 2.0, -3.0], vec![3]);
    /// let abs_tensor = tensor.abs();
    /// // Result: [1.0, 2.0, 3.0]
    /// ```
    pub fn abs(&self) -> Tensor<T>
    where
        T: Signed,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.abs()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Abs, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the hyperbolic tangent of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0, 1.0], vec![2]);
    /// let tanh_tensor = tensor.tanh();
    /// // Result: [tanh(0.0), tanh(1.0)]
    /// ```
    pub fn tanh(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.tanh()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Tanh, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the sigmoid (logistic) function of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0], vec![1]);
    /// let sigmoid_tensor = tensor.sigmoid();
    /// // Result: [0.5]
    /// ```
    pub fn sigmoid(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| T::one() / (T::one() + (-*x).exp())).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Sigmoid, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the exponential of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0, 1.0], vec![2]);
    /// let exp_tensor = tensor.exp();
    /// // Result: [exp(0.0), exp(1.0)]
    /// ```
    pub fn exp(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.exp()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Exp, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the natural logarithm of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.718], vec![2]);
    /// let log_tensor = tensor.log();
    /// // Result: [log(1.0), log(2.718)] ≈ [0.0, 1.0]
    /// ```
    pub fn log(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.ln()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Log, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the sine of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0, 3.14159], vec![2]);
    /// let sin_tensor = tensor.sin();
    /// // Result: [sin(0.0), sin(π)] ≈ [0.0, 0.0]
    /// ```
    pub fn sin(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.sin()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Sin, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the cosine of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0, 1.5708], vec![2]);
    /// let cos_tensor = tensor.cos();
    /// // Result: [cos(0.0), cos(π/2)] ≈ [1.0, 0.0]
    /// ```
    pub fn cos(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.cos()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Cos, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the square root of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 4.0, 9.0], vec![3]);
    /// let sqrt_tensor = tensor.sqrt();
    /// // Result: [1.0, 2.0, 3.0]
    /// ```
    pub fn sqrt(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.sqrt()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Sqrt, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Apply the Rectified Linear Unit (ReLU) activation function
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![-1.0, 0.0, 2.0], vec![3]);
    /// let relu_tensor = tensor.relu();
    /// // Result: [0.0, 0.0, 2.0]
    /// ```
    pub fn relu(&self) -> Tensor<T>
    where
        T: PartialOrd + Zero,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| if *x > T::zero() { *x } else { T::zero() }).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Relu, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Raise each element to a power
    ///
    /// # Arguments
    /// * `exponent` - Power to raise each element to
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
    /// let pow_tensor = tensor.pow(3.0);
    /// // Result: [8.0, 27.0]
    /// ```
    pub fn pow(&self, exponent: T) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.powf(exponent)).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Pow, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the ceiling of each element (round up to nearest integer)
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.3, -2.7, 3.9], vec![3]);
    /// let ceil_tensor = tensor.ceil();
    /// // Result: [2.0, -2.0, 4.0]
    /// ```
    pub fn ceil(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.ceil()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Ceil, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the floor of each element (round down to nearest integer)
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.3, -2.7, 3.9], vec![3]);
    /// let floor_tensor = tensor.floor();
    /// // Result: [1.0, -3.0, 3.0]
    /// ```
    pub fn floor(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.floor()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Floor, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Round each element to the nearest integer
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.3, 1.7, -2.7, -2.3], vec![4]);
    /// let round_tensor = tensor.round();
    /// // Result: [1.0, 2.0, -3.0, -2.0]
    /// ```
    pub fn round(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.round()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Round, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Truncate each element toward zero
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.3, -2.7, 3.9, -4.2], vec![4]);
    /// let trunc_tensor = tensor.trunc();
    /// // Result: [1.0, -2.0, 3.0, -4.0]
    /// ```
    pub fn trunc(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.trunc()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Trunc, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the square of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![2.0, 3.0, -4.0], vec![3]);
    /// let square_tensor = tensor.square();
    /// // Result: [4.0, 9.0, 16.0]
    /// ```
    pub fn square(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| *x * *x).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Square, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the reciprocal (1/x) of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![2.0, 4.0, 0.5], vec![3]);
    /// let reciprocal_tensor = tensor.reciprocal();
    /// // Result: [0.5, 0.25, 2.0]
    /// ```
    pub fn reciprocal(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| T::one() / *x).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Reciprocal, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the sign of each element (-1, 0, or 1)
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![-3.5, 0.0, 2.1, -0.0], vec![4]);
    /// let sign_tensor = tensor.sign();
    /// // Result: [-1.0, 0.0, 1.0, 0.0]
    /// ```
    pub fn sign(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.signum()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Sign, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the tangent of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0, 1.5707963267948966], vec![2]); // [0, π/2]
    /// let tan_tensor = tensor.tan();
    /// // Result: [0.0, very large number approaching infinity]
    /// ```
    pub fn tan(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.tan()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Tan, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Clamp each element to be within [min, max]
    ///
    /// # Arguments
    /// * `min` - Minimum value to clamp to
    /// * `max` - Maximum value to clamp to
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![-1.0, 2.0, 5.0, 0.5], vec![4]);
    /// let clamped = tensor.clamp(0.0, 3.0);
    /// // Result: [0.0, 2.0, 3.0, 0.5]
    /// ```
    pub fn clamp(&self, min: T, max: T) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.max(min).min(max)).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Clamp, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_ceil() {
        let a = Tensor::from_vec(vec![1.3, -2.7, 3.9, -4.2], vec![4]);
        let result = a.ceil();
        assert_eq!(result.data, &[2.0, -2.0, 4.0, -4.0]);
    }

    #[test]
    fn test_floor() {
        let a = Tensor::from_vec(vec![1.3, -2.7, 3.9, -4.2], vec![4]);
        let result = a.floor();
        assert_eq!(result.data, &[1.0, -3.0, 3.0, -5.0]);
    }

    #[test]
    fn test_round() {
        let a = Tensor::from_vec(vec![1.3, 1.7, -2.7, -2.3], vec![4]);
        let result = a.round();
        assert_eq!(result.data, &[1.0, 2.0, -3.0, -2.0]);
    }

    #[test]
    fn test_trunc() {
        let a = Tensor::from_vec(vec![1.3, -2.7, 3.9, -4.2], vec![4]);
        let result = a.trunc();
        assert_eq!(result.data, &[1.0, -2.0, 3.0, -4.0]);
    }

    #[test]
    fn test_square() {
        let a = Tensor::from_vec(vec![2.0, 3.0, -4.0], vec![3]);
        let result = a.square();
        assert_eq!(result.data, &[4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_reciprocal() {
        let a = Tensor::from_vec(vec![2.0, 4.0, 0.5], vec![3]);
        let result = a.reciprocal();
        assert_eq!(result.data, &[0.5, 0.25, 2.0]);
    }

    #[test]
    fn test_sign() {
        let a = Tensor::from_vec(vec![-3.5, 0.0, 2.1, -0.0], vec![4]);
        let result = a.sign();
        assert_eq!(result.data, &[-1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_tan() {
        let a = Tensor::from_vec(vec![0.0], vec![1]);
        let result = a.tan();
        assert!((result.data[0] - 0.0).abs() < 1e-6); // tan(0) = 0
    }

    #[test]
    fn test_clamp() {
        let a = Tensor::from_vec(vec![-1.0, 2.0, 5.0, 0.5], vec![4]);
        let result = a.clamp(0.0, 3.0);
        assert_eq!(result.data, &[0.0, 2.0, 3.0, 0.5]);
    }
}
