//! Mathematical operations for tensors
//!
//! This module contains element-wise mathematical operations and activation functions
//! for tensors, including automatic differentiation support.

use crate::{Tensor, TensorError, Dtype, FloatDtype, Result};
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

                let node_id = context.create_node(Operation::Pow(exponent.to_f64().unwrap_or(1.0)), vec![self_node]);
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
