//! Linear (fully connected) neural network layer
//!
//! This module implements the linear transformation: `output = input @ weight.T + bias`
//!
//! ## Mathematical Foundation
//!
//! A linear layer performs an affine transformation:
//!
//! ```math
//! y = xW^T + b
//! ```
//!
//! Where:
//! - `x` is the input tensor of shape `(batch_size, in_features)`
//! - `W` is the weight matrix of shape `(out_features, in_features)`
//! - `b` is the bias vector of shape `(out_features,)`
//! - `y` is the output tensor of shape `(batch_size, out_features)`
//!
//! ## Gradient Computation
//!
//! The backward pass computes:
//!
//! ```math
//! ∂L/∂W = (∂L/∂y)^T @ x
//! ∂L/∂b = sum(∂L/∂y, axis=0)
//! ∂L/∂x = (∂L/∂y) @ W
//! ```
//!
//! ## References
//!
//! - [Deep Learning Book - Linear Algebra](https://www.deeplearningbook.org/contents/linear_algebra.html)
//! - [PyTorch Linear Layer](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
//! - [Neural Networks: A Systematic Introduction](https://page.mi.fu-berlin.de/rojas/neural/)

use crate::{Module, NNError, Result};
use coeus_backend::CpuBackend;
use coeus_tensor::{Dtype, FloatDtype, Tensor};
use rand::prelude::*;
use std::fmt;

/// Linear (fully connected) neural network layer
#[derive(Debug, Clone)]
pub struct Linear<T: FloatDtype> {
    /// Weight matrix of shape (out_features, in_features)
    pub weight: Tensor<T, CpuBackend>,
    /// Bias vector of shape (out_features,)
    pub bias: Option<Tensor<T, CpuBackend>>,
    /// Number of input features
    pub in_features: usize,
    /// Number of output features
    pub out_features: usize,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> Linear<T> {
    /// Create a new linear layer with Kaiming uniform initialization
    ///
    /// # Arguments
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::Linear;
    ///
    /// let layer: Linear<f32> = Linear::new(784, 128);
    /// assert_eq!(layer.in_features, 784);
    /// assert_eq!(layer.out_features, 128);
    /// ```
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self::new_with_bias(in_features, out_features, true)
    }

    /// Create a new linear layer with optional bias
    ///
    /// # Arguments
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    /// * `bias` - Whether to include a bias term
    pub fn new_with_bias(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Kaiming uniform initialization for weights
        // Based on: https://arxiv.org/abs/1502.01852
        let bound = (6.0 / (in_features + out_features) as f64).sqrt();
        let mut rng = rand::thread_rng();

        let weight_data: Vec<T> = (0..in_features * out_features)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();

        let mut weight = Tensor::from_vec(CpuBackend::default(), weight_data, vec![out_features, in_features]).unwrap();
        weight.set_requires_grad(true); // Linear weights need gradients for training

        let bias_tensor = if bias {
            let bias_data: Vec<T> = (0..out_features)
                .map(|_| {
                    let val: f64 = rng.gen_range(-bound..bound);
                    <T as Dtype>::from_f64(val).unwrap()
                })
                .collect();
            let mut bias_tensor = Tensor::from_vec(CpuBackend::default(), bias_data, vec![out_features]).unwrap();
            bias_tensor.set_requires_grad(true); // Linear bias needs gradients for training
            Some(bias_tensor)
        } else {
            None
        };

        Self {
            weight,
            bias: bias_tensor,
            in_features,
            out_features,
        }
    }

    /// Create a linear layer with custom weight and bias initialization
    ///
    /// # Arguments
    /// * `weight` - Weight tensor of shape (out_features, in_features)
    /// * `bias` - Optional bias tensor of shape (out_features,)
    pub fn from_tensors(weight: Tensor<T, CpuBackend>, bias: Option<Tensor<T, CpuBackend>>) -> Result<Self> {
        let weight_shape = weight.shape();
        if weight_shape.len() != 2 {
            return Err(NNError::InvalidInput {
                message: "Weight tensor must be 2D".to_string(),
            });
        }

        let out_features = weight_shape[0];
        let in_features = weight_shape[1];

        if let Some(ref bias_tensor) = bias {
            let bias_shape = bias_tensor.shape();
            if bias_shape != [out_features] {
                return Err(NNError::ShapeMismatch {
                    expected: vec![out_features],
                    actual: bias_shape.to_vec(),
                });
            }
        }

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    /// Enable gradient computation for the layer parameters
    pub fn requires_grad(&mut self, requires_grad: bool) {
        self.weight.set_requires_grad(requires_grad);
        if let Some(ref mut bias) = self.bias {
            bias.set_requires_grad(requires_grad);
        }
    }
}

impl<T: FloatDtype> Module<T> for Linear<T> {
    /// Forward pass through the linear layer
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (..., in_features)
    ///
    /// # Returns
    /// Output tensor of shape (..., out_features)
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::{Linear, Module};
    /// use coeus_tensor::Tensor;
    ///
    /// let layer: Linear<f32> = Linear::new(10, 5);
    /// let input = Tensor::from_vec(CpuBackend::default(), vec![1.0; 10], vec![10]).unwrap();
    /// let output = layer.forward(&input).unwrap();
    /// assert_eq!(output.shape(), &[5]);
    /// ```
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        // Ensure input has correct shape
        let input_shape = input.shape();

        let input_features: usize =
            input_shape
                .last()
                .copied()
                .ok_or_else(|| crate::NNError::InvalidInput {
                    message: "Cannot get input features from empty shape for Linear layer"
                        .to_string(),
                })?;

        if input_features != self.in_features {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "Input features {} do not match layer input features {}",
                    input_features, self.in_features
                ),
            });
        }

        // Support multi-dimensional inputs by flattening all but the last dimension
        // This matches PyTorch behavior: (..., in_features) -> (..., out_features)
        let batch_dims: usize = input_shape.iter().take(input_shape.len() - 1).product();
        let flattened_shape = vec![batch_dims, input_features];

        // Reshape input for matrix multiplication
        let input_reshaped = input.reshape(flattened_shape)?;

        // Simple matrix multiplication: (batch_size, in_features) @ (out_features, in_features).T
        let weight_t = self.weight.t().map_err(|e| crate::NNError::InvalidInput {
            message: format!("Failed to transpose weights in Linear layer: {}", e),
        })?;

        let output_reshaped = input_reshaped.matmul(&weight_t)?;

        // Reshape output back to original batch dimensions plus output features
        let mut output_shape: Vec<usize> = input_shape.to_vec();
        output_shape[input_shape.len() - 1] = self.out_features;
        let mut output = output_reshaped.reshape(output_shape.clone())?;

        // Add bias if present
        if let Some(ref bias) = self.bias {
            // Use broadcasting addition from arithmetic module
            use coeus_tensor::ops::arithmetic;
            output = arithmetic::add(&output, bias)?;
        }

        // Set up autograd graph if any input requires gradients
        if input.requires_grad() || self.weight.requires_grad() || (self.bias.is_some() && self.bias.as_ref().unwrap().requires_grad()) {
            output.set_requires_grad(true);

            // Create a custom operation for Linear backward pass
            use coeus_tensor::core::tensor::{with_autograd_context, Operation};
            with_autograd_context(|context| {
                let input_node = input.node_id().unwrap_or_else(|| {
                    context.create_leaf_node()
                });
                let weight_node = self.weight.node_id().unwrap_or_else(|| {
                    context.create_leaf_node()
                });

                let mut input_nodes = vec![input_node, weight_node];
                let mut input_data = vec![
                    input.data().iter().map(|&x| <T as Dtype>::to_f64(&x).unwrap_or(0.0)).collect::<Vec<f64>>(),
                    self.weight.data().iter().map(|&x| <T as Dtype>::to_f64(&x).unwrap_or(0.0)).collect::<Vec<f64>>(),
                ];
                let mut input_shapes = vec![input.shape().to_vec(), self.weight.shape().to_vec()];

                if let Some(ref bias) = self.bias {
                    input_nodes.push(bias.node_id().unwrap_or_else(|| {
                        context.create_leaf_node()
                    }));
                    input_data.push(bias.data().iter().map(|&x| <T as Dtype>::to_f64(&x).unwrap_or(0.0)).collect::<Vec<f64>>());
                    input_shapes.push(bias.shape().to_vec());
                }

                let result_node = context.create_node_with_data_and_shapes(
                    Operation::Linear,
                    input_nodes,
                    input_data,
                    input_shapes
                );
                output.set_node_id(result_node);
            });
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
}

impl<T: FloatDtype> fmt::Display for Linear<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Linear(in_features={}, out_features={}, bias={})",
            self.in_features,
            self.out_features,
            self.bias.is_some()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_linear_creation() {
        let layer = Linear::<f32>::new(10, 5);
        assert_eq!(layer.in_features, 10);
        assert_eq!(layer.out_features, 5);
        assert_eq!(layer.weight.shape(), &[5, 10]);
        assert!(layer.bias.is_some());
        assert_eq!(layer.bias.as_ref().unwrap().shape(), &[5]);
    }

    #[test]
    fn test_linear_forward() {
        let layer = Linear::<f32>::new(3, 2);

        // Simple input
        let input = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let output = layer
            .forward(&input)
            .expect("Linear forward should succeed");

        assert_eq!(output.shape(), &[2]);
        assert_eq!(output.numel(), 2);
    }

    #[test]
    fn test_linear_batch_forward() {
        let layer = Linear::<f32>::new(3, 2);

        // Batch input
        let input = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let output = layer
            .forward(&input)
            .expect("Linear batch forward should succeed");

        assert_eq!(output.shape(), &[2, 2]);
    }

    #[test]
    fn test_linear_without_bias() {
        let layer = Linear::<f32>::new_with_bias(3, 2, false);
        assert!(layer.bias.is_none());

        let input = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let output = layer
            .forward(&input)
            .expect("Linear without bias forward should succeed");

        assert_eq!(output.shape(), &[2]);
    }

    #[test]
    fn test_linear_gradient_flow() {
        let mut layer = Linear::<f32>::new(3, 2);
        layer.requires_grad(true);

        let mut input = Tensor::from_vec_with_grad(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let output = layer
            .forward(&input)
            .expect("Linear gradient flow forward should succeed");

        // Compute some loss (sum of outputs)
        let loss = output.sum();

        // Backward pass
        let _ = loss.backward();

        // Debug output
        println!("Weight requires_grad: {}", layer.weight.requires_grad());
        println!("Weight grad: {:?}", layer.weight.grad().is_some());
        println!("Input grad: {:?}", input.grad().is_some());
        println!("Loss grad: {:?}", loss.grad().is_some());

        // Check that gradients exist and flow properly
        assert!(loss.grad().is_some(), "Loss tensor should have gradients");

        // Weight gradients should flow through transpose -> matmul -> sum operations
        assert!(
            layer.weight.grad().is_some(),
            "Weight tensor should have gradients"
        );
        if let Some(ref bias) = layer.bias {
            assert!(bias.grad().is_some(), "Bias tensor should have gradients");
        }

        // Input gradients should also flow
        assert!(input.grad().is_some(), "Input tensor should have gradients");

        // For now, just ensure the weight still requires gradients (setup is correct)
        assert!(
            layer.weight.requires_grad(),
            "Weight should still require gradients"
        );
    }

    #[test]
    fn test_linear_numerical_validation() {
        // Test that the linear transformation is mathematically correct
        let layer = Linear::<f64>::from_tensors(
            Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(),
            Some(Tensor::from_vec(CpuBackend::default(), vec![0.1, 0.2], vec![2]).unwrap()),
        )
        .unwrap();

        let input = Tensor::from_vec(CpuBackend::default(), vec![1.0, 1.0], vec![2]).unwrap();
        let output = layer
            .forward(&input)
            .expect("Linear manual computation forward should succeed");

        // Manual computation for weight [[1, 2], [3, 4]] and input [1, 1]:
        // output[0] = 1*1 + 2*1 + 0.1 = 3.1
        // output[1] = 3*1 + 4*1 + 0.2 = 7.2
        let expected = [3.1, 7.2];

        for (i, &expected_val) in expected.iter().enumerate() {
            assert_relative_eq!(output.data()[i], expected_val, epsilon = 1e-10);
        }
    }
}


