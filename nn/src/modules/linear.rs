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
use coeus_tensor::{FloatDtype, Tensor};
use rand::prelude::*;
use std::fmt;
use std::ops::Add;

/// Linear (fully connected) neural network layer
#[derive(Debug, Clone)]
pub struct Linear<T: FloatDtype> {
    /// Weight matrix of shape (out_features, in_features)
    pub weight: Tensor<T>,
    /// Bias vector of shape (out_features,)
    pub bias: Option<Tensor<T>>,
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
                T::from_f64(val).unwrap()
            })
            .collect();

        let weight = Tensor::from_vec(weight_data, vec![out_features, in_features]);

        let bias_tensor = if bias {
            let bias_data: Vec<T> = (0..out_features)
                .map(|_| {
                    let val: f64 = rng.gen_range(-bound..bound);
                    T::from_f64(val).unwrap()
                })
                .collect();
            Some(Tensor::from_vec(bias_data, vec![out_features]))
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
    pub fn from_tensors(weight: Tensor<T>, bias: Option<Tensor<T>>) -> Result<Self> {
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
    /// let input = Tensor::from_vec(vec![1.0; 10], vec![10]);
    /// let output = layer.forward(&input).unwrap();
    /// assert_eq!(output.shape(), &[5]);
    /// ```
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // Ensure input has correct shape
        let input_shape = input.shape();

        let input_features =
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

        // Support multi-dimensional inputs with proper reshaping
        // Handle different input shapes by flattening all but last dimension
        let original_shape = input_shape;
        let input_2d = if input_shape.len() > 2 {
            let batch_size: usize = input_shape.iter().take(input_shape.len() - 1).product();
            let in_features = input_shape[input_shape.len() - 1];
            input.reshape(vec![batch_size, in_features]).map_err(|e| {
                crate::NNError::InvalidInput {
                    message: format!("Failed to reshape input for Linear layer: {}", e),
                }
            })?
        } else if input_shape.len() == 1 {
            // Handle 1D input by adding batch dimension
            input
                .unsqueeze(0)
                .map_err(|e| crate::NNError::InvalidInput {
                    message: format!("Failed to unsqueeze input for Linear layer: {}", e),
                })?
        } else {
            input.clone()
        };

        if input_2d.shape()[1] != self.in_features {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "Reshaped input features {} do not match layer input features {}",
                    input_2d.shape()[1],
                    self.in_features
                ),
            });
        }

        // Simple matrix multiplication: (batch_size, in_features) @ (out_features, in_features).T
        let weight_t = self.weight.t().map_err(|e| crate::NNError::InvalidInput {
            message: format!("Failed to transpose weights in Linear layer: {}", e),
        })?;

        let output_2d = input_2d.matmul(&weight_t)?;

        // Add bias if present using autograd-enabled operations
        let output_2d = if let Some(ref bias) = self.bias {
            // For 2D case, bias should be (out_features,) and we broadcast to (batch_size, out_features)
            let bias_expanded = bias
                .unsqueeze(0)
                .map_err(|e| crate::NNError::InvalidInput {
                    message: format!("Failed to unsqueeze bias in Linear layer: {}", e),
                })?
                .expand(vec![output_2d.shape()[0], self.out_features])
                .map_err(|e| crate::NNError::InvalidInput {
                    message: format!("Failed to expand bias in Linear layer: {}", e),
                })?;

            output_2d
                .add(&bias_expanded)
                .map_err(|e| crate::NNError::InvalidInput {
                    message: format!("Bias addition failed in Linear layer: {}", e),
                })?
        } else {
            output_2d
        };

        // Reshape output back to original shape if input was reshaped
        if original_shape.len() > 2 {
            let mut output_shape = original_shape.to_vec();
            output_shape[original_shape.len() - 1] = self.out_features;
            output_2d
                .reshape(output_shape)
                .map_err(|e| crate::NNError::InvalidInput {
                    message: format!("Failed to reshape output in Linear layer: {}", e),
                })
        } else if original_shape.len() == 1 {
            // Remove batch dimension for 1D input - use reshape to remove dimension
            let squeezed_shape = vec![self.out_features];
            output_2d
                .reshape(squeezed_shape)
                .map_err(|e| crate::NNError::InvalidInput {
                    message: format!(
                        "Failed to reshape output for 1D input in Linear layer: {}",
                        e
                    ),
                })
        } else {
            Ok(output_2d)
        }
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
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
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
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
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let output = layer
            .forward(&input)
            .expect("Linear batch forward should succeed");

        assert_eq!(output.shape(), &[2, 2]);
    }

    #[test]
    fn test_linear_without_bias() {
        let layer = Linear::<f32>::new_with_bias(3, 2, false);
        assert!(layer.bias.is_none());

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let output = layer
            .forward(&input)
            .expect("Linear without bias forward should succeed");

        assert_eq!(output.shape(), &[2]);
    }

    #[test]
    fn test_linear_gradient_flow() {
        let mut layer = Linear::<f32>::new(3, 2);
        layer.requires_grad(true);

        let input = Tensor::from_vec_with_grad(vec![1.0, 2.0, 3.0], vec![1, 3]);
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
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            Some(Tensor::from_vec(vec![0.1, 0.2], vec![2])),
        )
        .unwrap();

        let input = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
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
