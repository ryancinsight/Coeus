//! Activation functions for neural networks
//!
//! This module provides common activation functions used in neural networks.
//! All activations implement the `Module` trait for seamless integration.
//!
//! ## Available Activations
//!
//! - **ReLU**: Rectified Linear Unit, `max(0, x)`
//! - **Sigmoid**: Logistic function, `1 / (1 + exp(-x))`
//! - **Tanh**: Hyperbolic tangent, `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`
//! - **Softmax**: Normalized exponential function
//! - **LeakyReLU**: Leaky version of ReLU, `max(αx, x)`
//!
//! ## Mathematical Properties
//!
//! ### ReLU
//! ```math
//! ReLU(x) = max(0, x)
//!
//! ∂ReLU/∂x = {
//!     1  if x > 0
//!     0  if x ≤ 0
//! }
//! ```
//!
//! ### Sigmoid
//! ```math
//! σ(x) = 1 / (1 + exp(-x))
//!
//! ∂σ/∂x = σ(x) * (1 - σ(x))
//! ```
//!
//! ## References
//!
//! - [Deep Learning Book - Activation Functions](https://www.deeplearningbook.org/contents/mlp.html)
//! - [Glorot & Bengio, 2010 - Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)
//! - [Hendrycks & Gimpel, 2016 - Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)

use crate::Module;
use coeus_tensor::{FloatDtype, Tensor};
use std::fmt;

/// ReLU (Rectified Linear Unit) activation function
///
/// Formula: `ReLU(x) = max(0, x)`
///
/// This is the most commonly used activation function in modern neural networks
/// due to its simplicity and effectiveness in combating the vanishing gradient problem.
#[derive(Debug, Clone, Copy)]
pub struct ReLU;

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl ReLU {
    /// Create a new ReLU activation
    pub fn new() -> Self {
        ReLU
    }
}

impl<T: FloatDtype> Module<T> for ReLU {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // Use the tensor's relu method which has proper autograd integration
        Ok(input.relu())
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

impl fmt::Display for ReLU {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReLU()")
    }
}

/// Sigmoid activation function
///
/// Formula: `σ(x) = 1 / (1 + exp(-x))`
///
/// The sigmoid function squashes the input to the range (0, 1).
/// It's commonly used in binary classification problems.
#[derive(Debug, Clone, Copy)]
pub struct Sigmoid;

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Sigmoid {
    /// Create a new Sigmoid activation
    pub fn new() -> Self {
        Sigmoid
    }
}

impl<T: FloatDtype> Module<T> for Sigmoid {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // Sigmoid: 1 / (1 + exp(-x))
        let data = input
            .data()
            .iter()
            .map(|&x| T::one() / (T::one() + (-x).exp()))
            .collect();

        let mut result = Tensor::from_vec(data, input.shape().to_vec());

        if input.requires_grad() {
            result.set_requires_grad(true);
        }

        Ok(result)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

impl fmt::Display for Sigmoid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sigmoid()")
    }
}

/// Tanh (Hyperbolic Tangent) activation function
///
/// Formula: `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
///
/// Tanh squashes the input to the range (-1, 1) and is zero-centered,
/// making it preferable to sigmoid in many cases.
#[derive(Debug, Clone, Copy)]
pub struct Tanh;

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Tanh {
    /// Create a new Tanh activation
    pub fn new() -> Self {
        Tanh
    }
}

impl<T: FloatDtype> Module<T> for Tanh {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        let data = input.data().iter().map(|&x| x.tanh()).collect();

        let mut result = Tensor::from_vec(data, input.shape().to_vec());

        if input.requires_grad() {
            result.set_requires_grad(true);
        }

        Ok(result)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

impl fmt::Display for Tanh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tanh()")
    }
}

/// Softmax activation function
///
/// Formula: `Softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)`
///
/// Softmax converts a vector of real numbers into a probability distribution.
/// It's commonly used in the output layer of classification networks.
#[derive(Debug, Clone, Copy)]
pub struct Softmax<T: FloatDtype> {
    /// Dimension along which to apply softmax
    dim: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: FloatDtype> Default for Softmax<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatDtype> Softmax<T> {
    /// Create a new Softmax activation with default dimension (-1, last dimension)
    pub fn new() -> Self {
        Self {
            dim: usize::MAX,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a new Softmax activation along a specific dimension
    pub fn new_with_dim(dim: usize) -> Self {
        Self {
            dim,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: FloatDtype + Clone> Module<T> for Softmax<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        let dim = if self.dim == usize::MAX {
            input.ndim() - 1
        } else {
            self.dim
        };

        // For simplicity, handle 1D and 2D cases
        if input.ndim() == 1 {
            Ok(self.softmax_1d(input))
        } else if input.ndim() == 2 && dim == 1 {
            Ok(self.softmax_2d(input))
        } else {
            // For higher dimensions, apply softmax along the specified dimension
            Ok(self.softmax_nd(input, dim))
        }
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

impl<T: FloatDtype + Clone> Softmax<T> {
    fn softmax_1d(&self, input: &Tensor<T>) -> Tensor<T> {
        // exp(x) / sum(exp(x))
        let exp_data: Vec<T> = input.data().iter().map(|&x| x.exp()).collect();
        let sum_exp: T = exp_data.iter().fold(T::zero(), |acc, &x| acc + x);

        let softmax_data: Vec<T> = exp_data.iter().map(|&x| x / sum_exp).collect();

        let mut result = Tensor::from_vec(softmax_data, input.shape().to_vec());

        if input.requires_grad() {
            result.set_requires_grad(true);
        }

        result
    }

    fn softmax_2d(&self, input: &Tensor<T>) -> Tensor<T> {
        let shape = input.shape();
        let batch_size = shape[0];
        let num_classes = shape[1];

        let mut result_data = vec![T::zero(); input.numel()];

        for b in 0..batch_size {
            // Extract the row for this batch
            let start_idx = b * num_classes;
            let end_idx = (b + 1) * num_classes;
            let row_data = &input.data()[start_idx..end_idx];

            // Compute exp and sum
            let exp_row: Vec<T> = row_data.iter().map(|&x| x.exp()).collect();
            let sum_exp: T = exp_row.iter().fold(T::zero(), |acc, &x| acc + x);

            // Compute softmax
            for i in 0..num_classes {
                result_data[start_idx + i] = exp_row[i] / sum_exp;
            }
        }

        let mut result = Tensor::from_vec(result_data, shape.to_vec());

        if input.requires_grad() {
            result.set_requires_grad(true);
        }

        result
    }

    fn softmax_nd(&self, input: &Tensor<T>, dim: usize) -> Tensor<T> {
        // For general N-dimensional softmax, we need to handle broadcasting
        // This is a simplified implementation for common cases
        let shape = input.shape();
        let result_shape = shape.to_vec();

        // Compute the size along the softmax dimension
        let dim_size = shape[dim];

        // Compute total number of softmax operations needed
        let outer_size: usize = shape.iter().take(dim).product();
        let inner_size: usize = shape.iter().skip(dim + 1).product();

        let mut result_data = vec![T::zero(); input.numel()];

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                // Extract the vector along the softmax dimension
                let mut vector = vec![T::zero(); dim_size];
                #[allow(clippy::needless_range_loop)]
                for i in 0..dim_size {
                    let mut indices = vec![0; shape.len()];
                    // Set outer dimensions
                    let mut remaining = outer;
                    for d in (0..dim).rev() {
                        indices[d] = remaining % shape[d];
                        remaining /= shape[d];
                    }
                    // Set inner dimensions
                    remaining = inner;
                    for d in (dim + 1..shape.len()).rev() {
                        indices[d] = remaining % shape[d];
                        remaining /= shape[d];
                    }
                    indices[dim] = i;

                    // Convert to flat index
                    let mut flat_idx = 0;
                    let mut stride = 1;
                    for (d, &idx) in indices.iter().enumerate().rev() {
                        flat_idx += idx * stride;
                        stride *= shape[shape.len() - 1 - d];
                    }

                    vector[i] = input.data()[flat_idx];
                }

                // Apply softmax to this vector
                let exp_vector: Vec<T> = vector.iter().map(|&x| x.exp()).collect();
                let sum_exp: T = exp_vector.iter().fold(T::zero(), |acc, &x| acc + x);
                let softmax_vector: Vec<T> = exp_vector.iter().map(|&x| x / sum_exp).collect();

                // Store results
                #[allow(clippy::needless_range_loop)]
                for i in 0..dim_size {
                    let mut indices = vec![0; shape.len()];
                    // Set outer dimensions
                    let mut remaining = outer;
                    for d in (0..dim).rev() {
                        indices[d] = remaining % shape[d];
                        remaining /= shape[d];
                    }
                    // Set inner dimensions
                    remaining = inner;
                    for d in (dim + 1..shape.len()).rev() {
                        indices[d] = remaining % shape[d];
                        remaining /= shape[d];
                    }
                    indices[dim] = i;

                    // Convert to flat index
                    let mut flat_idx = 0;
                    let mut stride = 1;
                    for (d, &idx) in indices.iter().enumerate().rev() {
                        flat_idx += idx * stride;
                        stride *= shape[shape.len() - 1 - d];
                    }

                    result_data[flat_idx] = softmax_vector[i];
                }
            }
        }

        let mut result = Tensor::from_vec(result_data, result_shape);

        if input.requires_grad() {
            result.set_requires_grad(true);
        }

        result
    }
}

impl<T: FloatDtype> fmt::Display for Softmax<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.dim == usize::MAX {
            write!(f, "Softmax()")
        } else {
            write!(f, "Softmax(dim={})", self.dim)
        }
    }
}

/// Leaky ReLU activation function
///
/// Formula: `LeakyReLU(x) = max(αx, x)` where α is a small positive constant
///
/// Leaky ReLU allows a small gradient when the input is negative,
/// helping to mitigate the "dying ReLU" problem.
#[derive(Debug, Clone)]
pub struct LeakyReLU {
    /// Negative slope coefficient
    negative_slope: f64,
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl LeakyReLU {
    /// Create a new LeakyReLU with default negative slope (0.01)
    pub fn new() -> Self {
        Self::new_with_slope(0.01)
    }

    /// Create a new LeakyReLU with custom negative slope
    pub fn new_with_slope(negative_slope: f64) -> Self {
        Self { negative_slope }
    }
}

impl<T: FloatDtype> Module<T> for LeakyReLU {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        let alpha = T::from_f64(self.negative_slope).unwrap();

        let data = input
            .data()
            .iter()
            .map(|&x| if x > T::zero() { x } else { alpha * x })
            .collect();

        let mut result = Tensor::from_vec(data, input.shape().to_vec());

        if input.requires_grad() {
            result.set_requires_grad(true);
        }

        Ok(result)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

impl fmt::Display for LeakyReLU {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LeakyReLU(negative_slope={:.3})", self.negative_slope)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_relu_forward() {
        let relu = ReLU::new();
        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], vec![4]);
        let output = relu.forward(&input).expect("ReLU forward should succeed");

        let expected = [0.0, 0.0, 1.0, 2.0];
        for (i, &expected_val) in expected.iter().enumerate() {
            assert_relative_eq!(output.data()[i], expected_val, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_sigmoid_forward() {
        let sigmoid = Sigmoid::new();
        let input = Tensor::from_vec(vec![0.0], vec![1]);
        let output = sigmoid
            .forward(&input)
            .expect("Sigmoid forward should succeed");

        // σ(0) = 0.5
        assert_relative_eq!(output.data()[0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_tanh_forward() {
        let tanh = Tanh::new();
        let input = Tensor::from_vec(vec![0.0], vec![1]);
        let output = tanh.forward(&input).expect("Tanh forward should succeed");

        // tanh(0) = 0
        assert_relative_eq!(output.data()[0], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax_1d() {
        let softmax = Softmax::new();
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let output = softmax
            .forward(&input)
            .expect("Softmax forward should succeed");

        // Check that output sums to 1
        let sum: f32 = output.data().iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

        // Check that all values are positive
        for &val in output.data() {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_softmax_2d() {
        let softmax = Softmax::new();
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let output = softmax
            .forward(&input)
            .expect("Softmax forward should succeed");

        // Check that each row sums to 1
        for row in 0..2 {
            let start = row * 3;
            let end = (row + 1) * 3;
            let sum: f32 = output.data()[start..end].iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_leaky_relu() {
        let leaky_relu = LeakyReLU::new_with_slope(0.1);
        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let output = leaky_relu
            .forward(&input)
            .expect("LeakyReLU forward should succeed");

        let expected = [-0.2, -0.1, 0.0, 1.0, 2.0];
        for (i, &expected_val) in expected.iter().enumerate() {
            assert_relative_eq!(output.data()[i], expected_val, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_activation_gradient_flow() {
        let relu = ReLU::new();
        let input = Tensor::from_vec_with_grad(vec![-1.0, 0.5, 2.0], vec![3]);
        let output = relu.forward(&input).expect("ReLU forward should succeed");

        // Compute some loss
        let loss = output.sum();

        // Call backward and then manually check the autograd context
        let _ = loss.backward();

        // For now, check that the loss tensor has a gradient (it should)
        assert!(loss.grad().is_some());

        // The input gradient check is more complex due to current autograd limitations
        // We'll verify this works when we complete the autograd system
        // For now, just ensure the computation doesn't panic
        println!("Activation gradient flow test completed without panic");
    }
}
