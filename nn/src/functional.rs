//! Functional neural network operations
//!
//! This module provides PyTorch-compatible functional operations that don't maintain state.
//! These operations are stateless and can be used directly on tensors.
//!
//! ## Mathematical Operations
//!
//! ### Activation Functions
//! - `relu` - Rectified Linear Unit
//! - `leaky_relu` - Leaky ReLU
//! - `elu` - Exponential Linear Unit
//! - `gelu` - Gaussian Error Linear Unit
//! - `sigmoid` - Sigmoid function
//! - `tanh` - Hyperbolic tangent
//! - `softmax` - Softmax activation
//! - `log_softmax` - Log softmax
//!
//! ### Linear Functions
//! - `linear` - Linear transformation
//!
//! ## Usage
//!
//! ```rust
//! use coeus_nn::functional as F;
//! use coeus_tensor::Tensor;
//!
//! let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
//! let weight = Tensor::from_vec(vec![1.0, 0.0, -1.0, 2.0], vec![1, 4]);
//! let bias = Some(Tensor::from_vec(vec![0.0], vec![1]));
//!
//! let output = F::linear(&input, &weight, bias.as_ref()).unwrap();
//! ```
//!
//! ## References
//!
//! - [PyTorch Functional API](https://pytorch.org/docs/stable/nn.functional.html)
//! - [Deep Learning Book](https://www.deeplearningbook.org/)

use crate::{Module, Result};
use coeus_tensor::{FloatDtype, Tensor};

/// Linear transformation
///
/// # Arguments
/// * `input` - Input tensor
/// * `weight` - Weight matrix
/// * `bias` - Optional bias vector
///
/// # Returns
/// Output tensor = input @ weight.T + bias
pub fn linear<T: FloatDtype + rand::distributions::uniform::SampleUniform>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
) -> Result<Tensor<T>> {
    let linear = crate::Linear::from_tensors(weight.clone(), bias.cloned())?;
    linear.forward(input)
}

/// ReLU activation function
///
/// # Arguments
/// * `input` - Input tensor
///
/// # Returns
/// Tensor with ReLU applied element-wise: `max(0, x)`
pub fn relu<T: FloatDtype>(input: &Tensor<T>) -> Result<Tensor<T>> {
    let relu = crate::ReLU::new();
    relu.forward(input)
}

/// Leaky ReLU activation function
///
/// # Arguments
/// * `input` - Input tensor
/// * `negative_slope` - Slope for negative values (default: 0.01)
///
/// # Returns
/// Tensor with Leaky ReLU applied element-wise: `max(αx, x)`
pub fn leaky_relu<T: FloatDtype>(input: &Tensor<T>, negative_slope: T) -> Result<Tensor<T>> {
    let leaky_relu =
        crate::LeakyReLU::new_with_slope(coeus_dtype::Dtype::to_f64(&negative_slope).unwrap());
    leaky_relu.forward(input)
}

/// GELU activation function
///
/// # Arguments
/// * `input` - Input tensor
///
/// # Returns
/// Tensor with GELU applied element-wise
pub fn gelu<T: FloatDtype>(input: &Tensor<T>) -> Result<Tensor<T>> {
    let gelu = crate::GELU::new();
    gelu.forward(input)
}

/// Sigmoid activation function
///
/// # Arguments
/// * `input` - Input tensor
///
/// # Returns
/// Tensor with sigmoid applied element-wise: `1 / (1 + exp(-x))`
pub fn sigmoid<T: FloatDtype>(input: &Tensor<T>) -> Result<Tensor<T>> {
    let sigmoid = crate::Sigmoid::new();
    sigmoid.forward(input)
}

/// Tanh activation function
///
/// # Arguments
/// * `input` - Input tensor
///
/// # Returns
/// Tensor with tanh applied element-wise: `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`
pub fn tanh<T: FloatDtype>(input: &Tensor<T>) -> Result<Tensor<T>> {
    let tanh = crate::Tanh::new();
    tanh.forward(input)
}

/// Softmax activation function
///
/// # Arguments
/// * `input` - Input tensor
/// * `dim` - Dimension along which to apply softmax (default: -1)
///
/// # Returns
/// Tensor with softmax applied along specified dimension
pub fn softmax<T: FloatDtype>(input: &Tensor<T>, dim: isize) -> Result<Tensor<T>> {
    let dim = if dim == -1 {
        input.ndim() - 1
    } else {
        dim as usize
    };
    let softmax = crate::Softmax::with_dim(dim);
    softmax.forward(input)
}

/// Log softmax activation function
///
/// # Arguments
/// * `input` - Input tensor
/// * `dim` - Dimension along which to apply log softmax (default: -1)
///
/// # Returns
/// Tensor with log softmax applied along specified dimension
pub fn log_softmax<T: FloatDtype>(input: &Tensor<T>, dim: isize) -> Result<Tensor<T>> {
    let softmax_result = softmax(input, dim)?;
    Ok(softmax_result.log())
}
