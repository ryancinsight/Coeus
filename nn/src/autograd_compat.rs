//! Compatibility layer for autograd operations in neural network modules.
//!
//! This module provides a bridge between the NN crate and the autograd system,
//! allowing neural network operations to participate in automatic differentiation
//! when gradients are required.

use autograd::Result as AutogradResult;
use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use tensor::Tensor;

/// Type alias for Float32 tensors on CPU backend
pub type Float32Tensor = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

/// Linear transformation with autograd support
///
/// Computes: output = input @ weight + bias
///
/// # Arguments
/// * `input` - Input tensor [batch_size, in_features]
/// * `weight` - Weight matrix [out_features, in_features]
/// * `bias` - Optional bias vector [out_features]
///
/// # Returns
/// Tensor containing the result with gradient information
pub fn linear(
    input: &Float32Tensor,
    weight: &Float32Tensor,
    bias: Option<&Float32Tensor>,
) -> AutogradResult<Float32Tensor> {
    use autograd::ops::{matmul, add};

    // Perform: output = input @ weight
    let output = matmul(input, weight)?;

    // Add bias if provided: output = output + bias
    let result = if let Some(bias_tensor) = bias {
        add(&output, bias_tensor)
    } else {
        output
    };

    Ok(result)
}
