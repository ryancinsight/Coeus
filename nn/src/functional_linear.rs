//! Linear operations for neural networks.
//!
//! This module provides stateless linear transformation operations,
//! including dense and sparse linear layers.

use coeus_backend::Backend;
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use coeus_tensor::Tensor;

use crate::error::{NNError, Result};

/// Applies a linear transformation: `y = x @ weight.T + bias`
///
/// This is equivalent to `torch.nn.functional.linear` in PyTorch.
///
/// # Arguments
/// * `input` - Input tensor of shape `(..., in_features)`
/// * `weight` - Weight tensor of shape `(out_features, in_features)`
/// * `bias` - Optional bias tensor of shape `(out_features,)`
///
/// # Returns
/// Output tensor of shape `(..., out_features)`
///
/// # Examples
/// ```rust
/// use coeus_nn::functional_linear::linear;
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.0), Float32::new(2.0)],
///     &[1, 2]
/// ).unwrap();
///
/// let weight = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(0.5), Float32::new(1.0), Float32::new(1.5), Float32::new(2.0)],
///     &[2, 2]
/// ).unwrap();
///
/// let output = linear(&input, &weight, None).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 2]);
/// ```
pub fn linear<B, S, T>(
    input: &Tensor<B, S, T>,
    weight: &Tensor<B, S, T>,
    bias: Option<&Tensor<B, S, T>>,
) -> Result<Tensor<B, S, T>>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    if weight_shape.len() != 2 {
        return Err(NNError::InvalidInput {
            message: format!("Weight must be 2D, got shape {:?}", weight_shape),
        });
    }

    let out_features = weight_shape[0];
    let in_features = weight_shape[1];

    // Check input feature dimension
    let input_features = *input_shape.last().unwrap();
    if input_features != in_features {
        return Err(NNError::InvalidInput {
            message: format!(
                "Input feature dimension {} does not match weight in_features {}",
                input_features, in_features
            ),
        });
    }

    // Check bias shape if provided
    if let Some(b) = bias {
        let bias_shape = b.shape().dims();
        if bias_shape != &[out_features] {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Bias shape {:?} does not match out_features {}",
                    bias_shape, out_features
                ),
            });
        }
    }

    // Flatten input for matrix multiplication
    let batch_size: usize = input_shape[..input_shape.len() - 1].iter().product();
    let input_flat_shape = vec![batch_size, in_features];

    let input_flat = input.reshape(&input_flat_shape)?;

    // Perform matrix multiplication: (batch_size, in_features) @ (in_features, out_features)
    // Result: (batch_size, out_features)
    let weight_t = weight.transpose(1, 0)?;
    let output_flat = input_flat.matmul(&weight_t)?;

    // Add bias if provided
    let output = if let Some(b) = bias {
        let bias_expanded_shape = vec![batch_size, out_features];
        let bias_data = b.as_slice();
        let mut expanded_bias = Vec::with_capacity(batch_size * out_features);

        for _ in 0..batch_size {
            expanded_bias.extend_from_slice(bias_data);
        }

        let bias_tensor = Tensor::from_vec(expanded_bias, &bias_expanded_shape)?;
        output_flat + bias_tensor
    } else {
        output_flat
    };

    // Reshape output to match input batch dimensions
    let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
    output_shape.push(out_features);

    output.reshape(&output_shape)
}

/// Applies a sparse linear transformation using sparse weight matrices.
///
/// This is optimized for sparse neural networks where most weights are zero.
/// The sparse format allows for memory-efficient storage and computation.
///
/// # Arguments
/// * `input` - Input tensor of shape `(..., in_features)`
/// * `weight_data` - Non-zero weight values
/// * `weight_indices` - Indices of non-zero elements in the sparse matrix
/// * `weight_shape` - Shape of the sparse weight matrix `(out_features, in_features)`
/// * `bias` - Optional bias tensor of shape `(out_features,)`
///
/// # Returns
/// Output tensor of shape `(..., out_features)`
///
/// # Note
/// This is a placeholder implementation. Full sparse linear support
/// would require integration with sparse storage formats.
pub fn sparse_linear<T: DataType + FloatExt>(
    input: &Tensor<impl Backend, impl Storage<T>, T>,
    weight_data: &[T],
    weight_indices: &[(usize, usize)],
    weight_shape: (usize, usize),
    bias: Option<&Tensor<impl Backend, impl Storage<T>, T>>,
) -> Result<Tensor<impl Backend, impl Storage<T>, T>>
where
    T: Clone,
{
    // For now, convert sparse representation to dense and use regular linear
    // In a full implementation, this would use sparse matrix multiplication algorithms

    let (out_features, in_features) = weight_shape;
    let mut dense_weight_data = vec![T::from(0.0).unwrap(); out_features * in_features];

    // Populate dense matrix from sparse representation
    for (&val, &(row, col)) in weight_data.iter().zip(weight_indices.iter()) {
        if row < out_features && col < in_features {
            dense_weight_data[row * in_features + col] = val.clone();
        }
    }

    let dense_weight = Tensor::from_vec(dense_weight_data, &[out_features, in_features])?;
    linear(input, &dense_weight, bias)
}
