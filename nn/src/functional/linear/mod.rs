//! Linear operations for neural networks.
//!
//! This module provides stateless linear transformation operations,
//! including dense and sparse linear layers.

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::core::error::{NNError, Result};

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
/// use nn::functional_linear::linear;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.0), Float32::new(2.0)],
///     &[1, 2]
/// ).unwrap();
///
/// let weight = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
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
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    if weight_shape.len() != 2usize {
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

    // Convert all tensors to dense for computation
    let input_dense = input.to_dense_generic()?;
    let weight_dense = weight.to_dense_generic()?;
    let bias_dense = bias.map(|b| b.to_dense_generic()).transpose()?;

    // Check bias shape if provided
    if let Some(b) = bias {
        let bias_shape = b.shape().dims();
        if bias_shape != [out_features] {
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
    let input_flat_shape = vec![batch_size as isize, in_features as isize];

    let input_flat = input_dense.reshape(&input_flat_shape)?;

    // Perform matrix multiplication: (batch_size, in_features) @ (in_features, out_features)
    // Result: (batch_size, out_features)
    let weight_t = weight_dense.transpose(1, 0)?;
    let output_flat = input_flat.matmul(&weight_t)?;

    // Add bias if provided
    let output = if let Some(bias_tensor) = &bias_dense {
        let _bias_shape = bias_tensor.shape().dims();
        let bias_data = bias_tensor.as_slice();
        let mut expanded_bias = Vec::with_capacity(batch_size * out_features);

        for _ in 0..batch_size {
            expanded_bias.extend_from_slice(bias_data);
        }

        let bias_expanded =
            Tensor::<B, DenseStorage<T>, T>::from_vec(expanded_bias, &[batch_size, out_features])?;
        &output_flat + &bias_expanded
    } else {
        output_flat
    };

    // Reshape output to match input batch dimensions
    let mut output_shape: Vec<isize> = input_shape[..input_shape.len() - 1]
        .iter()
        .map(|&x| x as isize)
        .collect();
    output_shape.push(out_features as isize);

    Ok(output.reshape(&output_shape)?)
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
/// # Implementation Notes
/// This implementation converts sparse weight matrices to dense format for computation.
/// For large sparse matrices, a dedicated sparse storage format with optimized
/// sparse-dense matrix multiplication would provide better performance.
/// The current approach is a valid fallback that works correctly but may not be optimal for all use cases.
pub fn sparse_linear<
    B: Backend<Data = T> + Default,
    S: Storage<T> + StorageToDense<T> + Clone + StorageFromVec<T> + 'static,
    T,
>(
    input: &Tensor<B, S, T>,
    weight_data: &[T],
    weight_indices: &[(usize, usize)],
    weight_shape: (usize, usize),
    bias: Option<&Tensor<B, S, T>>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    T: DataType + FloatExt + Clone,
{
    // For now, convert sparse representation to dense and use regular linear
    // In a full implementation, this would use sparse matrix multiplication algorithms

    let (out_features, in_features) = weight_shape;
    let mut dense_weight_data = vec![T::from(0.0).unwrap(); out_features * in_features];

    // Populate dense matrix from sparse representation
    for (&val, &(row, col)) in weight_data.iter().zip(weight_indices.iter()) {
        if row < out_features && col < in_features {
            dense_weight_data[row * in_features + col] = val;
        }
    }

    let dense_weight = Tensor::from_vec(dense_weight_data, &[out_features, in_features])?;
    linear(input, &dense_weight, bias)
}
