//! Utility functions for backward passes

use crate::Tensor;
use backend::Backend;
use dtype::DataType;
use storage::{DenseStorage, StorageFromVec};

/// Broadcasts a dense gradient output to a target input shape by summing across broadcasted dimensions.
pub fn unbroadcast_dense<B, T>(
    grad_output: &Tensor<B, DenseStorage<T>, T>,
    input_shape: &[usize],
) -> anyhow::Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + Clone + Copy + num_traits::Zero + std::ops::Add<Output = T>,
{
    let out_shape = grad_output.shape().dims();

    let out_len = out_shape.len();
    let in_len = input_shape.len();
    let mut input_padded = vec![1; out_len.saturating_sub(in_len)];
    input_padded.extend_from_slice(input_shape);

    let mut reduce_axes = Vec::new();
    for i in 0..out_len {
        let in_dim = input_padded[i];
        let out_dim = out_shape[i];
        if in_dim == 1 && out_dim != 1 {
            reduce_axes.push(i);
        }
    }

    let mut reduced = if reduce_axes.is_empty() {
        grad_output.clone()
    } else {
        // sum_dims is available in tensor_core or through ops
        // We'll use the one from crate::ops::reduction if available or a manual loop
        // sum_generic is available in implementations/reduction.rs
        grad_output
            .sum_generic(Some(&reduce_axes), true)
            .map_err(|e| anyhow::anyhow!("Sum error: {e:?}"))?

    };

    let current_shape = reduced.shape().dims();
    if &current_shape[..] != &input_shape[..] {
        let data = reduced.as_slice().to_vec();
        reduced = Tensor::<B, DenseStorage<T>, T>::from_vec_with_backend(
            data,
            input_shape,
            grad_output.backend.clone(),
        )
        .map_err(|e| anyhow::anyhow!("Reshape error: {e:?}"))?;
    }

    Ok(reduced)
}

/// Helper to convert a dense tensor back to the original storage type without breaking the graph
pub fn to_storage_preserving_graph<B, S, T>(
    dense: Tensor<B, DenseStorage<T>, T>,
) -> anyhow::Result<Tensor<B, S, T>>
where
    B: Backend<Data = T>,
    S: StorageFromVec<T>,
    T: DataType + Clone,
{
    let data = dense.as_slice().to_vec();
    let dims = dense.shape().dims().to_vec();
    Ok(Tensor::from_vec_with_backend(data, &dims, dense.backend().clone())?)
}
