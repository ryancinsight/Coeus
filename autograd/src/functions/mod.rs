//! Autograd backward functions organized by category
//!
//! This module provides differentiable function implementations for automatic
//! differentiation. Each category has its own submodule with individual files
//! per operation for easy forward/backward parity verification.
//!
//! # Structure
//!
//! ```text
//! functions/
//! ├── arithmetic/     # Add, Sub, Mul, Div, Neg
//! ├── activation/     # Sigmoid, ReLU, Tanh, LeakyReLU, GELU
//! ├── reduction/      # Sum, Mean, Max
//! ├── math/           # Exp, Log, Sin, Cos, Sqrt, Pow
//! ├── classification/ # Softmax, CrossEntropy, NLLLoss
//! ├── layout/         # Reshape, Transpose, Cat
//! └── rnn/            # RNN
//! ```

extern crate alloc;

use backend::Backend;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
pub use tensor::{DifferentiableFunction, Function, Tensor};

// Category modules (now mostly re-exports from tensor)
pub use tensor::functions::{
    activation, arithmetic, classification, layout, linalg, math, reduction, rnn,
    GeluFunction, LeakyReluFunction, ReluFunction, SigmoidFunction, TanhFunction,
    AddFunction, DivFunction, MulFunction, NegFunction, SubFunction,
    CrossEntropyFunction, NLLLossFunction, SoftmaxFunction,
    CatFunction, ReshapeFunction, TransposeFunction,

    MatMulFunction,
    CosFunction, ExpFunction, LogFunction, PowFunction, SinFunction, SqrtFunction,
    MaxFunction, MeanFunction, SumFunction,
    RNNFunction,
};


/// Type alias for tensor references used in automatic differentiation
pub type TensorRef<B, S, T> = alloc::sync::Arc<tensor::Tensor<B, S, T>>;

/// Type-erased function reference for tensor `grad_fn` fields
pub type FunctionRef<B, S, T> = alloc::sync::Arc<dyn DifferentiableFunction<B, S, T>>;

// ============================================================================
// Shared helper functions
// ============================================================================

pub(crate) fn canonical_reduce_dims(
    dim: Option<&[usize]>,
    input_ndim: usize,
) -> anyhow::Result<Vec<usize>> {
    let mut dims = if let Some(d) = dim {
        d.to_vec()
    } else {
        (0..input_ndim).collect()
    };

    dims.sort_unstable();
    dims.dedup();

    if dims.iter().any(|&d| d >= input_ndim) {
        return Err(anyhow::anyhow!(
            "reduction dim out of range: dim={dims:?}, input_ndim={input_ndim}"
        ));
    }

    Ok(dims)
}

pub(crate) fn kept_shape_for_reduction(
    input_shape: &[usize],
    reduce_dims: &[usize],
    out_shape: &[usize],
) -> anyhow::Result<Vec<usize>> {
    if input_shape.is_empty() {
        return Ok(Vec::new());
    }

    if reduce_dims.len() == input_shape.len() {
        if out_shape.is_empty() || (out_shape.len() == 1 && out_shape[0] == 1) {
            return Ok(vec![1; input_shape.len()]);
        }
        return Err(anyhow::anyhow!(
            "reduction output shape mismatch: input_shape={input_shape:?}, reduce_dims={reduce_dims:?}, out_shape={out_shape:?}"
        ));
    }

    let mut kept = Vec::with_capacity(input_shape.len());
    let mut out_i = 0usize;
    for axis in 0..input_shape.len() {
        if reduce_dims.binary_search(&axis).is_ok() {
            kept.push(1);
        } else {
            let Some(&d) = out_shape.get(out_i) else {
                return Err(anyhow::anyhow!(
                    "reduction output shape mismatch: input_shape={input_shape:?}, reduce_dims={reduce_dims:?}, out_shape={out_shape:?}"
                ));
            };
            kept.push(d);
            out_i += 1;
        }
    }

    if out_i != out_shape.len() {
        return Err(anyhow::anyhow!(
            "reduction output shape mismatch: input_shape={input_shape:?}, reduce_dims={reduce_dims:?}, out_shape={out_shape:?}"
        ));
    }

    Ok(kept)
}

pub(crate) fn unbroadcast_dense<B, T>(
    grad_output: &Tensor<B, DenseStorage<T>, T>,
    input_shape: &[usize],
) -> anyhow::Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    T: DataType,
{
    let out_shape = grad_output.shape().dims();

    let out_len = out_shape.len();
    let in_len = input_shape.len();
    let mut input_padded = vec![1; out_len.saturating_sub(in_len)];
    input_padded.extend_from_slice(input_shape);

    let sum_dims: Vec<usize> = input_padded
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| {
            if i < out_shape.len() && d == 1 && out_shape[i] > 1 {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    let mut result = if sum_dims.is_empty() {
        grad_output.clone()
    } else {
        tensor::ops::reduction::sum(grad_output, Some(&sum_dims), true)
            .map_err(|e| anyhow::anyhow!("Sum error during unbroadcast: {e:?}"))?
    };

    if out_len > in_len {
        let data = result.storage().as_slice().to_vec();
        result = Tensor::from_vec_with_backend(data, input_shape, result.backend().clone())
            .map_err(|e| anyhow::anyhow!("Reshape error during unbroadcast: {e:?}"))?;
    }

    Ok(result)
}

pub(crate) fn to_storage_preserving_graph<B, S, T>(
    dense: Tensor<B, DenseStorage<T>, T>,
) -> anyhow::Result<Tensor<B, S, T>>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    let data = dense.storage().as_slice().to_vec();
    let dims = dense.shape().dims();
    Tensor::from_vec_with_backend(data, dims, dense.backend().clone())
        .map_err(|e| anyhow::anyhow!(e.to_string()))
}

pub(crate) fn to_dense_preserving_graph_identity<B, S, T>(
    tensor: &Tensor<B, S, T>,
) -> anyhow::Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + 'static,
    T: DataType + Clone + 'static,
{
    tensor
        .to_dense_preserving_identity()
        .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))
}
