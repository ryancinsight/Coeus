use crate::functions::layout::TransposeFunction;
use crate::{Tensor, TensorError};
use backend::{Backend, DataType};
use std::sync::Arc;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Standalone transpose logic with Autograd integration
pub fn transpose<B, T, S>(
    tensor: &Tensor<B, S, T>,
    dim0: usize,
    dim1: usize,
) -> crate::Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + std::ops::Neg<Output = T> + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + crate::ops::TensorStorageOps<T> + Clone + 'static,
{
    let shape = tensor.shape().dims();
    let ndim = shape.len();

    if dim0 >= ndim || dim1 >= ndim {
        return Err(TensorError::ShapeError {
            expected: ndim,
            actual: std::cmp::max(dim0, dim1),
            message: format!(
                "Dimension out of range for transpose: dim0={}, dim1={}",
                dim0, dim1
            ),
        });
    }

    // If transposing the same dimension, return a copy (identity operation)
    // BUT we still need to track gradients if required?
    // Identity transpose is just a copy. If we want to track it as an op, we can,
    // but usually identity doesn't need a grad fn other than identity.
    // However, if we return early here, we might break the graph if downstream expects a node.
    // For now, let's treat it as a real op or just clone. If we clone, we need to manually propagate requires_grad
    // and maybe add an Identity function... or just use TransposeFunction with same dims (it should handle it).

    // Let's proceed with implementation.

    let mut new_shape = shape.to_vec();
    new_shape.swap(dim0, dim1);

    let input_dense = tensor.to_dense_generic()?;
    let input_data = input_dense.as_slice();
    let mut result_data = vec![T::zero(); input_data.len()];

    let mut input_strides = vec![1; ndim];
    for i in (1..ndim).rev() {
        input_strides[i - 1] = input_strides[i] * shape[i];
    }

    let mut output_strides = vec![1; ndim];
    for i in (1..ndim).rev() {
        output_strides[i - 1] = output_strides[i] * new_shape[i];
    }

    // Permutation for index mapping
    let mut perm = (0..ndim).collect::<Vec<_>>();
    perm.swap(dim0, dim1);

    for i in 0..input_data.len() {
        let mut temp_idx = i;
        let mut coords = vec![0; ndim];
        for j in (0..ndim).rev() {
            coords[j] = temp_idx % shape[j];
            temp_idx /= shape[j];
        }

        let mut out_idx = 0;
        for j in 0..ndim {
            out_idx += coords[perm[j]] * output_strides[j];
        }
        result_data[out_idx] = input_data[i];
    }

    let mut result =
        Tensor::from_vec_with_backend(result_data, &new_shape, tensor.backend.clone())?;

    if crate::tensor_core::grad_enabled() && tensor.requires_grad() {
        let grad_fn = TransposeFunction::new(Arc::new(tensor.clone()), dim0, dim1);
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}

impl<B, S, T> Tensor<B, S, T>
where
    B: Backend<Data = T> + Default + Clone + 'static,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
    T: DataType + Clone + std::ops::Neg<Output = T> + 'static,
{
    /// Transposes dimensions of the tensor.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> crate::Result<Self> {
        transpose(self, dim0, dim1)
    }
}
