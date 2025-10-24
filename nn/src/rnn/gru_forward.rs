//! GRU forward pass implementation.
//!
//! This module contains the forward computation logic for GRU layers.

use coeus_backend::{Backend, CpuBackend};
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::{DenseStorage, Storage, StorageFromVec};
use coeus_tensor::Tensor;

use crate::error::Result;

/// Type aliases to reduce complexity
type CpuTensor<T> = Tensor<CpuBackend<T>, DenseStorage<T>, T>;

/// GRU forward pass utilities
pub trait GRUForward<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    /// Transpose dimensions 0 and 1 of a 3D tensor.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape `(dim0, dim1, dim2)`
    ///
    /// # Returns
    /// Transposed tensor of shape `(dim1, dim0, dim2)`
    fn transpose_3d(
        input: &CpuTensor<T>,
        dim0: usize,
        dim1: usize,
        dim2: usize,
    ) -> Result<CpuTensor<T>> {
        let input_data = input.as_slice();
        let mut transposed_data = Vec::with_capacity(dim0 * dim1 * dim2);

        // Transpose: (dim0, dim1, dim2) ? (dim1, dim0, dim2)
        for b in 0..dim1 {
            for t in 0..dim0 {
                let start = (t * dim1 + b) * dim2;
                let end = start + dim2;
                transposed_data.extend_from_slice(&input_data[start..end]);
            }
        }

        Ok(Tensor::from_vec(transposed_data, &[dim1, dim0, dim2])?)
    }

    /// Reverse a sequence tensor along the time dimension (dim 0).
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape `(seq_len, batch_size, feature_size)`
    ///
    /// # Returns
    /// Reversed tensor of shape `(seq_len, batch_size, feature_size)`
    fn reverse_sequence(
        input: &CpuTensor<T>,
        seq_len: usize,
        batch_size: usize,
        feature_size: usize,
    ) -> Result<CpuTensor<T>> {
        let input_data = input.as_slice();
        let mut reversed_data = Vec::with_capacity(seq_len * batch_size * feature_size);

        // Reverse along seq_len dimension
        for t in (0..seq_len).rev() {
            let start = t * batch_size * feature_size;
            let end = (t + 1) * batch_size * feature_size;
            reversed_data.extend_from_slice(&input_data[start..end]);
        }

        Ok(Tensor::from_vec(
            reversed_data,
            &[seq_len, batch_size, feature_size],
        )?)
    }

    /// Forward pass for a single GRU layer (or direction).
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape `(seq_len, batch_size, input_size)`
    /// * `h` - Hidden state tensor
    /// * `weight_idx` - Index into weight arrays (for bidirectional: layer*2 or layer*2+1)
    /// * `dims` - Tuple of (seq_len, batch_size, input_size)
    ///
    /// # Returns
    /// Hidden output tensor
    fn forward_layer_unidirectional(
        &self,
        input: &CpuTensor<T>,
        h: &CpuTensor<T>,
        weight_idx: usize,
        dims: (usize, usize, usize),
    ) -> Result<CpuTensor<T>>;
}
