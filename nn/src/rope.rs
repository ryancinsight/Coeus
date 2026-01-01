//! Rotary Position Embedding (RoPE) for Transformers
//!
//! RoPE introduces relative position information by rotating token embeddings
//! based on their absolute positions. This provides better length generalization
//! and is used in modern transformer architectures like GPT-J, GPT-NeoX, and PaLM.

use crate::error::{NNError, Result};
use backend::Backend;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::{ops::tensor_ops::concatenate_tensors, FloatExt, Tensor};

use num_traits::FromPrimitive;
use std::cmp::Ordering;
use std::ops::Neg;

/// Rotary Position Embedding (RoPE) implementation
///
/// RoPE applies a rotation to query and key vectors based on their positions:
/// RoPE(x, m) = x * cos(mθ) + R(x) * sin(mθ)
/// where R(x) is the rotation of x by 90 degrees and θ is a function of position
pub struct RoPE<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static,
    T: DataType + FloatExt + Neg<Output = T> + Copy + num_traits::Num + FromPrimitive,
{
    /// Pre-computed cos and sin values for all positions
    cos_cache: Tensor<B, S, T>,
    sin_cache: Tensor<B, S, T>,
    /// Maximum sequence length this RoPE was initialized for
    max_seq_len: usize,
    /// Head dimension (must be even for complex number representation)
    head_dim: usize,
    /// Base for the frequency computation
    theta_base: f64,
}

type QkPair<B, S, T> = (Tensor<B, S, T>, Tensor<B, S, T>);

impl<B, S, T> RoPE<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static,
    T: DataType + FloatExt + Neg<Output = T> + Copy + num_traits::Num + FromPrimitive,
{
    /// Create a new RoPE instance
    ///
    /// # Arguments
    /// * `head_dim` - Head dimension (must be even)
    /// * `max_seq_len` - Maximum sequence length to pre-compute
    /// * `theta_base` - Base for frequency computation (typically 10000.0)
    pub fn new(head_dim: usize, max_seq_len: usize, theta_base: f64) -> Result<Self> {
        if head_dim % 2 != 0 {
            return Err(NNError::InvalidConfiguration {
                message: "Head dimension must be even for RoPE".to_string(),
            });
        }
        if theta_base.partial_cmp(&0.0) != Some(Ordering::Greater) {
            return Err(NNError::InvalidConfiguration {
                message: "Theta base must be positive for RoPE".to_string(),
            });
        }

        let (cos_cache, sin_cache) =
            Self::precompute_rope_cache(head_dim, max_seq_len, theta_base)?;

        Ok(Self {
            cos_cache,
            sin_cache,
            max_seq_len,
            head_dim,
            theta_base,
        })
    }

    /// Apply RoPE to query tensor
    ///
    /// # Arguments
    /// * `q` - Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
    /// * `positions` - Optional position indices (uses 0..seq_len if None)
    pub fn apply_to_query(
        &self,
        q: &Tensor<B, S, T>,
        positions: Option<&[usize]>,
    ) -> Result<Tensor<B, S, T>> {
        self.apply_rope(q, positions, true)
    }

    /// Apply RoPE to key tensor
    ///
    /// # Arguments
    /// * `k` - Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
    /// * `positions` - Optional position indices (uses 0..seq_len if None)
    pub fn apply_to_key(
        &self,
        k: &Tensor<B, S, T>,
        positions: Option<&[usize]>,
    ) -> Result<Tensor<B, S, T>> {
        self.apply_rope(k, positions, false)
    }

    /// Apply RoPE to both query and key tensors
    ///
    /// # Arguments
    /// * `q` - Query tensor
    /// * `k` - Key tensor
    /// * `positions` - Optional position indices
    pub fn apply_to_qk(
        &self,
        q: &Tensor<B, S, T>,
        k: &Tensor<B, S, T>,
        positions: Option<&[usize]>,
    ) -> Result<QkPair<B, S, T>> {
        let q_out = self.apply_to_query(q, positions)?;
        let k_out = self.apply_to_key(k, positions)?;
        Ok((q_out, k_out))
    }

    /// Helper to apply RoPE
    fn apply_rope(
        &self,
        x: &Tensor<B, S, T>,
        positions: Option<&[usize]>,
        is_query: bool,
    ) -> Result<Tensor<B, S, T>> {
        // Default positions if None: 0..seq_len
        let seq_len = x.shape().dims()[1];
        if seq_len > self.max_seq_len {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Sequence length {seq_len} exceeds RoPE max_seq_len {}",
                    self.max_seq_len
                ),
            });
        }
        let head_dim = x.shape().dims()[3];
        if head_dim != self.head_dim {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Head dimension {head_dim} does not match RoPE head_dim {}",
                    self.head_dim
                ),
            });
        }
        let default_positions: Vec<usize> = (0..seq_len).collect();
        let pos = positions.unwrap_or(&default_positions);
        if pos.iter().any(|&p| p >= self.max_seq_len) {
            return Err(NNError::InvalidInput {
                message: format!(
                    "RoPE position index exceeds max_seq_len {}",
                    self.max_seq_len
                ),
            });
        }

        // Check if we need to extend cache (omitted for now as we assume max_seq_len is sufficient or we handle it)
        // In a real implementation, we would resize cache if max(pos) >= max_seq_len

        self.apply_rope_rotation(x, pos, is_query)
    }

    #[must_use]
    pub fn theta_base(&self) -> f64 {
        self.theta_base
    }

    /// Pre-compute cos and sin caches
    fn precompute_rope_cache(
        head_dim: usize,
        max_seq_len: usize,
        theta_base: f64,
    ) -> Result<QkPair<B, S, T>> {
        let mut cos_data = Vec::with_capacity(max_seq_len * head_dim);
        let mut sin_data = Vec::with_capacity(max_seq_len * head_dim);

        for pos in 0..max_seq_len {
            for i in 0..(head_dim / 2) {
                let theta = 1.0 / theta_base.powf((2 * i) as f64 / head_dim as f64);
                let idx = pos as f64 * theta;
                let c = T::from_f64(idx.cos()).unwrap();
                let s = T::from_f64(idx.sin()).unwrap();

                // RoPE cache interleaving: [c, c] for each pair?
                // Standard RoPE usually pairs [i, i+half] or [2i, 2i+1]
                // The implementation of rotate_half assumes [-x2, x1].
                // So the cache should match the pairing.
                // For [-x2, x1], we want x * cos + [-x2, x1] * sin.
                // This implies pairs (x1, x2) are rotated by same theta.
                // So we store [cos(theta), ..., cos(theta)] for first half and second half?
                // Actually usually: cos = [cos(theta_0), ..., cos(theta_d/2-1), cos(theta_0), ..., cos(theta_d/2-1)]
                // and sin same.

                cos_data.push(c);
                sin_data.push(s);
            }
        }

        // We generated head_dim/2 values per pos. We need to duplicate them for the two halves.
        // Wait, the loop above generates head_dim/2 values.
        // We need to store them such that they align with x1 and x2.
        // If x is [x1, x2], then cos should be [cos_vec, cos_vec].

        // Let's redo generation properly:
        let mut full_cos_data = Vec::with_capacity(max_seq_len * head_dim);
        let mut full_sin_data = Vec::with_capacity(max_seq_len * head_dim);

        for pos in 0..max_seq_len {
            let mut row_cos = Vec::with_capacity(head_dim);
            let mut row_sin = Vec::with_capacity(head_dim);

            for i in 0..(head_dim / 2) {
                let theta = 1.0 / theta_base.powf((2 * i) as f64 / head_dim as f64);
                let val = pos as f64 * theta;
                row_cos.push(T::from_f64(val.cos()).unwrap());
                row_sin.push(T::from_f64(val.sin()).unwrap());
            }

            // Concatenate [cos, cos] and [sin, sin]
            full_cos_data.extend_from_slice(&row_cos);
            full_cos_data.extend_from_slice(&row_cos);
            full_sin_data.extend_from_slice(&row_sin);
            full_sin_data.extend_from_slice(&row_sin);
        }

        let shape = [max_seq_len, head_dim];
        let cos_tensor = Tensor::from_vec(full_cos_data, &shape)?;
        let sin_tensor = Tensor::from_vec(full_sin_data, &shape)?;

        Ok((cos_tensor, sin_tensor))
    }

    /// Apply the actual RoPE rotation computation
    fn apply_rope_rotation(
        &self,
        x: &Tensor<B, S, T>,
        positions: &[usize],
        _is_query: bool,
    ) -> Result<Tensor<B, S, T>> {
        // Convert input to dense for advanced indexing support
        let x_dense = x.to_dense_generic()?;
        let x_shape = x.shape().dims();
        let seq_len = x_shape[1];
        let head_dim = x_shape[3];

        // Helper to gather rows from cache (converts to dense internally)
        let gather_rows = |tensor: &Tensor<B, S, T>,
                           indices: &[usize]|
         -> Result<Tensor<B, DenseStorage<T>, T>> {
            let dense_cache = tensor.to_dense_generic()?;
            let dim = tensor.shape().dims()[1]; // head_dim
            let mut flat_indices = Vec::with_capacity(indices.len() * dim);
            for &idx in indices {
                let start = idx * dim;
                for i in 0..dim {
                    flat_indices.push((start + i) as i32);
                }
            }
            Ok(dense_cache.fancy_index(&flat_indices)?)
        };

        let cos = gather_rows(&self.cos_cache, positions)?;
        let sin = gather_rows(&self.sin_cache, positions)?;

        // Reshape for broadcasting: [seq_len, head_dim] -> [1, seq_len, 1, head_dim]
        // This assumes x is [batch, seq, heads, head_dim] and we want to broadcast across batch and heads
        let cos = cos.reshape(&[1, seq_len as isize, 1, head_dim as isize])?;
        let sin = sin.reshape(&[1, seq_len as isize, 1, head_dim as isize])?;

        // Rotate half (on dense tensor)
        let rotated_x = self.rotate_half_dense(&x_dense)?;

        // Apply formula: x * cos + rotated_x * sin
        let term1 = x_dense.mul(&cos)?;
        let term2 = rotated_x.mul(&sin)?;

        let result_dense = term1.add(&term2)?;

        // Convert back to original storage type S
        let data = result_dense.as_slice().to_vec();
        let dims = result_dense.shape().dims();
        Ok(Tensor::from_vec(data, dims)?)
    }

    /// Rotate half of the vector using RoPE formula: [-x2, x1]
    fn rotate_half_dense(
        &self,
        x: &Tensor<B, DenseStorage<T>, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>> {
        let shape = x.shape().dims();
        let ndim = shape.len();
        let head_dim = shape[ndim - 1];
        let half_dim = head_dim / 2;

        // Slice x1: [..., 0..half_dim]
        let mut slices1 = vec![(None, None, 1); ndim];
        slices1[ndim - 1] = (Some(0), Some(half_dim as i32), 1);
        let x1 = x.advanced_slice(&slices1)?;

        // Slice x2: [..., half_dim..head_dim]
        let mut slices2 = vec![(None, None, 1); ndim];
        slices2[ndim - 1] = (Some(half_dim as i32), None, 1);
        let x2 = x.advanced_slice(&slices2)?;

        // -x2
        let neg_x2 = x2.neg()?;

        // Concat [-x2, x1]
        Ok(concatenate_tensors(&[neg_x2, x1], ndim - 1)?)
    }
}
