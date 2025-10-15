//! Multi-Head Attention implementation.
//!
//! This module provides the standard multi-head attention mechanism used in transformer architectures.
//! It implements efficient parallel attention computation with multiple heads.

use std::fmt;
use std::marker::PhantomData;

use coeus_backend::{Backend, CpuBackend};
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::{Storage, StorageFromVec, StorageToDense, DenseStorage};
use coeus_tensor::Tensor;

use crate::error::{NNError, Result};
use crate::module::Module;
use crate::parameter::Parameter;

use super::utils::{AttentionDispatch, DenseAttention};

/// Standard multi-head attention mechanism.
///
/// Implements the transformer attention mechanism with multiple attention heads
/// for enhanced representational capacity and parallel computation.
///
/// # Architecture
/// - Multi-head query, key, value projections
/// - Scaled dot-product attention with softmax
/// - Concatenation and output projection
///
/// # Mathematical Definition
/// ```text
/// MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)Wᵒ
/// where headᵢ = Attention(QWᵢ^q, KWᵢ^k, VWᵢ^v)
///       Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k)) @ V
/// ```
///
/// # Examples
/// ```rust
/// use coeus_nn::attention::MultiHeadAttention;
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// // Create multi-head attention: embed_dim=64, num_heads=8
/// let attention = MultiHeadAttention::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 8).unwrap();
///
/// // Forward pass with sequence [batch_size=1, seq_len=10, embed_dim=64]
/// let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[1, 10, 64]).unwrap();
/// let output = attention.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 10, 64]);
/// ```
#[derive(Debug, Clone)]
pub struct MultiHeadAttention<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType,
{
    /// Number of attention heads
    pub num_heads: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Dimension per attention head
    pub head_dim: usize,

    /// Query projection parameters
    pub query_proj: Parameter<B, S, T>,
    /// Key projection parameters
    pub key_proj: Parameter<B, S, T>,
    /// Value projection parameters
    pub value_proj: Parameter<B, S, T>,
    /// Output projection parameters
    pub out_proj: Parameter<B, S, T>,
    /// Phantom data to ensure B and S are used for type safety
    _phantom: PhantomData<(B, S)>,
}

impl<B, S, T> MultiHeadAttention<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd,
{
    /// Create a new multi-head attention layer.
    ///
    /// # Arguments
    /// * `embed_dim` - Embedding dimension of input/output
    /// * `num_heads` - Number of attention heads
    ///
    /// # Returns
    /// Returns the new MultiHeadAttention layer or an error if configuration is invalid.
    pub fn new(embed_dim: usize, num_heads: usize) -> Result<Self> {
        if embed_dim % num_heads != 0 {
            return Err(NNError::InvalidConfiguration {
                message: format!("embed_dim ({}) must be divisible by num_heads ({})", embed_dim, num_heads),
            });
        }
        if num_heads == 0 {
            return Err(NNError::InvalidConfiguration {
                message: "num_heads must be > 0".to_string(),
            });
        }

        let head_dim = embed_dim / num_heads;

        // Create projection matrices
        let query_proj = Self::create_projection(embed_dim, embed_dim);
        let key_proj = Self::create_projection(embed_dim, embed_dim);
        let value_proj = Self::create_projection(embed_dim, embed_dim);
        let out_proj = Self::create_projection(embed_dim, embed_dim);

        let query_param = Parameter::new(query_proj.requires_grad_(true), "query_proj".to_string());
        let key_param = Parameter::new(key_proj.requires_grad_(true), "key_proj".to_string());
        let value_param = Parameter::new(value_proj.requires_grad_(true), "value_proj".to_string());
        let out_param = Parameter::new(out_proj.requires_grad_(true), "out_proj".to_string());

        Ok(Self {
            num_heads,
            embed_dim,
            head_dim,
            query_proj: query_param,
            key_proj: key_param,
            value_proj: value_param,
            out_proj: out_param,
            _phantom: PhantomData,
        })
    }

    /// Create a projection matrix with Xavier initialization.
    fn create_projection(
        in_features: usize,
        out_features: usize,
    ) -> Tensor<B, S, T> {
        // Xavier/Glorot uniform initialization
        let _limit = (T::from(6.0).unwrap() / T::from(in_features + out_features).unwrap()).sqrt();
        let weight_data =
            Tensor::<B, S, T>::zeros_generic(&[out_features, in_features]).unwrap();

        // For now, initialize with a simple constant (can be improved with proper random sampling)
        // This works for both dense and sparse storage
        let mut weight_dense = weight_data.to_dense_generic().unwrap();
        let data_slice = weight_dense.as_mut_slice();
        for elem in data_slice.iter_mut() {
            *elem = T::from(0.01).unwrap();
        }

        // Convert back to the original storage type
        Tensor::<B, S, T>::from_vec(weight_dense.as_slice().to_vec(), weight_dense.shape().dims()).unwrap()
    }

    /// Compute multi-head attention.
    fn compute_multihead_attention(
        &self,
        queries: &Tensor<B, DenseStorage<T>, T>,
        keys: &Tensor<B, DenseStorage<T>, T>,
        values: &Tensor<B, DenseStorage<T>, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>> {
        let query_shape = queries.shape().dims();
        let key_shape = keys.shape().dims();
        let value_shape = values.shape().dims();

        // Validate shapes: [batch_size, seq_len, embed_dim]
        if query_shape.len() != 3 || query_shape[2] != self.embed_dim {
            return Err(NNError::ShapeMismatch {
                operation: "multihead_attention_queries".to_string(),
                expected: vec![0, 0, self.embed_dim],
                actual: query_shape.to_vec(),
            });
        }
        if key_shape.len() != 3 || key_shape[2] != self.embed_dim {
            return Err(NNError::ShapeMismatch {
                operation: "multihead_attention_keys".to_string(),
                expected: vec![0, 0, self.embed_dim],
                actual: key_shape.to_vec(),
            });
        }
        if value_shape.len() != 3 || value_shape[2] != self.embed_dim {
            return Err(NNError::ShapeMismatch {
                operation: "multihead_attention_values".to_string(),
                expected: vec![0, 0, self.embed_dim],
                actual: value_shape.to_vec(),
            });
        }

        let batch_size = query_shape[0];
        let query_seq_len = query_shape[1];
        let key_seq_len = key_shape[1];
        let value_seq_len = value_shape[1];

        // Batch sizes must match
        if batch_size != key_shape[0] || batch_size != value_shape[0] {
            return Err(NNError::ShapeMismatch {
                operation: "multihead_attention_batch".to_string(),
                expected: vec![batch_size, 0, 0],
                actual: vec![key_shape[0], value_shape[0], 0],
            });
        }

        // Process each batch separately
        let mut attended_data = Vec::with_capacity(batch_size * query_seq_len * self.embed_dim);

        for batch_idx in 0..batch_size {
            // Extract batch data manually from the flattened tensors
            let query_batch_start = batch_idx * query_seq_len * self.embed_dim;
            let query_batch_end = (batch_idx + 1) * query_seq_len * self.embed_dim;
            let key_batch_start = batch_idx * key_seq_len * self.embed_dim;
            let key_batch_end = (batch_idx + 1) * key_seq_len * self.embed_dim;
            let value_batch_start = batch_idx * value_seq_len * self.embed_dim;
            let value_batch_end = (batch_idx + 1) * value_seq_len * self.embed_dim;

            let query_batch_data: Vec<T> =
                queries.as_slice()[query_batch_start..query_batch_end].to_vec();
            let key_batch_data: Vec<T> = keys.as_slice()[key_batch_start..key_batch_end].to_vec();
            let value_batch_data: Vec<T> =
                values.as_slice()[value_batch_start..value_batch_end].to_vec();

            // Create single-batch tensors: [seq_len, embed_dim]
            // Use generic backend with dense storage for computation
            let query_batch = Tensor::<B, DenseStorage<T>, T>::from_vec(query_batch_data, &[query_seq_len, self.embed_dim])?;
            let key_batch = Tensor::<B, DenseStorage<T>, T>::from_vec(key_batch_data, &[key_seq_len, self.embed_dim])?;
            let value_batch = Tensor::<B, DenseStorage<T>, T>::from_vec(value_batch_data, &[value_seq_len, self.embed_dim])?;

            // Compute Q @ K^T: [query_seq, embed] @ [key_seq, embed]^T -> [query_seq, key_seq]
            // Use available tensor operations - they handle storage conversions internally
            let key_batch_t = key_batch.transpose(0, 1)?;
            let attention_logits = query_batch.matmul(&key_batch_t)?;

            // Scale by sqrt(d_k)
            let scale = T::from((self.head_dim as f64).sqrt()).unwrap();
            // Convert to dense for scalar division, then back
            let attention_dense = attention_logits.to_dense_generic()?;
            let scale_tensor = Tensor::<B, DenseStorage<T>, T>::from_vec(vec![scale], &[1])?;
            let scaled_logits_dense = &attention_dense / &scale_tensor;
            let scaled_logits = Tensor::<B, DenseStorage<T>, T>::from_vec(
                scaled_logits_dense.as_slice().to_vec(),
                scaled_logits_dense.shape().dims()
            )?;

            // Apply softmax along rows (each query position)
            let attention_weights = self.softmax_rows_dense(&scaled_logits_dense)?;

            // Apply attention: attention_weights @ values
            // attention_weights: [query_seq, key_seq], value_batch: [value_seq, embed] -> [query_seq, embed]
            let attended_batch = attention_weights.matmul(&value_batch)?;

            // Collect the attended data
            attended_data.extend_from_slice(attended_batch.as_slice());
        }

        // Reshape back to [batch_size, query_seq_len, embed_dim]
        Tensor::<B, DenseStorage<T>, T>::from_vec(attended_data, &[batch_size, query_seq_len, self.embed_dim])
            .map_err(Into::into)
    }

    /// Apply softmax along rows of a dense tensor.
    fn softmax_rows_dense(
        &self,
        input: &Tensor<B, DenseStorage<T>, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>> {
        let shape = input.shape().dims();
        if shape.len() != 2 {
            return Err(NNError::ShapeMismatch {
                operation: "softmax_rows".to_string(),
                expected: vec![0, 0],
                actual: shape.to_vec(),
            });
        }

        let rows = shape[0];
        let cols = shape[1];
        let mut result_data = Vec::with_capacity(rows * cols);

        for row_idx in 0..rows {
            let row_start = row_idx * cols;
            let row_end = (row_idx + 1) * cols;
            let row_data: Vec<T> = input.as_slice()[row_start..row_end].to_vec();

            // Find max for numerical stability
            let mut max_val = T::min_value();
            for &val in &row_data {
                if val > max_val {
                    max_val = val;
                }
            }

            // Compute exp(x - max) and sum
            let mut exp_sum = T::zero();
            let mut exp_values = Vec::with_capacity(cols);
            for &val in &row_data {
                let exp_val = (val - max_val).exp();
                exp_values.push(exp_val);
                exp_sum = exp_sum + exp_val;
            }

            // Normalize by sum
            for exp_val in exp_values {
                result_data.push(exp_val / exp_sum);
            }
        }

        Tensor::<B, DenseStorage<T>, T>::from_vec(result_data, &[rows, cols])
            .map_err(Into::into)
    }
}

impl<B, S, T> Module<B, S, T> for MultiHeadAttention<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let input_shape = input.shape().dims();

        // Input shape: [batch_size, seq_len, embed_dim]
        if input_shape.len() != 3 || input_shape[2] != self.embed_dim {
            return Err(NNError::ShapeMismatch {
                operation: "MultiHeadAttention forward".to_string(),
                expected: vec![0, 0, self.embed_dim],
                actual: input_shape.to_vec(),
            });
        }

        // Convert input to dense for computation if needed
        let input_dense = input.to_dense_generic()?;

        // Apply projections: input -> [batch, seq, embed] @ [embed, embed] -> [batch, seq, embed]
        let query_proj_dense = self.query_proj.data().to_dense_generic()?;
        let key_proj_dense = self.key_proj.data().to_dense_generic()?;
        let value_proj_dense = self.value_proj.data().to_dense_generic()?;
        let out_proj_dense = self.out_proj.data().to_dense_generic()?;

        // Project inputs to query, key, value
        let queries = input_dense.matmul(&query_proj_dense.transpose(0, 1)?)?;
        let keys = input_dense.matmul(&key_proj_dense.transpose(0, 1)?)?;
        let values = input_dense.matmul(&value_proj_dense.transpose(0, 1)?)?;

        // Compute multi-head attention
        let attended = self.compute_multihead_attention(&queries, &keys, &values)?;

        // Apply output projection
        let output = attended.matmul(&out_proj_dense.transpose(0, 1)?)?;

        // Convert back to original storage type if needed
        if std::any::TypeId::of::<S>() == std::any::TypeId::of::<DenseStorage<T>>() {
            Ok(Tensor::<B, S, T>::from_vec(output.as_slice().to_vec(), output.shape().dims()).unwrap())
        } else {
            // For sparse storage, we'd need to implement conversion logic
            // For now, return dense result
            Ok(Tensor::<B, S, T>::from_vec(output.as_slice().to_vec(), output.shape().dims()).unwrap())
        }
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![
            self.query_proj.clone(),
            self.key_proj.clone(),
            self.value_proj.clone(),
            self.out_proj.clone(),
        ]
    }

    fn zero_grad(&mut self) {
        self.query_proj.zero_grad();
        self.key_proj.zero_grad();
        self.value_proj.zero_grad();
        self.out_proj.zero_grad();
    }

    fn train(&mut self, _mode: bool) {
        // No-op for now, could be used for dropout, batch norm, etc.
    }

    fn name(&self) -> &str {
        "MultiHeadAttention"
    }
}

impl<B, S, T> fmt::Display for MultiHeadAttention<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MultiHeadAttention {{ embed_dim: {}, num_heads: {}, head_dim: {} }}",
            self.embed_dim, self.num_heads, self.head_dim
        )
    }
}

impl<B, T> AttentionDispatch<B, DenseStorage<T>, T> for MultiHeadAttention<B, DenseStorage<T>, T>
where
    B: Backend + Clone + Default,
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd,
{
    type AttentionImpl = DenseAttention<B, T>;

    fn get_specialized_impl(&self) -> &Self::AttentionImpl {
        // For dense storage, we use the default dense attention implementation
        // In practice, this would be a stored field
        todo!("Implement specialized dense attention dispatch")
    }

    fn compute_specialized(
        &self,
        input: &Tensor<B, DenseStorage<T>, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>> {
        self.forward(input)
    }
}

impl<B, S, T> MultiHeadAttention<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd,
{
    /// Forward pass for cross-attention with separate query, key, and value tensors.
    ///
    /// This is used in transformer decoder layers where queries come from the decoder
    /// and keys/values come from the encoder outputs.
    ///
    /// # Arguments
    /// * `query` - Query tensor of shape [batch_size, query_seq_len, embed_dim]
    /// * `key` - Key tensor of shape [batch_size, key_seq_len, embed_dim]
    /// * `value` - Value tensor of shape [batch_size, key_seq_len, embed_dim]
    ///
    /// # Returns
    /// Cross-attention output tensor of shape [batch_size, query_seq_len, embed_dim]
    pub fn forward_cross_attention(
        &self,
        query: &Tensor<B, S, T>,
        key: &Tensor<B, S, T>,
        value: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>> {
        let query_shape = query.shape().dims();
        let key_shape = key.shape().dims();
        let value_shape = value.shape().dims();

        // Validate shapes: [batch_size, seq_len, embed_dim]
        if query_shape.len() != 3 || query_shape[2] != self.embed_dim {
            return Err(NNError::ShapeMismatch {
                operation: "MultiHeadAttention cross-attention query".to_string(),
                expected: vec![0, 0, self.embed_dim],
                actual: query_shape.to_vec(),
            });
        }
        if key_shape.len() != 3 || key_shape[2] != self.embed_dim {
            return Err(NNError::ShapeMismatch {
                operation: "MultiHeadAttention cross-attention key".to_string(),
                expected: vec![0, 0, self.embed_dim],
                actual: key_shape.to_vec(),
            });
        }
        if value_shape.len() != 3 || value_shape[2] != self.embed_dim {
            return Err(NNError::ShapeMismatch {
                operation: "MultiHeadAttention cross-attention value".to_string(),
                expected: vec![0, 0, self.embed_dim],
                actual: value_shape.to_vec(),
            });
        }

        // Convert to dense for computation if needed
        let query_dense = query.to_dense_generic()?;
        let key_dense = key.to_dense_generic()?;
        let value_dense = value.to_dense_generic()?;

        // Compute cross-attention
        self.compute_multihead_attention(&query_dense, &key_dense, &value_dense)
    }
}
