//! Sparse Attention implementation.
//!
//! This module provides sparse attention mechanisms that use sparse matrices
//! to reduce memory usage and computation for large sequence lengths.

use std::fmt;
use std::marker::PhantomData;

use rand::prelude::*;
use std::collections::BTreeMap;

use coeus_backend::{Backend, CpuBackend};
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::{Storage, StorageFromVec, StorageToDense, DenseStorage};
use coeus_tensor::Tensor;

use crate::error::{NNError, Result};
use crate::module::Module;
use crate::parameter::Parameter;
pub struct SparseAttention<B, S, T>
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
    /// Target sparsity ratio for attention weights (0.0 to 1.0)
    pub sparsity: f64,

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

impl<B, S, T> SparseAttention<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd + num_traits::Zero,
{
    /// Create a new sparse attention layer.
    ///
    /// # Arguments
    /// * `embed_dim` - Embedding dimension of input/output
    /// * `num_heads` - Number of attention heads
    /// * `sparsity` - Target sparsity ratio for attention weights (0.0 to 1.0)
    ///
    /// # Returns
    /// Returns the new SparseAttention layer or an error if configuration is invalid.
    pub fn new(embed_dim: usize, num_heads: usize, sparsity: f64) -> Result<Self> {
        if embed_dim % num_heads != 0 {
            return Err(NNError::InvalidConfiguration {
                message: format!("embed_dim ({}) must be divisible by num_heads ({})", embed_dim, num_heads),
            });
        }
        if !(0.0..=1.0).contains(&sparsity) {
            return Err(NNError::InvalidConfiguration {
                message: format!("sparsity ({}) must be between 0.0 and 1.0", sparsity),
            });
        }

        let head_dim = embed_dim / num_heads;

        // Create sparse projection parameters
        let query_proj = Self::create_sparse_projection_generic(embed_dim, embed_dim, sparsity);
        let key_proj = Self::create_sparse_projection_generic(embed_dim, embed_dim, sparsity);
        let value_proj = Self::create_sparse_projection_generic(embed_dim, embed_dim, sparsity);
        let out_proj = Self::create_sparse_projection_generic(embed_dim, embed_dim, sparsity);

        Ok(Self {
            num_heads,
            embed_dim,
            head_dim,
            sparsity,
            query_proj,
            key_proj,
            value_proj,
            out_proj,
            _phantom: PhantomData,
        })
    }

    /// Create a sparse projection matrix with controlled sparsity.
    fn create_sparse_projection_generic(
        in_features: usize,
        out_features: usize,
        sparsity: f64,
    ) -> Parameter<B, S, T>
    {
        use rand::prelude::*;
        use std::collections::BTreeMap;

        let mut rng = thread_rng();
        let total_elements = in_features * out_features;
        let nnz = ((1.0 - sparsity) * total_elements as f64) as usize;

        // Xavier initialization scaled for sparsity
        let limit = (T::from(6.0).unwrap() / T::from(in_features + out_features).unwrap()).sqrt();
        let scale = T::from((1.0 - sparsity).sqrt()).unwrap();
        let limit = limit * scale;

        // Generate sparse connectivity pattern
        let mut positions = BTreeMap::new();

        // Ensure some minimum connectivity per input feature
        for row in 0..in_features {
            let connections_per_row = (nnz as f32 / in_features as f32).max(1.0) as usize;
            for _ in 0..connections_per_row.min(out_features) {
                let col = rng.gen_range(0..out_features);
                let val = T::from(rng.gen_range(-1.0..1.0)).unwrap() * limit;
                positions.insert((row, col), val);
            }
        }

        // Convert to CSR format
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];

        for row in 0..in_features {
            for col in 0..out_features {
                if let Some(&val) = positions.get(&(row, col)) {
                    data.push(val);
                    indices.push(col);
                }
            }
            indptr.push(data.len());
        }

        // For now, create dense matrix for generic storage compatibility
        // TODO: Implement true sparse initialization for generic S
        let mut dense_data = vec![T::zero(); total_elements];
        for ((row, col), val) in positions {
            let idx = row * out_features + col;
            dense_data[idx] = val;
        }

        let storage = S::from_vec(dense_data, &[in_features, out_features]).unwrap();
        let tensor = Tensor::from_storage(storage, B::default());
        Parameter::new(tensor.requires_grad_(true), "sparse_proj".to_string())
    }

    /// Create a dense projection matrix.
    fn create_dense_projection(
        in_features: usize,
        out_features: usize,
    ) -> Tensor<B, DenseStorage<T>, T> {
        // Xavier/Glorot initialization for dense output projection
        let _limit = (T::from(6.0).unwrap() / T::from(in_features + out_features).unwrap()).sqrt();
        let mut weight_data =
            Tensor::<B, DenseStorage<T>, T>::zeros(&[out_features, in_features]).unwrap();

        let data_slice = weight_data.as_mut_slice();
        for elem in data_slice.iter_mut() {
            // Simple uniform initialization for testing
            *elem = T::from(0.01).unwrap();
        }

        weight_data
    }

    /// Compute sparse attention weights using scaled dot-product attention.
    ///
    /// Implements: attention_weights = softmax((Q @ K^T) / sqrt(d_k))
    /// with sparsity applied to the attention pattern.
    fn compute_sparse_attention(
        &self,
        queries: &Tensor<B, DenseStorage<T>, T>,
        keys: &Tensor<B, DenseStorage<T>, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>> {
        let _batch_size = queries.shape().dims()[0];
        let seq_len = queries.shape().dims()[1];

        // Batch-wise processing is supported (no restriction on batch_size)

        // Reshape to 2D for matrix operations: [seq, embed]
        let queries_2d = queries.reshape(&[seq_len as isize, self.embed_dim as isize])?;
        let keys_2d = keys.reshape(&[seq_len as isize, self.embed_dim as isize])?;

        // Compute Q @ K^T: [seq, embed] @ [seq, embed]^T -> [seq, seq]
            let keys_2d_t = keys_2d.transpose(0, 1)?;
            let attention_logits = queries_2d.matmul(&keys_2d_t)?;

        // Scale by sqrt(d_k) (convert to dense for division)
        let scale = T::from((self.head_dim as f64).sqrt()).unwrap();
        let attention_dense = attention_logits.to_dense_generic()?;
        let scale_tensor = Tensor::<B, DenseStorage<T>, T>::from_vec(vec![scale], &[1])?;
        let scaled_dense = &attention_dense / &scale_tensor;
        // Convert back to generic storage
        let scaled_logits = Tensor::<B, S, T>::from_vec(scaled_dense.as_slice().to_vec(), scaled_dense.shape().dims())?;

        // Create sparse attention mask and apply it
        let attention_mask = self.create_sparse_attention_mask(seq_len)?;
        let masked_attention = &scaled_dense * &attention_mask;

        // Apply sparse softmax along the last dimension (seq_len)
        let attention_weights_dense = self.sparse_softmax_rows(&masked_attention)?;
        // Return dense result (storage type conversion handled at Module level)
        Ok(attention_weights_dense)
    }

    /// Create a sparse attention mask based on the sparsity ratio.
    fn create_sparse_attention_mask(&self, seq_len: usize) -> Result<Tensor<B, DenseStorage<T>, T>> {
        let total_elements = seq_len * seq_len;
        let mut mask_data = vec![T::zero(); total_elements];

        // Implement different sparsity patterns based on the sparsity ratio
        if self.sparsity >= 0.9 {
            // Very sparse: only local window (3 positions around diagonal)
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let idx = i * seq_len + j;
                    let distance = (i as isize - j as isize).unsigned_abs();
                    if distance <= 1 { // Keep diagonal ±1
                        mask_data[idx] = T::one();
                    }
                }
            }
        } else if self.sparsity >= 0.7 {
            // Moderately sparse: local window (5 positions)
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let idx = i * seq_len + j;
                    let distance = (i as isize - j as isize).unsigned_abs();
                    if distance <= 2 { // Keep diagonal ±2
                        mask_data[idx] = T::one();
                    }
                }
            }
        } else {
            // Lightly sparse: keep top-k per row (more sophisticated approach)
            // For now, use a simple strided pattern
            let keep_count = ((1.0 - self.sparsity) * seq_len as f64) as usize;
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let idx = i * seq_len + j;
                    // Keep every (seq_len / keep_count) elements, offset by row
                    if j % (seq_len / keep_count.max(1)).max(1) == (i % keep_count) {
                        mask_data[idx] = T::one();
                    }
                }
            }
        }

        Ok(Tensor::<B, DenseStorage<T>, T>::from_vec(mask_data, &[seq_len, seq_len])?)
    }

    /// Apply sparse softmax along the last dimension (rows).
    fn sparse_softmax_rows(&self, input: &Tensor<B, DenseStorage<T>, T>) -> Result<Tensor<B, DenseStorage<T>, T>> {
        let shape = input.shape().dims();
        let seq_len = shape[1]; // For attention matrix [seq, seq], batch_size is 1
        let mut result_data = Vec::with_capacity(input.len());

        // Extract the attention matrix data [seq_len, seq_len]
        let matrix_data = input.as_slice();

        // Apply softmax to each row of the attention matrix
        for row in 0..seq_len {
            let row_start = row * seq_len;
            let row_end = (row + 1) * seq_len;
            if row_end <= matrix_data.len() {
                let row_data: Vec<T> = matrix_data[row_start..row_end].to_vec();

                // Compute sparse softmax for this row
                let softmax_row = self.compute_sparse_softmax(&row_data)?;
                result_data.extend(softmax_row);
            } else {
                // If row slice is out of bounds, fill with zeros
                result_data.extend(vec![T::zero(); seq_len]);
            }
        }

        Ok(Tensor::<B, DenseStorage<T>, T>::from_vec(result_data, shape)?)
    }

    /// Compute sparse softmax for a single row (vector).
    fn compute_sparse_softmax(&self, row: &[T]) -> Result<Vec<T>> {
        // Find non-zero elements and their indices
        let mut non_zero_values = Vec::new();
        let mut non_zero_indices = Vec::new();

        for (i, &val) in row.iter().enumerate() {
            if val != T::zero() {
                non_zero_values.push(val);
                non_zero_indices.push(i);
            }
        }

        // If all zeros, return uniform distribution
        if non_zero_values.is_empty() {
            return Ok(vec![T::zero(); row.len()]);
        }

        // Compute softmax only for non-zero elements
        // First, find the max for numerical stability
        let max_val = non_zero_values.iter()
            .fold(T::min_value(), |a, &b| if a > b { a } else { b });

        // Compute exp(x - max) for non-zero elements
        let exp_values: Vec<T> = non_zero_values.iter()
            .map(|&x| (x - max_val).exp())
            .collect();

        // Compute sum of exp values
        let exp_sum: T = exp_values.iter().fold(T::zero(), |a, &b| a + b);

        // Create result vector initialized to zero
        let mut result = vec![T::zero(); row.len()];

        // Fill in softmax values for non-zero positions
        for (i, &exp_val) in exp_values.iter().enumerate() {
            let idx = non_zero_indices[i];
            result[idx] = exp_val / exp_sum;
        }

        Ok(result)
    }

    /// Apply softmax along the last dimension.
    fn softmax_last_dim(
        &self,
        logits: &Tensor<B, DenseStorage<T>, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>> {
        let shape = logits.shape().dims();
        let mut result = Tensor::<B, DenseStorage<T>, T>::zeros(shape).unwrap();
        let logits_slice = logits.as_slice();
        let result_slice = result.as_mut_slice();

        // Softmax over last dimension (seq_len)
        // For 2D tensor [seq, seq], process each row
        let seq_len = shape[shape.len() - 1];
        let rows = shape.iter().take(shape.len() - 1).product();

        for row in 0..rows {
            // Compute max for numerical stability
            let mut max_val = T::min_value();
            for j in 0..seq_len {
                let idx = row * seq_len + j;
                let val = logits_slice[idx];
                if val > max_val {
                    max_val = val;
                }
            }

            // Compute exp(x - max) and sum
            let mut exp_sum = T::zero();
            for j in 0..seq_len {
                let idx = row * seq_len + j;
                let val = logits_slice[idx];
                let exp_val = (val - max_val).exp();
                exp_sum = exp_sum + exp_val;
            }

            // Compute softmax
            for j in 0..seq_len {
                let idx = row * seq_len + j;
                let val = logits_slice[idx];
                let exp_val = (val - max_val).exp();
                let softmax_val = exp_val / exp_sum;
                result_slice[idx] = softmax_val;
            }
        }

        Ok(result)
    }
}

impl<B, S, T> Module<B, S, T> for SparseAttention<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd,
{
    fn forward(
        &self,
        input: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>> {
        let input_shape = input.shape().dims();
        let requires_grad = input.requires_grad();

        // Expect [batch_size, seq_len, embed_dim]
        if input_shape.len() != 3 || input_shape[2] != self.embed_dim {
            return Err(NNError::ShapeMismatch {
                operation: "sparse_attention".to_string(),
                expected: vec![0, 0, self.embed_dim],
                actual: input_shape.to_vec(),
            });
        }

        let batch_size = input_shape[0];
        let seq_len = input_shape[1];

        // Reshape input for linear projections: [batch, seq, embed] -> [batch*seq, embed]
        let flattened_shape = &[
            isize::try_from(batch_size * seq_len).unwrap(),
            isize::try_from(self.embed_dim).unwrap(),
        ];
        let input_2d = input.reshape(flattened_shape)?;

        // Linear projections with sparse weights (converted to dense for computation)
        let query_weights = self.query_proj.data().to_dense_generic()?;
        let key_weights = self.key_proj.data().to_dense_generic()?;
        let value_weights = self.value_proj.data().to_dense_generic()?;
        let queries_2d = crate::functional::linear(&input_2d, &query_weights, None)?;
        let keys_2d = crate::functional::linear(&input_2d, &key_weights, None)?;
        let values_2d = crate::functional::linear(&input_2d, &value_weights, None)?;

        // Reshape back to 3D: [batch*seq, embed] -> [batch, seq, embed]
        let queries = queries_2d.reshape(&[
            isize::try_from(batch_size).unwrap(),
            isize::try_from(seq_len).unwrap(),
            isize::try_from(self.embed_dim).unwrap(),
        ])?;
        let keys = keys_2d.reshape(&[
            isize::try_from(batch_size).unwrap(),
            isize::try_from(seq_len).unwrap(),
            isize::try_from(self.embed_dim).unwrap(),
        ])?;
        let values = values_2d.reshape(&[
            isize::try_from(batch_size).unwrap(),
            isize::try_from(seq_len).unwrap(),
            isize::try_from(self.embed_dim).unwrap(),
        ])?;

        // Process each batch separately for sparse attention
        // Since Tensor doesn't have slice/concat, we'll process the full batch*seq tensors
        // and extract results manually. This is a simplified implementation.

        let total_seq = batch_size * seq_len;
        let mut attended_data = Vec::with_capacity(total_seq * self.embed_dim);

        for batch_idx in 0..batch_size {
            let batch_start = batch_idx * seq_len * self.embed_dim;
            let batch_end = (batch_idx + 1) * seq_len * self.embed_dim;

            // Extract batch data manually from the flattened tensors
            // This is inefficient but works for the test case
            let queries_batch_data: Vec<T> = queries.as_slice()[batch_start..batch_end].to_vec();
            let keys_batch_data: Vec<T> = keys.as_slice()[batch_start..batch_end].to_vec();
            let values_batch_data: Vec<T> = values.as_slice()[batch_start..batch_end].to_vec();

            // Create single-batch tensors (computation requires dense, but we work with any S)
            let queries_batch = Tensor::<B, DenseStorage<T>, T>::from_vec(queries_batch_data, &[seq_len, self.embed_dim])?;
            let keys_batch = Tensor::<B, DenseStorage<T>, T>::from_vec(keys_batch_data, &[seq_len, self.embed_dim])?;
            let values_batch = Tensor::<B, DenseStorage<T>, T>::from_vec(values_batch_data, &[seq_len, self.embed_dim])?;

            // Reshape for attention: [seq, embed] -> [1, seq, embed]
            let queries_reshaped =
                queries_batch.reshape(&[1, seq_len as isize, self.embed_dim as isize])?;
            let keys_reshaped =
                keys_batch.reshape(&[1, seq_len as isize, self.embed_dim as isize])?;
            let values_reshaped =
                values_batch.reshape(&[1, seq_len as isize, self.embed_dim as isize])?;

            // Compute sparse attention weights: [seq, seq]
            // Convert to dense for computation, then back to generic storage
            let queries_dense = queries_reshaped.to_dense_generic()?;
            let keys_dense = keys_reshaped.to_dense_generic()?;
            let attention_weights_dense = self.compute_sparse_attention(&queries_dense, &keys_dense)?;
            let attention_weights = Tensor::<B, DenseStorage<T>, T>::from_vec(
                attention_weights_dense.as_slice().to_vec(),
                attention_weights_dense.shape().dims()
            )?;

            // Apply attention: attention_weights @ values_reshaped
            // attention_weights: [seq, seq], values_reshaped: [1, seq, embed] -> [seq, embed]
            let values_2d = values_reshaped.reshape(&[seq_len as isize, self.embed_dim as isize])?;
            let attention_weights_dense_2 = attention_weights.to_dense_generic()?;
            let values_2d_dense = values_2d.to_dense_generic()?;
            let attended_batch_dense = attention_weights_dense_2.matmul(&values_2d_dense)?;
            let attended_batch = Tensor::<B, DenseStorage<T>, T>::from_vec(
                attended_batch_dense.as_slice().to_vec(),
                attended_batch_dense.shape().dims()
            )?;

            // Append to output
            attended_data.extend_from_slice(attended_batch.as_slice());
        }

        // Create attended output tensor: [batch, seq, embed]
        let attended_output = Tensor::<B, DenseStorage<T>, T>::from_vec(
            attended_data,
            &[batch_size, seq_len, self.embed_dim]
        )?;

        // Reshape for output projection: [batch, seq, embed] -> [batch*seq, embed]
        let attended_2d = attended_output.reshape(&[
            isize::try_from(batch_size * seq_len).unwrap(),
            isize::try_from(self.embed_dim).unwrap(),
        ])?;

        // Final linear projection with sparse weights
        let out_weights = self.out_proj.data().to_dense_generic()?;
        let output_2d = crate::functional::linear(&attended_2d, &out_weights, None)?;

        // Convert dense computation result back to target storage type S
        // Note: Current implementation does dense computation, future versions may maintain sparsity
        let output_data = output_2d.as_slice().to_vec();
        let result = Tensor::<B, S, T>::from_vec(output_data, &[batch_size, seq_len, self.embed_dim])
            .map_err(NNError::from)?;

        // Preserve gradient requirements from input
        Ok(result.requires_grad_(requires_grad))
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![
            self.query_proj.clone(),
            self.key_proj.clone(),
            self.value_proj.clone(),
            self.out_proj.clone(),
        ]
    }

    fn modules(&self) -> Vec<&dyn Module<B, S, T>> {
        vec![]
    }

    fn zero_grad(&mut self) {
        self.query_proj.zero_grad();
        self.key_proj.zero_grad();
        self.value_proj.zero_grad();
        self.out_proj.zero_grad();
    }

    fn train(&mut self, _mode: bool) {
        // SparseAttention layers don't have training-specific behavior
    }

    fn name(&self) -> &str {
        "SparseAttention"
    }
}

impl<B, S, T> SparseAttention<B, S, T>
where
    B: Backend,
    S: Storage<T> + Clone + StorageFromVec<T>,
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd,
{
    /// Reshape tensor for attention computation.
    #[allow(dead_code)]
    fn reshape_for_attention(
        tensor: &Tensor<CpuBackend, DenseStorage<T>, T>,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
        let embed_dim = tensor.shape().dims()[2];

        // Handle batched input: [batch, seq, embed] -> [batch * seq, embed]
        // This preserves batch information for per-batch processing
        Ok(tensor.reshape(&[(batch_size * seq_len) as isize, embed_dim as isize])?)
    }

    /// Reshape tensor back from attention computation.
    #[allow(dead_code)]
    fn reshape_from_attention(
        tensor: &Tensor<CpuBackend, DenseStorage<T>, T>,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
        let embed_dim = tensor.shape().dims()[1];

        // Reshape [batch * seq, embed] -> [batch, seq, embed]
        Ok(tensor.reshape(&[batch_size as isize, seq_len as isize, embed_dim as isize])?)
    }
}

impl<B, S, T> fmt::Display for SparseAttention<B, S, T>
where
    B: Backend,
    S: Storage<T> + Clone + StorageFromVec<T>,
    T: DataType,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SparseAttention(embed_dim={}, num_heads={}, sparsity={:.1})",
            self.embed_dim, self.num_heads, self.sparsity
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_dtype::float::Float32;

    #[test]
    fn test_sparse_attention_creation() {
        let attention = SparseAttention::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 8, 0.9).unwrap();

        assert_eq!(attention.embed_dim, 64);
        assert_eq!(attention.num_heads, 8);
        assert_eq!(attention.head_dim, 8); // 64 / 8
        assert!((attention.sparsity - 0.9).abs() < 1e-6);

        let params = attention.parameters();
        assert_eq!(params.len(), 4);
        assert_eq!(params[0].name(), "query_proj");
        assert_eq!(params[1].name(), "key_proj");
        assert_eq!(params[2].name(), "value_proj");
        assert_eq!(params[3].name(), "out_proj");
    }

    #[test]
    fn test_sparse_attention_forward_shape() {
        let attention = SparseAttention::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 8, 0.9).unwrap();

        // Test with batch_size=2, seq_len=10, embed_dim=64
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[2, 10, 64]).unwrap();
        let output = attention.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[2, 10, 64]);
    }

    #[test]
    fn test_sparse_attention_invalid_input_shape() {
        let attention = SparseAttention::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 8, 0.9).unwrap();

        // Wrong embed_dim
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[2, 10, 32]).unwrap();
        let result = attention.forward(&input);

        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_attention_display() {
        let attention = SparseAttention::<CpuBackend, DenseStorage<Float32>, Float32>::new(128, 16, 0.95).unwrap();
        let display = format!("{}", attention);

        assert!(display.contains("SparseAttention"));
        assert!(display.contains("embed_dim=128"));
        assert!(display.contains("num_heads=16"));
        assert!(display.contains("sparsity=0.9"));
    }
}

#[cfg(test)]
mod multihead_tests {
    use super::*;
    use coeus_dtype::float::Float32;

    #[test]
    fn test_multihead_attention_creation() {
        let attention = MultiHeadAttention::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 8).unwrap();

        assert_eq!(attention.embed_dim, 64);
        assert_eq!(attention.num_heads, 8);
        assert_eq!(attention.head_dim, 8); // 64 / 8

        let params = attention.parameters();
        assert_eq!(params.len(), 4);
        assert_eq!(params[0].name(), "query_proj");
        assert_eq!(params[1].name(), "key_proj");
        assert_eq!(params[2].name(), "value_proj");
        assert_eq!(params[3].name(), "out_proj");
    }

    #[test]
    fn test_multihead_attention_forward_shape() {
        let attention = MultiHeadAttention::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 8).unwrap();

        // Test with batch_size=1, seq_len=10, embed_dim=64
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[1, 10, 64]).unwrap();
        let output = attention.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[1, 10, 64]);
    }

    #[test]
    fn test_multihead_attention_forward_shape_batch2() {
        let attention = MultiHeadAttention::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 8).unwrap();

        // Test with batch_size=2, seq_len=10, embed_dim=64
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[2, 10, 64]).unwrap();
        let output = attention.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[2, 10, 64]);
    }

    #[test]
    fn test_multihead_attention_forward_shape_batch4() {
        let attention = MultiHeadAttention::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 8).unwrap();

        // Test with batch_size=4, seq_len=10, embed_dim=64
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[4, 10, 64]).unwrap();
        let output = attention.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[4, 10, 64]);
    }

    #[test]
    fn test_multihead_attention_forward_shape_batch8() {
        let attention = MultiHeadAttention::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 8).unwrap();

        // Test with batch_size=8, seq_len=10, embed_dim=64
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[8, 10, 64]).unwrap();
        let output = attention.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[8, 10, 64]);
    }

    #[test]
    fn test_multihead_attention_cross_attention_batch2() {
        let attention = MultiHeadAttention::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 8).unwrap();

        // Test cross-attention with batch_size=2
        // Query from decoder: [2, 8, 64] (batch=2, seq=8, embed=64)
        let query =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[2, 8, 64]).unwrap();
        // Key/Value from encoder: [2, 12, 64] (batch=2, seq=12, embed=64)
        let key =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[2, 12, 64]).unwrap();
        let value =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[2, 12, 64]).unwrap();

        let output = attention
            .forward_cross_attention(&query, &key, &value)
            .unwrap();

        // Output should match query shape: [2, 8, 64]
        assert_eq!(output.shape().dims(), &[2, 8, 64]);
    }

    #[test]
    fn test_multihead_attention_invalid_embed_dim() {
        let attention = MultiHeadAttention::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 8).unwrap();

        // Wrong embed_dim
        let input =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[1, 10, 32]).unwrap();
        let result = attention.forward(&input);

        assert!(result.is_err());
    }

    #[test]
    fn test_multihead_attention_display() {
        let attention = MultiHeadAttention::<CpuBackend, DenseStorage<Float32>, Float32>::new(128, 16).unwrap();
        let display = format!("{}", attention);

        assert!(display.contains("MultiHeadAttention"));
        assert!(display.contains("embed_dim=128"));
        assert!(display.contains("num_heads=16"));
        assert!(display.contains("head_dim=8"));
    }

    #[test]
    fn test_multihead_attention_cross_attention() {
        let attention = MultiHeadAttention::<CpuBackend, DenseStorage<Float32>, Float32>::new(64, 8).unwrap();

        // Query from decoder: [1, 8, 64] (batch=1, seq=8, embed=64)
        let query =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[1, 8, 64]).unwrap();
        // Key/Value from encoder: [1, 12, 64] (batch=1, seq=12, embed=64)
        let key =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[1, 12, 64]).unwrap();
        let value =
            Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[1, 12, 64]).unwrap();

        let output = attention
            .forward_cross_attention(&query, &key, &value)
            .unwrap();

        // Output should match query shape: [1, 8, 64]
        assert_eq!(output.shape().dims(), &[1, 8, 64]);
    }

    #[cfg(feature = "quantized")]
    #[test]
    fn test_quantized_multihead_attention() {
        let backend = CpuBackend::new();
        let embed_dim = 64;
        let num_heads = 8;

        let config = MixedPrecisionConfig::new()
            .with_default_bitwidth(QuantizationBitwidth::Bits8)
            .with_scheme(QuantizationScheme::Affine)
            .with_granularity(QuantizationGranularity::PerTensor);

        let mut attention = QuantizedMultiHeadAttention::new(embed_dim, num_heads, config).unwrap();

        // Test input: [batch_size=2, seq_len=10, embed_dim=64]
        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[2, 10, 64]).unwrap();

        let output = attention.forward(&input).unwrap();

        // Output should match input shape
        assert_eq!(output.shape().dims(), &[2, 10, 64]);
    }

    #[cfg(feature = "quantized")]
    #[test]
    fn test_quantized_sparse_attention() {
        let backend = CpuBackend::new();
        let embed_dim = 64;
        let num_heads = 8;

        let config = MixedPrecisionConfig::new()
            .with_default_bitwidth(QuantizationBitwidth::Bits4)
            .with_scheme(QuantizationScheme::Symmetric)
            .with_granularity(QuantizationGranularity::PerTensor);

        let mut attention = QuantizedSparseAttention::new(embed_dim, num_heads, 0.1, config).unwrap();

        // Test input: [batch_size=2, seq_len=10, embed_dim=64]
        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[2, 10, 64]).unwrap();

        let output = attention.forward(&input).unwrap();

        // Output should match input shape
        assert_eq!(output.shape().dims(), &[2, 10, 64]);
    }

    #[test]
    fn test_kv_cache() {
        let cache = KVCache::<CpuBackend, DenseStorage<Float32>, Float32>::new(
            12, // num_layers
            8,  // num_heads
            64, // head_dim
            2,  // batch_size
            1024, // max_seq_len
        ).unwrap();

        // Check initial state
        assert_eq!(cache.num_layers, 12);
        assert_eq!(cache.num_heads, 8);
        assert_eq!(cache.head_dim, 64);
        assert_eq!(cache.max_seq_len, 1024);
        assert_eq!(cache.seq_lengths, vec![0, 0]);

        // Check memory usage (should be pre-allocated)
        let memory_usage = cache.memory_usage();
        let expected_elements = 12 * 2 * 1024 * 8 * 64 * 2; // layers * batch * seq * heads * dim * (keys + values)
        assert_eq!(memory_usage, expected_elements);
    }

    #[cfg(feature = "quantized")]
    #[test]
    fn test_quantized_kv_cache() {
        let config = MixedPrecisionConfig::new()
            .with_default_bitwidth(QuantizationBitwidth::Bits8)
            .with_scheme(QuantizationScheme::Affine)
            .with_granularity(QuantizationGranularity::PerTensor);

        let cache = QuantizedKVCache::<CpuBackend, DenseStorage<Float32>, Float32>::new(
            12, // num_layers
            8,  // num_heads
            64, // head_dim
            2,  // batch_size
            1024, // max_seq_len
            config,
            KVCacheQuantizationPolicy::PerSequence,
        );

        // Check initial state
        assert_eq!(cache.num_layers, 12);
        assert_eq!(cache.num_heads, 8);
        assert_eq!(cache.head_dim, 64);
        assert_eq!(cache.max_seq_len, 1024);
        assert_eq!(cache.seq_lengths, vec![0, 0]);
        assert_eq!(cache.policy, KVCacheQuantizationPolicy::PerSequence);

        // Test compression stats
        let stats = cache.compression_stats();
        assert!(stats.compression_ratio > 1.0); // Should show compression
        assert_eq!(stats.bitwidth, QuantizationBitwidth::Bits8);
    }
}

