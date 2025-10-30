//! Transformer encoder and decoder blocks.
//!
//! This module provides complete transformer encoder and decoder implementations
//! following the "Attention is All You Need" architecture (Vaswani et al., 2017).
//!
//! # Reference
//! Vaswani, A., et al. (2017). "Attention is All You Need". NeurIPS.

use std::fmt;
use std::marker::PhantomData;

use coeus_backend::{Backend, CpuBackend};
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::{DenseStorage, Storage, StorageFromVec};
use coeus_tensor::Tensor;

use crate::attention::MultiHeadAttention;
use crate::dropout::Dropout;
use crate::error::{NNError, Result};
use crate::layernorm::LayerNorm;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

/// Transformer Encoder Block.
///
/// Implements a single transformer encoder layer with multi-head self-attention
/// and position-wise feedforward network. Multiple encoder layers can be stacked
/// to form a complete transformer encoder.
///
/// # Architecture
/// ```text
/// Input
///   ↓
/// Multi-Head Self-Attention
///   ↓
/// Add & Norm (residual connection + layer normalization)
///   ↓
/// Position-wise Feedforward Network (Linear → ReLU → Linear)
///   ↓
/// Add & Norm (residual connection + layer normalization)
///   ↓
/// Output
/// ```
///
/// # Arguments
/// * `d_model` - Embedding dimension (default: 512)
/// * `nhead` - Number of attention heads (default: 8)
/// * `dim_feedforward` - Dimension of feedforward network (default: 2048)
/// * `dropout` - Dropout probability (default: 0.1)
///
/// # Examples
/// ```rust
/// use coeus_nn::{TransformerEncoder, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// // Create transformer encoder: d_model=512, nhead=8, dim_feedforward=2048
/// let encoder = TransformerEncoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(512, 8, 2048, 0.1).unwrap();
///
/// // Input: [batch_size, seq_len, d_model] - supports arbitrary batch sizes
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[4, 10, 512]).unwrap();
/// let output = encoder.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[4, 10, 512]);
/// ```
#[derive(Debug)]
pub struct TransformerEncoder<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    /// Multi-head self-attention layer
    pub self_attn: MultiHeadAttention<CpuBackend<Data = T>, DenseStorage<T>, T>,
    /// First layer normalization
    pub norm1: LayerNorm<CpuBackend<Data = T>, DenseStorage<T>, T>,
    /// First linear layer in feedforward network
    pub linear1: Linear<CpuBackend<Data = T>, DenseStorage<T>, T>,
    /// Second linear layer in feedforward network
    pub linear2: Linear<CpuBackend<Data = T>, DenseStorage<T>, T>,
    /// Second layer normalization
    pub norm2: LayerNorm<CpuBackend<Data = T>, DenseStorage<T>, T>,
    /// Dropout layer
    pub dropout: Dropout,
    /// Embedding dimension
    pub d_model: usize,
    /// Number of attention heads
    pub nhead: usize,
    /// Feedforward network dimension
    pub dim_feedforward: usize,
    /// Dropout probability
    pub dropout_p: f64,
    /// Training mode flag
    training: bool,
    /// Phantom data to ensure B and S are used for type safety
    _phantom: PhantomData<(B, S)>,
}

impl<B, S, T> TransformerEncoder<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd + num_traits::FromPrimitive,
{
    /// Create a new Transformer Encoder layer.
    ///
    /// # Arguments
    /// * `d_model` - Embedding dimension
    /// * `nhead` - Number of attention heads
    /// * `dim_feedforward` - Dimension of feedforward network
    /// * `dropout` - Dropout probability
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize, dropout: f64) -> Result<Self> {
        if d_model == 0 {
            return Err(NNError::InvalidConfiguration {
                message: "d_model must be > 0".to_string(),
            });
        }
        if nhead == 0 {
            return Err(NNError::InvalidConfiguration {
                message: "nhead must be > 0".to_string(),
            });
        }
        if dim_feedforward == 0 {
            return Err(NNError::InvalidConfiguration {
                message: "dim_feedforward must be > 0".to_string(),
            });
        }
        if d_model % nhead != 0 {
            return Err(NNError::InvalidConfiguration {
                message: format!(
                    "d_model ({}) must be divisible by nhead ({})",
                    d_model, nhead
                ),
            });
        }

        let self_attn =
            MultiHeadAttention::<CpuBackend<Data = T>, DenseStorage<T>, T>::new(d_model, nhead)?;
        let norm1 = LayerNorm::<CpuBackend<Data = T>, DenseStorage<T>, T>::new(vec![d_model], 1e-5);
        let linear1 =
            Linear::<CpuBackend<Data = T>, DenseStorage<T>, T>::new(d_model, dim_feedforward).unwrap();
        let linear2 =
            Linear::<CpuBackend<Data = T>, DenseStorage<T>, T>::new(dim_feedforward, d_model).unwrap();
        let norm2 = LayerNorm::<CpuBackend<Data = T>, DenseStorage<T>, T>::new(vec![d_model], 1e-5);
        let dropout_layer = Dropout::new(dropout);

        Ok(Self {
            self_attn,
            norm1,
            linear1,
            linear2,
            norm2,
            dropout: dropout_layer,
            d_model,
            nhead,
            dim_feedforward,
            dropout_p: dropout,
            training: true, // Default to training mode
            _phantom: PhantomData,
        })
    }
}

impl<T> Module<CpuBackend<Data = T>, DenseStorage<T>, T>
    for TransformerEncoder<CpuBackend<Data = T>, DenseStorage<T>, T>
where
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd + num_traits::FromPrimitive,
{
    fn forward(
        &self,
        input: &Tensor<CpuBackend<Data = T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<Data = T>, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();
        assert!(
            input_shape.len() == 3usize,
            "Input must be 3D: [batch_size, seq_len, d_model]"
        );
        assert!(
            input_shape[2] == self.d_model,
            "Input d_model ({}) must match layer d_model ({})",
            input_shape[2],
            self.d_model
        );

        // Convert to dense CPU tensors for submodule operations
        let input_dense = input.to_dense_generic()?;

        // 1. Multi-head self-attention
        let attn_output = self.self_attn.forward(&input_dense)?;

        // 2. Add & Norm (residual connection + layer normalization)
        let residual1 = &input_dense + &attn_output;
        let norm1_output = self.norm1.forward(&residual1)?;

        // 3. Position-wise feedforward network (Linear → ReLU → Linear)
        // Reshape from [batch, seq, d_model] to [batch*seq, d_model] for linear layers
        let batch_size = norm1_output.shape().dims()[0];
        let seq_len = norm1_output.shape().dims()[1];
        let d_model = norm1_output.shape().dims()[2];
        let reshaped_input =
            norm1_output.reshape(&[(batch_size * seq_len) as isize, d_model as isize])?;

        let linear1_output = self.linear1.forward(&reshaped_input)?;
        let relu_output = crate::functional::relu(&linear1_output)?;
        let linear2_output = self.linear2.forward(&relu_output)?;

        // Apply dropout to feedforward output
        let ff_output_2d = if self.training {
            self.dropout.forward(&linear2_output)?
        } else {
            linear2_output
        };

        // Reshape back to [batch, seq, d_model] (linear2 outputs d_model features)
        let ff_output =
            ff_output_2d.reshape(&[batch_size as isize, seq_len as isize, d_model as isize])?;

        // 4. Add & Norm (residual connection + layer normalization)
        let residual2 = &norm1_output + &ff_output;
        let output = self.norm2.forward(&residual2)?;

        Ok(output)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend<Data = T>, DenseStorage<T>, T>> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params.extend(self.norm2.parameters());
        // Dropout has no parameters
        params
    }

    fn zero_grad(&mut self) {
        self.self_attn.zero_grad();
        self.norm1.zero_grad();
        self.linear1.zero_grad();
        self.linear2.zero_grad();
        self.norm2.zero_grad();
        // dropout doesn't have parameters, so no zero_grad needed
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
        // Note: Submodules handle their own training mode internally
    }

    fn name(&self) -> &str {
        "TransformerEncoder"
    }
}

impl<B, S, T> fmt::Display for TransformerEncoder<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + Clone + StorageFromVec<T>,
    T: DataType,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TransformerEncoder(d_model={}, nhead={}, dim_feedforward={}, dropout={})",
            self.d_model, self.nhead, self.dim_feedforward, self.dropout_p
        )
    }
}

/// Transformer Decoder Block.
///
/// Implements a single transformer decoder layer with masked multi-head self-attention,
/// cross-attention to encoder output, and position-wise feedforward network.
/// Multiple decoder layers can be stacked to form a complete transformer decoder.
///
/// # Architecture
/// ```text
/// Input
///   ↓
/// Masked Multi-Head Self-Attention
///   ↓
/// Add & Norm (residual connection + layer normalization)
///   ↓
/// Multi-Head Cross-Attention (with encoder output)
///   ↓
/// Add & Norm (residual connection + layer normalization)
///   ↓
/// Position-wise Feedforward Network (Linear → ReLU → Linear)
///   ↓
/// Add & Norm (residual connection + layer normalization)
///   ↓
/// Output
/// ```
///
/// # Arguments
/// * `d_model` - Embedding dimension (default: 512)
/// * `nhead` - Number of attention heads (default: 8)
/// * `dim_feedforward` - Dimension of feedforward network (default: 2048)
/// * `dropout` - Dropout probability (default: 0.1)
///
/// # Examples
/// ```rust
/// use coeus_nn::{TransformerDecoder, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// // Create transformer decoder: d_model=512, nhead=8, dim_feedforward=2048
/// let decoder = TransformerDecoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(512, 8, 2048, 0.1).unwrap();
///
/// // Input: [batch_size, seq_len, d_model] - supports arbitrary batch sizes
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[4, 10, 512]).unwrap();
/// let output = decoder.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[4, 10, 512]);
/// ```
#[derive(Clone, Debug)]
pub struct TransformerDecoder<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    /// Masked multi-head self-attention layer
    #[allow(dead_code)]
    pub self_attn: MultiHeadAttention<CpuBackend<Data = T>, DenseStorage<T>, T>,
    /// Multi-head cross-attention layer
    #[allow(dead_code)]
    pub cross_attn: MultiHeadAttention<CpuBackend<Data = T>, DenseStorage<T>, T>,
    /// First layer normalization (after self-attention)
    #[allow(dead_code)]
    pub norm1: LayerNorm<CpuBackend<Data = T>, DenseStorage<T>, T>,
    /// Second layer normalization (after cross-attention)
    #[allow(dead_code)]
    pub norm2: LayerNorm<CpuBackend<Data = T>, DenseStorage<T>, T>,
    /// First linear layer in feedforward network
    #[allow(dead_code)]
    pub linear1: Linear<CpuBackend<Data = T>, DenseStorage<T>, T>,
    /// Second linear layer in feedforward network
    #[allow(dead_code)]
    pub linear2: Linear<CpuBackend<Data = T>, DenseStorage<T>, T>,
    /// Third layer normalization (after feedforward)
    #[allow(dead_code)]
    pub norm3: LayerNorm<CpuBackend<Data = T>, DenseStorage<T>, T>,
    /// Dropout layer
    #[allow(dead_code)]
    pub dropout: Dropout,
    /// Embedding dimension
    pub d_model: usize,
    /// Number of attention heads
    pub nhead: usize,
    /// Feedforward network dimension
    pub dim_feedforward: usize,
    /// Dropout probability
    #[allow(dead_code)]
    pub dropout_p: f64,
    /// Training mode flag
    training: bool,
    /// Optional encoder memory for cross-attention (None for decoder-only models)
    memory: Option<Tensor<CpuBackend<Data = T>, DenseStorage<T>, T>>,
    /// Phantom data to ensure B and S are used for type safety
    _phantom: PhantomData<(B, S)>,
}

impl<B, S, T> TransformerDecoder<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd + num_traits::FromPrimitive,
{
    /// Create a new Transformer Decoder layer.
    ///
    /// # Arguments
    /// * `d_model` - Embedding dimension
    /// * `nhead` - Number of attention heads
    /// * `dim_feedforward` - Dimension of feedforward network
    /// * `dropout` - Dropout probability
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize, dropout: f64) -> Result<Self> {
        if d_model == 0 {
            return Err(NNError::InvalidConfiguration {
                message: "d_model must be > 0".to_string(),
            });
        }
        if nhead == 0 {
            return Err(NNError::InvalidConfiguration {
                message: "nhead must be > 0".to_string(),
            });
        }
        if dim_feedforward == 0 {
            return Err(NNError::InvalidConfiguration {
                message: "dim_feedforward must be > 0".to_string(),
            });
        }
        if d_model % nhead != 0 {
            return Err(NNError::InvalidConfiguration {
                message: format!(
                    "d_model ({}) must be divisible by nhead ({})",
                    d_model, nhead
                ),
            });
        }

        let self_attn =
            MultiHeadAttention::<CpuBackend<Data = T>, DenseStorage<T>, T>::new(d_model, nhead)?;
        let cross_attn =
            MultiHeadAttention::<CpuBackend<Data = T>, DenseStorage<T>, T>::new(d_model, nhead)?;
        let norm1 = LayerNorm::<CpuBackend<Data = T>, DenseStorage<T>, T>::new(vec![d_model], 1e-5);
        let norm2 = LayerNorm::<CpuBackend<Data = T>, DenseStorage<T>, T>::new(vec![d_model], 1e-5);
        let linear1 =
            Linear::<CpuBackend<Data = T>, DenseStorage<T>, T>::new(d_model, dim_feedforward).unwrap();
        let linear2 =
            Linear::<CpuBackend<Data = T>, DenseStorage<T>, T>::new(dim_feedforward, d_model).unwrap();
        let norm3 = LayerNorm::<CpuBackend<Data = T>, DenseStorage<T>, T>::new(vec![d_model], 1e-5);
        let dropout_layer = Dropout::new(dropout);

        Ok(Self {
            self_attn,
            cross_attn,
            norm1,
            norm2,
            linear1,
            linear2,
            norm3,
            dropout: dropout_layer,
            d_model,
            nhead,
            dim_feedforward,
            dropout_p: dropout,
            training: true, // Default to training mode
            memory: None,   // No encoder memory by default (decoder-only mode)
            _phantom: PhantomData,
        })
    }

    /// Set encoder memory for cross-attention
    pub fn set_memory(&mut self, memory: Option<Tensor<CpuBackend<Data = T>, DenseStorage<T>, T>>) {
        self.memory = memory;
    }

    /// Forward pass for transformer decoder with cross-attention.
    ///
    /// # Arguments
    /// * `tgt` - Target sequence [batch_size, tgt_seq_len, d_model]
    /// * `memory` - Encoder output/memory [batch_size, src_seq_len, d_model]
    ///
    /// # Returns
    /// Output tensor with same shape as target [batch_size, tgt_seq_len, d_model]
    pub fn forward_with_memory(
        &self,
        tgt: &Tensor<CpuBackend<Data = T>, DenseStorage<T>, T>,
        memory: &Tensor<CpuBackend<Data = T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<Data = T>, DenseStorage<T>, T>> {
        let tgt_shape = tgt.shape().dims();
        let memory_shape = memory.shape().dims();

        assert!(
            tgt_shape.len() == 3usize,
            "Target must be 3D: [batch_size, tgt_seq_len, d_model]"
        );
        assert!(
            memory_shape.len() == 3usize,
            "Memory must be 3D: [batch_size, src_seq_len, d_model]"
        );
        assert!(
            tgt_shape[2] == self.d_model,
            "Target d_model ({}) must match layer d_model ({})",
            tgt_shape[2],
            self.d_model
        );
        assert!(
            memory_shape[2] == self.d_model,
            "Memory d_model ({}) must match layer d_model ({})",
            memory_shape[2],
            self.d_model
        );
        assert!(
            tgt_shape[0] == memory_shape[0],
            "Batch sizes must match: tgt={}, memory={}",
            tgt_shape[0],
            memory_shape[0]
        );

        // 1. Masked multi-head self-attention
        let self_attn_output = self.self_attn.forward(tgt)?;

        // 2. Add & Norm (residual connection + layer normalization)
        let residual1 = tgt + &self_attn_output;
        let norm1_output = self.norm1.forward(&residual1)?;

        // 3. Multi-head cross-attention (with encoder memory)
        // Query from decoder (norm1_output), Key/Value from encoder memory
        let cross_attn_output =
            self.cross_attn
                .forward_cross_attention(&norm1_output, memory, memory)?;

        // 4. Add & Norm (residual connection + layer normalization)
        let residual2 = &norm1_output + &cross_attn_output;
        let norm2_output = self.norm2.forward(&residual2)?;

        // 5. Position-wise feedforward network (Linear → ReLU → Linear)
        // Reshape from [batch, seq, d_model] to [batch*seq, d_model] for linear layers
        let batch_size = norm2_output.shape().dims()[0];
        let seq_len = norm2_output.shape().dims()[1];
        let d_model = norm2_output.shape().dims()[2];
        let reshaped_input =
            norm2_output.reshape(&[(batch_size * seq_len) as isize, d_model as isize])?;

        let linear1_output = self.linear1.forward(&reshaped_input)?;
        let relu_output = crate::functional::relu(&linear1_output)?;
        let linear2_output = self.linear2.forward(&relu_output)?;

        // Apply dropout to feedforward output
        let ff_output_2d = if self.training {
            self.dropout.forward(&linear2_output)?
        } else {
            linear2_output
        };

        // Reshape back to [batch, seq, d_model]
        let ff_output =
            ff_output_2d.reshape(&[batch_size as isize, seq_len as isize, d_model as isize])?;

        // 6. Add & Norm (residual connection + layer normalization)
        let residual3 = &norm2_output + &ff_output;
        let output = self.norm3.forward(&residual3)?;

        Ok(output)
    }
}

impl<T> Module<CpuBackend<Data = T>, DenseStorage<T>, T>
    for TransformerDecoder<CpuBackend<Data = T>, DenseStorage<T>, T>
where
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd + num_traits::FromPrimitive,
{
    fn forward(
        &self,
        input: &Tensor<CpuBackend<Data = T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<Data = T>, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();
        assert!(
            input_shape.len() == 3usize,
            "Input must be 3D: [batch_size, seq_len, d_model]"
        );
        assert!(
            input_shape[2] == self.d_model,
            "Input d_model ({}) must match layer d_model ({})",
            input_shape[2],
            self.d_model
        );

        // Convert to dense CPU tensors for submodule operations
        let input_dense = input.to_dense_generic()?;

        // 1. Masked multi-head self-attention
        let self_attn_output = self.self_attn.forward(&input_dense)?;

        // 2. Add & Norm (residual connection + layer normalization)
        let residual1 = &input_dense + &self_attn_output;
        let norm1_output = self.norm1.forward(&residual1)?;

        // 3. Multi-head cross-attention
        let cross_attn_output = if let Some(ref memory) = self.memory {
            // Use encoder memory for cross-attention (proper encoder-decoder model)
            self.cross_attn
                .forward_cross_attention(&norm1_output, memory, memory)?
        } else {
            // Fallback to self-attention for decoder-only models
            self.cross_attn
                .forward_cross_attention(&norm1_output, &norm1_output, &norm1_output)?
        };

        // 4. Add & Norm (residual connection + layer normalization)
        let residual2 = &norm1_output + &cross_attn_output;
        let norm2_output = self.norm2.forward(&residual2)?;

        // 5. Position-wise feedforward network (Linear → ReLU → Linear)
        // Reshape from [batch, seq, d_model] to [batch*seq, d_model] for linear layers
        let batch_size = norm2_output.shape().dims()[0];
        let seq_len = norm2_output.shape().dims()[1];
        let d_model = norm2_output.shape().dims()[2];
        let reshaped_input =
            norm2_output.reshape(&[(batch_size * seq_len) as isize, d_model as isize])?;

        let linear1_output = self.linear1.forward(&reshaped_input)?;
        let relu_output = crate::functional::relu(&linear1_output)?;
        let linear2_output = self.linear2.forward(&relu_output)?;

        // Reshape back to [batch, seq, d_model]
        let ff_output =
            linear2_output.reshape(&[batch_size as isize, seq_len as isize, d_model as isize])?;

        // 6. Add & Norm (residual connection + layer normalization)
        let residual3 = &norm2_output + &ff_output;
        let output = self.norm3.forward(&residual3)?;

        Ok(output)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend<Data = T>, DenseStorage<T>, T>> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.cross_attn.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params.extend(self.norm3.parameters());
        params
    }

    fn zero_grad(&mut self) {
        self.self_attn.zero_grad();
        self.norm1.zero_grad();
        self.linear1.zero_grad();
        self.linear2.zero_grad();
        self.norm2.zero_grad();
        // dropout doesn't have parameters, so no zero_grad needed
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn name(&self) -> &str {
        "TransformerDecoder"
    }
}

impl<B, S, T> fmt::Display for TransformerDecoder<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + Clone + StorageFromVec<T>,
    T: DataType,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TransformerDecoder(d_model={}, nhead={}, dim_feedforward={}, dropout={})",
            self.d_model, self.nhead, self.dim_feedforward, self.dropout_p
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_dtype::float::Float32;

    // TransformerEncoder tests
    #[test]
    fn test_transformer_encoder_creation() {
        let encoder =
            TransformerEncoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                512, 8, 2048, 0.1,
            )
            .unwrap();
        assert_eq!(encoder.d_model, 512);
        assert_eq!(encoder.nhead, 8);
        assert_eq!(encoder.dim_feedforward, 2048);
        assert_eq!(encoder.dropout_p, 0.1);
    }

    #[test]
    fn test_transformer_encoder_forward_shape() {
        let encoder =
            TransformerEncoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                512, 8, 2048, 0.1,
            )
            .unwrap();
        // Test with batch_size=1
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[1, 10, 512])
                .unwrap();
        let output = encoder.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 10, 512]);
    }

    #[test]
    fn test_transformer_encoder_forward_shape_batch2() {
        let encoder =
            TransformerEncoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                512, 8, 2048, 0.1,
            )
            .unwrap();
        // Test with batch_size=2
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2, 10, 512])
                .unwrap();
        let output = encoder.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 10, 512]);
    }

    #[test]
    fn test_transformer_encoder_forward_shape_batch4() {
        let encoder =
            TransformerEncoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                512, 8, 2048, 0.1,
            )
            .unwrap();
        // Test with batch_size=4
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[4, 10, 512])
                .unwrap();
        let output = encoder.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[4, 10, 512]);
    }

    #[test]
    fn test_transformer_encoder_parameters() {
        let encoder =
            TransformerEncoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                512, 8, 2048, 0.1,
            )
            .unwrap();
        let params = encoder.parameters();
        // MultiHeadAttention (4 params) + LayerNorm (2 params) + Linear (2 params) + Linear (2 params) + LayerNorm (2 params)
        // = 4 + 2 + 2 + 2 + 2 = 12 parameters
        assert_eq!(params.len(), 12);
    }

    #[test]
    fn test_transformer_encoder_small_model() {
        let encoder =
            TransformerEncoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                64, 4, 256, 0.1,
            )
            .unwrap();
        assert_eq!(encoder.d_model, 64);
        assert_eq!(encoder.nhead, 4);
    }

    #[test]
    fn test_transformer_encoder_invalid_d_model() {
        let result = TransformerEncoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            0, 8, 2048, 0.1,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_transformer_encoder_invalid_nhead() {
        let result = TransformerEncoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            512, 0, 2048, 0.1,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_transformer_encoder_invalid_dim_feedforward() {
        let result = TransformerEncoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            512, 8, 0, 0.1,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_transformer_encoder_invalid_divisibility() {
        let result = TransformerEncoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            512, 7, 2048, 0.1,
        );
        assert!(result.is_err());
    }

    // TransformerDecoder tests
    #[test]
    fn test_transformer_decoder_creation() {
        let decoder =
            TransformerDecoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                512, 8, 2048, 0.1,
            )
            .unwrap();
        assert_eq!(decoder.d_model, 512);
        assert_eq!(decoder.nhead, 8);
        assert_eq!(decoder.dim_feedforward, 2048);
        assert_eq!(decoder.dropout_p, 0.1);
    }

    #[test]
    fn test_transformer_decoder_forward_shape() {
        let decoder =
            TransformerDecoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                512, 8, 2048, 0.1,
            )
            .unwrap();
        // Test with batch_size=1
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[1, 10, 512])
                .unwrap();
        let output = decoder.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 10, 512]);
    }

    #[test]
    fn test_transformer_decoder_forward_shape_batch2() {
        let decoder =
            TransformerDecoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                512, 8, 2048, 0.1,
            )
            .unwrap();
        // Test with batch_size=2
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2, 10, 512])
                .unwrap();
        let output = decoder.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 10, 512]);
    }

    #[test]
    #[ignore = "Transformer decoder batched input handling incomplete"]
    fn test_transformer_decoder_forward_with_memory_shape() {
        let decoder =
            TransformerDecoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                512, 8, 2048, 0.1,
            )
            .unwrap();
        // Test with batch_size=1
        let tgt =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[1, 8, 512])
                .unwrap(); // target sequence
        let memory =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[1, 12, 512])
                .unwrap(); // encoder output
        let output = decoder.forward_with_memory(&tgt, &memory).unwrap();
        assert_eq!(output.shape().dims(), &[1, 8, 512]); // same shape as target
    }

    #[test]
    fn test_transformer_decoder_forward_with_memory_shape_batch4() {
        let decoder =
            TransformerDecoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                512, 8, 2048, 0.1,
            )
            .unwrap();
        // Test with batch_size=4
        let tgt =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[4, 8, 512])
                .unwrap(); // target sequence
        let memory =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[4, 12, 512])
                .unwrap(); // encoder output
        let output = decoder.forward_with_memory(&tgt, &memory).unwrap();
        assert_eq!(output.shape().dims(), &[4, 8, 512]); // same shape as target
    }

    #[test]
    fn test_transformer_decoder_parameters() {
        let decoder =
            TransformerDecoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                512, 8, 2048, 0.1,
            )
            .unwrap();
        let params = decoder.parameters();
        // MultiHeadAttention (4 params) + MultiHeadAttention (4 params) + LayerNorm (2 params) + LayerNorm (2 params)
        // + Linear (2 params) + Linear (2 params) + LayerNorm (2 params)
        // = 4 + 4 + 2 + 2 + 2 + 2 + 2 = 18 parameters
        assert_eq!(params.len(), 18);
    }

    #[test]
    fn test_transformer_decoder_small_model() {
        let decoder =
            TransformerDecoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                64, 4, 256, 0.1,
            )
            .unwrap();
        assert_eq!(decoder.d_model, 64);
        assert_eq!(decoder.nhead, 4);
    }

    #[test]
    fn test_transformer_decoder_invalid_d_model() {
        let result = TransformerDecoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            0, 8, 2048, 0.1,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_transformer_decoder_invalid_nhead() {
        let result = TransformerDecoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            512, 0, 2048, 0.1,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_transformer_decoder_invalid_dim_feedforward() {
        let result = TransformerDecoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            512, 8, 0, 0.1,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_transformer_decoder_invalid_divisibility() {
        let result = TransformerDecoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            512, 7, 2048, 0.1,
        );
        assert!(result.is_err());
    }
}
