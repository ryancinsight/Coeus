//! Complete Transformer (Encoder-Decoder) Architecture
//!
//! This module provides the full transformer architecture combining encoder and decoder
//! components for sequence-to-sequence tasks like machine translation.

use crate::{Module, Result};
use coeus_dtype::Dtype;
use std::fmt;
use coeus_tensor::{Tensor, FloatDtype};
use coeus_backend::{Backend, CpuBackend};

/// Transformer (Encoder-Decoder Architecture)
///
/// A complete transformer model consisting of an encoder and decoder.
/// This is the standard architecture used in machine translation and other sequence-to-sequence tasks.
///
/// # Backend Integration
///
/// This transformer uses backend-agnostic tensor operations, supporting both CPU and GPU
/// operations through the tensor system's backend abstraction. All tensor operations
/// are performed through the backend system for optimal performance and device flexibility.
///
/// # Architecture
///
/// The transformer consists of:
/// - **Encoder**: Processes source sequences into contextual representations
/// - **Decoder**: Generates target sequences using encoded context
/// - **Cross-Attention**: Allows decoder to attend to encoder outputs
///
/// # Examples
///
/// ```rust,no_run
/// use coeus_nn::modules::Transformer;
/// use coeus_tensor::Tensor;
///
/// // Create transformer with default backend
/// let transformer = Transformer::<f32>::new(
///     512, 8, 6, 6, 2048, 0.1
/// );
///
/// // Use in sequence-to-sequence tasks
/// let source_seq = Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap();
/// let target_seq = Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap();
///
/// let output = transformer.forward(&source_seq, &target_seq, None, None, None);
/// ```
#[derive(Debug, Clone)]
pub struct Transformer<T: FloatDtype + rand::distributions::uniform::SampleUniform> {
    /// Transformer encoder
    pub encoder: crate::modules::transformer_encoder::TransformerEncoder<T>,
    /// Transformer decoder
    pub decoder: crate::modules::transformer_decoder::TransformerDecoder<T>,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> Transformer<T> {
    /// Create a new Transformer
    ///
    /// # Arguments
    /// * `d_model` - Model dimension (embedding size)
    /// * `nhead` - Number of attention heads
    /// * `num_encoder_layers` - Number of encoder layers
    /// * `num_decoder_layers` - Number of decoder layers
    /// * `dim_feedforward` - Dimension of feed-forward networks
    /// * `dropout` - Dropout probability
    ///
    /// # Returns
    /// A new Transformer instance
    pub fn new(
        d_model: usize,
        nhead: usize,
        num_encoder_layers: usize,
        num_decoder_layers: usize,
        dim_feedforward: usize,
        dropout: f64,
    ) -> Result<Self> {
        let encoder_layer = crate::modules::transformer_encoder::TransformerEncoderLayer::new(d_model, nhead, dim_feedforward, dropout)?;
        let encoder = crate::modules::transformer_encoder::TransformerEncoder::new(encoder_layer, num_encoder_layers, None);

        let decoder_layer = crate::modules::transformer_decoder::TransformerDecoderLayer::new(d_model, nhead, dim_feedforward, dropout)?;
        let decoder = crate::modules::transformer_decoder::TransformerDecoder::new(decoder_layer, num_decoder_layers, None);

        Ok(Self { encoder, decoder })
    }

    /// Forward pass for sequence-to-sequence tasks
    ///
    /// # Arguments
    /// * `src` - Source sequence tensor of shape (batch_size, src_seq_len, d_model)
    /// * `tgt` - Target sequence tensor of shape (batch_size, tgt_seq_len, d_model)
    /// * `src_mask` - Optional source mask for encoder self-attention
    /// * `tgt_mask` - Optional target mask for decoder self-attention (causal mask)
    /// * `memory_mask` - Optional memory mask for encoder-decoder cross-attention
    ///
    /// # Returns
    /// Output tensor of shape (batch_size, tgt_seq_len, d_model)
    ///
    /// # Errors
    /// Returns an error if tensor operations fail or shapes are incompatible
    pub fn forward(
        &self,
        src: &Tensor<T, CpuBackend>,
        tgt: &Tensor<T, CpuBackend>,
        _src_mask: Option<&Tensor<T, CpuBackend>>,
        tgt_mask: Option<&Tensor<T, CpuBackend>>,
        memory_mask: Option<&Tensor<T, CpuBackend>>,
    ) -> crate::Result<Tensor<T, CpuBackend>> {
        // Encode source sequence
        let memory = self.encoder.forward(src)?;

        // Decode target sequence using encoded memory
        let output = self
            .decoder
            .forward(tgt, Some(&memory), tgt_mask, memory_mask)?;

        Ok(output)
    }
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> Module<T> for Transformer<T> {
    fn forward(&self, input: &coeus_tensor::Tensor<T, CpuBackend>) -> crate::Result<coeus_tensor::Tensor<T, CpuBackend>> {
        // For Module trait, assume encoder-only operation (no decoder)
        self.encoder.forward(input)
    }

    fn parameters(&self) -> Vec<&coeus_tensor::Tensor<T, CpuBackend>> {
        let mut params = Vec::new();
        params.extend(self.encoder.parameters());
        params.extend(self.decoder.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut coeus_tensor::Tensor<T, CpuBackend>> {
        let mut params = Vec::new();
        params.extend(self.encoder.parameters_mut());
        params.extend(self.decoder.parameters_mut());
        params
    }
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> fmt::Display for Transformer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Transformer(d_model={}, nhead={}, num_encoder_layers={}, num_decoder_layers={}, dim_feedforward={})",
            self.encoder.layers[0].self_attn.embed_dim,
            self.encoder.layers[0].self_attn.num_heads,
            self.encoder.layers.len(),
            self.decoder.layers.len(),
            self.encoder.layers[0].linear1.out_features
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_transformer_creation() {
        // Test basic transformer creation with standard parameters
        let transformer = Transformer::<f32>::new(512, 8, 6, 6, 2048, 0.1);

        // Verify encoder configuration
        assert_eq!(transformer.encoder.layers.len(), 6);
        assert_eq!(transformer.encoder.layers[0].self_attn.embed_dim, 512);
        assert_eq!(transformer.encoder.layers[0].self_attn.num_heads, 8);

        // Verify decoder configuration
        assert_eq!(transformer.decoder.layers.len(), 6);
        assert_eq!(transformer.decoder.layers[0].self_attn.embed_dim, 512);
        assert_eq!(transformer.decoder.layers[0].self_attn.num_heads, 8);
    }

    #[test]
    fn test_transformer_forward_encoder_only() {
        // Test encoder-only forward pass (Module trait implementation)
        let transformer = Transformer::<f32>::new(64, 4, 2, 2, 128, 0.0);

        // Create input tensor (batch_size=2, seq_len=10, d_model=64)
        let input = Tensor::from_vec(CpuBackend::default(), CpuBackend::default(), vec![0.1; 2 * 10 * 64], vec![2, 10, 64]).unwrap();

        // Forward pass through encoder only (using Module trait forward method)
        let output = Module::<f32>::forward(&transformer, &input).unwrap();

        // Verify output shape
        assert_eq!(output.shape(), &[2, 10, 64]);
        assert!(!output.shape().is_empty());
    }

    #[test]
    fn test_transformer_forward_sequence_to_sequence() {
        // Test full sequence-to-sequence forward pass
        let transformer = Transformer::<f32>::new(64, 4, 2, 2, 128, 0.0);

        // Create source sequence (batch_size=2, src_seq_len=8, d_model=64)
        let src = Tensor::from_vec(CpuBackend::default(), CpuBackend::default(), vec![0.1; 2 * 8 * 64], vec![2, 8, 64]).unwrap();

        // Create target sequence (batch_size=2, tgt_seq_len=6, d_model=64)
        let tgt = Tensor::from_vec(CpuBackend::default(), CpuBackend::default(), vec![0.05; 2 * 6 * 64], vec![2, 6, 64]).unwrap();

        // Forward pass (using custom forward method)
        let output = transformer.forward(&src, &tgt, None, None, None);

        // Verify output shape (should match target sequence shape)
        assert_eq!(output.shape(), &[2, 6, 64]);
    }

    #[test]
    fn test_transformer_parameter_count() {
        // Test parameter counting
        let transformer = Transformer::<f32>::new(128, 8, 3, 3, 512, 0.0);

        let params = transformer.parameters();
        assert!(!params.is_empty());

        // Each transformer has encoder and decoder parameters
        // Encoder: 3 layers × (self-attention + feed-forward)
        // Decoder: 3 layers × (self-attention + cross-attention + feed-forward)
        // Rough estimate: should have many parameters
        assert!(params.len() > 10);
    }

    #[test]
    fn test_transformer_display() {
        // Test Display implementation
        let transformer = Transformer::<f32>::new(256, 4, 6, 6, 1024, 0.1);

        let display_str = format!("{}", transformer);
        assert!(display_str.contains("Transformer"));
        assert!(display_str.contains("d_model=256"));
        assert!(display_str.contains("nhead=4"));
        assert!(display_str.contains("num_encoder_layers=6"));
        assert!(display_str.contains("num_decoder_layers=6"));
        assert!(display_str.contains("dim_feedforward=1024"));
    }

    #[test]
    fn test_transformer_gradient_flow() {
        // Test that gradients flow through the transformer
        let transformer = Transformer::<f32>::new(32, 2, 1, 1, 64, 0.0);

        // Create input tensors
        let src = Tensor::from_vec(CpuBackend::default(), CpuBackend::default(), vec![0.1; 1 * 4 * 32], vec![1, 4, 32]).unwrap();

        let tgt = Tensor::from_vec(CpuBackend::default(), CpuBackend::default(), vec![0.05; 1 * 3 * 32], vec![1, 3, 32]).unwrap();

        // Enable gradients
        let mut src_grad = src.clone();
        src_grad.set_requires_grad(true);
        let mut tgt_grad = tgt.clone();
        tgt_grad.set_requires_grad(true);

        // Forward pass
        let output = transformer.forward(&src_grad, &tgt_grad, None, None, None);

        // Backward pass
        let loss = output.sum();
        loss.backward();

        // Verify gradients are computed
        assert!(src_grad.grad().is_some());
        assert!(tgt_grad.grad().is_some());
    }

    #[test]
    fn test_transformer_different_dtypes() {
        // Test transformer works with different floating-point types
        let transformer_f64 = Transformer::<f64>::new(64, 4, 2, 2, 128, 0.0);

        let input = Tensor::from_vec(CpuBackend::default(), CpuBackend::default(), vec![0.1f64; 1 * 5 * 64], vec![1, 5, 64]).unwrap();

        let output = Module::<f64>::forward(&transformer_f64, &input).unwrap();
        assert_eq!(output.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_transformer_memory_safety() {
        // Test that transformer operations don't cause memory issues
        let transformer = Transformer::<f32>::new(128, 8, 4, 4, 512, 0.0);

        // Large tensors to stress test memory
        let src = Tensor::from_vec(CpuBackend::default(), CpuBackend::default(), vec![0.01; 4 * 20 * 128], vec![4, 20, 128]).unwrap(); // batch=4, seq=20, d_model=128

        let tgt = Tensor::from_vec(CpuBackend::default(), CpuBackend::default(), vec![0.01; 4 * 15 * 128], vec![4, 15, 128]).unwrap(); // batch=4, seq=15, d_model=128

        // Multiple forward passes
        for _ in 0..3 {
            let output = transformer.forward$1.unwrap_grad();
            assert_eq!(output.shape(), &[4, 15, 128]);
        }
    }
}


