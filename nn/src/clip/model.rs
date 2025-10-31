//! CLIP model implementation
//!
//! This module contains the main CLIP model architecture with vision and text encoders,
//! projection heads, and inference methods.

use std::fmt;
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::error::{NNError, Result};
use crate::module::Module;
use crate::parameter::Parameter;
use crate::attention::MultiHeadAttention;
use crate::linear::Linear;
use crate::layernorm::LayerNorm;
use crate::activation::GELU;
use crate::conv2d::Conv2D; // For patch extraction in vision transformer

use super::config::{ClipConfig, VisionConfig, TextConfig};
use super::loss::InfoNCELoss;

/// Vision Transformer encoder for CLIP
#[derive(Debug)]
pub struct VisionTransformer<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Patch embedding layer
    patch_embed: Linear<B, S, T>,
    /// Position embeddings
    position_embed: Parameter<B, S, T>,
    /// Transformer layers
    layers: Vec<VisionTransformerLayer<B, S, T>>,
    /// Final layer norm
    norm: LayerNorm<B, S, T>,
    /// Configuration
    config: VisionConfig,
}

#[derive(Debug)]
pub struct VisionTransformerLayer<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Self-attention mechanism
    attention: MultiHeadAttention<B, S, T>,
    /// First layer norm (pre-attention)
    norm1: LayerNorm<B, S, T>,
    /// Feed-forward network
    mlp: Vec<Linear<B, S, T>>,
    /// Second layer norm (pre-MLP)
    norm2: LayerNorm<B, S, T>,
    /// GELU activation
    gelu: GELU,
}

/// Vision encoder wrapper for CLIP
#[derive(Debug)]
pub struct VisionEncoder<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Vision Transformer model
    vision_model: VisionTransformer<B, S, T>,
    /// Projection head to CLIP embedding space
    projection_head: Linear<B, S, T>,
    /// Config
    config: VisionConfig,
}

impl<B, S, T> VisionEncoder<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Create new vision encoder
    pub fn new(config: &VisionConfig, clip_embed_dim: usize) -> Result<Self> {
        let vision_model = VisionTransformer::new(config)?;
        let projection_head = Linear::new(config.hidden_size, clip_embed_dim)?;

        Ok(Self {
            vision_model,
            projection_head,
            config: config.clone(),
        })
    }

    /// Forward pass: image -> CLIP embedding
    pub fn forward(&self, pixel_values: &[f32], batch_size: usize) -> Result<Tensor<B, DenseStorage<T>, T>> {
        // Convert pixel values to correct format
        let image_tensor = Tensor::<B, S, T>::from_vec(
            pixel_values.iter().map(|&x| T::from(x).unwrap()).collect(),
            &[batch_size, self.config.image_size, self.config.image_size, self.config.num_channels],
        )?;

        // Forward through ViT
        let hidden_states = self.vision_model.forward(&image_tensor)?;

        // Apply projection head: [batch_size, embed_dim] -> [batch_size, clip_embed_dim]
        let projected = self.projection_head.forward(&hidden_states)?;

        Ok(projected)
    }

    fn create_projection_matrix(in_features: usize, out_features: usize) -> Tensor<B, S, T> {
        // Xavier initialization
        let _limit = (T::from(6.0).unwrap() / T::from(in_features + out_features).unwrap()).sqrt();
        Tensor::<B, S, T>::zeros_generic(&[out_features, in_features]).unwrap()
    }

    /// Get projection head parameters
    pub fn projection_head(&self) -> &Parameter<B, S, T> {
        &self.projection_head
    }
}

impl<B, S, T> VisionTransformer<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Create new Vision Transformer
    pub fn new(config: &VisionConfig) -> Result<Self> {
        let image_size = config.image_size;
        let patch_size = config.patch_size;
        let num_patches = (image_size / patch_size) * (image_size / patch_size);
        let embed_dim = config.hidden_size;

        // Patch embedding: flatten patches and project to embed_dim
        let patch_embed = Linear::new(
            config.num_channels * patch_size * patch_size,
            embed_dim,
        )?;

        // Position embeddings for each patch + class token
        let position_embed = Parameter::new(
            &[num_patches + 1, embed_dim],
            &mut rand::thread_rng(),
        )?;

        // Create transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(VisionTransformerLayer::new(
                embed_dim,
                config.num_heads,
                embed_dim * 4, // MLP expansion
            )?);
        }

        let norm = LayerNorm::new(embed_dim, 1e-6)?;

        Ok(Self {
            patch_embed,
            position_embed,
            layers,
            norm,
            config: config.clone(),
        })
    }

    /// Forward pass through Vision Transformer (simplified for implementation)
    pub fn forward(&self, pixel_values: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // pixel_values: [batch_size, height, width, channels]
        let batch_size = pixel_values.shape().dims()[0];
        let embed_dim = self.config.hidden_size;
        let num_patches = (self.config.image_size / self.config.patch_size).pow(2);

        // Simplified ViT implementation for GPU acceleration
        // For now, we use the patch embedding linear layer directly
        // This is a temporary simplification until proper 2D convolution is implemented

        // Flatten image patches: [batch_size, H, W, C] -> [batch_size * num_patches, patch_size * patch_size * channels]
        let patch_data = pixel_values.as_slice().to_vec();
        let patch_sequence = Tensor::<B, S, T>::from_vec(patch_data, &[batch_size * num_patches, self.config.patch_size * self.config.patch_size * self.config.num_channels])?;

        // Apply patch embedding: [batch_size * num_patches, embed_dim]
        let patch_embeddings = self.patch_embed.forward(&patch_sequence)?;

        // Reshape back to [batch_size, num_patches, embed_dim]
        let patch_embeddings_data = patch_embeddings.as_slice().to_vec();
        let patch_embeddings_reshaped = Tensor::<B, S, T>::from_vec(patch_embeddings_data, &[batch_size, num_patches, embed_dim])?;

        // Create class token and expand to batch
        let cls_token_data = vec![T::from(1.0).unwrap(); embed_dim]; // Initialize to non-zero for learning
        let cls_token = Tensor::<B, S, T>::from_vec(cls_token_data, &[1, embed_dim])?;
        let cls_tokens = cls_token.broadcast_to(&[batch_size, 1, embed_dim])?;

        // Concatenate class token + patch embeddings: [batch_size, num_patches + 1, embed_dim]
        let sequence_embeddings = &[&cls_tokens, &patch_embeddings_reshaped];
        let mut hidden_states = crate::tensor::ops::tensor_ops::concatenate_tensors(sequence_embeddings, 1)?;

        // Add position embeddings (simplified - just add positional offset)
        let position_embeddings = self.position_embed.data();
        let position_embeddings_broadcasted = position_embeddings.broadcast_to(&[batch_size, num_patches + 1, embed_dim])?;
        hidden_states = &hidden_states + &position_embeddings_broadcasted;

        // Apply transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        // Apply final layer norm
        hidden_states = self.norm.forward(&hidden_states)?;

        // Extract class token: [batch_size, num_patches + 1, embed_dim] -> [batch_size, embed_dim]
        let hidden_shape = hidden_states.shape().dims();
        let hidden_data = hidden_states.as_slice();

        // Extract cls token manually: first embed_dim elements per batch
        let mut cls_output = Vec::new();
        let seq_length = hidden_shape[1];
        for b in 0..batch_size {
            for e in 0..embed_dim {
                let idx = b * seq_length * embed_dim + e;
                cls_output.push(hidden_data[idx]);
            }
        }

        Tensor::<B, S, T>::from_vec(cls_output, &[batch_size, embed_dim])
    }
}

impl<B, S, T> VisionTransformerLayer<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    pub fn new(embed_dim: usize, num_heads: usize, mlp_dim: usize) -> Result<Self> {
        let attention = MultiHeadAttention::new(embed_dim, num_heads)?;
        let norm1 = LayerNorm::new(embed_dim, 1e-6)?;
        let norm2 = LayerNorm::new(embed_dim, 1e-6)?;

        // MLP: embed_dim -> mlp_dim -> embed_dim
        let mlp = vec![
            Linear::new(embed_dim, mlp_dim)?,
            Linear::new(mlp_dim, embed_dim)?,
        ];

        Ok(Self {
            attention,
            norm1,
            mlp,
            norm2,
            gelu: GELU::new(),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Pre-norm attention
        let normed_hidden = self.norm1.forward(hidden_states)?;
        let attn_output = self.attention.forward(&normed_hidden, &normed_hidden, &normed_hidden)?;
        let hidden_states = hidden_states + &attn_output; // Residual

        // Pre-norm MLP
        let normed_hidden = self.norm2.forward(&hidden_states)?;
        let mlp_hidden = self.mlp[0].forward(&normed_hidden)?;
        let mlp_hidden = self.gelu.forward(&mlp_hidden)?;
        let mlp_output = self.mlp[1].forward(&mlp_hidden)?;
        let hidden_states = hidden_states + &mlp_output; // Residual

        Ok(hidden_states)
    }
}

/// Text Transformer encoder for CLIP
#[derive(Debug)]
pub struct TextTransformer<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Token embeddings
    token_embed: Parameter<B, S, T>,
    /// Position embeddings
    position_embed: Parameter<B, S, T>,
    /// Transformer layers
    layers: Vec<TextTransformerLayer<B, S, T>>,
    /// Final layer norm
    norm: LayerNorm<B, S, T>,
    /// Configuration
    config: TextConfig,
}

#[derive(Debug)]
pub struct TextTransformerLayer<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Self-attention mechanism (causal)
    attention: MultiHeadAttention<B, S, T>,
    /// First layer norm (pre-attention)
    norm1: LayerNorm<B, S, T>,
    /// Feed-forward network
    mlp: Vec<Linear<B, S, T>>,
    /// Second layer norm (pre-MLP)
    norm2: LayerNorm<B, S, T>,
    /// GELU activation
    gelu: GELU,
}

/// Text encoder wrapper for CLIP
#[derive(Debug)]
pub struct TextEncoder<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Text Transformer model
    text_model: TextTransformer<B, S, T>,
    /// Projection head to CLIP embedding space
    projection_head: Linear<B, S, T>,
    /// Config
    config: TextConfig,
}

impl<B, S, T> TextEncoder<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Create new text encoder
    pub fn new(config: &TextConfig, clip_embed_dim: usize) -> Result<Self> {
        let text_model = TextTransformer::new(config)?;
        let projection_head = Linear::new(config.hidden_size, clip_embed_dim)?;

        Ok(Self {
            text_model,
            projection_head,
            config: config.clone(),
        })
    }

    /// Forward pass: token_ids -> CLIP embedding
    pub fn forward(&self, input_ids: &[u32], batch_size: usize) -> Result<Tensor<B, DenseStorage<T>, T>> {
        // Forward through GPT model (placeholder - would need adaptation)
        let hidden_states = self.text_model_forward_placeholder(input_ids, batch_size)?;

        // For text, we typically use the [EOS] token representation or average pooling
        // For CLIP, it's usually the [EOS] token: last position
        let eos_representation = self.extract_eos_token(&hidden_states);

        // Apply projection head
        let projected = eos_representation.matmul(&self.projection_head.data().to_dense_generic()?.transpose(0, 1)?)?;

        Ok(projected)
    }

    fn text_model_forward_placeholder(&self, _input_ids: &[u32], batch_size: usize) -> Result<Tensor<B, DenseStorage<T>, T>> {
        // Placeholder - would integrate with actual GPT forward pass
        // Return sequence of embeddings: [batch_size, seq_len, hidden_size]
        Ok(Tensor::<B, DenseStorage<T>, T>::zeros(&[batch_size, self.config.max_position_embeddings, self.config.hidden_size])?)
    }

    fn extract_eos_token(&self, sequence_output: &Tensor<B, DenseStorage<T>, T>) -> Tensor<B, DenseStorage<T>, T> {
        // Extract last token representation: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
        let shape = sequence_output.shape().dims();
        let batch_size = shape[0];
        let hidden_size = shape[2];

        // For simplicity, use the first token (normally would be conditioned on EOS token position)
        let mut eos_features = Vec::with_capacity(batch_size * hidden_size);

        for b in 0..batch_size {
            for h in 0..hidden_size {
                // Take last position: [b, seq_len-1, h]
                let seq_len = shape[1];
                eos_features.push(sequence_output.as_slice()[(b * seq_len + seq_len - 1) * hidden_size + h]);
            }
        }

        Tensor::<B, DenseStorage<T>, T>::from_vec(eos_features, &[batch_size, hidden_size]).unwrap()
    }

    fn create_projection_matrix(in_features: usize, out_features: usize) -> Tensor<B, S, T> {
        // Xavier initialization
        let _limit = (T::from(6.0).unwrap() / T::from(in_features + out_features).unwrap()).sqrt();
        Tensor::<B, S, T>::zeros_generic(&[out_features, in_features]).unwrap()
    }

    /// Get projection head parameters
    pub fn projection_head(&self) -> &Parameter<B, S, T> {
        &self.projection_head
    }
}

impl<B, S, T> TextTransformer<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Create new Text Transformer
    pub fn new(config: &TextConfig) -> Result<Self> {
        let vocab_size = config.vocab_size;
        let embed_dim = config.hidden_size;
        let max_seq_len = config.max_position_embeddings;

        // Token embeddings
        let token_embed = Parameter::new(
            &[vocab_size, embed_dim],
            &mut rand::thread_rng(),
        )?;

        // Position embeddings
        let position_embed = Parameter::new(
            &[max_seq_len, embed_dim],
            &mut rand::thread_rng(),
        )?;

        // Create transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(TextTransformerLayer::new(
                embed_dim,
                config.num_heads,
                embed_dim * 4, // MLP expansion
            )?);
        }

        let norm = LayerNorm::new(embed_dim, 1e-6)?;

        Ok(Self {
            token_embed,
            position_embed,
            layers,
            norm,
            config: config.clone(),
        })
    }

    /// Forward pass through Text Transformer
    pub fn forward(&self, input_ids: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // input_ids: [batch_size, seq_len]
        let batch_size = input_ids.shape().dims()[0];
        let seq_len = input_ids.shape().dims()[1];
        let embed_dim = self.config.hidden_size;

        // Get token embeddings: [batch_size, seq_len, embed_dim]
        // TODO: Implement proper embedding lookup
        let mut token_embeddings = Tensor::zeros(&[batch_size, seq_len, embed_dim])?;

        // Add position embeddings
        // TODO: Add position embeddings to token_embeddings

        // Apply transformer layers
        let mut hidden_states = token_embeddings;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        // Apply final layer norm
        hidden_states = self.norm.forward(&hidden_states)?;

        // Extract last token representation (for CLIP, we use the [EOS] token)
        // TODO: Extract last token from hidden_states

        Ok(hidden_states)
    }
}

impl<B, S, T> TextTransformerLayer<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    pub fn new(embed_dim: usize, num_heads: usize, mlp_dim: usize) -> Result<Self> {
        let attention = MultiHeadAttention::new(embed_dim, num_heads)?;
        let norm1 = LayerNorm::new(embed_dim, 1e-6)?;
        let norm2 = LayerNorm::new(embed_dim, 1e-6)?;

        // MLP: embed_dim -> mlp_dim -> embed_dim
        let mlp = vec![
            Linear::new(embed_dim, mlp_dim)?,
            Linear::new(mlp_dim, embed_dim)?,
        ];

        Ok(Self {
            attention,
            norm1,
            mlp,
            norm2,
            gelu: GELU::new(),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Pre-norm attention (causal for text)
        let normed_hidden = self.norm1.forward(hidden_states)?;
        let attn_output = self.attention.forward(&normed_hidden, &normed_hidden, &normed_hidden)?;
        let hidden_states = hidden_states + &attn_output; // Residual

        // Pre-norm MLP
        let normed_hidden = self.norm2.forward(&hidden_states)?;
        let mlp_hidden = self.mlp[0].forward(&normed_hidden)?;
        let mlp_hidden = self.gelu.forward(&mlp_hidden)?;
        let mlp_output = self.mlp[1].forward(&mlp_hidden)?;
        let hidden_states = hidden_states + &mlp_output; // Residual

        Ok(hidden_states)
    }
}

/// Main CLIP model
#[derive(Debug)]
pub struct ClipModel<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    /// Vision encoder
    vision_encoder: VisionEncoder<B, S, T>,
    /// Text encoder
    text_encoder: TextEncoder<B, S, T>,
    /// CLIP configuration
    config: ClipConfig,
    /// InfoNCE loss function
    loss_fn: InfoNCELoss<T>,
    /// Temperature parameter (learnable)
    temperature: Parameter<B, S, T>,
}

impl<B, S, T> ClipModel<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Create new CLIP model
    pub fn new(config: ClipConfig) -> Result<Self> {
        let vision_encoder = VisionEncoder::new(&config.vision_config, config.embed_dim)?;
        let text_encoder = TextEncoder::new(&config.text_config, config.embed_dim)?;

        let loss_fn = InfoNCELoss::new(config.temperature);

        // Create learnable temperature parameter
        let temp_value = T::from(config.temperature).unwrap();
        let temperature = Parameter::new(
            Tensor::<B, S, T>::from_vec(vec![temp_value], &[1])?.requires_grad_(true),
            "temperature".to_string(),
        );

        Ok(Self {
            vision_encoder,
            text_encoder,
            config,
            loss_fn,
            temperature,
        })
    }

    /// Forward pass for training: (images, texts) -> loss
    pub fn forward_train(
        &self,
        images: &[f32],
        texts: &[u32],
        batch_size: usize,
    ) -> Result<Tensor<B, DenseStorage<T>, T>> {
        // Encode images and texts
        let image_features = self.vision_encoder.forward(images, batch_size)?;
        let text_features = self.text_encoder.forward(texts, batch_size)?;

        // Get current temperature
        let temp_value = self.temperature.data().as_slice()[0];
        let temp_f64 = temp_value.to_f64().unwrap();

        // Compute InfoNCE loss
        self.loss_fn.forward(&image_features, &text_features)
    }

    /// Forward pass for inference: encode image
    pub fn encode_image(&self, images: &[f32], batch_size: usize) -> Result<Tensor<B, DenseStorage<T>, T>> {
        self.vision_encoder.forward(images, batch_size)
    }

    /// Forward pass for inference: encode text
    pub fn encode_text(&self, texts: &[&str]) -> Result<Tensor<B, DenseStorage<T>, T>> {
        // This would need text tokenization - placeholder implementation
        let dummy_tokens = vec![1u32; texts.len() * self.config.text_config.max_position_embeddings];
        self.text_encoder.forward(&dummy_tokens, texts.len())
    }

    /// Get similarity between image and text embeddings
    pub fn get_similarity(
        &self,
        image_features: &Tensor<B, DenseStorage<T>, T>,
        text_features: &Tensor<B, DenseStorage<T>, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>> {
        // Normalize features
        let image_norm = self.normalize_features(image_features)?;
        let text_norm = self.normalize_features(text_features)?;

        // Compute similarity matrix: image_features @ text_features.T
        let text_norm_t = text_norm.transpose(0, 1)?;
        image_norm.matmul(&text_norm_t)
    }

    fn normalize_features(&self, features: &Tensor<B, DenseStorage<T>, T>) -> Result<Tensor<B, DenseStorage<T>, T>> {
        // L2 normalize along the last dimension
        let shape = features.shape().dims();
        let feature_dim = *shape.last().unwrap();

        // Compute L2 norms
        let norm_sq = features.pow(T::from(2.0).unwrap())?;
        let norms = norm_sq.sum_dim(shape.len() - 1, true)?.sqrt()?;

        // Avoid division by zero
        let epsilon = T::from(1e-10).unwrap();
        let safe_norms = norms.maximum(&Tensor::<B, DenseStorage<T>, T>::from_vec(vec![epsilon], &[1])?)?;

        // Normalize
        features.div(&safe_norms)
    }

    /// Get all model parameters
    pub fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        let mut params = Vec::new();
        params.extend(self.vision_encoder.parameters());
        params.extend(self.text_encoder.parameters());
        params.push(self.temperature.clone());
        params
    }

    /// Get vision encoder (read-only)
    pub fn vision_encoder(&self) -> &VisionEncoder<B, S, T> {
        &self.vision_encoder
    }

    /// Get text encoder (read-only)
    pub fn text_encoder(&self) -> &TextEncoder<B, S, T> {
        &self.text_encoder
    }

    /// Get configuration
    pub fn config(&self) -> &ClipConfig {
        &self.config
    }

    /// Get temperature parameter
    pub fn temperature(&self) -> f64 {
        let temp_value = self.temperature.data().as_slice()[0];
        temp_value.to_f64().unwrap()
    }
}

impl<B, S, T> fmt::Display for ClipModel<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ClipModel(vision={}, text={}, embed_dim={})",
            self.config.vision_config.patch_size,
            self.config.text_config.hidden_size,
            self.config.embed_dim
        )
    }
}

// Implement Module trait for CLIP in the future
impl<B, S, T> VisionEncoder<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![self.projection_head.clone()]
    }
}

impl<B, S, T> TextEncoder<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![self.projection_head.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;

    type TestBackend = CpuBackend<Float32>;
    type TestStorage = DenseStorage<Float32>;

    #[test]
    fn test_clip_model_creation() {
        let config = ClipConfig::vit_b32();
        let model = ClipModel::<TestBackend, TestStorage, Float32>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_clip_config_validation() {
        let config = ClipConfig::vit_b32();
        assert_eq!(config.vision_config.patch_size, 32);
        assert_eq!(config.vision_config.num_patches, 49);
        assert_eq!(config.text_config.vocab_size, 49408);
        assert_eq!(config.embed_dim, 512);
    }
}
