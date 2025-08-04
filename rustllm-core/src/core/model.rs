//! Model traits and abstractions.
//!
//! This module defines the core model interfaces for building and using
//! language models with zero-copy abstractions.

use crate::foundation::{
    error::Result,
    types::{ModelDim, HeadCount, LayerCount, Shape, ModelFloat},
};
use core::fmt::Debug;
use std::any::Any;

/// Trait representing a model configuration.
pub trait ModelConfig: Debug + Clone + Send + Sync {
    /// Returns the model dimension (hidden size).
    fn model_dim(&self) -> ModelDim;
    
    /// Returns the number of attention heads.
    fn head_count(&self) -> HeadCount;
    
    /// Returns the number of layers.
    fn layer_count(&self) -> LayerCount;
    
    /// Validates the configuration.
    fn validate(&self) -> Result<()>;

    /// Returns a reference to the underlying configuration.
    fn as_any(&self) -> &dyn Any;
}

/// Basic model configuration implementation.
#[derive(Debug, Clone)]
pub struct BasicModelConfig {
    /// Model dimension (hidden size).
    pub model_dim: ModelDim,
    
    /// Number of attention heads.
    pub head_count: HeadCount,
    
    /// Number of layers.
    pub layer_count: LayerCount,
    
    /// Maximum sequence length.
    pub max_seq_len: usize,
    
    /// Vocabulary size.
    pub vocab_size: usize,
    
    /// Dropout probability.
    pub dropout: f32,
    
    /// Layer normalization epsilon.
    pub layer_norm_eps: f32,
}

impl Default for BasicModelConfig {
    fn default() -> Self {
        Self {
            model_dim: 512,
            head_count: 8,
            layer_count: 6,
            max_seq_len: 512,
            vocab_size: 32000,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        }
    }
}

impl ModelConfig for BasicModelConfig {
    fn model_dim(&self) -> ModelDim {
        self.model_dim
    }
    
    fn head_count(&self) -> HeadCount {
        self.head_count
    }
    
    fn layer_count(&self) -> LayerCount {
        self.layer_count
    }
    
    fn validate(&self) -> Result<()> {
        if self.model_dim == 0 {
            return Err(crate::foundation::error::Error::Config(
                "Model dimension must be greater than 0".to_string()
            ));
        }
        
        if self.model_dim % self.head_count != 0 {
            return Err(crate::foundation::error::Error::Config(
                "Model dimension must be divisible by head count".to_string()
            ));
        }
        
        if self.layer_count == 0 {
            return Err(crate::foundation::error::Error::Config(
                "Layer count must be greater than 0".to_string()
            ));
        }
        
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Configuration for a ~250M parameter transformer model.
/// 
/// This configuration is optimized for rapid prototyping and experimentation:
/// - 768 hidden dimensions (same as BERT-base)
/// - 12 layers
/// - 12 attention heads
/// - ~250M total parameters
#[derive(Debug, Clone, PartialEq)]
pub struct Transformer250MConfig {
    /// Hidden dimension size (768 for 250M model).
    pub hidden_dim: usize,
    
    /// Number of transformer layers (12 for 250M model).
    pub num_layers: usize,
    
    /// Number of attention heads (12 for 250M model).
    pub num_heads: usize,
    
    /// Intermediate/feedforward dimension (typically 4x hidden_dim).
    pub intermediate_dim: usize,
    
    /// Vocabulary size.
    pub vocab_size: usize,
    
    /// Maximum sequence length.
    pub max_seq_len: usize,
    
    /// Dropout probability.
    pub dropout: f32,
    
    /// Attention dropout probability.
    pub attention_dropout: f32,
    
    /// Layer normalization epsilon.
    pub layer_norm_eps: f32,
    
    /// Whether to use bias in linear layers.
    pub use_bias: bool,
    
    /// Activation function type.
    pub activation: String,
}

impl Transformer250MConfig {
    /// Creates a new 250M parameter configuration.
    pub fn new() -> Self {
        Self {
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_dim: 3072, // 4 * 768
            vocab_size: 50257, // GPT-2 vocabulary size
            max_seq_len: 1024,
            dropout: 0.1,
            attention_dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_bias: true,
            activation: "gelu".to_string(),
        }
    }
    
    /// Calculates the approximate number of parameters.
    pub fn num_parameters(&self) -> usize {
        let mut params = 0;
        
        // Embedding parameters
        params += self.vocab_size * self.hidden_dim; // Token embeddings
        params += self.max_seq_len * self.hidden_dim; // Position embeddings
        
        // Transformer layers
        for _ in 0..self.num_layers {
            // Self-attention
            params += 4 * self.hidden_dim * self.hidden_dim; // Q, K, V, O projections
            if self.use_bias {
                params += 4 * self.hidden_dim; // Biases
            }
            
            // Layer norm 1
            params += 2 * self.hidden_dim; // Scale and bias
            
            // Feedforward
            params += self.hidden_dim * self.intermediate_dim; // Up projection
            params += self.intermediate_dim * self.hidden_dim; // Down projection
            if self.use_bias {
                params += self.intermediate_dim + self.hidden_dim; // Biases
            }
            
            // Layer norm 2
            params += 2 * self.hidden_dim; // Scale and bias
        }
        
        // Final layer norm
        params += 2 * self.hidden_dim;
        
        // Output projection
        params += self.hidden_dim * self.vocab_size;
        if self.use_bias {
            params += self.vocab_size;
        }
        
        params
    }
    
    /// Validates the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_dim % self.num_heads != 0 {
            return Err(crate::foundation::error::Error::InvalidInput(format!(
                "Hidden dimension {} must be divisible by number of heads {}",
                self.hidden_dim, self.num_heads
            )));
        }
        
        if self.num_layers == 0 {
            return Err(crate::foundation::error::Error::InvalidInput("Number of layers must be > 0".to_string()));
        }
        
        if self.vocab_size == 0 {
            return Err(crate::foundation::error::Error::InvalidInput("Vocabulary size must be > 0".to_string()));
        }
        
        if self.max_seq_len == 0 {
            return Err(crate::foundation::error::Error::InvalidInput("Maximum sequence length must be > 0".to_string()));
        }
        
        Ok(())
    }
}

impl Default for Transformer250MConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelConfig for Transformer250MConfig {
    fn model_dim(&self) -> ModelDim {
        self.hidden_dim as ModelDim
    }
    
    fn head_count(&self) -> HeadCount {
        self.num_heads as HeadCount
    }
    
    fn layer_count(&self) -> LayerCount {
        self.num_layers as LayerCount
    }
    
    fn validate(&self) -> Result<()> {
        self.validate()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Trait representing a language model.
pub trait Model: Send + Sync {
    /// The input type for the model.
    type Input;
    
    /// The output type for the model.
    type Output;
    
    /// The configuration type for the model.
    type Config: ModelConfig;
    
    /// Performs a forward pass through the model.
    fn forward(&self, input: Self::Input) -> Result<Self::Output>;
    
    /// Returns the model configuration.
    fn config(&self) -> &Self::Config;
    
    /// Returns the number of parameters in the model.
    fn num_parameters(&self) -> usize;
}

/// Trait for models that support inference.
pub trait InferenceModel: Model {
    /// Performs inference on a single input.
    fn infer(&self, input: Self::Input) -> Result<Self::Output> {
        self.forward(input)
    }
    
    /// Performs batched inference.
    fn infer_batch(&self, inputs: Vec<Self::Input>) -> Result<Vec<Self::Output>> {
        inputs.into_iter()
            .map(|input| self.infer(input))
            .collect()
    }
}

/// Trait for models that support generation.
pub trait GenerativeModel: Model {
    /// Generation configuration type.
    type GenerationConfig: GenerationConfig;
    
    /// Generates output based on input and configuration.
    fn generate(
        &self,
        input: Self::Input,
        config: Self::GenerationConfig,
    ) -> Result<Self::Output>;
}

/// Configuration for text generation.
pub trait GenerationConfig: Debug + Clone + Send + Sync {
    /// Returns the maximum number of tokens to generate.
    fn max_length(&self) -> usize;
    
    /// Returns the temperature for sampling.
    fn temperature(&self) -> f32;
    
    /// Returns the top-k value for sampling.
    fn top_k(&self) -> Option<usize>;
    
    /// Returns the top-p value for sampling.
    fn top_p(&self) -> Option<f32>;
}

/// Basic generation configuration.
#[derive(Debug, Clone)]
pub struct BasicGenerationConfig {
    /// Maximum length of generated sequence.
    pub max_length: usize,
    
    /// Temperature for sampling.
    pub temperature: f32,
    
    /// Top-k sampling parameter.
    pub top_k: Option<usize>,
    
    /// Top-p (nucleus) sampling parameter.
    pub top_p: Option<f32>,
    
    /// Repetition penalty.
    pub repetition_penalty: f32,
    
    /// Whether to use sampling or greedy decoding.
    pub do_sample: bool,
}

impl Default for BasicGenerationConfig {
    fn default() -> Self {
        Self {
            max_length: 100,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            do_sample: true,
        }
    }
}

impl GenerationConfig for BasicGenerationConfig {
    fn max_length(&self) -> usize {
        self.max_length
    }
    
    fn temperature(&self) -> f32 {
        self.temperature
    }
    
    fn top_k(&self) -> Option<usize> {
        self.top_k
    }
    
    fn top_p(&self) -> Option<f32> {
        self.top_p
    }
}

/// Trait for model builders.
pub trait ModelBuilder: Send + Sync {
    /// The model type to build.
    type Model: Model;
    
    /// The configuration type.
    type Config: ModelConfig;
    
    /// The error type.
    #[cfg(feature = "std")]
    type Error: std::error::Error + Send + Sync + 'static;
    
    #[cfg(not(feature = "std"))]
    type Error: core::fmt::Debug + core::fmt::Display + Send + Sync + 'static;
    
    /// Builds a model with the given configuration.
    fn build(&self, config: Self::Config) -> Result<Self::Model>;
}

/// Trait for models that can be saved and loaded.
pub trait PersistentModel: Model {
    /// Saves the model to bytes.
    fn save(&self) -> Result<Vec<u8>>;
    
    /// Loads the model from bytes.
    fn load(data: &[u8]) -> Result<Self>
    where
        Self: Sized;
}

/// Trait for models that support quantization.
pub trait QuantizableModel: Model {
    /// Quantization configuration type.
    type QuantConfig: Debug + Clone;
    
    /// Quantizes the model with the given configuration.
    fn quantize(&mut self, config: Self::QuantConfig) -> Result<()>;
    
    /// Checks if the model is quantized.
    fn is_quantized(&self) -> bool;
}

/// Represents a tensor in the model.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// The shape of the tensor.
    pub shape: Shape,
    
    /// The data of the tensor.
    pub data: Vec<ModelFloat>,
}

impl Tensor {
    /// Creates a new tensor with the given shape and data.
    pub fn new(shape: Shape, data: Vec<ModelFloat>) -> Result<Self> {
        if shape.numel() != data.len() {
            return Err(crate::foundation::error::Error::Model(
                crate::foundation::error::ModelError::ShapeMismatch {
                    expected: format!("numel={}", shape.numel()),
                    actual: format!("len={}", data.len()),
                }
            ));
        }
        
        Ok(Self { shape, data })
    }
    
    /// Creates a tensor filled with zeros.
    pub fn zeros(shape: Shape) -> Self {
        let data = vec![0.0; shape.numel()];
        Self { shape, data }
    }
    
    /// Creates a tensor filled with ones.
    pub fn ones(shape: Shape) -> Self {
        let data = vec![1.0; shape.numel()];
        Self { shape, data }
    }
    
    /// Returns a view of the tensor data.
    pub fn as_slice(&self) -> &[ModelFloat] {
        &self.data
    }
    
    /// Returns a mutable view of the tensor data.
    pub fn as_mut_slice(&mut self) -> &mut [ModelFloat] {
        &mut self.data
    }
}

/// Trait for model layers.
pub trait Layer: Send + Sync {
    /// The input type.
    type Input;
    
    /// The output type.
    type Output;
    
    /// Performs a forward pass through the layer.
    fn forward(&self, input: Self::Input) -> Result<Self::Output>;
    
    /// Returns the number of parameters in the layer.
    fn num_parameters(&self) -> usize;
}

/// Trait for attention mechanisms.
pub trait Attention: Layer {
    /// Applies attention with the given query, key, and value.
    fn attend(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor>;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_model_config() {
        let config = BasicModelConfig::default();
        assert_eq!(config.model_dim(), 512);
        assert_eq!(config.head_count(), 8);
        assert_eq!(config.layer_count(), 6);
        assert!(config.validate().is_ok());
        
        let mut invalid_config = config.clone();
        invalid_config.model_dim = 0;
        assert!(invalid_config.validate().is_err());
        
        let mut invalid_config2 = config.clone();
        invalid_config2.model_dim = 511; // Not divisible by head_count
        assert!(invalid_config2.validate().is_err());
    }
    
    #[test]
    fn test_transformer_250m_config() {
        let config = Transformer250MConfig::default();
        assert_eq!(config.hidden_dim, 768);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.intermediate_dim, 3072);
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.max_seq_len, 1024);
        assert_eq!(config.dropout, 0.1);
        assert_eq!(config.attention_dropout, 0.1);
        assert_eq!(config.layer_norm_eps, 1e-5);
        assert!(config.use_bias);
        assert_eq!(config.activation, "gelu");
        assert!(config.validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.hidden_dim = 767; // Not divisible by num_heads
        assert!(invalid_config.validate().is_err());

        let mut invalid_config2 = config.clone();
        invalid_config2.num_layers = 0;
        assert!(invalid_config2.validate().is_err());

        let mut invalid_config3 = config.clone();
        invalid_config3.vocab_size = 0;
        assert!(invalid_config3.validate().is_err());

        let mut invalid_config4 = config.clone();
        invalid_config4.max_seq_len = 0;
        assert!(invalid_config4.validate().is_err());
    }
    
    #[test]
    fn test_tensor_creation() {
        let shape = Shape::new(vec![2, 3]);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(shape.clone(), data).unwrap();
        
        assert_eq!(tensor.shape.dims(), &[2, 3]);
        assert_eq!(tensor.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        let zeros = Tensor::zeros(shape.clone());
        assert_eq!(zeros.as_slice(), &[0.0; 6]);
        
        let ones = Tensor::ones(shape);
        assert_eq!(ones.as_slice(), &[1.0; 6]);
    }
    
    #[test]
    fn test_generation_config() {
        let config = BasicGenerationConfig::default();
        assert_eq!(config.max_length(), 100);
        assert_eq!(config.temperature(), 1.0);
        assert_eq!(config.top_k(), None);
        assert_eq!(config.top_p(), None);
    }
}