//! Model traits and abstractions.
//!
//! This module defines the core model interfaces for building and using
//! language models with zero-copy abstractions.
//!
//! ## Design Principles
//!
//! - **Single Responsibility**: Each trait handles one aspect of model behavior
//! - **Open/Closed**: Models are extensible through traits without modification
//! - **Liskov Substitution**: All model implementations are interchangeable
//! - **Interface Segregation**: Models implement only needed interfaces
//! - **Dependency Inversion**: Depend on abstractions, not concrete types

use crate::foundation::{
    error::Result,
    types::{HeadCount, LayerCount, ModelDim, ModelFloat, Shape},
};
use core::fmt::Debug;
use std::any::Any;

// ============================================================================
// Configuration Traits (Interface Segregation Principle)
// ============================================================================

/// Core model configuration trait.
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

/// Configuration for models with vocabulary.
pub trait VocabularyConfig: ModelConfig {
    /// Returns the vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Returns the padding token ID.
    fn pad_token_id(&self) -> Option<usize>;
}

/// Configuration for sequence models.
pub trait SequenceConfig: ModelConfig {
    /// Returns the maximum sequence length.
    fn max_seq_len(&self) -> usize;

    /// Returns whether to use positional encoding.
    fn use_positional_encoding(&self) -> bool;
}

/// Configuration for models with dropout.
pub trait DropoutConfig: ModelConfig {
    /// Returns the dropout probability.
    fn dropout(&self) -> f32;

    /// Returns the attention dropout probability.
    fn attention_dropout(&self) -> Option<f32> {
        Some(self.dropout())
    }
}

// ============================================================================
// Model Traits (Single Responsibility Principle)
// ============================================================================

/// Base model trait.
pub trait Model: Send + Sync + Debug {
    /// Returns the model configuration.
    type Config: ModelConfig;

    /// Gets the model configuration.
    fn config(&self) -> &Self::Config;

    /// Returns the model name.
    fn name(&self) -> &str;
}

/// Model that can perform forward passes.
pub trait ForwardModel: Model {
    /// Input type for the model.
    type Input;

    /// Output type for the model.
    type Output;

    /// Performs a forward pass.
    fn forward(&self, input: Self::Input) -> Result<Self::Output>;

    /// Returns the number of parameters in the model.
    fn num_parameters(&self) -> usize;
}

/// Model that can be trained.
pub trait TrainableModel: Model {
    /// Optimizer state type.
    type OptimizerState;

    /// Loss type.
    type Loss;

    /// Training step input.
    type TrainingInput;

    /// Training data type.
    type TrainingData;

    /// Performs a training step.
    fn train_step(
        &mut self,
        input: Self::TrainingInput,
        optimizer_state: &mut Self::OptimizerState,
    ) -> Result<Self::Loss>;

    /// Updates model parameters.
    fn update_parameters(&mut self, gradients: &[ModelFloat]) -> Result<()>;

    /// Computes loss for training data.
    fn compute_loss(&self, data: &Self::TrainingData) -> Result<ModelFloat>;

    /// Returns whether the model is in training mode.
    fn is_training(&self) -> bool;

    /// Sets the training mode.
    fn set_training(&mut self, training: bool);
}

/// Model that can generate sequences.
pub trait GenerativeModel: ForwardModel {
    /// Generation configuration type.
    type GenerationConfig;

    /// Generates tokens based on input.
    fn generate(&self, input: Self::Input, config: &Self::GenerationConfig) -> Result<Vec<usize>>;
}

/// Model that supports batched operations.
pub trait BatchedModel: Model {
    /// Batch input type.
    type BatchInput;

    /// Batch output type.
    type BatchOutput;

    /// Processes a batch of inputs.
    fn forward_batch(&self, batch: Self::BatchInput) -> Result<Self::BatchOutput>;
}

// ============================================================================
// Model Builder Trait (Abstract Factory Pattern)
// ============================================================================

/// Trait for building models.
pub trait ModelBuilder: Send + Sync {
    /// The type of model this builder creates.
    type Model: Model;

    /// The configuration type.
    type Config: ModelConfig;

    /// Builds a model from configuration.
    fn build(&self, config: Self::Config) -> Result<Self::Model>;

    /// Validates a configuration without building.
    fn validate_config(&self, config: &Self::Config) -> Result<()> {
        config.validate()
    }
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
            return Err(crate::foundation::error::Error::Processing(
                crate::foundation::error::ProcessingError::ShapeMismatch {
                    expected: format!("numel={}", shape.numel()),
                    actual: format!("len={}", data.len()),
                },
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

// ============================================================================
// Concrete Implementations (following SOLID principles)
// ============================================================================

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
                crate::foundation::error::ConfigError::Invalid {
                    key: "model_dim".to_string(),
                    value: self.model_dim.to_string(),
                    error: "Model dimension must be greater than 0".to_string(),
                },
            ));
        }

        if self.model_dim % self.head_count != 0 {
            return Err(crate::foundation::error::Error::Config(
                crate::foundation::error::ConfigError::Invalid {
                    key: "model_dim".to_string(),
                    value: format!("{} (head_count={})", self.model_dim, self.head_count),
                    error: "Model dimension must be divisible by head count".to_string(),
                },
            ));
        }

        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl VocabularyConfig for BasicModelConfig {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn pad_token_id(&self) -> Option<usize> {
        None
    }
}

impl SequenceConfig for BasicModelConfig {
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn use_positional_encoding(&self) -> bool {
        true
    }
}

impl DropoutConfig for BasicModelConfig {
    fn dropout(&self) -> f32 {
        self.dropout
    }
}

/// Configuration for a ~250M parameter transformer model.
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
    /// Creates a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculates the estimated number of parameters for this configuration.
    pub fn num_parameters(&self) -> usize {
        // Embeddings
        let embedding_params = self.vocab_size * self.hidden_dim;

        // Position embeddings
        let position_params = self.max_seq_len * self.hidden_dim;

        // Per layer parameters
        let per_layer = {
            // Multi-head attention
            let qkv_params = 3 * self.hidden_dim * self.hidden_dim;
            let qkv_bias = if self.use_bias {
                3 * self.hidden_dim
            } else {
                0
            };
            let out_proj_params = self.hidden_dim * self.hidden_dim;
            let out_proj_bias = if self.use_bias { self.hidden_dim } else { 0 };

            // Feedforward
            let ff1_params = self.hidden_dim * self.intermediate_dim;
            let ff1_bias = if self.use_bias {
                self.intermediate_dim
            } else {
                0
            };
            let ff2_params = self.intermediate_dim * self.hidden_dim;
            let ff2_bias = if self.use_bias { self.hidden_dim } else { 0 };

            // Layer norms (2 per layer)
            let ln_params = 2 * self.hidden_dim * 2; // weight and bias

            qkv_params
                + qkv_bias
                + out_proj_params
                + out_proj_bias
                + ff1_params
                + ff1_bias
                + ff2_params
                + ff2_bias
                + ln_params
        };

        let transformer_params = per_layer * self.num_layers;

        // Final layer norm
        let final_ln_params = self.hidden_dim * 2;

        // Output projection (tied with embeddings or separate)
        let output_params = self.vocab_size * self.hidden_dim;

        embedding_params + position_params + transformer_params + final_ln_params + output_params
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<()> {
        use crate::foundation::error::{ConfigError, Error};

        if self.hidden_dim == 0 {
            return Err(Error::Config(ConfigError::Invalid {
                key: "hidden_dim".to_string(),
                value: self.hidden_dim.to_string(),
                error: "Hidden dimension must be greater than 0".to_string(),
            }));
        }

        if self.hidden_dim % self.num_heads != 0 {
            return Err(Error::Config(ConfigError::Invalid {
                key: "hidden_dim".to_string(),
                value: format!("{} (num_heads={})", self.hidden_dim, self.num_heads),
                error: "Hidden dimension must be divisible by number of heads".to_string(),
            }));
        }

        if self.num_layers == 0 {
            return Err(Error::Config(ConfigError::Invalid {
                key: "num_layers".to_string(),
                value: self.num_layers.to_string(),
                error: "Number of layers must be greater than 0".to_string(),
            }));
        }

        if self.vocab_size == 0 {
            return Err(Error::Config(ConfigError::Invalid {
                key: "vocab_size".to_string(),
                value: self.vocab_size.to_string(),
                error: "Vocabulary size must be greater than 0".to_string(),
            }));
        }

        if self.max_seq_len == 0 {
            return Err(Error::Config(ConfigError::Invalid {
                key: "max_seq_len".to_string(),
                value: self.max_seq_len.to_string(),
                error: "Maximum sequence length must be greater than 0".to_string(),
            }));
        }

        Ok(())
    }
}

impl Default for Transformer250MConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_dim: 3072,
            vocab_size: 50257,
            max_seq_len: 1024,
            dropout: 0.1,
            attention_dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_bias: true,
            activation: "gelu".to_string(),
        }
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
        if self.num_layers == 0 {
            return Err(crate::foundation::error::Error::Config(
                crate::foundation::error::ConfigError::Invalid {
                    key: "num_layers".to_string(),
                    value: self.num_layers.to_string(),
                    error: "Number of layers must be greater than 0".to_string(),
                },
            ));
        }

        if self.vocab_size == 0 {
            return Err(crate::foundation::error::Error::Config(
                crate::foundation::error::ConfigError::Invalid {
                    key: "vocab_size".to_string(),
                    value: self.vocab_size.to_string(),
                    error: "Vocabulary size must be greater than 0".to_string(),
                },
            ));
        }

        if self.max_seq_len == 0 {
            return Err(crate::foundation::error::Error::Config(
                crate::foundation::error::ConfigError::Invalid {
                    key: "max_seq_len".to_string(),
                    value: self.max_seq_len.to_string(),
                    error: "Maximum sequence length must be greater than 0".to_string(),
                },
            ));
        }

        if self.hidden_dim % self.num_heads != 0 {
            return Err(crate::foundation::error::Error::Config(
                crate::foundation::error::ConfigError::Invalid {
                    key: "hidden_dim".to_string(),
                    value: format!("{} (num_heads={})", self.hidden_dim, self.num_heads),
                    error: format!(
                        "Hidden dimension {} must be divisible by number of heads {}",
                        self.hidden_dim, self.num_heads
                    ),
                },
            ));
        }

        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl VocabularyConfig for Transformer250MConfig {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn pad_token_id(&self) -> Option<usize> {
        Some(0)
    }
}

impl SequenceConfig for Transformer250MConfig {
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn use_positional_encoding(&self) -> bool {
        true
    }
}

impl DropoutConfig for Transformer250MConfig {
    fn dropout(&self) -> f32 {
        self.dropout
    }

    fn attention_dropout(&self) -> Option<f32> {
        Some(self.attention_dropout)
    }
}

// ============================================================================
// Legacy Traits (for backward compatibility)
// ============================================================================

/// Legacy inference model trait.
pub trait InferenceModel: ForwardModel {
    /// Performs inference on a single input.
    fn infer(&self, input: Self::Input) -> Result<Self::Output> {
        self.forward(input)
    }
}

/// Legacy diffusion model trait.
pub trait DiffusionModel: ForwardModel {
    /// The type of noise schedule.
    type NoiseSchedule: NoiseSchedule;

    /// The type of sampler.
    type Sampler: DiffusionSampler;

    /// Adds noise to the input.
    fn add_noise(
        &self,
        input: &Self::Input,
        timestep: usize,
        noise_schedule: &Self::NoiseSchedule,
    ) -> Result<Self::Input>;

    /// Predicts noise from noisy input.
    fn predict_noise(&self, noisy_input: &Self::Input, timestep: usize) -> Result<Self::Output>;

    /// Performs a single denoising step.
    fn denoise_step(
        &self,
        noisy_input: &Self::Input,
        timestep: usize,
        noise_schedule: &Self::NoiseSchedule,
        sampler: &Self::Sampler,
    ) -> Result<Self::Input>;

    /// Generates samples from noise.
    fn generate_samples(
        &self,
        num_samples: usize,
        noise_schedule: &Self::NoiseSchedule,
        sampler: &Self::Sampler,
    ) -> Result<Vec<Self::Output>>;
}

/// Trait for optimizer state.
pub trait OptimizerState: Send + Sync {
    /// Resets the optimizer state.
    fn reset(&mut self);

    /// Returns the current learning rate.
    fn learning_rate(&self) -> ModelFloat;

    /// Sets the learning rate.
    fn set_learning_rate(&mut self, lr: ModelFloat);

    /// Returns the current step count.
    fn step_count(&self) -> usize;

    /// Increments the step count.
    fn increment_step(&mut self);
}

/// Trait for loss functions.
pub trait Loss: Send + Sync {
    /// Computes the loss given predictions and targets.
    fn compute(&self, predictions: &[ModelFloat], targets: &[ModelFloat]) -> Result<ModelFloat>;

    /// Computes the gradient of the loss.
    fn gradient(
        &self,
        predictions: &[ModelFloat],
        targets: &[ModelFloat],
    ) -> Result<Vec<ModelFloat>>;
}

/// Trait for noise schedules.
pub trait NoiseSchedule: Send + Sync {
    /// Returns the number of diffusion steps.
    fn num_steps(&self) -> usize;

    /// Returns the noise level at a timestep.
    fn beta(&self, timestep: usize) -> ModelFloat;

    /// Returns the cumulative product of (1 - beta) up to timestep.
    fn alpha_bar(&self, timestep: usize) -> ModelFloat;
}

/// Trait for diffusion samplers.
pub trait DiffusionSampler: Send + Sync {
    /// Performs a sampling step.
    fn sample_step(
        &self,
        current: &[ModelFloat],
        predicted_noise: &[ModelFloat],
        timestep: usize,
        noise_schedule: &dyn NoiseSchedule,
    ) -> Result<Vec<ModelFloat>>;

    /// Returns whether the sampler is deterministic.
    fn is_deterministic(&self) -> bool;
}

/// Trait for generation configuration.
pub trait GenerationConfig: Debug + Clone + Send + Sync {
    /// Returns the maximum number of tokens to generate.
    fn max_length(&self) -> usize;

    /// Returns the temperature for sampling.
    fn temperature(&self) -> f32;
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
}

impl Default for BasicGenerationConfig {
    fn default() -> Self {
        Self {
            max_length: 100,
            temperature: 1.0,
            top_k: None,
            top_p: None,
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
        assert_eq!(config.top_k, None);
        assert_eq!(config.top_p, None);
    }
}
