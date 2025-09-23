//! # Coeus Models

#![allow(clippy::bool_assert_comparison)]
//!
//! Llama.cpp-like model loading, inference, and quantization functionality for the Coeus tensor library.
//!
//! This crate provides:
//! - **GGUF Format Support**: Load and parse GGUF model files (Llama.cpp format)
//! - **Efficient Inference**: Memory-optimized transformer inference with batching
//! - **Quantization**: Multiple quantization schemes (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)
//! - **Model Registry**: Integration with Coeus Hub for model downloading and caching
//! - **Tokenizer Integration**: Seamless integration with Coeus tokenizer ecosystem
//!
//! ## Architecture
//!
//! The models crate is organized into several key modules:
//!
//! - **`format`**: GGUF file format parsing and validation
//! - **`quantization`**: Quantization schemes and dequantization
//! - **`inference`**: Efficient transformer model inference
//! - **`registry`**: Model metadata and registry management
//! - **`loader`**: Model loading and caching infrastructure
//! - **`config`**: Model configuration and hyperparameters
//!
//! ## Example Usage
//!
//! ### Loading GGUF Models (llama.cpp Compatible)
//!
//! ```rust,no_run
//! use coeus_models::{ModelLoader, InferenceConfig, ModelConfig, ModelType};
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Method 1: Load from Hub (PyTorch Hub compatible)
//! let loader = ModelLoader::new();
//! let mut model = loader.load_from_hub("meta-llama", "llama-2-7b-q4_0").await?;
//!
//! // Method 2: Load from local file
//! let mut model = loader.load_from_file("path/to/model.gguf").await?;
//!
//! // Create inference configuration (PyTorch-like API)
//! let config = InferenceConfig::new()
//!     .with_max_new_tokens(50)
//!     .with_temperature(0.8)
//!     .with_top_p(0.9)
//!     .with_repetition_penalty(1.1);
//!
//! // Generate text
//! let prompt = "The future of AI is";
//! let generated = Arc::get_mut(&mut model).unwrap().generate(prompt, &config)?;
//!
//! println!("Generated: {}", generated);
//! # Ok(())
//! # }
//! ```
//!
//! ### PyTorch Hub Integration
//!
//! ```rust,no_run
//! use coeus_hub::Hub;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // PyTorch Hub compatible loading
//! let hub = Hub::new();
//!
//! // Load a model (equivalent to torch.hub.load)
//! let state_dict = hub.load("pytorch/vision", "resnet18", false, None, false).await?;
//!
//! // Load from specific branch
//! let state_dict = hub.load("pytorch/vision", "resnet18", false, Some("v0.12.0"), true).await?;
//!
//! // Load with force reload
//! let state_dict = hub.load("pytorch/vision", "resnet18", true, None, false).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Advanced Configuration
//!
//! ```rust,no_run
//! use coeus_models::{ModelLoader, ModelLoadConfig, QuantizationScheme};
//!
//! # fn example() {
//! // Create custom loading configuration
//! let config = ModelLoadConfig::new()
//!     .with_gpu(true)
//!     .with_memory_map(true)
//!     .with_preferred_quantization(QuantizationScheme::Q4_0);
//!
//! // Use configuration for loading
//! let loader = ModelLoader::with_config(config);
//! # }
//! ```
//!
//! ## GGUF Format Support
//!
//! The crate provides comprehensive support for the GGUF (Georgi Gerganov Universal Format)
//! used by llama.cpp:
//!
//! - **File Parsing**: Complete GGUF file format parsing with validation
//! - **Metadata Extraction**: Model architecture, vocabulary, and hyperparameters
//! - **Tensor Loading**: Efficient tensor loading with memory mapping
//! - **Quantization Support**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 quantization schemes
//! - **Multi-Architecture**: Support for Llama, GPT-2, and other transformer architectures
//!
//! ## Performance Features
//!
//! - **Memory Mapping**: Zero-copy tensor loading using memory-mapped files
//! - **Batch Processing**: Efficient batch inference for multiple sequences
//! - **Quantized Inference**: Reduced memory usage with maintained accuracy
//! - **Parallel Processing**: Rayon-based parallel computation for multi-core systems
//! - **GPU Acceleration**: Optional GPU support via wgpu backend
//! - **Streaming**: Token-by-token generation with early stopping
//!
//! ## Safety Guarantees
//!
//! - **Memory Safety**: Zero unsafe code blocks, guaranteed by Rust ownership system
//! - **Thread Safety**: Safe concurrent model inference with proper synchronization
//! - **Validation**: Comprehensive validation of model files and parameters
//! - **Error Handling**: Detailed error propagation with descriptive messages
//!
//! ## Integration
//!
//! The models crate integrates seamlessly with the broader Coeus ecosystem:
//!
//! - **Tensor Operations**: Uses Coeus tensor operations for efficient computation
//! - **Neural Networks**: Leverages Coeus NN modules for transformer components
//! - **Model Hub**: Integrates with Coeus Hub for model downloading and caching
//! - **Tokenization**: Works with Coeus tokenizer for text preprocessing
//! - **Python Bindings**: Full Python API compatibility through PyCoeus
//!
//! ## PyTorch API Compatibility
//!
//! This crate provides PyTorch-like APIs for seamless migration:
//!
//! - **`ModelLoader`**: Equivalent to loading models from files or URLs
//! - **`InferenceConfig`**: Similar to PyTorch's generation parameters
//! - **`ModelConfig`**: Comparable to model configuration dictionaries
//! - **Quantization Schemes**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 (llama.cpp compatible)
//! - **Memory Mapping**: Zero-copy tensor loading for large models
//! - **GPU Support**: Optional GPU acceleration via wgpu backend
//!
//! ### Migration from PyTorch
//!
//! ```rust,no_run
//! // PyTorch equivalent:
//! // model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b", torch_dtype=torch.float16)
//! // outputs = model.generate(inputs, max_new_tokens=50, temperature=0.8)
//!
//! // Coeus equivalent:
//! use coeus_models::{ModelLoader, InferenceConfig};
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut model = ModelLoader::new()
//!     .load_from_hub("meta-llama", "llama-2-7b-q4_0").await?;
//!
//! let config = InferenceConfig::new()
//!     .with_max_new_tokens(50)
//!     .with_temperature(0.8);
//!
//! let outputs = Arc::get_mut(&mut model).unwrap()
//!     .generate("Hello world", &config)?;
//! # Ok(())
//! # }
//! ```

pub mod config;
pub mod error;
pub mod format;
pub mod inference;
pub mod loader;
pub mod quantization;

pub use error::ModelError;
pub use config::ModelLoadConfig;
pub use format::{GgufFormat, ModelArchitecture, ModelMetadata, TensorInfo};
pub use inference::{InferenceConfig, InferenceEngine, InferenceResult, LlamaModel};
pub use loader::{ModelLoader, ModelSource};
pub use quantization::{QuantizationScheme, QuantizationType, QuantizedTensor};

/// Result type for model operations
pub type Result<T> = std::result::Result<T, ModelError>;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Supported model architectures
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelType {
    /// Llama model family
    Llama,
    /// GPT-2 model family
    Gpt2,
    /// Generic transformer model
    Transformer,
    /// Other/unknown model type
    Other(String),
}

impl ModelType {
    /// Detect model type from architecture string
    pub fn from_architecture(arch: &str) -> Self {
        match arch.to_lowercase().as_str() {
            "llama" => Self::Llama,
            "gpt2" | "gpt-2" => Self::Gpt2,
            "transformer" => Self::Transformer,
            _ => Self::Other(arch.to_string()),
        }
    }

    /// Get the architecture string
    pub fn architecture(&self) -> &str {
        match self {
            Self::Llama => "llama",
            Self::Gpt2 => "gpt2",
            Self::Transformer => "transformer",
            Self::Other(s) => s,
        }
    }
}

/// Model loading configuration
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model architecture type
    pub model_type: ModelType,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Model dimensionality
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Quantization scheme (None for full precision)
    pub quantization: Option<QuantizationScheme>,
    /// Use GPU acceleration if available
    pub use_gpu: bool,
    /// Use memory mapping for tensor loading
    pub use_memory_map: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::Llama,
            max_seq_len: 2048,
            hidden_size: 4096,
            num_heads: 32,
            num_layers: 32,
            vocab_size: 32000,
            quantization: Some(QuantizationScheme::Q4_0),
            use_gpu: false,
            use_memory_map: true,
        }
    }
}

impl ModelConfig {
    /// Create a new model configuration
    pub fn new(model_type: ModelType) -> Self {
        Self {
            model_type,
            ..Default::default()
        }
    }

    /// Set maximum sequence length
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    /// Set hidden size
    pub fn with_hidden_size(mut self, hidden_size: usize) -> Self {
        self.hidden_size = hidden_size;
        self
    }

    /// Set number of attention heads
    pub fn with_num_heads(mut self, num_heads: usize) -> Self {
        self.num_heads = num_heads;
        self
    }

    /// Set number of layers
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Set vocabulary size
    pub fn with_vocab_size(mut self, vocab_size: usize) -> Self {
        self.vocab_size = vocab_size;
        self
    }

    /// Set quantization scheme
    pub fn with_quantization(mut self, quantization: QuantizationScheme) -> Self {
        self.quantization = Some(quantization);
        self
    }

    /// Enable GPU acceleration
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Enable/disable memory mapping
    pub fn with_memory_map(mut self, use_memory_map: bool) -> Self {
        self.use_memory_map = use_memory_map;
        self
    }
}

/// Initialize the models system
pub fn init() -> Result<()> {
    env_logger::init();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_detection() {
        assert_eq!(ModelType::from_architecture("llama"), ModelType::Llama);
        assert_eq!(ModelType::from_architecture("gpt2"), ModelType::Gpt2);
        assert_eq!(ModelType::from_architecture("gpt-2"), ModelType::Gpt2);
        assert_eq!(
            ModelType::from_architecture("transformer"),
            ModelType::Transformer
        );
        assert_eq!(
            ModelType::from_architecture("unknown"),
            ModelType::Other("unknown".to_string())
        );
    }

    #[test]
    fn test_model_config_builder() {
        let config = ModelConfig::new(ModelType::Llama)
            .with_max_seq_len(1024)
            .with_hidden_size(2048)
            .with_num_heads(16)
            .with_num_layers(24)
            .with_vocab_size(50000)
            .with_quantization(QuantizationScheme::Q8_0)
            .with_gpu(true)
            .with_memory_map(false);

        assert_eq!(config.model_type, ModelType::Llama);
        assert_eq!(config.max_seq_len, 1024);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.vocab_size, 50000);
        assert_eq!(config.quantization, Some(QuantizationScheme::Q8_0));
        assert_eq!(config.use_gpu, true);
        assert_eq!(config.use_memory_map, false);
    }

    #[test]
    fn test_model_config_defaults() {
        let config = ModelConfig::default();
        assert_eq!(config.model_type, ModelType::Llama);
        assert_eq!(config.max_seq_len, 2048);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.quantization, Some(QuantizationScheme::Q4_0));
        assert_eq!(config.use_gpu, false);
        assert_eq!(config.use_memory_map, true);
    }
}
