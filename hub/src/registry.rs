//! Model registry for managing available models
//!
//! This module provides an extensible model registry that supports both traditional
//! PyTorch models and modern GGUF (llama.cpp) models with quantization support.

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model type enumeration
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ModelType {
    /// Traditional PyTorch model
    PyTorch,
    /// GGUF format model (llama.cpp)
    Gguf,
    /// ONNX model
    Onnx,
    /// SafeTensors format
    SafeTensors,
    /// Other/unknown format
    Other(String),
}

#[allow(clippy::derivable_impls)]
impl Default for ModelType {
    fn default() -> Self {
        Self::PyTorch
    }
}

/// Quantization scheme for GGUF models
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum QuantizationScheme {
    /// No quantization (full precision)
    None,
    /// 4-bit quantization with zero point
    Q4_0,
    /// 4-bit quantization with improved zero handling
    Q4_1,
    /// 5-bit quantization with zero point
    Q5_0,
    /// 5-bit quantization with improved zero handling
    Q5_1,
    /// 8-bit quantization with zero point
    Q8_0,
    /// 8-bit quantization with improved accuracy
    Q8_1,
}

impl From<QuantizationScheme> for String {
    fn from(scheme: QuantizationScheme) -> Self {
        match scheme {
            QuantizationScheme::None => "none".to_string(),
            QuantizationScheme::Q4_0 => "q4_0".to_string(),
            QuantizationScheme::Q4_1 => "q4_1".to_string(),
            QuantizationScheme::Q5_0 => "q5_0".to_string(),
            QuantizationScheme::Q5_1 => "q5_1".to_string(),
            QuantizationScheme::Q8_0 => "q8_0".to_string(),
            QuantizationScheme::Q8_1 => "q8_1".to_string(),
        }
    }
}

/// Information about a model in the registry
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    /// Repository name
    pub repo: String,
    /// Model description
    pub description: String,
    /// Model URL
    pub url: String,
    /// Hash for verification (optional)
    pub hash: Option<String>,
    /// Model size in bytes (optional)
    pub size: Option<u64>,
    /// Default configuration
    pub config: HashMap<String, serde_json::Value>,
    /// Model type (PyTorch, GGUF, etc.)
    #[serde(default)]
    pub model_type: ModelType,
    /// Quantization scheme (for GGUF models)
    #[serde(default)]
    pub quantization: Option<QuantizationScheme>,
    /// Model architecture (llama, gpt2, etc.)
    pub architecture: Option<String>,
    /// Context length
    pub context_length: Option<usize>,
    /// Vocabulary size
    pub vocab_size: Option<usize>,
    /// Number of parameters (in millions)
    pub parameters_millions: Option<f64>,
    /// Whether the model requires authentication
    pub requires_auth: bool,
    /// Model license
    pub license: Option<String>,
    /// Author/organization
    pub author: Option<String>,
    /// Creation date
    pub created_at: Option<String>,
    /// Last updated
    pub updated_at: Option<String>,
}

impl ModelInfo {
    /// Create a new model info
    pub fn new(name: String, repo: String, description: String, url: String) -> Self {
        Self {
            name,
            repo,
            description,
            url,
            hash: None,
            size: None,
            config: HashMap::new(),
            model_type: ModelType::PyTorch,
            quantization: None,
            architecture: None,
            context_length: None,
            vocab_size: None,
            parameters_millions: None,
            requires_auth: false,
            license: None,
            author: None,
            created_at: None,
            updated_at: None,
        }
    }

    /// Create a GGUF model info
    pub fn new_gguf(
        name: String,
        repo: String,
        description: String,
        url: String,
        architecture: String,
        quantization: QuantizationScheme,
    ) -> Self {
        Self {
            name,
            repo,
            description,
            url,
            model_type: ModelType::Gguf,
            quantization: Some(quantization),
            architecture: Some(architecture),
            ..Default::default()
        }
    }

    /// Set the hash for verification
    pub fn with_hash(mut self, hash: String) -> Self {
        self.hash = Some(hash);
        self
    }

    /// Set the size
    pub fn with_size(mut self, size: u64) -> Self {
        self.size = Some(size);
        self
    }

    /// Add a configuration value
    pub fn with_config(mut self, key: String, value: serde_json::Value) -> Self {
        self.config.insert(key, value);
        self
    }

    /// Set model type
    pub fn with_model_type(mut self, model_type: ModelType) -> Self {
        self.model_type = model_type;
        self
    }

    /// Set quantization scheme
    pub fn with_quantization(mut self, quantization: QuantizationScheme) -> Self {
        self.quantization = Some(quantization);
        self
    }

    /// Set architecture
    pub fn with_architecture(mut self, architecture: String) -> Self {
        self.architecture = Some(architecture);
        self
    }

    /// Set context length
    pub fn with_context_length(mut self, context_length: usize) -> Self {
        self.context_length = Some(context_length);
        self
    }

    /// Set vocabulary size
    pub fn with_vocab_size(mut self, vocab_size: usize) -> Self {
        self.vocab_size = Some(vocab_size);
        self
    }

    /// Set parameters in millions
    pub fn with_parameters_millions(mut self, parameters: f64) -> Self {
        self.parameters_millions = Some(parameters);
        self
    }

    /// Set authentication requirement
    pub fn requires_auth(mut self, requires_auth: bool) -> Self {
        self.requires_auth = requires_auth;
        self
    }

    /// Set license
    pub fn with_license(mut self, license: String) -> Self {
        self.license = Some(license);
        self
    }

    /// Set author
    pub fn with_author(mut self, author: String) -> Self {
        self.author = Some(author);
        self
    }

    /// Set creation date
    pub fn created_at(mut self, created_at: String) -> Self {
        self.created_at = Some(created_at);
        self
    }

    /// Set update date
    pub fn updated_at(mut self, updated_at: String) -> Self {
        self.updated_at = Some(updated_at);
        self
    }

    /// Check if this is a GGUF model
    pub fn is_gguf(&self) -> bool {
        matches!(self.model_type, ModelType::Gguf)
    }

    /// Check if this is a quantized model
    pub fn is_quantized(&self) -> bool {
        matches!(
            self.quantization,
            Some(QuantizationScheme::Q4_0)
                | Some(QuantizationScheme::Q4_1)
                | Some(QuantizationScheme::Q5_0)
                | Some(QuantizationScheme::Q5_1)
                | Some(QuantizationScheme::Q8_0)
                | Some(QuantizationScheme::Q8_1)
        )
    }

    /// Get memory usage estimate in bytes
    pub fn estimated_memory_usage(&self) -> Option<u64> {
        match self.quantization {
            Some(QuantizationScheme::Q4_0) | Some(QuantizationScheme::Q4_1) => {
                self.size.map(|s| s * 4) // ~4x reduction
            }
            Some(QuantizationScheme::Q5_0) | Some(QuantizationScheme::Q5_1) => {
                self.size.map(|s| (s * 5).div_ceil(8)) // ~5/8 of original
            }
            Some(QuantizationScheme::Q8_0) | Some(QuantizationScheme::Q8_1) => {
                self.size.map(|s| s / 2) // ~2x reduction
            }
            _ => self.size, // No reduction for full precision
        }
    }
}

/// Model registry containing available models
#[derive(Clone, Debug)]
pub struct ModelRegistry {
    models: HashMap<String, HashMap<String, ModelInfo>>,
}

impl ModelRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Get all GGUF models
    pub fn gguf_models(&self) -> Vec<&ModelInfo> {
        self.all_models()
            .filter_map(|(_, _, info)| if info.is_gguf() { Some(info) } else { None })
            .collect()
    }

    /// Get all quantized models
    pub fn quantized_models(&self) -> Vec<&ModelInfo> {
        self.all_models()
            .filter_map(|(_, _, info)| {
                if info.is_quantized() {
                    Some(info)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get models by architecture
    pub fn models_by_architecture(&self, architecture: &str) -> Vec<&ModelInfo> {
        self.all_models()
            .filter_map(|(_, _, info)| {
                if let Some(ref arch) = info.architecture {
                    if arch == architecture {
                        Some(info)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get models by quantization scheme
    pub fn models_by_quantization(&self, quantization: QuantizationScheme) -> Vec<&ModelInfo> {
        self.all_models()
            .filter_map(|(_, _, info)| {
                if let Some(ref q) = info.quantization {
                    if q == &quantization {
                        Some(info)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get models requiring authentication
    pub fn auth_required_models(&self) -> Vec<&ModelInfo> {
        self.all_models()
            .filter_map(|(_, _, info)| if info.requires_auth { Some(info) } else { None })
            .collect()
    }

    /// Search models by name or description
    pub fn search_models(&self, query: &str) -> Vec<&ModelInfo> {
        let query = query.to_lowercase();
        self.all_models()
            .filter_map(|(_, _, info)| {
                let name_match = info.name.to_lowercase().contains(&query);
                let desc_match = info.description.to_lowercase().contains(&query);
                let arch_match = info
                    .architecture
                    .as_ref()
                    .map(|a| a.to_lowercase().contains(&query))
                    .unwrap_or(false);

                if name_match || desc_match || arch_match {
                    Some(info)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get model statistics
    pub fn statistics(&self) -> ModelStats {
        let mut total_models = 0;
        let mut gguf_models = 0;
        let mut quantized_models = 0;
        let mut total_size: u64 = 0;
        let mut auth_required = 0;
        let mut architectures = std::collections::HashSet::new();
        let mut quantizations = std::collections::HashSet::new();

        for (_, _, info) in self.all_models() {
            total_models += 1;
            if info.is_gguf() {
                gguf_models += 1;
            }
            if info.is_quantized() {
                quantized_models += 1;
            }
            if let Some(size) = info.size {
                total_size += size;
            }
            if info.requires_auth {
                auth_required += 1;
            }
            if let Some(ref arch) = info.architecture {
                architectures.insert(arch.clone());
            }
            if let Some(ref quant) = info.quantization {
                quantizations.insert(format!("{:?}", quant));
            }
        }

        ModelStats {
            total_models,
            gguf_models,
            quantized_models,
            total_size_bytes: total_size,
            auth_required_models: auth_required,
            unique_architectures: architectures.len(),
            unique_quantizations: quantizations.len(),
            architectures: architectures.into_iter().collect(),
            quantizations: quantizations.into_iter().collect(),
        }
    }

    /// Add a model to the registry
    pub fn add_model(&mut self, model: ModelInfo) {
        self.models
            .entry(model.repo.clone())
            .or_default()
            .insert(model.name.clone(), model);
    }

    /// Get a model by repo and name
    pub fn get_model(&self, repo: &str, name: &str) -> Option<&ModelInfo> {
        self.models
            .get(repo)
            .and_then(|repo_models| repo_models.get(name))
    }

    /// Get all models in a repository
    pub fn get_repo_models(&self, repo: &str) -> Option<&HashMap<String, ModelInfo>> {
        self.models.get(repo)
    }

    /// Get all repositories
    pub fn repos(&self) -> impl Iterator<Item = &String> {
        self.models.keys()
    }

    /// Get all models across all repositories
    pub fn all_models(&self) -> impl Iterator<Item = (&String, &String, &ModelInfo)> {
        self.models
            .iter()
            .flat_map(|(repo, models)| models.iter().map(move |(name, info)| (repo, name, info)))
    }

    /// Check if a model exists
    pub fn contains(&self, repo: &str, name: &str) -> bool {
        self.get_model(repo, name).is_some()
    }

    /// Remove a model
    pub fn remove_model(&mut self, repo: &str, name: &str) -> bool {
        if let Some(repo_models) = self.models.get_mut(repo) {
            repo_models.remove(name).is_some()
        } else {
            false
        }
    }

    /// Load registry from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        let models: HashMap<String, HashMap<String, ModelInfo>> = serde_json::from_str(json)?;
        Ok(Self { models })
    }

    /// Export registry to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.models).map_err(Into::into)
    }
}

/// Model registry statistics
#[derive(Debug, Clone)]
pub struct ModelStats {
    /// Total number of models
    pub total_models: usize,
    /// Number of GGUF models
    pub gguf_models: usize,
    /// Number of quantized models
    pub quantized_models: usize,
    /// Total size of all models in bytes
    pub total_size_bytes: u64,
    /// Number of models requiring authentication
    pub auth_required_models: usize,
    /// Number of unique architectures
    pub unique_architectures: usize,
    /// Number of unique quantization schemes
    pub unique_quantizations: usize,
    /// List of architectures
    pub architectures: Vec<String>,
    /// List of quantization schemes
    pub quantizations: Vec<String>,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Built-in comprehensive model registry with PyTorch and GGUF support
pub fn comprehensive_registry() -> ModelRegistry {
    let mut registry = ModelRegistry::new();

    // PyTorch Vision Models
    add_pytorch_vision_models(&mut registry);

    // GGUF Models (llama.cpp format)
    add_gguf_models(&mut registry);

    registry
}

/// Add PyTorch Vision models to registry
fn add_pytorch_vision_models(registry: &mut ModelRegistry) {
    let base_url = crate::PYTORCH_HUB_URL;

    // ResNet models
    registry.add_model(
        ModelInfo::new(
            "resnet18".to_string(),
            "pytorch/vision".to_string(),
            "ResNet-18 model".to_string(),
            format!("{}resnet18-5c106cde.pth", base_url),
        )
        .with_parameters_millions(11.7)
        .with_license("BSD-3-Clause".to_string())
        .with_author("Facebook Research".to_string()),
    );

    registry.add_model(
        ModelInfo::new(
            "resnet34".to_string(),
            "pytorch/vision".to_string(),
            "ResNet-34 model".to_string(),
            format!("{}resnet34-333f7ec4.pth", base_url),
        )
        .with_parameters_millions(21.8)
        .with_license("BSD-3-Clause".to_string())
        .with_author("Facebook Research".to_string()),
    );

    registry.add_model(
        ModelInfo::new(
            "resnet50".to_string(),
            "pytorch/vision".to_string(),
            "ResNet-50 model".to_string(),
            format!("{}resnet50-19c8e357.pth", base_url),
        )
        .with_parameters_millions(25.6)
        .with_license("BSD-3-Clause".to_string())
        .with_author("Facebook Research".to_string()),
    );

    // VGG models
    registry.add_model(
        ModelInfo::new(
            "vgg16".to_string(),
            "pytorch/vision".to_string(),
            "VGG-16 model".to_string(),
            format!("{}vgg16-397923af.pth", base_url),
        )
        .with_parameters_millions(138.4)
        .with_license("BSD-3-Clause".to_string())
        .with_author("Oxford University".to_string()),
    );

    registry.add_model(
        ModelInfo::new(
            "vgg19".to_string(),
            "pytorch/vision".to_string(),
            "VGG-19 model".to_string(),
            format!("{}vgg19-dcbb9e9d.pth", base_url),
        )
        .with_parameters_millions(143.7)
        .with_license("BSD-3-Clause".to_string())
        .with_author("Oxford University".to_string()),
    );
}

/// Add GGUF models to registry
fn add_gguf_models(registry: &mut ModelRegistry) {
    // Base URL for GGUF models (example - would be actual hosting in production)
    let gguf_base_url = "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/";

    // Llama 2 models (GGUF format)
    registry.add_model(
        ModelInfo::new_gguf(
            "llama-2-7b-q4_0".to_string(),
            "meta-llama/Llama-2-7b".to_string(),
            "Llama 2 7B model in Q4_0 quantization".to_string(),
            format!("{}llama-2-7b-q4_0.gguf", gguf_base_url),
            "llama".to_string(),
            QuantizationScheme::Q4_0,
        )
        .with_parameters_millions(6738.0)
        .with_context_length(4096)
        .with_vocab_size(32000)
        .with_size(3584 * 1024 * 1024) // ~3.5GB
        .with_license("Llama 2 Community License".to_string())
        .with_author("Meta".to_string()),
    );

    registry.add_model(
        ModelInfo::new_gguf(
            "llama-2-7b-q5_0".to_string(),
            "meta-llama/Llama-2-7b".to_string(),
            "Llama 2 7B model in Q5_0 quantization".to_string(),
            format!("{}llama-2-7b-q5_0.gguf", gguf_base_url),
            "llama".to_string(),
            QuantizationScheme::Q5_0,
        )
        .with_parameters_millions(6738.0)
        .with_context_length(4096)
        .with_vocab_size(32000)
        .with_size(4480 * 1024 * 1024) // ~4.4GB
        .with_license("Llama 2 Community License".to_string())
        .with_author("Meta".to_string()),
    );

    registry.add_model(
        ModelInfo::new_gguf(
            "llama-2-13b-q4_0".to_string(),
            "meta-llama/Llama-2-13b".to_string(),
            "Llama 2 13B model in Q4_0 quantization".to_string(),
            format!("{}llama-2-13b-q4_0.gguf", gguf_base_url),
            "llama".to_string(),
            QuantizationScheme::Q4_0,
        )
        .with_parameters_millions(13000.0)
        .with_context_length(4096)
        .with_vocab_size(32000)
        .with_size(6738 * 1024 * 1024) // ~6.6GB
        .with_license("Llama 2 Community License".to_string())
        .with_author("Meta".to_string()),
    );

    // GPT-2 models
    registry.add_model(
        ModelInfo::new_gguf(
            "gpt2-q4_0".to_string(),
            "openai/gpt2".to_string(),
            "GPT-2 model in Q4_0 quantization".to_string(),
            format!("{}gpt2-q4_0.gguf", gguf_base_url),
            "gpt2".to_string(),
            QuantizationScheme::Q4_0,
        )
        .with_parameters_millions(124.4)
        .with_context_length(1024)
        .with_vocab_size(50257)
        .with_size(510 * 1024 * 1024) // ~510MB
        .with_license("MIT".to_string())
        .with_author("OpenAI".to_string()),
    );

    registry.add_model(
        ModelInfo::new_gguf(
            "gpt2-medium-q4_0".to_string(),
            "openai/gpt2-medium".to_string(),
            "GPT-2 Medium model in Q4_0 quantization".to_string(),
            format!("{}gpt2-medium-q4_0.gguf", gguf_base_url),
            "gpt2".to_string(),
            QuantizationScheme::Q4_0,
        )
        .with_parameters_millions(354.8)
        .with_context_length(1024)
        .with_vocab_size(50257)
        .with_size(1408 * 1024 * 1024) // ~1.4GB
        .with_license("MIT".to_string())
        .with_author("OpenAI".to_string()),
    );

    // Code Llama models
    registry.add_model(
        ModelInfo::new_gguf(
            "codellama-7b-q4_0".to_string(),
            "meta-llama/CodeLlama-7b".to_string(),
            "Code Llama 7B model in Q4_0 quantization".to_string(),
            format!("{}codellama-7b-q4_0.gguf", gguf_base_url),
            "llama".to_string(),
            QuantizationScheme::Q4_0,
        )
        .with_parameters_millions(6738.0)
        .with_context_length(16384)
        .with_vocab_size(32016)
        .with_size(3584 * 1024 * 1024) // ~3.5GB
        .with_license("Llama 2 Community License".to_string())
        .with_author("Meta".to_string()),
    );
}

/// Built-in PyTorch models registry (backwards compatibility)
pub fn pytorch_registry() -> ModelRegistry {
    let mut registry = ModelRegistry::new();
    add_pytorch_vision_models(&mut registry);
    registry
}

/// Built-in GGUF models registry
pub fn gguf_registry() -> ModelRegistry {
    let mut registry = ModelRegistry::new();
    add_gguf_models(&mut registry);
    registry
}
