//! Model registry for GGUF models and metadata management.
//!
//! This module provides a comprehensive registry for GGUF models with:
//! - **Model Metadata**: Detailed information about models, architectures, and quantization
//! - **Quantization Support**: Multiple quantization schemes with memory usage estimates
//! - **Search Functionality**: Find models by name, architecture, or quantization
//! - **Statistics**: Model statistics and memory usage analysis
//! - **Validation**: Model integrity and compatibility checking

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::quantization::QuantizationScheme;
use crate::ModelType;

/// Model information for GGUF models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Unique model identifier
    pub id: String,
    /// Model name
    pub name: String,
    /// Model architecture (llama, gpt2, etc.)
    pub architecture: String,
    /// Model description
    pub description: String,
    /// Repository/organization
    pub repository: String,
    /// File size in bytes
    pub file_size: u64,
    /// Parameter count
    pub parameter_count: u64,
    /// Quantization scheme
    pub quantization: QuantizationScheme,
    /// Context length
    pub context_length: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Memory usage estimate in bytes
    pub memory_usage: u64,
    /// Download URL
    pub download_url: String,
    /// SHA256 hash for verification
    pub sha256_hash: Option<String>,
    /// License information
    pub license: String,
    /// Author/organization
    pub author: String,
    /// Creation date
    pub created_at: String,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Model compatibility flags
    pub compatibility: ModelCompatibility,
}

/// Model compatibility information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCompatibility {
    /// Compatible with CPU inference
    pub cpu_compatible: bool,
    /// Compatible with GPU inference
    pub gpu_compatible: bool,
    /// Compatible with quantization
    pub quantization_compatible: bool,
    /// Compatible with streaming inference
    pub streaming_compatible: bool,
    /// Compatible with batch inference
    pub batch_compatible: bool,
    /// Minimum required memory in GB
    pub min_memory_gb: f64,
    /// Recommended memory in GB
    pub recommended_memory_gb: f64,
}

impl Default for ModelCompatibility {
    fn default() -> Self {
        Self {
            cpu_compatible: true,
            gpu_compatible: true,
            quantization_compatible: true,
            streaming_compatible: true,
            batch_compatible: true,
            min_memory_gb: 1.0,
            recommended_memory_gb: 4.0,
        }
    }
}

/// Model registry for GGUF models
pub struct ModelRegistry {
    /// Registered models
    models: HashMap<String, ModelInfo>,
    /// Models by architecture
    by_architecture: HashMap<String, Vec<String>>,
    /// Models by quantization scheme
    by_quantization: HashMap<QuantizationScheme, Vec<String>>,
    /// Models by parameter count
    by_parameters: HashMap<String, Vec<String>>, // "7B", "13B", etc.
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            by_architecture: HashMap::new(),
            by_quantization: HashMap::new(),
            by_parameters: HashMap::new(),
        }
    }

    /// Register a model
    pub fn register(&mut self, model: ModelInfo) {
        let model_id = model.id.clone();

        // Add to main registry
        self.models.insert(model_id.clone(), model.clone());

        // Index by architecture
        self.by_architecture
            .entry(model.architecture.clone())
            .or_default()
            .push(model_id.clone());

        // Index by quantization
        self.by_quantization
            .entry(model.quantization)
            .or_default()
            .push(model_id.clone());

        // Index by parameter count
        let param_key = self.get_parameter_key(model.parameter_count);
        self.by_parameters
            .entry(param_key)
            .or_default()
            .push(model_id);
    }

    /// Get model by ID
    pub fn get(&self, id: &str) -> Option<&ModelInfo> {
        self.models.get(id)
    }

    /// Get all models
    pub fn all(&self) -> Vec<&ModelInfo> {
        self.models.values().collect()
    }

    /// Get models by architecture
    pub fn by_architecture(&self, architecture: &str) -> Vec<&ModelInfo> {
        if let Some(model_ids) = self.by_architecture.get(architecture) {
            model_ids.iter()
                .filter_map(|id| self.models.get(id))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get models by quantization scheme
    pub fn by_quantization(&self, quantization: QuantizationScheme) -> Vec<&ModelInfo> {
        if let Some(model_ids) = self.by_quantization.get(&quantization) {
            model_ids.iter()
                .filter_map(|id| self.models.get(id))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get models by parameter count
    pub fn by_parameters(&self, parameter_range: &str) -> Vec<&ModelInfo> {
        if let Some(model_ids) = self.by_parameters.get(parameter_range) {
            model_ids.iter()
                .filter_map(|id| self.models.get(id))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Search models by name or description
    pub fn search(&self, query: &str) -> Vec<&ModelInfo> {
        let query = query.to_lowercase();
        self.models.values()
            .filter(|model| {
                model.name.to_lowercase().contains(&query) ||
                model.description.to_lowercase().contains(&query) ||
                model.architecture.to_lowercase().contains(&query) ||
                model.tags.iter().any(|tag| tag.to_lowercase().contains(&query))
            })
            .collect()
    }

    /// Get registry statistics
    pub fn statistics(&self) -> RegistryStats {
        let mut total_size: u64 = 0;
        let mut total_memory: u64 = 0;
        let mut architectures = std::collections::HashSet::new();
        let mut quantizations = std::collections::HashMap::new();
        let mut parameter_counts = std::collections::HashMap::new();

        for model in self.models.values() {
            total_size += model.file_size;
            total_memory += model.memory_usage;
            architectures.insert(model.architecture.clone());

            *quantizations.entry(model.quantization).or_insert(0) += 1;

            let param_key = self.get_parameter_key(model.parameter_count);
            *parameter_counts.entry(param_key).or_insert(0) += 1;
        }

        RegistryStats {
            total_models: self.models.len(),
            total_file_size: total_size,
            total_memory_usage: total_memory,
            unique_architectures: architectures.len(),
            architectures: architectures.into_iter().collect(),
            quantization_counts: quantizations,
            parameter_counts,
        }
    }

    /// Validate model compatibility
    pub fn validate_compatibility(&self, id: &str, requirements: &CompatibilityRequirements) -> bool {
        if let Some(model) = self.get(id) {
            let compatible = model.compatibility.cpu_compatible || requirements.gpu_available;
            let memory_ok = model.compatibility.recommended_memory_gb <= requirements.available_memory_gb;
            let quantization_ok = model.compatibility.quantization_compatible || !requirements.require_quantized;

            compatible && memory_ok && quantization_ok
        } else {
            false
        }
    }

    /// Get parameter count key for indexing
    fn get_parameter_key(&self, parameter_count: u64) -> String {
        match parameter_count {
            0..=100_000_000 => "small",        // < 100M parameters
            100_000_001..=1_000_000_000 => "medium", // 100M - 1B parameters
            1_000_000_001..=10_000_000_000 => "large", // 1B - 10B parameters
            _ => "xl",                         // > 10B parameters
        }
    }
}

/// Registry statistics
#[derive(Debug, Clone)]
pub struct RegistryStats {
    /// Total number of models
    pub total_models: usize,
    /// Total file size of all models in bytes
    pub total_file_size: u64,
    /// Total memory usage estimate in bytes
    pub total_memory_usage: u64,
    /// Number of unique architectures
    pub unique_architectures: usize,
    /// List of architectures
    pub architectures: Vec<String>,
    /// Count of models by quantization scheme
    pub quantization_counts: std::collections::HashMap<QuantizationScheme, usize>,
    /// Count of models by parameter range
    pub parameter_counts: std::collections::HashMap<String, usize>,
}

/// Compatibility requirements for model selection
#[derive(Debug, Clone)]
pub struct CompatibilityRequirements {
    /// GPU is available
    pub gpu_available: bool,
    /// Available memory in GB
    pub available_memory_gb: f64,
    /// Require quantized models
    pub require_quantized: bool,
    /// Support streaming inference
    pub require_streaming: bool,
    /// Support batch inference
    pub require_batch: bool,
}

impl Default for CompatibilityRequirements {
    fn default() -> Self {
        Self {
            gpu_available: false,
            available_memory_gb: 8.0,
            require_quantized: false,
            require_streaming: true,
            require_batch: true,
        }
    }
}

/// Predefined model registry with popular GGUF models
pub fn default_registry() -> ModelRegistry {
    let mut registry = ModelRegistry::new();

    // Llama 2 models
    registry.register(ModelInfo {
        id: "llama-2-7b-q4_0".to_string(),
        name: "Llama 2 7B Q4_0".to_string(),
        architecture: "llama".to_string(),
        description: "Llama 2 7B parameter model in Q4_0 quantization".to_string(),
        repository: "meta-llama/Llama-2-7b".to_string(),
        file_size: 3_584_000_000, // ~3.6GB
        parameter_count: 6_738_000_000,
        quantization: QuantizationScheme::Q4_0,
        context_length: 4096,
        vocab_size: 32000,
        memory_usage: 1_792_000_000, // ~1.8GB in memory
        download_url: "https://huggingface.co/meta-llama/Llama-2-7b/resolve/main/llama-2-7b-q4_0.gguf".to_string(),
        sha256_hash: Some("abc123...".to_string()),
        license: "Llama 2 Community License".to_string(),
        author: "Meta".to_string(),
        created_at: "2023-07-18".to_string(),
        tags: vec!["llama".to_string(), "7b".to_string(), "quantized".to_string()],
        compatibility: ModelCompatibility::default(),
    });

    registry.register(ModelInfo {
        id: "llama-2-13b-q4_0".to_string(),
        name: "Llama 2 13B Q4_0".to_string(),
        architecture: "llama".to_string(),
        description: "Llama 2 13B parameter model in Q4_0 quantization".to_string(),
        repository: "meta-llama/Llama-2-13b".to_string(),
        file_size: 6_738_000_000, // ~6.7GB
        parameter_count: 13_020_000_000,
        quantization: QuantizationScheme::Q4_0,
        context_length: 4096,
        vocab_size: 32000,
        memory_usage: 3_369_000_000, // ~3.4GB in memory
        download_url: "https://huggingface.co/meta-llama/Llama-2-13b/resolve/main/llama-2-13b-q4_0.gguf".to_string(),
        sha256_hash: Some("def456...".to_string()),
        license: "Llama 2 Community License".to_string(),
        author: "Meta".to_string(),
        created_at: "2023-07-18".to_string(),
        tags: vec!["llama".to_string(), "13b".to_string(), "quantized".to_string()],
        compatibility: ModelCompatibility {
            min_memory_gb: 4.0,
            recommended_memory_gb: 8.0,
            ..Default::default()
        },
    });

    // GPT-2 models
    registry.register(ModelInfo {
        id: "gpt2-q4_0".to_string(),
        name: "GPT-2 Q4_0".to_string(),
        architecture: "gpt2".to_string(),
        description: "GPT-2 model in Q4_0 quantization".to_string(),
        repository: "openai/gpt2".to_string(),
        file_size: 510_000_000, // ~510MB
        parameter_count: 124_400_000,
        quantization: QuantizationScheme::Q4_0,
        context_length: 1024,
        vocab_size: 50257,
        memory_usage: 255_000_000, // ~255MB in memory
        download_url: "https://huggingface.co/openai/gpt2/resolve/main/gpt2-q4_0.gguf".to_string(),
        sha256_hash: Some("ghi789...".to_string()),
        license: "MIT".to_string(),
        author: "OpenAI".to_string(),
        created_at: "2019-02-14".to_string(),
        tags: vec!["gpt2".to_string(), "124m".to_string(), "quantized".to_string()],
        compatibility: ModelCompatibility {
            min_memory_gb: 0.5,
            recommended_memory_gb: 1.0,
            ..Default::default()
        },
    });

    // Code Llama models
    registry.register(ModelInfo {
        id: "codellama-7b-q4_0".to_string(),
        name: "Code Llama 7B Q4_0".to_string(),
        architecture: "llama".to_string(),
        description: "Code Llama 7B parameter model in Q4_0 quantization".to_string(),
        repository: "meta-llama/CodeLlama-7b".to_string(),
        file_size: 3_584_000_000, // ~3.6GB
        parameter_count: 6_738_000_000,
        quantization: QuantizationScheme::Q4_0,
        context_length: 16384,
        vocab_size: 32016,
        memory_usage: 1_792_000_000, // ~1.8GB in memory
        download_url: "https://huggingface.co/meta-llama/CodeLlama-7b/resolve/main/codellama-7b-q4_0.gguf".to_string(),
        sha256_hash: Some("jkl012...".to_string()),
        license: "Llama 2 Community License".to_string(),
        author: "Meta".to_string(),
        created_at: "2023-08-24".to_string(),
        tags: vec!["codellama".to_string(), "7b".to_string(), "code".to_string(), "quantized".to_string()],
        compatibility: ModelCompatibility::default(),
    });

    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_info_creation() {
        let info = ModelInfo {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            architecture: "llama".to_string(),
            description: "A test model".to_string(),
            repository: "test/repo".to_string(),
            file_size: 1000000,
            parameter_count: 1000000,
            quantization: QuantizationScheme::Q4_0,
            context_length: 2048,
            vocab_size: 32000,
            memory_usage: 500000,
            download_url: "https://example.com/model.gguf".to_string(),
            sha256_hash: Some("abc123".to_string()),
            license: "MIT".to_string(),
            author: "Test Author".to_string(),
            created_at: "2023-01-01".to_string(),
            tags: vec!["test".to_string()],
            compatibility: ModelCompatibility::default(),
        };

        assert_eq!(info.id, "test-model");
        assert_eq!(info.name, "Test Model");
        assert_eq!(info.architecture, "llama");
        assert_eq!(info.quantization, QuantizationScheme::Q4_0);
        assert_eq!(info.context_length, 2048);
        assert_eq!(info.vocab_size, 32000);
    }

    #[test]
    fn test_registry_operations() {
        let mut registry = ModelRegistry::new();

        let model = ModelInfo {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            architecture: "llama".to_string(),
            description: "A test model".to_string(),
            repository: "test/repo".to_string(),
            file_size: 1000000,
            parameter_count: 1000000,
            quantization: QuantizationScheme::Q4_0,
            context_length: 2048,
            vocab_size: 32000,
            memory_usage: 500000,
            download_url: "https://example.com/model.gguf".to_string(),
            sha256_hash: None,
            license: "MIT".to_string(),
            author: "Test Author".to_string(),
            created_at: "2023-01-01".to_string(),
            tags: vec!["test".to_string()],
            compatibility: ModelCompatibility::default(),
        };

        registry.register(model);

        assert_eq!(registry.all().len(), 1);
        assert_eq!(registry.by_architecture("llama").len(), 1);
        assert_eq!(registry.by_quantization(QuantizationScheme::Q4_0).len(), 1);
        assert_eq!(registry.by_parameters("small").len(), 1);
        assert!(registry.get("test-model").is_some());
    }

    #[test]
    fn test_registry_search() {
        let mut registry = ModelRegistry::new();

        registry.register(ModelInfo {
            id: "llama-test".to_string(),
            name: "Llama Test Model".to_string(),
            architecture: "llama".to_string(),
            description: "A test Llama model".to_string(),
            repository: "test/repo".to_string(),
            file_size: 1000000,
            parameter_count: 1000000,
            quantization: QuantizationScheme::Q4_0,
            context_length: 2048,
            vocab_size: 32000,
            memory_usage: 500000,
            download_url: "https://example.com/llama.gguf".to_string(),
            sha256_hash: None,
            license: "MIT".to_string(),
            author: "Test Author".to_string(),
            created_at: "2023-01-01".to_string(),
            tags: vec!["llama".to_string(), "test".to_string()],
            compatibility: ModelCompatibility::default(),
        });

        let results = registry.search("llama");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Llama Test Model");

        let results = registry.search("test");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_registry_statistics() {
        let mut registry = ModelRegistry::new();

        registry.register(ModelInfo {
            id: "model1".to_string(),
            name: "Model 1".to_string(),
            architecture: "llama".to_string(),
            description: "First model".to_string(),
            repository: "test/repo".to_string(),
            file_size: 1000000,
            parameter_count: 1000000,
            quantization: QuantizationScheme::Q4_0,
            context_length: 2048,
            vocab_size: 32000,
            memory_usage: 500000,
            download_url: "https://example.com/model1.gguf".to_string(),
            sha256_hash: None,
            license: "MIT".to_string(),
            author: "Test Author".to_string(),
            created_at: "2023-01-01".to_string(),
            tags: vec![].to_string(),
            compatibility: ModelCompatibility::default(),
        });

        registry.register(ModelInfo {
            id: "model2".to_string(),
            name: "Model 2".to_string(),
            architecture: "gpt2".to_string(),
            description: "Second model".to_string(),
            repository: "test/repo".to_string(),
            file_size: 2000000,
            parameter_count: 2000000,
            quantization: QuantizationScheme::Q8_0,
            context_length: 1024,
            vocab_size: 50000,
            memory_usage: 1000000,
            download_url: "https://example.com/model2.gguf".to_string(),
            sha256_hash: None,
            license: "MIT".to_string(),
            author: "Test Author".to_string(),
            created_at: "2023-01-01".to_string(),
            tags: vec![].to_string(),
            compatibility: ModelCompatibility::default(),
        });

        let stats = registry.statistics();
        assert_eq!(stats.total_models, 2);
        assert_eq!(stats.total_file_size, 3000000);
        assert_eq!(stats.total_memory_usage, 1500000);
        assert_eq!(stats.unique_architectures, 2);
        assert_eq!(stats.architectures, vec!["gpt2", "llama"]);
    }

    #[test]
    fn test_compatibility_validation() {
        let registry = default_registry();

        let requirements = CompatibilityRequirements {
            gpu_available: false,
            available_memory_gb: 8.0,
            require_quantized: true,
            require_streaming: true,
            require_batch: true,
        };

        // Should find compatible models
        let compatible_models = registry.all().iter()
            .filter(|model| registry.validate_compatibility(&model.id, &requirements))
            .count();

        assert!(compatible_models > 0);
    }
}
