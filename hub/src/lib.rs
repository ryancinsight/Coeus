//! # Coeus Hub
//!
//! PyTorch Hub-compatible model loading and management for the Coeus tensor library,
//! with comprehensive support for GGUF (llama.cpp) models and quantization.
//!
//! This crate provides functionality to load pre-trained models from remote repositories,
//! similar to PyTorch Hub, enabling users to easily access and use state-of-the-art models.
//!
//! ## Features
//!
//! - **Model Loading**: Load pre-trained models from remote repositories
//! - **PyTorch Compatibility**: Load models saved in PyTorch format (.pth, .pt)
//! - **GGUF Format Support**: Load and manage GGUF models (llama.cpp format)
//! - **Quantization Support**: Multiple quantization schemes (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)
//! - **State Dict Management**: Load and manage model state dictionaries
//! - **Model Registry**: Centralized registry of available models with metadata
//! - **Caching**: Automatic caching of downloaded models with integrity verification
//! - **Multi-Architecture**: Support for Llama, GPT-2, and other transformer architectures
//! - **Memory Optimization**: Efficient loading with memory mapping
//!
//! ## GGUF Model Support
//!
//! The hub provides comprehensive support for GGUF models:
//!
//! - **Llama Models**: Llama 2, Code Llama, and variants in multiple quantizations
//! - **GPT Models**: GPT-2 and GPT-Neo models with quantization support
//! - **Memory Efficient**: Quantized models reduce memory usage by up to 75%
//! - **Performance**: Optimized inference with SIMD acceleration
//! - **Metadata**: Rich model metadata including architecture, vocabulary, and licensing
//!
//! ## Example Usage
//!
//! ### PyTorch Models
//! ```rust,no_run
//! use coeus_hub::Hub;
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//!
//! // Load a pre-trained PyTorch model's state dict
//! let hub = Hub::new();
//! let state_dict = hub.load("pytorch/vision", "resnet18", false, None, false).await?;
//!
//! // Apply to your model (assuming you have a model that implements load_state_dict)
//! // model.load_state_dict(&state_dict)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### GGUF Models (llama.cpp)
//! ```rust,no_run
//! use coeus_hub::{Hub, comprehensive_registry};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Use the comprehensive registry with GGUF models
//! let registry = comprehensive_registry();
//! let hub = Hub::with_registry(registry.clone());
//!
//! // List available GGUF models (access registry directly)
//! let gguf_models = registry.gguf_models();
//! println!("Available GGUF models: {}", gguf_models.len());
//!
//! // Get model information
//! let llama_info = hub.model_info("meta-llama/Llama-2-7b", "llama-2-7b-q4_0").unwrap();
//! println!("Model: {}", llama_info.name);
//! println!("Architecture: {:?}", llama_info.architecture);
//! println!("Quantization: {:?}", llama_info.quantization);
//! println!("Memory usage: {} MB", llama_info.estimated_memory_usage().unwrap() / 1024 / 1024);
//!
//! // Load a GGUF model
//! let state_dict = hub.load("meta-llama/Llama-2-7b", "llama-2-7b-q4_0", false, None, false).await?;
//! # Ok(())
//! # }
//! ```

pub mod error;
pub mod hub;
pub mod loader;
pub mod registry;
pub mod state_dict;

pub use error::HubError;
pub use hub::Hub;
pub use loader::{load_state_dict, load_state_dict_from_url, save_state_dict};
pub use registry::{
    comprehensive_registry, gguf_registry, pytorch_registry, ModelInfo, ModelRegistry, ModelStats,
    ModelType, QuantizationScheme,
};
pub use state_dict::StateDict;

/// Result type for hub operations
pub type Result<T> = std::result::Result<T, HubError>;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default PyTorch Hub URL
pub const PYTORCH_HUB_URL: &str = "https://pytorch.s3.amazonaws.com/models/";

/// Default cache directory
pub fn default_cache_dir() -> std::path::PathBuf {
    dirs::cache_dir()
        .unwrap_or(std::env::temp_dir())
        .join("coeus")
        .join("hub")
}

/// Initialize the global hub
pub fn init() -> Result<()> {
    Hub::init_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub::global_hub;
    use crate::loader::{load_json_state_dict, save_json_state_dict};
    use crate::registry::ModelInfo;
    use coeus_tensor::Tensor;

    #[test]
    fn test_state_dict_operations() {
        let mut state_dict = StateDict::new();

        // Test empty state dict
        assert!(state_dict.is_empty());
        assert_eq!(state_dict.len(), 0);

        // Add a parameter
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        state_dict.insert("layer.weight".to_string(), tensor);

        assert!(!state_dict.is_empty());
        assert_eq!(state_dict.len(), 1);
        assert!(state_dict.contains_key("layer.weight"));
        assert!(!state_dict.contains_key("nonexistent"));

        // Get parameter
        let retrieved = state_dict.get("layer.weight").unwrap();
        assert_eq!(retrieved.data(), &[1.0, 2.0, 3.0]);
        assert_eq!(retrieved.shape(), &[3]);

        // Test iteration
        let keys: Vec<_> = state_dict.keys().collect();
        assert_eq!(keys, &["layer.weight"]);

        let values_count = state_dict.values().count();
        assert_eq!(values_count, 1);

        // Test mutable access
        {
            let mut_tensor = state_dict.get_mut("layer.weight").unwrap();
            // Modify the tensor data (simplified test)
            assert_eq!(mut_tensor.shape(), &[3]);
        }

        // Remove parameter
        let removed = state_dict.remove("layer.weight");
        assert!(removed.is_some());
        assert!(state_dict.is_empty());

        // Test subset creation
        let mut state_dict = StateDict::new();
        state_dict.insert("param1".to_string(), Tensor::from_vec(vec![1.0], vec![1]));
        state_dict.insert("param2".to_string(), Tensor::from_vec(vec![2.0], vec![1]));
        state_dict.insert("param3".to_string(), Tensor::from_vec(vec![3.0], vec![1]));

        let subset = state_dict.subset(&["param1", "param3"]);
        assert_eq!(subset.len(), 2);
        assert!(subset.contains_key("param1"));
        assert!(subset.contains_key("param3"));
        assert!(!subset.contains_key("param2"));
    }

    #[test]
    fn test_model_registry_operations() {
        let mut registry = ModelRegistry::new();

        // Test empty registry
        assert!(!registry.contains("test_repo", "test_model"));

        // Add a model
        let model_info = ModelInfo::new(
            "test_model".to_string(),
            "test_repo".to_string(),
            "A test model".to_string(),
            "https://example.com/model.pth".to_string(),
        );
        registry.add_model(model_info);

        // Test model existence
        assert!(registry.contains("test_repo", "test_model"));
        assert!(!registry.contains("test_repo", "nonexistent"));
        assert!(!registry.contains("nonexistent", "test_model"));

        // Get model info
        let retrieved = registry.get_model("test_repo", "test_model").unwrap();
        assert_eq!(retrieved.name, "test_model");
        assert_eq!(retrieved.repo, "test_repo");
        assert_eq!(retrieved.description, "A test model");

        // Test repository listing
        let repos: Vec<_> = registry.repos().collect();
        assert_eq!(repos.len(), 1);
        assert_eq!(repos[0], "test_repo");

        // Test model listing in repo
        let models = registry.get_repo_models("test_repo").unwrap();
        assert_eq!(models.len(), 1);
        assert!(models.contains_key("test_model"));

        // Test all models iteration
        let all_models: Vec<_> = registry.all_models().collect();
        assert_eq!(all_models.len(), 1);
        let (repo, name, info) = &all_models[0];
        assert_eq!(*repo, "test_repo");
        assert_eq!(*name, "test_model");
        assert_eq!(info.name, "test_model");

        // Remove model
        let removed = registry.remove_model("test_repo", "test_model");
        assert!(removed);
        assert!(!registry.contains("test_repo", "test_model"));

        // Test removing nonexistent model
        let removed_nonexistent = registry.remove_model("test_repo", "nonexistent");
        assert!(!removed_nonexistent);
    }

    #[test]
    fn test_model_info_builder_pattern() {
        let model_info = ModelInfo::new(
            "resnet18".to_string(),
            "pytorch/vision".to_string(),
            "ResNet-18 model".to_string(),
            "https://example.com/resnet18.pth".to_string(),
        )
        .with_hash("abc123".to_string())
        .with_size(1024)
        .with_config("num_classes".to_string(), serde_json::json!(1000));

        assert_eq!(model_info.name, "resnet18");
        assert_eq!(model_info.repo, "pytorch/vision");
        assert_eq!(model_info.hash, Some("abc123".to_string()));
        assert_eq!(model_info.size, Some(1024));
        assert_eq!(model_info.config["num_classes"], 1000);
    }

    #[test]
    fn test_pytorch_registry_initialization() {
        let registry = pytorch_registry();

        // Should contain some models
        assert!(registry.contains("pytorch/vision", "resnet18"));
        assert!(registry.contains("pytorch/vision", "resnet34"));
        assert!(registry.contains("pytorch/vision", "resnet50"));
        assert!(registry.contains("pytorch/vision", "vgg16"));
        assert!(registry.contains("pytorch/vision", "vgg19"));

        // Should not contain nonexistent models
        assert!(!registry.contains("pytorch/vision", "nonexistent"));
        assert!(!registry.contains("nonexistent", "resnet18"));
    }

    #[test]
    fn test_registry_serialization() {
        let mut registry = ModelRegistry::new();
        let model_info = ModelInfo::new(
            "test_model".to_string(),
            "test_repo".to_string(),
            "Test model".to_string(),
            "https://example.com/test.pth".to_string(),
        )
        .with_size(2048);
        registry.add_model(model_info);

        // Serialize to JSON
        let json = registry.to_json().unwrap();
        assert!(json.contains("test_model"));
        assert!(json.contains("test_repo"));

        // Deserialize from JSON
        let deserialized = ModelRegistry::from_json(&json).unwrap();
        assert!(deserialized.contains("test_repo", "test_model"));

        let retrieved = deserialized.get_model("test_repo", "test_model").unwrap();
        assert_eq!(retrieved.size, Some(2048));
    }

    #[test]
    fn test_state_dict_json_serialization() {
        let mut state_dict = StateDict::new();
        state_dict.insert(
            "param1".to_string(),
            Tensor::from_vec(vec![1.0, 2.0], vec![2]),
        );
        state_dict.insert(
            "param2".to_string(),
            Tensor::from_vec(vec![3.0, 4.0, 5.0], vec![3]),
        );

        // Serialize to JSON
        let json = save_json_state_dict(&state_dict).unwrap();
        assert!(json.contains("param1"));
        assert!(json.contains("param2"));

        // Deserialize from JSON
        let deserialized = load_json_state_dict(&json).unwrap();
        assert_eq!(deserialized.len(), 2);

        let param1 = deserialized.get("param1").unwrap();
        assert_eq!(param1.data(), &[1.0, 2.0]);
        assert_eq!(param1.shape(), &[2]);

        let param2 = deserialized.get("param2").unwrap();
        assert_eq!(param2.data(), &[3.0, 4.0, 5.0]);
        assert_eq!(param2.shape(), &[3]);
    }

    #[test]
    fn test_hub_basic_operations() {
        let hub = Hub::new();

        // Test model existence
        assert!(hub.has_model("pytorch/vision", "resnet18"));
        assert!(!hub.has_model("pytorch/vision", "nonexistent"));

        // Test model listing
        let resnet_models = hub.list_models("pytorch/vision");
        assert!(resnet_models.contains(&"resnet18".to_string()));
        assert!(resnet_models.contains(&"resnet50".to_string()));

        // Test repo listing
        let repos = hub.list_repos();
        assert!(repos.contains(&"pytorch/vision".to_string()));

        // Test model info retrieval
        let resnet_info = hub.model_info("pytorch/vision", "resnet18").unwrap();
        assert_eq!(resnet_info.name, "resnet18");
        assert!(resnet_info.url.contains("resnet18"));
    }

    #[test]
    fn test_hub_with_custom_registry() {
        let mut registry = ModelRegistry::new();
        let model_info = ModelInfo::new(
            "custom_model".to_string(),
            "custom_repo".to_string(),
            "Custom model".to_string(),
            "https://example.com/custom.pth".to_string(),
        );
        registry.add_model(model_info);

        let hub = Hub::with_registry(registry);

        assert!(hub.has_model("custom_repo", "custom_model"));
        assert!(!hub.has_model("pytorch/vision", "resnet18"));
    }

    #[test]
    fn test_global_hub_access() {
        // Test that global hub is accessible
        let _global_hub = global_hub();

        // Test global load functions exist (though they will fail without network)
        // These are compile-time tests - actual async tests would require mocking
    }

    #[test]
    fn test_error_conditions() {
        // Test empty registry operations
        let registry = ModelRegistry::new();
        assert!(!registry.contains("nonexistent", "model"));
        assert!(registry.get_model("nonexistent", "model").is_none());
        assert!(registry.get_repo_models("nonexistent").is_none());

        // Test registry serialization with empty registry
        let json = registry.to_json().unwrap();
        let deserialized = ModelRegistry::from_json(&json).unwrap();
        assert_eq!(deserialized.all_models().count(), 0); // Should have no models

        // Test invalid JSON for registry
        assert!(ModelRegistry::from_json("invalid json").is_err());

        // Test empty state dict operations
        let state_dict = StateDict::new();
        assert!(state_dict.is_empty());
        assert_eq!(state_dict.len(), 0);
        assert!(state_dict.get("nonexistent").is_none());
        assert!(!state_dict.contains_key("nonexistent"));

        // Test invalid JSON for state dict
        assert!(load_json_state_dict("invalid json").is_err());
        assert!(load_json_state_dict("{}").is_ok()); // Empty JSON should work

        // Test pickle loading via URL (will fail with network error, but tests the path)
        // Note: Actual pickle loading tests would require proper pickle data, which is complex
        // For now, we test that the function exists and the error handling works through other paths
    }

    #[test]
    fn test_models_hub_integration() {
        use coeus_models::{ModelConfig, ModelType, QuantizationScheme};
        use coeus_tensor::Tensor;

        // Test integration between models and hub crates
        let registry = pytorch_registry();
        let hub = Hub::with_registry(registry);

        // Verify hub can access model registry
        assert!(hub.has_model("pytorch/vision", "resnet18"));

        // Test model config integration
        let config = ModelConfig::new(ModelType::Llama)
            .with_max_seq_len(1024)
            .with_hidden_size(4096)
            .with_quantization(QuantizationScheme::Q4_0);

        assert_eq!(config.model_type, ModelType::Llama);
        assert_eq!(config.max_seq_len, 1024);
        assert_eq!(config.quantization, Some(QuantizationScheme::Q4_0));

        // Test state dict operations
        let mut state_dict = StateDict::new();
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        state_dict.insert("test.weight".to_string(), tensor);

        assert_eq!(state_dict.len(), 1);
        assert!(state_dict.contains_key("test.weight"));
    }

    #[tokio::test]
    async fn test_enhanced_error_handling() {
        use crate::HubError;

        // Test invalid model error
        let registry = ModelRegistry::new();
        let hub = Hub::with_registry(registry);

        // Should return ModelNotFound error
        let result = hub
            .load("nonexistent/repo", "nonexistent_model", false, None, true)
            .await;
        assert!(matches!(result, Err(HubError::ModelNotFound { .. })));

        // Test empty state dict error handling
        let empty_state_dict = StateDict::new();
        assert!(empty_state_dict.is_empty());

        // Test URL parsing in error context
        let invalid_url = "not-a-valid-url";
        let url_error = HubError::download_error(invalid_url.to_string(), "Invalid URL format");
        assert!(url_error.to_string().contains("not-a-valid-url"));
    }

    #[test]
    fn test_edge_cases() {
        // Test model info with minimal data
        let minimal_info = ModelInfo::new(
            "".to_string(),          // Empty name
            "".to_string(),          // Empty repo
            "".to_string(),          // Empty description
            "not-a-url".to_string(), // Invalid URL
        );
        assert_eq!(minimal_info.name, "");
        assert_eq!(minimal_info.repo, "");
        assert_eq!(minimal_info.description, "");

        // Test state dict with empty tensors
        let mut state_dict = StateDict::new();
        let empty_tensor = Tensor::from_vec(Vec::<f32>::new(), vec![0]);
        state_dict.insert("empty".to_string(), empty_tensor);
        assert_eq!(state_dict.len(), 1);
        let retrieved = state_dict.get("empty").unwrap();
        assert_eq!(retrieved.numel(), 0);

        // Test registry with special characters in names
        let mut registry = ModelRegistry::new();
        let special_info = ModelInfo::new(
            "model_with_underscores_and-dashes".to_string(),
            "repo/with/slashes".to_string(),
            "Description with spaces".to_string(),
            "https://example.com/model.pth".to_string(),
        );
        registry.add_model(special_info);
        assert!(registry.contains("repo/with/slashes", "model_with_underscores_and-dashes"));

        // Test hub with empty cache directory path
        let hub = Hub::with_cache_dir("");
        assert_eq!(hub.cache_dir().to_str().unwrap(), "");

        // Test model removal
        let mut registry = ModelRegistry::new();
        let model_info = ModelInfo::new(
            "test".to_string(),
            "repo".to_string(),
            "test".to_string(),
            "https://example.com/test.pth".to_string(),
        );
        registry.add_model(model_info.clone());
        assert!(registry.contains("repo", "test"));

        // Remove existing model
        assert!(registry.remove_model("repo", "test"));
        assert!(!registry.contains("repo", "test"));

        // Try to remove again (should return false)
        assert!(!registry.remove_model("repo", "test"));
    }
}
