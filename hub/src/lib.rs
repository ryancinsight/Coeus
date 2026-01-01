//! # Coeus Model Hub
//!
//! A comprehensive model hub for pretrained neural networks in Rust.
//! Provides PyTorch Hub-compatible functionality with memory safety guarantees.
//!
//! ## Features
//!
//! - **Model Registry**: Centralized, versioned catalog of pretrained models
//! - **Safe Loading**: Memory-safe model deserialization and weight loading
//! - **Intelligent Caching**: Local storage with automatic cleanup and versioning
//! - **Model Validation**: Integrity verification and performance validation
//! - **PyTorch Compatibility**: API compatible with torch.hub
//!
//! ## Usage
//!
//! ```rust,no_run
//! use hub::{Hub, Task as ModelTask};
//!
//! #[tokio::test]
//! async fn load_resnet() {
//!     let hub = Hub::new();
//!
//!     // Load a pretrained ResNet-50 model
//!     let model = hub.load("resnet50", ModelTask::Classification).await.unwrap();
//!
//!     // Use the model for inference
//!     // let output = model.forward(&input).unwrap();
//! }
//! ```
//!
//! ## Architecture
//!
//! The hub operates in layers:
//! 1. **Registry**: Model discovery and metadata management
//! 2. **Cache**: Local storage and retrieval of model artifacts
//! 3. **Loader**: Safe deserialization and instantiation of models
//! 4. **Validator**: Model integrity and performance verification

pub mod cache;
pub mod error;
pub mod loader;
pub mod models;
pub mod registry;
pub mod validator;

pub use cache::ModelCache;
pub use error::{HubError, Result};
pub use loader::{
    HuggingFaceFile, HuggingFaceLoader, HuggingFaceModelInfo, LoadConfig, LoadedModel, ModelLoader,
};
pub use models::*;
pub use registry::{ModelEntry, ModelMetadata, ModelRegistry, Task as ModelTask};
pub use validator::{ModelValidator, ValidationResult};

// Re-export commonly used traits
pub use backend::Backend;
pub use dtype::DataType;
pub use nn::Module;

/// Main hub interface for model management
#[derive(Debug)]
#[allow(dead_code)]
pub struct Hub {
    registry: ModelRegistry,
    loader: ModelLoader,
    cache: ModelCache,
    validator: ModelValidator,
}

impl Hub {
    /// Create a new model hub instance
    pub fn new() -> Self {
        Self {
            registry: ModelRegistry::new(),
            loader: ModelLoader::new(),
            cache: ModelCache::new(),
            validator: ModelValidator::new(),
        }
    }

    /// Load a pretrained model by name and task
    pub async fn load<M, B: Backend<Data = T>, S, T>(
        &self,
        model_name: &str,
        task: ModelTask,
    ) -> Result<LoadedModel<M, B, T>>
    where
        M: Module<B, S, T>,
        B: Backend,
        S: storage::Storage<T>
            + Clone
            + 'static
            + storage::StorageFromVec<T>
            + storage::StorageToDense<T>,
        T: DataType + dtype::FloatExt,
    {
        let config = LoadConfig {
            task,
            force_reload: false,
            validate: true,
        };

        self.loader.load(model_name, config).await
    }

    /// List available models for a given task
    pub fn list_models(&self, task: Option<ModelTask>) -> Vec<&ModelEntry> {
        self.registry.list_models(task)
    }

    /// Get detailed information about a specific model
    pub fn model_info(&self, model_name: &str) -> Option<&ModelEntry> {
        self.registry.get_model(model_name)
    }

    /// Clear the model cache
    pub fn clear_cache(&mut self) -> Result<()> {
        self.cache.clear()
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> cache::CacheStats {
        self.cache.stats()
    }
}

impl Default for Hub {
    fn default() -> Self {
        Self::new()
    }
}

/// Re-export commonly used types for convenience
pub use crate::registry::Task;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hub_creation() {
        let hub = Hub::new();
        assert!(hub.list_models(None).is_empty());
    }

    #[test]
    fn test_cache_operations() {
        let hub = Hub::new();
        let stats = hub.cache_stats();
        assert_eq!(stats.total_entries, 0);
    }
}
