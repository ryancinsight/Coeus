//! Model registry for managing available models

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Information about a model in the registry
#[derive(Clone, Debug, Serialize, Deserialize)]
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

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Built-in PyTorch models registry
pub fn pytorch_registry() -> ModelRegistry {
    let mut registry = ModelRegistry::new();

    // Add some common PyTorch models
    // ResNet models
    registry.add_model(ModelInfo::new(
        "resnet18".to_string(),
        "pytorch/vision".to_string(),
        "ResNet-18 model".to_string(),
        format!("{}resnet18-5c106cde.pth", crate::PYTORCH_HUB_URL),
    ));

    registry.add_model(ModelInfo::new(
        "resnet34".to_string(),
        "pytorch/vision".to_string(),
        "ResNet-34 model".to_string(),
        format!("{}resnet34-333f7ec4.pth", crate::PYTORCH_HUB_URL),
    ));

    registry.add_model(ModelInfo::new(
        "resnet50".to_string(),
        "pytorch/vision".to_string(),
        "ResNet-50 model".to_string(),
        format!("{}resnet50-19c8e357.pth", crate::PYTORCH_HUB_URL),
    ));

    // VGG models
    registry.add_model(ModelInfo::new(
        "vgg16".to_string(),
        "pytorch/vision".to_string(),
        "VGG-16 model".to_string(),
        format!("{}vgg16-397923af.pth", crate::PYTORCH_HUB_URL),
    ));

    registry.add_model(ModelInfo::new(
        "vgg19".to_string(),
        "pytorch/vision".to_string(),
        "VGG-19 model".to_string(),
        format!("{}vgg19-dcbb9e9d.pth", crate::PYTORCH_HUB_URL),
    ));

    registry
}
