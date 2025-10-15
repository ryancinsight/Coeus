//! Model registry for discovery and metadata management

use crate::error::{HubError, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Task types that models can perform
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Task {
    Classification,
    Detection,
    Segmentation,
    Generation,
    Embedding,
    Other,
}

impl std::fmt::Display for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Task::Classification => write!(f, "classification"),
            Task::Detection => write!(f, "detection"),
            Task::Segmentation => write!(f, "segmentation"),
            Task::Generation => write!(f, "generation"),
            Task::Embedding => write!(f, "embedding"),
            Task::Other => write!(f, "other"),
        }
    }
}

/// Model metadata containing comprehensive information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub description: String,
    pub author: String,
    pub license: String,
    pub parameters: usize,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub dtype: String,
    pub tags: Vec<String>,
    pub paper_url: Option<String>,
    pub code_url: Option<String>,
}

/// Registry entry for a specific model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub version: String, // Store as string for serialization
    pub architecture: String,
    pub task: Task,
    pub metrics: HashMap<String, f32>,
    pub metadata: ModelMetadata,
    pub download_url: String,
    pub checksum: String,
    pub file_size: u64,
}

/// Model registry managing available models
#[derive(Debug)]
pub struct ModelRegistry {
    models: IndexMap<String, ModelEntry>,
    by_task: HashMap<Task, Vec<String>>,
}

impl ModelRegistry {
    /// Create a new empty model registry
    pub fn new() -> Self {
        Self {
            models: IndexMap::new(),
            by_task: HashMap::new(),
        }
    }

    /// Register a new model in the registry
    pub fn register_model(&mut self, entry: ModelEntry) -> Result<()> {
        // Validate the entry
        self.validate_entry(&entry)?;

        let model_id = entry.id.clone();
        let task = entry.task;

        // Insert into main registry
        self.models.insert(model_id.clone(), entry);

        // Update task index
        self.by_task.entry(task).or_default().push(model_id);

        Ok(())
    }

    /// Get a model by its identifier
    pub fn get_model(&self, model_id: &str) -> Option<&ModelEntry> {
        self.models.get(model_id)
    }

    /// Resolve a model name/version to a specific entry
    pub fn resolve(&self, model_spec: &str) -> Result<&ModelEntry> {
        // Parse model specification (name@version or just name)
        let (name, version_req) = if let Some(at_pos) = model_spec.find('@') {
            let name = &model_spec[..at_pos];
            let version_str = &model_spec[at_pos + 1..];
            // For now, just store the version string - full semver parsing can be added later
            (name, Some(version_str.to_string()))
        } else {
            (model_spec, None)
        };

        // Find matching models
        let candidates: Vec<_> = self
            .models
            .values()
            .filter(|entry| entry.name == name)
            .collect();

        if candidates.is_empty() {
            return Err(HubError::ModelNotFound {
                name: name.to_string(),
            });
        }

        // If no version specified, return latest (simplified)
        if version_req.is_none() {
            return Ok(candidates.first().ok_or_else(|| HubError::ModelNotFound {
                name: name.to_string(),
            })?);
        }

        // Find version that matches requirement (simplified string comparison)
        let version_req = version_req.unwrap();
        for candidate in candidates {
            if candidate.version == version_req {
                return Ok(candidate);
            }
        }

        Err(HubError::VersionNotFound {
            name: name.to_string(),
            version: version_req.to_string(),
        })
    }

    /// List all available models, optionally filtered by task
    pub fn list_models(&self, task_filter: Option<Task>) -> Vec<&ModelEntry> {
        match task_filter {
            Some(task) => self
                .by_task
                .get(&task)
                .map(|model_ids| {
                    model_ids
                        .iter()
                        .filter_map(|id| self.models.get(id))
                        .collect()
                })
                .unwrap_or_default(),
            None => self.models.values().collect(),
        }
    }

    /// Get models sorted by popularity (simplified metric)
    pub fn popular_models(&self, limit: usize) -> Vec<&ModelEntry> {
        let mut models: Vec<_> = self.models.values().collect();
        // Sort by parameter count as a simple popularity proxy
        models.sort_by(|a, b| b.metadata.parameters.cmp(&a.metadata.parameters));
        models.into_iter().take(limit).collect()
    }

    /// Search models by name or description
    pub fn search(&self, query: &str) -> Vec<&ModelEntry> {
        let query_lower = query.to_lowercase();
        self.models
            .values()
            .filter(|entry| {
                entry.name.to_lowercase().contains(&query_lower)
                    || entry
                        .metadata
                        .description
                        .to_lowercase()
                        .contains(&query_lower)
                    || entry
                        .metadata
                        .tags
                        .iter()
                        .any(|tag| tag.to_lowercase().contains(&query_lower))
            })
            .collect()
    }

    /// Get statistics about the registry
    pub fn stats(&self) -> RegistryStats {
        let total_models = self.models.len();
        let models_by_task: HashMap<Task, usize> = self
            .by_task
            .iter()
            .map(|(task, models)| (*task, models.len()))
            .collect();

        let total_parameters: usize = self
            .models
            .values()
            .map(|entry| entry.metadata.parameters)
            .sum();

        RegistryStats {
            total_models,
            models_by_task,
            total_parameters,
        }
    }

    /// Validate a model entry before registration
    fn validate_entry(&self, entry: &ModelEntry) -> Result<()> {
        // Check required fields
        if entry.name.is_empty() {
            return Err(HubError::InvalidMetadata {
                field: "name".to_string(),
                reason: "cannot be empty".to_string(),
            });
        }

        if entry.download_url.is_empty() {
            return Err(HubError::InvalidMetadata {
                field: "download_url".to_string(),
                reason: "cannot be empty".to_string(),
            });
        }

        if entry.metadata.parameters == 0 {
            return Err(HubError::InvalidMetadata {
                field: "parameters".to_string(),
                reason: "must be greater than zero".to_string(),
            });
        }

        // Check for duplicate IDs
        if self.models.contains_key(&entry.id) {
            return Err(HubError::RegistryError {
                message: format!("Model with ID '{}' already exists", entry.id),
            });
        }

        Ok(())
    }
}

/// Registry statistics
#[derive(Debug, Clone)]
pub struct RegistryStats {
    pub total_models: usize,
    pub models_by_task: HashMap<Task, usize>,
    pub total_parameters: usize,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_entry(name: &str, task: Task) -> ModelEntry {
        ModelEntry {
            id: name.to_string(),
            name: name.to_string(),
            version: "1.0.0".to_string(),
            architecture: "TestArch".to_string(),
            task,
            metrics: HashMap::from([("accuracy".to_string(), 0.95)]),
            metadata: ModelMetadata {
                description: format!("Test {} model", name),
                author: "Test Author".to_string(),
                license: "MIT".to_string(),
                parameters: 1000000,
                input_shape: vec![224, 224, 3],
                output_shape: vec![1000],
                dtype: "f32".to_string(),
                tags: vec!["test".to_string()],
                paper_url: None,
                code_url: None,
            },
            download_url: format!("https://example.com/{}.bin", name),
            checksum: "abcd1234".to_string(),
            file_size: 1024000,
        }
    }

    #[test]
    fn test_registry_operations() {
        let mut registry = ModelRegistry::new();

        // Register models
        let resnet_entry = create_test_entry("resnet50", Task::Classification);
        let bert_entry = create_test_entry("bert-base", Task::Embedding);

        registry.register_model(resnet_entry).unwrap();
        registry.register_model(bert_entry).unwrap();

        assert_eq!(registry.list_models(None).len(), 2);

        // Test resolution
        let resolved = registry.resolve("resnet50").unwrap();
        assert_eq!(resolved.name, "resnet50");

        // Test task filtering
        let classification_models = registry.list_models(Some(Task::Classification));
        assert_eq!(classification_models.len(), 1);
        assert_eq!(classification_models[0].name, "resnet50");

        // Test search
        let search_results = registry.search("resnet");
        assert_eq!(search_results.len(), 1);
        assert_eq!(search_results[0].name, "resnet50");
    }

    #[test]
    fn test_registry_validation() {
        let mut registry = ModelRegistry::new();

        // Test invalid entry
        let invalid_entry = ModelEntry {
            id: "invalid".to_string(),
            name: "".to_string(), // Empty name should fail
            version: "1.0.0".to_string(),
            architecture: "Test".to_string(),
            task: Task::Classification,
            metrics: HashMap::new(),
            metadata: ModelMetadata {
                description: "Test".to_string(),
                author: "Test".to_string(),
                license: "MIT".to_string(),
                parameters: 1000,
                input_shape: vec![224, 224, 3],
                output_shape: vec![1000],
                dtype: "f32".to_string(),
                tags: vec![],
                paper_url: None,
                code_url: None,
            },
            download_url: "https://example.com/test.bin".to_string(),
            checksum: "test".to_string(),
            file_size: 1000,
        };

        assert!(registry.register_model(invalid_entry).is_err());
    }

    #[test]
    fn test_registry_stats() {
        let mut registry = ModelRegistry::new();

        registry
            .register_model(create_test_entry("model1", Task::Classification))
            .unwrap();
        registry
            .register_model(create_test_entry("model2", Task::Generation))
            .unwrap();

        let stats = registry.stats();
        assert_eq!(stats.total_models, 2);
        assert_eq!(stats.models_by_task.get(&Task::Classification), Some(&1));
        assert_eq!(stats.models_by_task.get(&Task::Generation), Some(&1));
        assert_eq!(stats.total_parameters, 2000000);
    }
}
