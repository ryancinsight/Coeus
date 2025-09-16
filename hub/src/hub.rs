//! Main Hub interface for model loading and management

use crate::{
    default_cache_dir, load_state_dict_from_url, pytorch_registry, HubError, ModelRegistry, Result,
    StateDict,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Main Hub interface for loading and managing pre-trained models
#[derive(Clone)]
pub struct Hub {
    /// Model registry
    registry: Arc<RwLock<ModelRegistry>>,
    /// Cache directory
    cache_dir: PathBuf,
    /// Whether to use cache
    use_cache: bool,
}

impl Hub {
    /// Create a new hub with default settings
    pub fn new() -> Self {
        Self {
            registry: Arc::new(RwLock::new(pytorch_registry())),
            cache_dir: default_cache_dir(),
            use_cache: true,
        }
    }

    /// Create a hub with custom cache directory
    pub fn with_cache_dir<P: AsRef<Path>>(cache_dir: P) -> Self {
        Self {
            registry: Arc::new(RwLock::new(pytorch_registry())),
            cache_dir: cache_dir.as_ref().to_path_buf(),
            use_cache: true,
        }
    }

    /// Create a hub with custom registry
    pub fn with_registry(registry: ModelRegistry) -> Self {
        Self {
            registry: Arc::new(RwLock::new(registry)),
            cache_dir: default_cache_dir(),
            use_cache: true,
        }
    }

    /// Initialize with default configuration
    pub fn init_default() -> Result<()> {
        let hub = Self::new();
        std::fs::create_dir_all(&hub.cache_dir)?;
        Ok(())
    }

    /// Load a model from the hub
    pub async fn load(&self, repo: &str, model: &str, force_reload: bool) -> Result<StateDict> {
        // Get model info (release lock before await)
        let model_info = {
            let registry = self.registry.read().unwrap();
            registry
                .get_model(repo, model)
                .ok_or_else(|| HubError::ModelNotFound {
                    repo: repo.to_string(),
                    model: model.to_string(),
                })?
                .clone()
        };

        // Check cache first
        if self.use_cache && !force_reload {
            if let Some(cached) = self.load_from_cache(repo, model).await? {
                return Ok(cached);
            }
        }

        // Download and load
        let state_dict = load_state_dict_from_url(&model_info.url).await?;

        // Cache the result
        if self.use_cache {
            self.save_to_cache(repo, model, &state_dict).await?;
        }

        Ok(state_dict)
    }

    /// Load a model with default settings (no force reload)
    pub async fn load_default(&self, repo: &str, model: &str) -> Result<StateDict> {
        self.load(repo, model, false).await
    }

    /// List available models in a repository
    pub fn list_models(&self, repo: &str) -> Vec<String> {
        let registry = self.registry.read().unwrap();
        registry
            .get_repo_models(repo)
            .map(|models| models.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// List all repositories
    pub fn list_repos(&self) -> Vec<String> {
        let registry = self.registry.read().unwrap();
        registry.repos().cloned().collect()
    }

    /// Get model information
    pub fn model_info(&self, repo: &str, model: &str) -> Option<crate::registry::ModelInfo> {
        let registry = self.registry.read().unwrap();
        registry.get_model(repo, model).cloned()
    }

    /// Check if a model exists
    pub fn has_model(&self, repo: &str, model: &str) -> bool {
        let registry = self.registry.read().unwrap();
        registry.contains(repo, model)
    }

    /// Clear cache for a specific model
    pub fn clear_cache(&self, repo: &str, model: &str) -> Result<()> {
        let cache_path = self.cache_path(repo, model);
        if cache_path.exists() {
            std::fs::remove_file(cache_path)?;
        }
        Ok(())
    }

    /// Clear entire cache
    pub fn clear_all_cache(&self) -> Result<()> {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir)?;
            std::fs::create_dir_all(&self.cache_dir)?;
        }
        Ok(())
    }

    /// Set cache directory
    pub fn set_cache_dir<P: AsRef<Path>>(&mut self, cache_dir: P) {
        self.cache_dir = cache_dir.as_ref().to_path_buf();
    }

    /// Enable/disable caching
    pub fn set_cache_enabled(&mut self, enabled: bool) {
        self.use_cache = enabled;
    }

    /// Get cache directory
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Load from cache
    async fn load_from_cache(&self, repo: &str, model: &str) -> Result<Option<StateDict>> {
        let cache_path = self.cache_path(repo, model);
        if cache_path.exists() {
            match crate::load_state_dict(&cache_path) {
                Ok(state_dict) => Ok(Some(state_dict)),
                Err(_) => {
                    // Cache file corrupted, remove it
                    let _ = std::fs::remove_file(&cache_path);
                    Ok(None)
                }
            }
        } else {
            Ok(None)
        }
    }

    /// Save to cache
    async fn save_to_cache(&self, repo: &str, model: &str, state_dict: &StateDict) -> Result<()> {
        let cache_path = self.cache_path(repo, model);

        // Create cache directory if needed
        if let Some(parent) = cache_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        crate::save_state_dict(state_dict, &cache_path)
    }

    /// Get cache path for a model
    fn cache_path(&self, repo: &str, model: &str) -> PathBuf {
        self.cache_dir.join(repo).join(format!("{}.pth", model))
    }

    /// Add a custom model to the registry
    pub fn add_model(&self, model_info: crate::registry::ModelInfo) {
        let mut registry = self.registry.write().unwrap();
        registry.add_model(model_info);
    }

    /// Remove a model from the registry
    pub fn remove_model(&self, repo: &str, model: &str) -> bool {
        let mut registry = self.registry.write().unwrap();
        registry.remove_model(repo, model)
    }
}

impl Default for Hub {
    fn default() -> Self {
        Self::new()
    }
}

/// Global hub instance using thread-safe lazy initialization
use std::sync::OnceLock;

static GLOBAL_HUB: OnceLock<Hub> = OnceLock::new();

/// Get the global hub instance
pub fn global_hub() -> &'static Hub {
    GLOBAL_HUB.get_or_init(Hub::new)
}

/// Load a model using the global hub
pub async fn load(repo: &str, model: &str) -> Result<StateDict> {
    global_hub().load_default(repo, model).await
}

/// Load a model with force reload using the global hub
pub async fn load_force(repo: &str, model: &str) -> Result<StateDict> {
    global_hub().load(repo, model, true).await
}
