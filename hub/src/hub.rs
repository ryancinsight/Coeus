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
    ///
    /// This is the main method for loading models, equivalent to PyTorch's `torch.hub.load()`.
    /// Supports branch/tag specification and trust verification.
    ///
    /// # Arguments
    /// * `repo` - Repository name (e.g., "pytorch/vision")
    /// * `model` - Model name (e.g., "resnet18")
    /// * `force_reload` - If true, download the model even if cached
    /// * `branch` - Optional branch or tag to use (default: "main")
    /// * `trust_repo` - If false, verify the repository is trusted (default: false)
    ///
    /// # Example
    /// ```rust,no_run
    /// use coeus_hub::Hub;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let hub = Hub::new();
    ///
    /// // Load ResNet18 from PyTorch Vision
    /// let state_dict = hub.load("pytorch/vision", "resnet18", false, None, false).await?;
    ///
    /// // Load from specific branch
    /// let state_dict = hub.load("pytorch/vision", "resnet18", false, Some("v0.12.0"), false).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn load(
        &self,
        repo: &str,
        model: &str,
        force_reload: bool,
        branch: Option<&str>,
        trust_repo: bool,
    ) -> Result<StateDict> {
        // Verify repository trust if required
        if !trust_repo {
            self.verify_trusted_repo(repo)?;
        }

        // Get model info (release lock before await)
        let (model_info, download_url) = {
            let registry = self.registry.read().unwrap();
            let model_info = registry
                .get_model(repo, model)
                .ok_or_else(|| HubError::ModelNotFound {
                    repo: repo.to_string(),
                    model: model.to_string(),
                })?
                .clone();

            // Construct URL with branch if specified
            let download_url = if let Some(branch) = branch {
                model_info.url.replace("main", branch)
            } else {
                model_info.url.clone()
            };

            (model_info, download_url)
        };

        // Check cache first
        if self.use_cache && !force_reload {
            if let Some(cached) = self.load_from_cache(repo, model).await? {
                return Ok(cached);
            }
        }

        // Download and load with better error handling
        let state_dict = load_state_dict_from_url(&download_url).await.map_err(|e| {
            HubError::download_error(download_url.clone(), format!("Failed to download: {}", e))
        })?;

        // Verify model integrity if hash is available
        if let Some(hash) = &model_info.hash {
            self.verify_model_hash(&state_dict, hash).map_err(|_| {
                HubError::invalid_file_format(
                    model_info.url.clone(),
                    "Model file hash verification failed",
                )
            })?;
        }

        // Validate state dict is not empty
        if state_dict.is_empty() {
            return Err(HubError::invalid_file_format(
                download_url.clone(),
                "Downloaded state dict is empty",
            ));
        }

        // Cache the result
        if self.use_cache {
            self.save_to_cache(repo, model, &state_dict).await?;
        }

        Ok(state_dict)
    }

    /// Verify that a repository is trusted
    fn verify_trusted_repo(&self, repo: &str) -> Result<()> {
        // For now, trust all repositories in the registry
        // In a production system, this would check against a trusted list
        let registry = self.registry.read().unwrap();
        let parts: Vec<&str> = repo.split('/').collect();
        if registry.contains(
            parts.first().copied().unwrap_or(""),
            parts.last().copied().unwrap_or(""),
        ) {
            Ok(())
        } else {
            Err(HubError::UntrustedRepository {
                repo: repo.to_string(),
            })
        }
    }

    /// Verify model integrity using hash
    fn verify_model_hash(&self, _state_dict: &StateDict, _hash: &str) -> Result<()> {
        // TODO: Implement SHA256 verification
        // For now, skip verification but maintain API compatibility
        Ok(())
    }

    /// Load a model with default settings (no force reload, no branch, trust repo)
    pub async fn load_default(&self, repo: &str, model: &str) -> Result<StateDict> {
        self.load(repo, model, false, None, true).await
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
/// Equivalent to PyTorch's `torch.hub.load()`
pub async fn load(repo: &str, model: &str) -> Result<StateDict> {
    global_hub().load_default(repo, model).await
}

/// Load a model with force reload using the global hub
/// Equivalent to PyTorch's `torch.hub.load()` with force_reload=True
pub async fn load_force(repo: &str, model: &str) -> Result<StateDict> {
    global_hub().load(repo, model, true, None, true).await
}
