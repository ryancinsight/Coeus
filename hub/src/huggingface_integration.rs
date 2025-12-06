//! HuggingFace Hub Integration for Coeus
//!
//! This module provides comprehensive integration with the HuggingFace Hub,
//! enabling seamless model sharing, discovery, and deployment.

use crate::error::{HubError, Result};
use crate::models::{ModelInfo, ModelMetadata};
use crate::{Cache, Registry, Validator};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use reqwest::Client;

/// Configuration for HuggingFace Hub integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceConfig {
    /// API token for authentication
    pub api_token: Option<String>,
    /// Base URL for the Hub API
    pub api_base_url: String,
    /// Cache directory for downloaded models
    pub cache_dir: PathBuf,
    /// Default organization for uploads
    pub default_org: Option<String>,
    /// Enable automatic model conversion
    pub auto_convert: bool,
    /// Enable model validation before upload
    pub validate_before_upload: bool,
}

impl Default for HuggingFaceConfig {
    fn default() -> Self {
        Self {
            api_token: None,
            api_base_url: "https://huggingface.co".to_string(),
            cache_dir: PathBuf::from("./hf_cache"),
            default_org: None,
            auto_convert: true,
            validate_before_upload: true,
        }
    }
}

/// HuggingFace Hub client for model operations
pub struct HuggingFaceHub {
    /// HTTP client
    client: Client,
    /// Configuration
    config: HuggingFaceConfig,
    /// Local cache
    cache: Cache,
    /// Model registry
    registry: Registry,
    /// Model validator
    validator: Validator,
}

impl HuggingFaceHub {
    /// Create new HuggingFace Hub client
    pub fn new(config: HuggingFaceConfig) -> Result<Self> {
        let client = Client::new();

        // Create cache directory if it doesn't exist
        if !config.cache_dir.exists() {
            std::fs::create_dir_all(&config.cache_dir)?;
        }

        let cache = Cache::new(config.cache_dir.clone());
        let registry = Registry::new();
        let validator = Validator::new();

        Ok(Self {
            client,
            config,
            cache,
            registry,
            validator,
        })
    }

    /// Download a model from the Hub
    pub async fn download_model(&self, model_id: &str, revision: Option<&str>) -> Result<ModelInfo> {
        println!("📥 Downloading model: {}", model_id);

        // Get model metadata first
        let metadata = self.get_model_info(model_id, revision).await?;

        // Check if model is already cached
        if let Some(cached_model) = self.cache.get_model(model_id, revision)? {
            println!("✅ Using cached model: {}", model_id);
            return Ok(cached_model);
        }

        // Download model files
        let model_files = self.download_model_files(&metadata).await?;

        // Validate downloaded model
        self.validator.validate_model(&model_files)?;

        // Convert to Coeus format if needed
        let converted_model = if self.config.auto_convert {
            self.convert_to_coeus_format(&model_files).await?
        } else {
            model_files
        };

        // Cache the model
        self.cache.store_model(model_id, revision, &converted_model)?;

        // Register the model
        self.registry.register_model(&converted_model)?;

        println!("✅ Successfully downloaded and cached model: {}", model_id);
        Ok(converted_model)
    }

    /// Upload a model to the Hub
    pub async fn upload_model(&self, model_info: &ModelInfo, private: bool) -> Result<String> {
        println!("📤 Uploading model: {}", model_info.id);

        // Validate model before upload
        if self.config.validate_before_upload {
            self.validator.validate_model(model_info)?;
        }

        // Convert to Hub format
        let hub_model = self.convert_from_coeus_format(model_info).await?;

        // Create model repository
        let repo_id = self.create_model_repo(&model_info.id, private).await?;

        // Upload model files
        self.upload_model_files(&repo_id, &hub_model).await?;

        // Update model card
        self.update_model_card(&repo_id, model_info).await?;

        println!("✅ Successfully uploaded model: {}", repo_id);
        Ok(repo_id)
    }

    /// Search for models on the Hub
    pub async fn search_models(&self, query: &str, filter: Option<SearchFilter>) -> Result<Vec<ModelInfo>> {
        println!("🔍 Searching models: {}", query);

        let search_url = format!("{}/api/models", self.api_base_url);
        let mut params = HashMap::new();
        params.insert("search".to_string(), query.to_string());

        if let Some(filter) = filter {
            if let Some(task) = filter.task {
                params.insert("pipeline_tag".to_string(), task);
            }
            if let Some(library) = filter.library {
                params.insert("library".to_string(), library);
            }
            if let Some(language) = filter.language {
                params.insert("language".to_string(), language);
            }
        }

        let response = self.client
            .get(&search_url)
            .query(&params)
            .send()
            .await
            .map_err(|e| HubError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(HubError::ApiError(format!("Search failed: {}", response.status())));
        }

        let search_results: Vec<HuggingFaceModelInfo> = response
            .json()
            .await
            .map_err(|e| HubError::ParseError(e.to_string()))?;

        // Convert to Coeus ModelInfo format
        let mut results = Vec::new();
        for hf_model in search_results {
            let model_info = self.convert_hf_to_coeus_model(&hf_model)?;
            results.push(model_info);
        }

        println!("✅ Found {} models matching query", results.len());
        Ok(results)
    }

    /// Get model information from the Hub
    pub async fn get_model_info(&self, model_id: &str, revision: Option<&str>) -> Result<ModelMetadata> {
        let revision = revision.unwrap_or("main");
        let info_url = format!("{}/api/models/{}/revision/{}", self.api_base_url, model_id, revision);

        let response = self.client
            .get(&info_url)
            .send()
            .await
            .map_err(|e| HubError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(HubError::ApiError(format!("Failed to get model info: {}", response.status())));
        }

        let hf_model: HuggingFaceModelInfo = response
            .json()
            .await
            .map_err(|e| HubError::ParseError(e.to_string()))?;

        Ok(ModelMetadata {
            id: hf_model.id,
            author: hf_model.author,
            sha: hf_model.sha,
            created_at: hf_model.created_at,
            last_modified: hf_model.last_modified,
            private: hf_model.private,
            disabled: hf_model.disabled,
            downloads: hf_model.downloads,
            likes: hf_model.likes,
            library_name: hf_model.library_name,
            tags: hf_model.tags,
            pipeline_tag: hf_model.pipeline_tag,
            mask_token: hf_model.mask_token,
            card_data: hf_model.card_data,
            widget_data: hf_model.widget_data,
            model_index: hf_model.model_index,
            config: hf_model.config,
            transformers_info: hf_model.transformers_info,
            siblings: hf_model.siblings.into_iter().map(|s| crate::models::Sibling {
                rfilename: s.rfilename,
            }).collect(),
        })
    }

    /// List user's models
    pub async fn list_user_models(&self, username: &str) -> Result<Vec<ModelInfo>> {
        let list_url = format!("{}/api/models", self.api_base_url);
        let params = &[("author", username)];

        let response = self.client
            .get(&list_url)
            .query(params)
            .send()
            .await
            .map_err(|e| HubError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(HubError::ApiError(format!("Failed to list models: {}", response.status())));
        }

        let models: Vec<HuggingFaceModelInfo> = response
            .json()
            .await
            .map_err(|e| HubError::ParseError(e.to_string()))?;

        let mut results = Vec::new();
        for hf_model in models {
            let model_info = self.convert_hf_to_coeus_model(&hf_model)?;
            results.push(model_info);
        }

        Ok(results)
    }

    /// Delete a model from the Hub
    pub async fn delete_model(&self, model_id: &str) -> Result<()> {
        println!("🗑️  Deleting model: {}", model_id);

        let delete_url = format!("{}/api/models/{}", self.api_base_url, model_id);

        let response = self.client
            .delete(&delete_url)
            .header("Authorization", format!("Bearer {}", self.config.api_token.as_deref().unwrap_or("")))
            .send()
            .await
            .map_err(|e| HubError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(HubError::ApiError(format!("Failed to delete model: {}", response.status())));
        }

        // Remove from local cache
        self.cache.remove_model(model_id)?;

        println!("✅ Successfully deleted model: {}", model_id);
        Ok(())
    }

    /// Get model metrics and statistics
    pub async fn get_model_stats(&self, model_id: &str) -> Result<ModelStats> {
        let stats_url = format!("{}/api/models/{}/stats", self.api_base_url, model_id);

        let response = self.client
            .get(&stats_url)
            .send()
            .await
            .map_err(|e| HubError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(HubError::ApiError(format!("Failed to get stats: {}", response.status())));
        }

        let stats: ModelStats = response
            .json()
            .await
            .map_err(|e| HubError::ParseError(e.to_string()))?;

        Ok(stats)
    }

    // Private helper methods

    async fn download_model_files(&self, metadata: &ModelMetadata) -> Result<ModelInfo> {
        let mut model_files = Vec::new();

        for sibling in &metadata.siblings {
            let file_url = format!("{}/api/models/{}/raw/main/{}", self.api_base_url, metadata.id, sibling.rfilename);

            let response = self.client
                .get(&file_url)
                .send()
                .await
                .map_err(|e| HubError::NetworkError(e.to_string()))?;

            if !response.status().is_success() {
                return Err(HubError::DownloadError(format!("Failed to download {}: {}", sibling.rfilename, response.status())));
            }

            let content = response
                .bytes()
                .await
                .map_err(|e| HubError::NetworkError(e.to_string()))?;

            model_files.push(crate::models::ModelFile {
                filename: sibling.rfilename.clone(),
                content: content.to_vec(),
            });
        }

        Ok(ModelInfo {
            id: metadata.id.clone(),
            metadata: metadata.clone(),
            files: model_files,
        })
    }

    async fn convert_to_coeus_format(&self, model_info: &ModelInfo) -> Result<ModelInfo> {
        // This would implement conversion from HuggingFace format to Coeus format
        // For now, return as-is
        Ok(model_info.clone())
    }

    async fn convert_from_coeus_format(&self, model_info: &ModelInfo) -> Result<ModelInfo> {
        // This would implement conversion from Coeus format to HuggingFace format
        // For now, return as-is
        Ok(model_info.clone())
    }

    async fn create_model_repo(&self, model_id: &str, private: bool) -> Result<String> {
        let create_url = format!("{}/api/repos/create", self.api_base_url);

        let mut payload = HashMap::new();
        payload.insert("name", model_id);
        payload.insert("type", "model");
        payload.insert("private", &private.to_string());

        if let Some(org) = &self.config.default_org {
            payload.insert("organization", org);
        }

        let response = self.client
            .post(&create_url)
            .header("Authorization", format!("Bearer {}", self.config.api_token.as_deref().unwrap_or("")))
            .json(&payload)
            .send()
            .await
            .map_err(|e| HubError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(HubError::ApiError(format!("Failed to create repo: {}", response.status())));
        }

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| HubError::ParseError(e.to_string()))?;

        let repo_id = result["url"]
            .as_str()
            .ok_or_else(|| HubError::ParseError("Invalid repo creation response".to_string()))?
            .split('/')
            .last()
            .ok_or_else(|| HubError::ParseError("Invalid repo URL".to_string()))?;

        Ok(repo_id.to_string())
    }

    async fn upload_model_files(&self, repo_id: &str, model_info: &ModelInfo) -> Result<()> {
        for file in &model_info.files {
            let upload_url = format!("{}/api/models/{}/upload/main/{}", self.api_base_url, repo_id, file.filename);

            let response = self.client
                .post(&upload_url)
                .header("Authorization", format!("Bearer {}", self.config.api_token.as_deref().unwrap_or("")))
                .body(file.content.clone())
                .send()
                .await
                .map_err(|e| HubError::NetworkError(e.to_string()))?;

            if !response.status().is_success() {
                return Err(HubError::UploadError(format!("Failed to upload {}: {}", file.filename, response.status())));
            }
        }

        Ok(())
    }

    async fn update_model_card(&self, repo_id: &str, model_info: &ModelInfo) -> Result<()> {
        let card_url = format!("{}/api/models/{}/upload/main/README.md", self.api_base_url, repo_id);

        let model_card = self.generate_model_card(model_info);

        let response = self.client
            .post(&card_url)
            .header("Authorization", format!("Bearer {}", self.config.api_token.as_deref().unwrap_or("")))
            .body(model_card)
            .send()
            .await
            .map_err(|e| HubError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(HubError::UploadError(format!("Failed to upload model card: {}", response.status())));
        }

        Ok(())
    }

    fn generate_model_card(&self, model_info: &ModelInfo) -> String {
        format!(
            "# {}\n\n\
            Model converted and uploaded using Coeus.\n\n\
            ## Model Details\n\
            - **Model ID**: {}\n\
            - **Library**: coeus\n\
            - **Tags**: {}\n\
            - **Uploaded**: {}\n\n\
            ## Usage\n\
            ```python\n\
            from coeus import load_model\n\
            model = load_model(\"{}\")\n\
            ```",
            model_info.id,
            model_info.id,
            model_info.metadata.tags.join(", "),
            chrono::Utc::now().format("%Y-%m-%d"),
            model_info.id
        )
    }

    fn convert_hf_to_coeus_model(&self, hf_model: &HuggingFaceModelInfo) -> Result<ModelInfo> {
        let metadata = ModelMetadata {
            id: hf_model.id.clone(),
            author: hf_model.author.clone(),
            sha: hf_model.sha.clone(),
            created_at: hf_model.created_at.clone(),
            last_modified: hf_model.last_modified.clone(),
            private: hf_model.private,
            disabled: hf_model.disabled,
            downloads: hf_model.downloads,
            likes: hf_model.likes,
            library_name: hf_model.library_name.clone(),
            tags: hf_model.tags.clone(),
            pipeline_tag: hf_model.pipeline_tag.clone(),
            mask_token: hf_model.mask_token.clone(),
            card_data: hf_model.card_data.clone(),
            widget_data: hf_model.widget_data.clone(),
            model_index: hf_model.model_index.clone(),
            config: hf_model.config.clone(),
            transformers_info: hf_model.transformers_info.clone(),
            siblings: hf_model.siblings.iter().map(|s| crate::models::Sibling {
                rfilename: s.rfilename.clone(),
            }).collect(),
        };

        Ok(ModelInfo {
            id: hf_model.id.clone(),
            metadata,
            files: Vec::new(), // Files would be downloaded separately
        })
    }
}

/// Search filter for model discovery
#[derive(Debug, Clone, Default)]
pub struct SearchFilter {
    /// Pipeline tag (task type)
    pub task: Option<String>,
    /// Library name
    pub library: Option<String>,
    /// Language
    pub language: Option<String>,
}

/// Model statistics from the Hub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStats {
    /// Number of downloads
    pub downloads: u64,
    /// Number of likes
    pub likes: u64,
    /// Recent download trends
    pub download_trends: HashMap<String, u64>,
}

/// HuggingFace model information (API response format)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HuggingFaceModelInfo {
    pub id: String,
    pub author: String,
    pub sha: String,
    pub created_at: String,
    pub last_modified: String,
    pub private: bool,
    pub disabled: bool,
    pub downloads: u64,
    pub likes: u64,
    pub library_name: Option<String>,
    pub tags: Vec<String>,
    pub pipeline_tag: Option<String>,
    pub mask_token: Option<String>,
    pub card_data: Option<serde_json::Value>,
    pub widget_data: Option<serde_json::Value>,
    pub model_index: Option<serde_json::Value>,
    pub config: Option<serde_json::Value>,
    pub transformers_info: Option<serde_json::Value>,
    pub siblings: Vec<HuggingFaceSibling>,
}

/// HuggingFace sibling file information
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HuggingFaceSibling {
    pub rfilename: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huggingface_config() {
        let config = HuggingFaceConfig::default();
        assert_eq!(config.api_base_url, "https://huggingface.co");
        assert!(config.auto_convert);
        assert!(config.validate_before_upload);
    }

    #[test]
    fn test_search_filter() {
        let filter = SearchFilter {
            task: Some("text-classification".to_string()),
            library: Some("transformers".to_string()),
            language: Some("en".to_string()),
        };

        assert_eq!(filter.task.as_deref(), Some("text-classification"));
        assert_eq!(filter.library.as_deref(), Some("transformers"));
        assert_eq!(filter.language.as_deref(), Some("en"));
    }

    #[tokio::test]
    async fn test_hub_creation() {
        let config = HuggingFaceConfig::default();
        let hub = HuggingFaceHub::new(config);
        assert!(hub.is_ok());
    }

    #[test]
    fn test_model_card_generation() {
        let config = HuggingFaceConfig::default();
        let hub = HuggingFaceHub::new(config).unwrap();

        // Create a mock model info for testing
        let model_info = ModelInfo {
            id: "test-model".to_string(),
            metadata: ModelMetadata {
                id: "test-model".to_string(),
                author: "test-author".to_string(),
                sha: "abc123".to_string(),
                created_at: "2023-01-01T00:00:00Z".to_string(),
                last_modified: "2023-01-01T00:00:00Z".to_string(),
                private: false,
                disabled: false,
                downloads: 100,
                likes: 10,
                library_name: Some("coeus".to_string()),
                tags: vec!["test".to_string(), "classification".to_string()],
                pipeline_tag: Some("text-classification".to_string()),
                mask_token: None,
                card_data: None,
                widget_data: None,
                model_index: None,
                config: None,
                transformers_info: None,
                siblings: vec![],
            },
            files: vec![],
        };

        let card = hub.generate_model_card(&model_info);
        assert!(card.contains("# test-model"));
        assert!(card.contains("Model converted and uploaded using Coeus"));
        assert!(card.contains("from coeus import load_model"));
    }
}
