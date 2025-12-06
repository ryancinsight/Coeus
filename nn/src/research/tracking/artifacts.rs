//! Experiment Artifact Storage System
//!
//! This module provides comprehensive artifact management for research experiments,
//! including data, models, visualizations, and research documents with automatic
//! versioning, compression, and retention policies.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};

/// Artifact storage and management system
#[derive(Debug, Clone, serde::Serialize)]
pub struct ArtifactStorage {
    /// Stored artifacts by ID
    pub artifacts: HashMap<String, Artifact>,
    /// Storage configuration
    storage_config: ArtifactStorageConfig,
    /// Storage statistics
    pub stats: ArtifactStorageStats,
}

impl ArtifactStorage {
    /// Create new artifact storage
    pub fn new() -> Self {
        Self {
            artifacts: HashMap::new(),
            storage_config: ArtifactStorageConfig::default(),
            stats: ArtifactStorageStats::default(),
        }
    }

    /// Store artifact with automatic metadata collection
    pub fn store_artifact(&mut self, name: String, artifact_type: ArtifactType, data: Vec<u8>) -> crate::error::Result<String> {
        let artifact_id = format!("{}_{}", name, chrono::Utc::now().timestamp());
        let size_bytes = data.len() as u64;

        // Check storage limits
        if !self.check_storage_limits(size_bytes) {
            return Err(crate::error::NNError::InvalidConfiguration {
                message: "Storage limit exceeded for artifacts".to_string(),
            });
        }

        // Create artifact metadata
        let content_type = self.infer_content_type(&artifact_type);
        let mut artifact = Artifact {
            id: artifact_id.clone(),
            name,
            artifact_type,
            content_type,
            size_bytes,
            checksum: self.calculate_checksum(&data),
            created_at: chrono::Utc::now(),
            description: None,
            tags: Vec::new(),
            metadata: HashMap::new(),
            data: if self.storage_config.compress_large_artifacts && size_bytes > 1024 * 1024 {
                self.compress_data(data)?
            } else {
                data
            },
            compressed: self.storage_config.compress_large_artifacts && size_bytes > 1024 * 1024,
            version: "1.0.0".to_string(),
            dependencies: Vec::new(),
        };

        // Update statistics
        self.stats.total_artifacts += 1;
        self.stats.total_size_bytes += artifact.size_bytes;
        *self.stats.by_type.entry(artifact.artifact_type.clone()).or_insert(0) += 1;

        // Store artifact
        self.artifacts.insert(artifact_id.clone(), artifact);

        // Clean up old artifacts if needed
        self.cleanup_old_artifacts()?;

        Ok(artifact_id)
    }

    /// Store artifact from file path
    pub fn store_artifact_from_file(&mut self, name: String, artifact_type: ArtifactType, file_path: &Path) -> crate::error::Result<String> {
        match std::fs::read(file_path) {
            Ok(data) => self.store_artifact(name, artifact_type, data),
            Err(e) => Err(crate::error::NNError::IoError { error: e }),
        }
    }

    /// Retrieve artifact by ID
    pub fn get_artifact(&self, id: &str) -> Option<&Artifact> {
        self.artifacts.get(id)
    }

    /// Get artifact data (decompressing if necessary)
    pub fn get_artifact_data(&self, id: &str) -> crate::error::Result<Vec<u8>> {
        if let Some(artifact) = self.artifacts.get(id) {
            if artifact.compressed {
                self.decompress_data(&artifact.data)
            } else {
                Ok(artifact.data.clone())
            }
        } else {
            Err(crate::error::NNError::NotFound {
                resource: id.to_string(),
            })
        }
    }

    /// List all artifacts
    pub fn list_artifacts(&self) -> Vec<&Artifact> {
        self.artifacts.values().collect()
    }

    /// List artifacts by type
    pub fn list_artifacts_by_type(&self, artifact_type: &ArtifactType) -> Vec<&Artifact> {
        self.artifacts.values()
            .filter(|a| &a.artifact_type == artifact_type)
            .collect()
    }

    /// List artifacts by tags
    pub fn list_artifacts_by_tags(&self, tags: &[String]) -> Vec<&Artifact> {
        self.artifacts.values()
            .filter(|a| tags.iter().any(|tag| a.tags.contains(tag)))
            .collect()
    }

    /// Delete artifact
    pub fn delete_artifact(&mut self, id: &str) -> crate::error::Result<()> {
        if let Some(artifact) = self.artifacts.remove(id) {
            // Update statistics
            self.stats.total_artifacts -= 1;
            self.stats.total_size_bytes -= artifact.size_bytes;
            if let Some(count) = self.stats.by_type.get_mut(&artifact.artifact_type) {
                *count -= 1;
            }
            Ok(())
        } else {
            Err(crate::error::NNError::NotFound {
                resource: id.to_string(),
            })
        }
    }

    /// Search artifacts by name pattern
    pub fn search_artifacts(&self, pattern: &str) -> Vec<&Artifact> {
        self.artifacts.values()
            .filter(|a| a.name.contains(pattern) || a.description.as_ref().map_or(false, |d| d.contains(pattern)))
            .collect()
    }

    /// Export artifact to file
    pub fn export_artifact(&self, id: &str, file_path: &Path) -> crate::error::Result<()> {
        let data = self.get_artifact_data(id)?;
        std::fs::write(file_path, data).map_err(|e| crate::error::NNError::IoError { error: e })
    }

    /// Create artifact bundle (zip multiple artifacts)
    pub fn create_artifact_bundle(&mut self, bundle_name: String, artifact_ids: Vec<String>) -> crate::error::Result<String> {
        let mut bundle_data = Vec::new();

        // Create a simple tar-like format (could be enhanced with actual compression)
        for id in artifact_ids {
            if let Some(artifact) = self.artifacts.get(&id) {
                let data = self.get_artifact_data(&id)?;
                // Store as: id_length (u32) + id + data_length (u64) + data + metadata_length (u32) + metadata_json
                let id_bytes = id.as_bytes();
                let metadata_json = serde_json::to_string(&artifact.metadata).unwrap_or_default();
                let metadata_bytes = metadata_json.as_bytes();

                bundle_data.extend_from_slice(&(id_bytes.len() as u32).to_be_bytes());
                bundle_data.extend_from_slice(id_bytes);
                bundle_data.extend_from_slice(&(data.len() as u64).to_be_bytes());
                bundle_data.extend_from_slice(&data);
                bundle_data.extend_from_slice(&(metadata_bytes.len() as u32).to_be_bytes());
                bundle_data.extend_from_slice(metadata_bytes);
            }
        }

        self.store_artifact(bundle_name, ArtifactType::Bundle, bundle_data)
    }

    /// Tag artifact
    pub fn tag_artifact(&mut self, id: &str, tag: String) -> crate::error::Result<()> {
        if let Some(artifact) = self.artifacts.get_mut(id) {
            if !artifact.tags.contains(&tag) {
                artifact.tags.push(tag);
            }
            Ok(())
        } else {
            Err(crate::error::NNError::NotFound {
                resource: id.to_string(),
            })
        }
    }

    /// Add metadata to artifact
    pub fn add_artifact_metadata(&mut self, id: &str, key: String, value: serde_json::Value) -> crate::error::Result<()> {
        if let Some(artifact) = self.artifacts.get_mut(id) {
            artifact.metadata.insert(key, value);
            Ok(())
        } else {
            Err(crate::error::NNError::NotFound {
                resource: id.to_string(),
            })
        }
    }

    /// Get storage statistics
    pub fn get_storage_stats(&self) -> &ArtifactStorageStats {
        &self.stats
    }

    /// Check if storage limits would be exceeded
    fn check_storage_limits(&self, additional_size: u64) -> bool {
        self.stats.total_artifacts < self.storage_config.max_artifacts &&
        self.stats.total_size_bytes + additional_size <= self.storage_config.max_total_size_bytes
    }

    /// Clean up old artifacts based on retention policy
    fn cleanup_old_artifacts(&mut self) -> crate::error::Result<()> {
        // Group artifacts by type for retention policy application
        let mut by_type: HashMap<ArtifactType, Vec<(chrono::DateTime<chrono::Utc>, String)>> = HashMap::new();

        for (id, artifact) in &self.artifacts {
            by_type.entry(artifact.artifact_type.clone())
                  .or_insert_with(Vec::new)
                  .push((artifact.created_at, id.clone()));
        }

        // Apply retention policies
        for (artifact_type, mut artifacts) in by_type {
            // Sort by creation time (oldest first)
            artifacts.sort_by(|a, b| a.0.cmp(&b.0));

            let max_keep = self.storage_config.retention_policies
                .get(&artifact_type)
                .copied()
                .unwrap_or(self.storage_config.default_max_per_type);

            // Remove oldest artifacts beyond limit
            let total_artifacts = artifacts.len();
            for (_, id) in artifacts.into_iter().take(total_artifacts.saturating_sub(max_keep)) {
                self.delete_artifact(&id)?;
            }
        }

        Ok(())
    }

    /// Calculate simple checksum for data integrity
    fn calculate_checksum(&self, data: &[u8]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Compress data (placeholder - would use actual compression like gzip)
    fn compress_data(&self, data: Vec<u8>) -> crate::error::Result<Vec<u8>> {
        // Placeholder implementation - would use a compression library like flate2
        // For now, just return the data unchanged
        Ok(data)
    }

    /// Decompress data (placeholder)
    fn decompress_data(&self, data: &[u8]) -> crate::error::Result<Vec<u8>> {
        // Placeholder implementation - would use a compression library like flate2
        Ok(data.to_vec())
    }

    /// Infer content type from artifact type
    fn infer_content_type(&self, artifact_type: &ArtifactType) -> String {
        match artifact_type {
            ArtifactType::Model => "application/octet-stream",
            ArtifactType::Dataset => "application/octet-stream",
            ArtifactType::Plot => "image/png",
            ArtifactType::Report => "text/html",
            ArtifactType::Config => "application/json",
            ArtifactType::Log => "text/plain",
            ArtifactType::Bundle => "application/octet-stream",
            ArtifactType::Custom(_) => "application/octet-stream",
        }.to_string()
    }
}

/// Stored artifact with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Unique artifact identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Artifact type category
    pub artifact_type: ArtifactType,
    /// MIME content type
    pub content_type: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// Data integrity checksum
    pub checksum: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Optional description
    pub description: Option<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Custom metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Artifact data (potentially compressed)
    pub data: Vec<u8>,
    /// Whether data is compressed
    pub compressed: bool,
    /// Version information
    pub version: String,
    /// Dependencies on other artifacts
    pub dependencies: Vec<String>,
}

/// Artifact type categories for research workflows
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArtifactType {
    /// Trained model files
    Model,
    /// Dataset files
    Dataset,
    /// Generated plots/visualizations
    Plot,
    /// Research reports and documentation
    Report,
    /// Configuration files
    Config,
    /// Log files
    Log,
    /// Bundled artifacts (zip/tar)
    Bundle,
    /// Custom artifact type
    Custom(String),
}

impl std::fmt::Display for ArtifactType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model => write!(f, "Model"),
            Self::Dataset => write!(f, "Dataset"),
            Self::Plot => write!(f, "Plot"),
            Self::Report => write!(f, "Report"),
            Self::Config => write!(f, "Config"),
            Self::Log => write!(f, "Log"),
            Self::Bundle => write!(f, "Bundle"),
            Self::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Artifact storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactStorageConfig {
    /// Maximum number of artifacts to store
    pub max_artifacts: usize,
    /// Maximum total storage size (bytes)
    pub max_total_size_bytes: u64,
    /// Default maximum artifacts per type
    pub default_max_per_type: usize,
    /// Retention policies by artifact type
    pub retention_policies: HashMap<ArtifactType, usize>,
    /// Whether to compress large artifacts
    pub compress_large_artifacts: bool,
    /// Minimum size for compression (bytes)
    pub compression_threshold: u64,
    /// Storage directory
    pub storage_dir: PathBuf,
    /// Auto-cleanup interval (seconds)
    pub cleanup_interval_seconds: u64,
}

impl Default for ArtifactStorageConfig {
    fn default() -> Self {
        let mut retention_policies = HashMap::new();
        retention_policies.insert(ArtifactType::Model, 10);
        retention_policies.insert(ArtifactType::Dataset, 5);
        retention_policies.insert(ArtifactType::Plot, 50);
        retention_policies.insert(ArtifactType::Report, 20);
        retention_policies.insert(ArtifactType::Config, 30);
        retention_policies.insert(ArtifactType::Log, 100);
        retention_policies.insert(ArtifactType::Bundle, 15);

        Self {
            max_artifacts: 1000,
            max_total_size_bytes: 10 * 1024 * 1024 * 1024, // 10GB
            default_max_per_type: 20,
            retention_policies,
            compress_large_artifacts: true,
            compression_threshold: 1024 * 1024, // 1MB
            storage_dir: PathBuf::from("./artifacts"),
            cleanup_interval_seconds: 3600, // 1 hour
        }
    }
}

/// Storage statistics and analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactStorageStats {
    /// Total number of artifacts stored
    pub total_artifacts: usize,
    /// Total storage used (bytes)
    pub total_size_bytes: u64,
    /// Artifacts by type
    pub by_type: HashMap<ArtifactType, usize>,
    /// Storage utilization percentage
    pub utilization_percent: f64,
    /// Average artifact size
    pub avg_artifact_size: f64,
    /// Largest artifact size
    pub largest_artifact_size: u64,
    /// Oldest artifact age (seconds)
    pub oldest_artifact_age_seconds: u64,
}

impl Default for ArtifactStorageStats {
    fn default() -> Self {
        Self {
            total_artifacts: 0,
            total_size_bytes: 0,
            by_type: HashMap::new(),
            utilization_percent: 0.0,
            avg_artifact_size: 0.0,
            largest_artifact_size: 0,
            oldest_artifact_age_seconds: 0,
        }
    }
}

impl ArtifactStorageStats {
    /// Update calculated statistics
    pub fn update_calculated_stats(&mut self, max_storage: u64) {
        self.utilization_percent = if max_storage > 0 {
            (self.total_size_bytes as f64 / max_storage as f64) * 100.0
        } else {
            0.0
        };

        self.avg_artifact_size = if self.total_artifacts > 0 {
            self.total_size_bytes as f64 / self.total_artifacts as f64
        } else {
            0.0
        };
    }
}

/// Artifact archive for long-term storage
pub struct ArtifactArchive {
    /// Archived artifacts
    archived: HashMap<String, ArchivedArtifact>,
    /// Archive storage path
    archive_path: PathBuf,
}

impl ArtifactArchive {
    /// Create new artifact archive
    pub fn new(archive_path: PathBuf) -> Self {
        Self {
            archived: HashMap::new(),
            archive_path,
        }
    }

    /// Archive artifact
    pub fn archive_artifact(&mut self, artifact: Artifact) -> crate::error::Result<String> {
        let artifact_id = artifact.id.clone();
        // Create archive entry
        let archived = ArchivedArtifact {
            original_id: artifact_id.clone(),
            archived_at: chrono::Utc::now(),
            artifact,
        };

        self.archived.insert(artifact_id.clone(), archived);
        Ok(artifact_id)
    }

    /// Retrieve archived artifact
    pub fn retrieve_archived(&self, id: &str) -> Option<&ArchivedArtifact> {
        self.archived.get(id)
    }

    /// List archived artifacts
    pub fn list_archived(&self) -> Vec<&ArchivedArtifact> {
        self.archived.values().collect()
    }

    /// Export archive to file
    pub fn export_archive(&self, export_path: &Path) -> crate::error::Result<()> {
        let archive_data = serde_json::to_vec_pretty(&self.archived)
            .map_err(|e| crate::error::NNError::SerializationError { message: format!("Failed to serialize archived data: {}", e) })?;
        std::fs::write(export_path, archive_data)
            .map_err(|e| crate::error::NNError::IoError { error: e })
    }
}

/// Archived artifact with preservation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchivedArtifact {
    /// Original artifact ID
    pub original_id: String,
    /// When artifact was archived
    pub archived_at: chrono::DateTime<chrono::Utc>,
    /// The archived artifact
    pub artifact: Artifact,
}

/// Convenience methods for common artifact operations
impl ArtifactStorage {
    /// Store model artifact
    pub fn store_model(&mut self, name: String, model_data: Vec<u8>) -> crate::error::Result<String> {
        self.store_artifact(name, ArtifactType::Model, model_data)
    }

    /// Store plot artifact
    pub fn store_plot(&mut self, name: String, plot_data: Vec<u8>) -> crate::error::Result<String> {
        self.store_artifact(name, ArtifactType::Plot, plot_data)
    }

    /// Store dataset artifact
    pub fn store_dataset(&mut self, name: String, dataset_data: Vec<u8>) -> crate::error::Result<String> {
        self.store_artifact(name, ArtifactType::Dataset, dataset_data)
    }

    /// Store configuration artifact
    pub fn store_config(&mut self, name: String, config_data: Vec<u8>) -> crate::error::Result<String> {
        self.store_artifact(name, ArtifactType::Config, config_data)
    }
}
