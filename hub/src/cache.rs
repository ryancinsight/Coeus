//! Model caching system for local storage and retrieval

use crate::error::{HubError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Cache entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    path: String, // Store as string for serialization
    size: u64,
    created: SystemTime,
    last_accessed: SystemTime,
    checksum: String,
}

/// Model cache for local storage management
#[derive(Debug)]
pub struct ModelCache {
    cache_dir: PathBuf,
    max_size: u64,
    index: HashMap<String, CacheEntry>,
    current_size: u64,
}

impl ModelCache {
    /// Create a new model cache with default settings
    pub fn new() -> Self {
        Self::with_capacity(1024 * 1024 * 1024) // 1GB default
    }

    /// Create a new model cache with specified capacity
    pub fn with_capacity(max_size: u64) -> Self {
        let cache_dir = Self::default_cache_dir();
        Self::with_directory_and_capacity(cache_dir, max_size)
    }

    /// Create a new model cache with custom directory and capacity
    pub fn with_directory_and_capacity<P: AsRef<Path>>(cache_dir: P, max_size: u64) -> Self {
        let cache_dir = cache_dir.as_ref().to_path_buf();

        // Create cache directory if it doesn't exist
        if let Err(e) = fs::create_dir_all(&cache_dir) {
            tracing::warn!("Failed to create cache directory: {}", e);
        }

        let mut cache = Self {
            cache_dir,
            max_size,
            index: HashMap::new(),
            current_size: 0,
        };

        // Load existing cache index
        if let Err(e) = cache.load_index() {
            tracing::warn!("Failed to load cache index: {}", e);
        }

        cache
    }

    /// Store model data in the cache
    pub fn store(&mut self, model_id: &str, data: &[u8]) -> Result<()> {
        let data_size = data.len() as u64;

        // Check if we need to evict entries
        self.ensure_capacity(data_size)?;

        let file_path = self.cache_dir.join(format!("{}.bin", model_id));
        let checksum = self.compute_checksum(data);

        // Write data to file
        fs::write(&file_path, data).map_err(|e| HubError::IoError {
            message: format!("Failed to write cache file: {}", e),
        })?;

        // Update index
        let entry = CacheEntry {
            path: file_path.to_string_lossy().to_string(),
            size: data_size,
            created: SystemTime::now(),
            last_accessed: SystemTime::now(),
            checksum,
        };

        // Remove old entry if it exists
        if let Some(old_entry) = self.index.remove(model_id) {
            self.current_size -= old_entry.size;
        }

        self.index.insert(model_id.to_string(), entry);
        self.current_size += data_size;

        // Save updated index
        self.save_index()?;

        tracing::debug!("Cached model {} ({} bytes)", model_id, data_size);
        Ok(())
    }

    /// Retrieve model data from the cache
    pub fn get(&mut self, model_id: &str) -> Result<Option<Vec<u8>>> {
        if let Some(entry) = self.index.get(model_id) {
            let file_path = PathBuf::from(&entry.path);
            match fs::read(&file_path) {
                Ok(data) => {
                    // Verify checksum
                    let computed_checksum = self.compute_checksum(&data);
                    if computed_checksum == entry.checksum {
                        // Update access time
                        if let Some(entry_mut) = self.index.get_mut(model_id) {
                            entry_mut.last_accessed = SystemTime::now();
                        }
                        // Save updated index
                        self.save_index()?;
                        tracing::debug!("Retrieved model {} from cache", model_id);
                        Ok(Some(data))
                    } else {
                        // Corrupted file - remove it
                        tracing::warn!("Cache corruption detected for model {}", model_id);
                        self.remove(model_id)?;
                        Ok(None)
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to read cached model {}: {}", model_id, e);
                    // Remove corrupted entry
                    let size = entry.size;
                    self.index.remove(model_id);
                    self.current_size -= size;
                    self.save_index()?;
                    Ok(None)
                }
            }
        } else {
            Ok(None)
        }
    }

    /// Check if a model is cached
    pub fn contains(&self, model_id: &str) -> bool {
        self.index.contains_key(model_id)
    }

    /// Remove a model from the cache
    pub fn remove(&mut self, model_id: &str) -> Result<()> {
        if let Some(entry) = self.index.remove(model_id) {
            // Remove file
            let path_buf = PathBuf::from(&entry.path);
            if path_buf.exists() {
                fs::remove_file(&path_buf).map_err(|e| HubError::IoError {
                    message: format!("Failed to remove cache file: {}", e),
                })?;
            }

            self.current_size -= entry.size;
            self.save_index()?;
        }
        Ok(())
    }

    /// Clear all cached models
    pub fn clear(&mut self) -> Result<()> {
        // Remove all files
        for entry in self.index.values() {
            let path_buf = PathBuf::from(&entry.path);
            if path_buf.exists() {
                let _ = fs::remove_file(&path_buf); // Ignore errors
            }
        }

        self.index.clear();
        self.current_size = 0;
        self.save_index()?;
        Ok(())
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            total_entries: self.index.len(),
            total_size: self.current_size,
            max_size: self.max_size,
            hit_rate: 0.0, // Would need to track hits/misses over time
        }
    }

    /// Get the default cache directory
    fn default_cache_dir() -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("coeus")
            .join("models")
    }

    /// Ensure there's enough capacity for new data
    fn ensure_capacity(&mut self, required_size: u64) -> Result<()> {
        while self.current_size + required_size > self.max_size && !self.index.is_empty() {
            self.evict_lru()?;
        }

        if self.current_size + required_size > self.max_size {
            return Err(HubError::CacheLimitExceeded {
                requested: required_size,
                limit: self.max_size,
            });
        }

        Ok(())
    }

    /// Evict least recently used entries
    fn evict_lru(&mut self) -> Result<()> {
        if self.index.is_empty() {
            return Ok(());
        }

        // Find the least recently used entry
        let lru_key = self
            .index
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(key, _)| key.clone())
            .unwrap();

        self.remove(&lru_key)?;
        tracing::debug!("Evicted LRU entry: {}", lru_key);
        Ok(())
    }

    /// Compute checksum for data integrity
    fn compute_checksum(&self, data: &[u8]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Load cache index from disk
    fn load_index(&mut self) -> Result<()> {
        let index_path = self.cache_dir.join("index.json");

        if !index_path.exists() {
            return Ok(());
        }

        let index_data = fs::read_to_string(&index_path).map_err(|e| HubError::IoError {
            message: format!("Failed to read cache index: {}", e),
        })?;

        let entries: HashMap<String, CacheEntry> =
            serde_json::from_str(&index_data).map_err(|e| HubError::JsonError {
                message: format!("Failed to parse cache index: {}", e),
            })?;

        // Validate entries and compute current size
        let mut current_size = 0u64;
        let mut valid_entries = HashMap::new();

        for (model_id, entry) in entries {
            let path_buf = PathBuf::from(&entry.path);
            if path_buf.exists() {
                // Verify file size matches
                if let Ok(metadata) = path_buf.metadata() {
                    if metadata.len() == entry.size {
                        current_size += entry.size;
                        valid_entries.insert(model_id, entry);
                    }
                }
            }
        }

        self.index = valid_entries;
        self.current_size = current_size;

        Ok(())
    }

    /// Save cache index to disk
    fn save_index(&self) -> Result<()> {
        let index_path = self.cache_dir.join("index.json");

        let json_data = serde_json::to_string_pretty(&self.index).map_err(|e| {
            HubError::SerializationError {
                message: format!("Failed to serialize cache index: {}", e),
            }
        })?;

        fs::write(&index_path, json_data).map_err(|e| HubError::IoError {
            message: format!("Failed to write cache index: {}", e),
        })?;

        Ok(())
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_size: u64,
    pub max_size: u64,
    pub hit_rate: f32,
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_temp_cache() -> (ModelCache, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let cache = ModelCache::with_directory_and_capacity(&temp_dir.path(), 1024 * 1024); // 1MB
        (cache, temp_dir)
    }

    #[test]
    fn test_cache_operations() {
        let (mut cache, _temp) = create_temp_cache();

        let test_data = b"test model data";

        // Store data
        cache.store("test_model", test_data).unwrap();

        // Verify it's cached
        assert!(cache.contains("test_model"));
        assert_eq!(cache.stats().total_entries, 1);

        // Retrieve data
        let retrieved = cache.get("test_model").unwrap().unwrap();
        assert_eq!(retrieved, test_data);

        // Remove data
        cache.remove("test_model").unwrap();
        assert!(!cache.contains("test_model"));
        assert_eq!(cache.stats().total_entries, 0);
    }

    #[test]
    fn test_cache_eviction() {
        let (mut cache, _temp) = create_temp_cache();

        // Fill cache with data that exceeds limit when combined
        let large_data = vec![0u8; 600 * 1024]; // 600KB
        let small_data = vec![1u8; 300 * 1024]; // 300KB

        cache.store("large_model", &large_data).unwrap();
        cache.store("small_model", &small_data).unwrap();

        // Both should be cached initially
        assert!(cache.contains("large_model"));
        assert!(cache.contains("small_model"));

        // Try to store another large model - should trigger eviction
        let another_large = vec![2u8; 600 * 1024];
        cache.store("another_model", &another_large).unwrap();

        // One of the previous models should be evicted
        let total_entries = cache.stats().total_entries;
        assert!(total_entries <= 2); // Should have evicted at least one
    }

    #[test]
    fn test_cache_clear() {
        let (mut cache, _temp) = create_temp_cache();

        cache.store("model1", b"data1").unwrap();
        cache.store("model2", b"data2").unwrap();

        assert_eq!(cache.stats().total_entries, 2);

        cache.clear().unwrap();
        assert_eq!(cache.stats().total_entries, 0);
    }

    #[test]
    fn test_cache_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path();

        {
            let mut cache = ModelCache::with_directory_and_capacity(cache_path, 1024 * 1024);
            cache.store("persistent_model", b"persistent data").unwrap();
        }

        // Create new cache instance - should load the index
        {
            let cache = ModelCache::with_directory_and_capacity(cache_path, 1024 * 1024);
            assert!(cache.contains("persistent_model"));
            assert_eq!(cache.stats().total_entries, 1);
        }
    }
}
