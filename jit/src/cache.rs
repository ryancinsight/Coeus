//! Kernel caching system for JIT compilation

use crate::compiler::CompiledKernel;
use crate::error::{JitError, Result};
use std::collections::HashMap;
use std::path::Path;

/// Kernel cache for storing compiled kernels
#[derive(Debug)]
pub struct KernelCache {
    memory_cache: HashMap<String, CompiledKernel>,
    disk_cache_path: Option<std::path::PathBuf>,
    max_memory_entries: usize,
    enable_disk_cache: bool,
}

impl KernelCache {
    /// Create a new kernel cache with memory-only storage
    pub fn new() -> Self {
        Self {
            memory_cache: HashMap::new(),
            disk_cache_path: None,
            max_memory_entries: 100,
            enable_disk_cache: false,
        }
    }

    /// Create a new kernel cache with disk persistence
    pub fn with_disk_cache<P: AsRef<Path>>(path: P, max_memory_entries: usize) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Create cache directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| JitError::CacheError {
                message: format!("Failed to create cache directory: {}", e),
            })?;
        }

        Ok(Self {
            memory_cache: HashMap::new(),
            disk_cache_path: Some(path),
            max_memory_entries,
            enable_disk_cache: true,
        })
    }

    /// Store a compiled kernel in the cache
    pub fn store(&mut self, kernel: CompiledKernel) -> Result<()> {
        let key = kernel.kernel_id.clone();

        // Check memory cache size limit
        if self.memory_cache.len() >= self.max_memory_entries {
            self.evict_lru()?;
        }

        // Store in memory cache
        self.memory_cache.insert(key.clone(), kernel);

        // Store on disk if enabled
        if self.enable_disk_cache {
            self.store_on_disk(&key)?;
        }

        Ok(())
    }

    /// Retrieve a compiled kernel from the cache
    pub fn retrieve(&self, kernel_id: &str) -> Option<&CompiledKernel> {
        // Try memory cache first
        if let Some(kernel) = self.memory_cache.get(kernel_id) {
            return Some(kernel);
        }

        // Try disk cache if enabled and not in memory
        if self.enable_disk_cache {
            if let Ok(Some(kernel)) = self.retrieve_from_disk(kernel_id) {
                // Note: In a full implementation, we would move this to memory cache
                return Some(Box::leak(Box::new(kernel)));
            }
        }

        None
    }

    /// Check if a kernel is cached
    pub fn contains(&self, kernel_id: &str) -> bool {
        self.memory_cache.contains_key(kernel_id)
            || (self.enable_disk_cache && self.disk_contains(kernel_id))
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let memory_entries = self.memory_cache.len();
        let disk_entries = if self.enable_disk_cache {
            self.disk_cache_size()
        } else {
            0
        };

        let memory_usage = self
            .memory_cache
            .values()
            .map(|k| k.memory_requirements)
            .sum();

        CacheStats {
            memory_entries,
            disk_entries,
            memory_usage,
            max_memory_entries: self.max_memory_entries,
        }
    }

    /// Clear all cached kernels
    pub fn clear(&mut self) -> Result<()> {
        self.memory_cache.clear();

        if self.enable_disk_cache {
            self.clear_disk_cache()?;
        }

        Ok(())
    }

    /// Evict least recently used kernels when cache is full
    fn evict_lru(&mut self) -> Result<()> {
        // Simple eviction strategy: remove oldest entries
        // In a full implementation, this would track access patterns

        if self.memory_cache.is_empty() {
            return Ok(());
        }

        // Remove 10% of entries
        let to_remove = (self.memory_cache.len() / 10).max(1);
        let keys_to_remove: Vec<String> =
            self.memory_cache.keys().take(to_remove).cloned().collect();

        for key in keys_to_remove {
            self.memory_cache.remove(&key);
        }

        Ok(())
    }

    /// Store a kernel on disk
    fn store_on_disk(&self, kernel_id: &str) -> Result<()> {
        if let Some(cache_path) = &self.disk_cache_path {
            if let Some(_kernel) = self.memory_cache.get(kernel_id) {
                let file_path = cache_path.with_extension("bin");

                // In a real implementation, this would serialize the kernel
                // For now, we just touch the file to indicate presence
                std::fs::File::create(&file_path).map_err(|e| JitError::CacheError {
                    message: format!("Failed to create cache file: {}", e),
                })?;
            }
        }
        Ok(())
    }

    /// Retrieve a kernel from disk
    fn retrieve_from_disk(&self, kernel_id: &str) -> Result<Option<CompiledKernel>> {
        if let Some(cache_path) = &self.disk_cache_path {
            let file_path = cache_path.with_extension("bin");

            if file_path.exists() {
                // In a real implementation, this would deserialize the kernel
                // For now, return a placeholder
                Ok(Some(CompiledKernel {
                    kernel_id: kernel_id.to_string(),
                    target_arch: crate::compiler::TargetArchitecture::X86_64,
                    optimization_level: crate::compiler::OptimizationLevel::Basic,
                    memory_requirements: 1024,
                    performance_estimate: 50.0,
                    machine_code: vec![0; 256],
                    function_ptr: Some(0),
                }))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Check if disk cache contains a kernel
    fn disk_contains(&self, _kernel_id: &str) -> bool {
        if let Some(cache_path) = &self.disk_cache_path {
            let file_path = cache_path.with_extension("bin");
            file_path.exists()
        } else {
            false
        }
    }

    /// Get disk cache size
    fn disk_cache_size(&self) -> usize {
        if let Some(cache_path) = &self.disk_cache_path {
            let cache_dir = cache_path.parent().unwrap_or(cache_path);
            if let Ok(entries) = std::fs::read_dir(cache_dir) {
                entries.count()
            } else {
                0
            }
        } else {
            0
        }
    }

    /// Clear disk cache
    fn clear_disk_cache(&self) -> Result<()> {
        if let Some(cache_path) = &self.disk_cache_path {
            let cache_dir = cache_path.parent().unwrap_or(cache_path);
            if cache_dir.exists() {
                std::fs::remove_dir_all(cache_dir).map_err(|e| JitError::CacheError {
                    message: format!("Failed to clear disk cache: {}", e),
                })?;
            }
        }
        Ok(())
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub memory_entries: usize,
    pub disk_entries: usize,
    pub memory_usage: usize,
    pub max_memory_entries: usize,
}

impl Default for KernelCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{CompiledKernel, OptimizationLevel, TargetArchitecture};

    fn create_test_kernel(id: &str) -> CompiledKernel {
        CompiledKernel {
            kernel_id: id.to_string(),
            target_arch: TargetArchitecture::X86_64,
            optimization_level: OptimizationLevel::Basic,
            memory_requirements: 1024,
            performance_estimate: 50.0,
            machine_code: vec![0; 256],
            function_ptr: Some(0),
        }
    }

    #[test]
    fn test_memory_cache_operations() {
        let mut cache = KernelCache::new();

        // Initially empty
        assert_eq!(cache.stats().memory_entries, 0);

        // Store a kernel
        let kernel = create_test_kernel("test_kernel");
        cache.store(kernel.clone()).unwrap();
        assert_eq!(cache.stats().memory_entries, 1);

        // Retrieve it
        let retrieved = cache.retrieve("test_kernel").unwrap();
        assert_eq!(retrieved.kernel_id, "test_kernel");

        // Clear cache
        cache.clear().unwrap();
        assert_eq!(cache.stats().memory_entries, 0);
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = KernelCache {
            memory_cache: HashMap::new(),
            disk_cache_path: None,
            max_memory_entries: 3,
            enable_disk_cache: false,
        };

        // Fill cache to limit
        for i in 0..3 {
            let kernel = create_test_kernel(&format!("kernel_{}", i));
            cache.store(kernel).unwrap();
        }

        assert_eq!(cache.stats().memory_entries, 3);

        // Add one more - should trigger eviction
        let kernel = create_test_kernel("kernel_evict");
        cache.store(kernel).unwrap();

        // Should have evicted some entries
        assert!(cache.stats().memory_entries <= 3);
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = KernelCache::new();

        let kernel1 = create_test_kernel("kernel1");
        let kernel2 = create_test_kernel("kernel2");

        cache.store(kernel1).unwrap();
        cache.store(kernel2).unwrap();

        let stats = cache.stats();
        assert_eq!(stats.memory_entries, 2);
        assert_eq!(stats.disk_entries, 0); // No disk cache
        assert_eq!(stats.memory_usage, 2048); // 2 * 1024
        assert_eq!(stats.max_memory_entries, 100);
    }

    #[test]
    fn test_disk_cache_creation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cache_path = temp_dir.path().join("test_cache.bin");

        let cache = KernelCache::with_disk_cache(&cache_path, 50).unwrap();
        assert!(cache.enable_disk_cache);
        assert_eq!(cache.max_memory_entries, 50);
    }
}
