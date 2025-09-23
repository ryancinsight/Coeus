//! Model configuration

#![allow(clippy::bool_assert_comparison)]
#![allow(clippy::empty_line_after_doc_comments)]

/// Configuration and hyperparameter management.
///
/// This module provides configuration structures for model loading, inference,
/// and performance optimization. It supports both PyTorch-style and GGUF-specific
/// configuration options.
use crate::quantization::QuantizationScheme;
use crate::ModelType;

/// Model loading configuration
#[derive(Debug, Clone)]
pub struct ModelLoadConfig {
    /// Use memory mapping for tensor loading
    pub use_memory_map: bool,
    /// Verify model integrity with SHA256
    pub verify_integrity: bool,
    /// Cache model files locally
    pub use_cache: bool,
    /// Maximum cache size in bytes
    pub max_cache_size: u64,
    /// Cache directory path
    pub cache_dir: Option<std::path::PathBuf>,
    /// Use GPU acceleration if available
    pub use_gpu: bool,
    /// Preferred quantization scheme
    pub preferred_quantization: Option<QuantizationScheme>,
    /// Enable debug logging
    pub debug: bool,
}

impl Default for ModelLoadConfig {
    fn default() -> Self {
        Self {
            use_memory_map: true,
            verify_integrity: true,
            use_cache: true,
            max_cache_size: 10 * 1024 * 1024 * 1024, // 10GB
            cache_dir: None,
            use_gpu: false,
            preferred_quantization: Some(QuantizationScheme::Q4_0),
            debug: false,
        }
    }
}

impl ModelLoadConfig {
    /// Create a new loading configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable/disable memory mapping
    pub fn with_memory_map(mut self, use_memory_map: bool) -> Self {
        self.use_memory_map = use_memory_map;
        self
    }

    /// Enable/disable integrity verification
    pub fn with_integrity_verification(mut self, verify_integrity: bool) -> Self {
        self.verify_integrity = verify_integrity;
        self
    }

    /// Enable/disable caching
    pub fn with_cache(mut self, use_cache: bool) -> Self {
        self.use_cache = use_cache;
        self
    }

    /// Set maximum cache size
    pub fn with_max_cache_size(mut self, max_cache_size: u64) -> Self {
        self.max_cache_size = max_cache_size;
        self
    }

    /// Set cache directory
    pub fn with_cache_dir(mut self, cache_dir: std::path::PathBuf) -> Self {
        self.cache_dir = Some(cache_dir);
        self
    }

    /// Enable GPU acceleration
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Set preferred quantization scheme
    pub fn with_preferred_quantization(mut self, quantization: QuantizationScheme) -> Self {
        self.preferred_quantization = Some(quantization);
        self
    }

    /// Enable debug logging
    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }
}

/// Performance optimization configuration
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Number of CPU threads to use
    pub num_threads: usize,
    /// Enable SIMD optimizations
    pub use_simd: bool,
    /// Enable parallel processing
    pub use_parallel: bool,
    /// Memory pool size
    pub memory_pool_size: usize,
    /// Use fast approximations
    pub use_fast_approx: bool,
    /// Enable profiling
    pub enable_profiling: bool,
    /// GPU memory fraction (0.0 to 1.0)
    pub gpu_memory_fraction: f32,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            use_simd: true,
            use_parallel: true,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            use_fast_approx: false,
            enable_profiling: false,
            gpu_memory_fraction: 0.9,
        }
    }
}

impl PerformanceConfig {
    /// Create a new performance configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of threads
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Enable/disable SIMD optimizations
    pub fn with_simd(mut self, use_simd: bool) -> Self {
        self.use_simd = use_simd;
        self
    }

    /// Enable/disable parallel processing
    pub fn with_parallel(mut self, use_parallel: bool) -> Self {
        self.use_parallel = use_parallel;
        self
    }

    /// Set memory pool size
    pub fn with_memory_pool_size(mut self, memory_pool_size: usize) -> Self {
        self.memory_pool_size = memory_pool_size;
        self
    }

    /// Enable/disable fast approximations
    pub fn with_fast_approx(mut self, use_fast_approx: bool) -> Self {
        self.use_fast_approx = use_fast_approx;
        self
    }

    /// Enable/disable profiling
    pub fn with_profiling(mut self, enable_profiling: bool) -> Self {
        self.enable_profiling = enable_profiling;
        self
    }

    /// Set GPU memory fraction
    pub fn with_gpu_memory_fraction(mut self, gpu_memory_fraction: f32) -> Self {
        self.gpu_memory_fraction = gpu_memory_fraction.clamp(0.1, 1.0);
        self
    }
}

/// Complete model configuration combining all aspects
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model architecture type
    pub model_type: ModelType,
    /// Model dimensionality
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Model loading configuration
    pub loading: ModelLoadConfig,
    /// Performance optimization configuration
    pub performance: PerformanceConfig,
    /// Inference parameters
    pub inference: crate::inference::InferenceConfig,
}

impl ModelConfig {
    /// Create a new comprehensive model configuration
    pub fn new(model_type: ModelType) -> Self {
        Self {
            model_type,
            hidden_size: 4096,
            num_heads: 32,
            num_layers: 32,
            vocab_size: 32000,
            loading: ModelLoadConfig::default(),
            performance: PerformanceConfig::default(),
            inference: crate::inference::InferenceConfig::default(),
        }
    }

    /// Set loading configuration
    pub fn with_loading_config(mut self, loading: ModelLoadConfig) -> Self {
        self.loading = loading;
        self
    }

    /// Set performance configuration
    pub fn with_performance_config(mut self, performance: PerformanceConfig) -> Self {
        self.performance = performance;
        self
    }

    /// Set inference configuration
    pub fn with_inference_config(mut self, inference: crate::inference::InferenceConfig) -> Self {
        self.inference = inference;
        self
    }

    /// Set vocabulary size
    pub fn with_vocab_size(mut self, vocab_size: usize) -> Self {
        self.vocab_size = vocab_size;
        self
    }

    /// Set hidden size
    pub fn with_hidden_size(mut self, hidden_size: usize) -> Self {
        self.hidden_size = hidden_size;
        self
    }

    /// Set number of heads
    pub fn with_num_heads(mut self, num_heads: usize) -> Self {
        self.num_heads = num_heads;
        self
    }

    /// Set number of layers
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Set maximum sequence length
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.inference.max_seq_len = max_seq_len;
        self
    }

    /// Create a high-performance configuration
    pub fn high_performance(model_type: ModelType) -> Self {
        Self {
            model_type,
            hidden_size: 4096,
            num_heads: 32,
            num_layers: 32,
            vocab_size: 32000,
            loading: ModelLoadConfig::default(),
            performance: PerformanceConfig {
                num_threads: num_cpus::get() * 2,
                use_simd: true,
                use_parallel: true,
                memory_pool_size: 2 * 1024 * 1024 * 1024, // 2GB
                use_fast_approx: false,
                enable_profiling: false,
                gpu_memory_fraction: 0.95,
            },
            inference: crate::inference::InferenceConfig::default(),
        }
    }

    /// Create a memory-efficient configuration
    pub fn memory_efficient(model_type: ModelType) -> Self {
        Self {
            model_type,
            hidden_size: 4096,
            num_heads: 32,
            num_layers: 32,
            vocab_size: 32000,
            loading: ModelLoadConfig {
                use_memory_map: true,
                preferred_quantization: Some(QuantizationScheme::Q4_0),
                ..Default::default()
            },
            performance: PerformanceConfig {
                num_threads: 1,
                use_simd: true,
                use_parallel: false,
                memory_pool_size: 512 * 1024 * 1024, // 512MB
                use_fast_approx: true,
                enable_profiling: false,
                gpu_memory_fraction: 0.8,
            },
            inference: crate::inference::InferenceConfig::default(),
        }
    }

    /// Create a GPU-optimized configuration
    pub fn gpu_optimized(model_type: ModelType) -> Self {
        Self {
            model_type,
            hidden_size: 4096,
            num_heads: 32,
            num_layers: 32,
            vocab_size: 32000,
            loading: ModelLoadConfig {
                use_gpu: true,
                ..Default::default()
            },
            performance: PerformanceConfig {
                num_threads: num_cpus::get(),
                use_simd: true,
                use_parallel: true,
                memory_pool_size: 1024 * 1024 * 1024, // 1GB
                use_fast_approx: false,
                enable_profiling: true,
                gpu_memory_fraction: 0.9,
            },
            inference: crate::inference::InferenceConfig::default().with_gpu(true),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_load_config_builder() {
        let config = ModelLoadConfig::new()
            .with_memory_map(false)
            .with_integrity_verification(false)
            .with_cache(false)
            .with_max_cache_size(5_000_000_000)
            .with_gpu(true)
            .with_preferred_quantization(QuantizationScheme::Q8_0)
            .with_debug(true);

        assert_eq!(config.use_memory_map, false);
        assert_eq!(config.verify_integrity, false);
        assert_eq!(config.use_cache, false);
        assert_eq!(config.max_cache_size, 5_000_000_000);
        assert_eq!(config.use_gpu, true);
        assert_eq!(
            config.preferred_quantization,
            Some(QuantizationScheme::Q8_0)
        );
        assert_eq!(config.debug, true);
    }

    #[test]
    fn test_performance_config_builder() {
        let config = PerformanceConfig::new()
            .with_num_threads(8)
            .with_simd(false)
            .with_parallel(false)
            .with_memory_pool_size(256 * 1024 * 1024)
            .with_fast_approx(true)
            .with_profiling(true)
            .with_gpu_memory_fraction(0.5);

        assert_eq!(config.num_threads, 8);
        assert_eq!(config.use_simd, false);
        assert_eq!(config.use_parallel, false);
        assert_eq!(config.memory_pool_size, 256 * 1024 * 1024);
        assert_eq!(config.use_fast_approx, true);
        assert_eq!(config.enable_profiling, true);
        assert_eq!(config.gpu_memory_fraction, 0.5);
    }

    #[test]
    fn test_model_config_presets() {
        let default_config = ModelConfig::new(ModelType::Llama);
        assert_eq!(default_config.model_type, ModelType::Llama);

        let high_perf_config = ModelConfig::high_performance(ModelType::Llama);
        assert_eq!(
            high_perf_config.performance.num_threads,
            num_cpus::get() * 2
        );

        let memory_eff_config = ModelConfig::memory_efficient(ModelType::Llama);
        assert_eq!(
            memory_eff_config.loading.preferred_quantization,
            Some(QuantizationScheme::Q4_0)
        );
        assert_eq!(memory_eff_config.performance.use_parallel, false);

        let gpu_config = ModelConfig::gpu_optimized(ModelType::Llama);
        assert_eq!(gpu_config.loading.use_gpu, true);
        assert_eq!(gpu_config.inference.use_gpu, true);
    }

    #[test]
    fn test_performance_config_validation() {
        let config = PerformanceConfig::new().with_gpu_memory_fraction(1.5); // Should be clamped

        assert_eq!(config.gpu_memory_fraction, 1.0);

        let config = PerformanceConfig::new().with_gpu_memory_fraction(-0.1); // Should be clamped

        assert_eq!(config.gpu_memory_fraction, 0.1);
    }
}
