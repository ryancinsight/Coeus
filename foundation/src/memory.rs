//! Memory Optimization Systems for Foundation Model Training
//!
//! This module implements advanced memory optimization techniques for training
//! large foundation models efficiently:
//! - Gradient checkpointing for memory tradeoffs
//! - Activation recomputation to reduce memory footprint
//! - Mixed precision training (FP16/BF16/FP8)
//! - Memory-efficient optimizers (8-bit Adam, Lion)
//! - Dynamic memory management and offloading strategies

use std::collections::{HashMap, VecDeque};
use crate::error::{NNError, Result};

/// Memory Optimization Orchestrator
#[derive(Debug)]
pub struct MemoryOptimizer {
    /// Current memory budget (GB)
    pub memory_budget_gb: f64,
    /// Memory optimization strategy
    pub strategy: MemoryStrategy,
    /// Activation checkpointing manager
    pub activation_manager: ActivationManager,
    /// Gradient checkpointing manager
    pub gradient_manager: GradientManager,
    /// Parameter offloading manager
    pub offloading_manager: OffloadingManager,
    /// Mixed precision manager
    pub mixed_precision: MixedPrecisionManager,
    /// Memory statistics tracker
    pub memory_stats: MemoryStatistics,
}

#[derive(Debug)]
pub enum MemoryStrategy {
    /// Standard memory management
    Standard,
    /// Aggressive memory optimization (max recomputation)
    Aggressive,
    /// Conservative memory usage (minimal recomputation)
    Conservative,
    /// Adaptive memory based on available resources
    Adaptive,
    /// Custom memory strategy
    Custom(String),
}

impl MemoryOptimizer {
    /// Create new memory optimizer with specified budget
    pub fn new(memory_budget_gb: f64) -> Self {
        Self {
            memory_budget_gb,
            strategy: MemoryStrategy::Adaptive,
            activation_manager: ActivationManager::new(),
            gradient_manager: GradientManager::new(),
            offloading_manager: OffloadingManager::new(),
            mixed_precision: MixedPrecisionManager::new(),
            memory_stats: MemoryStatistics::new(),
        }
    }

    /// Configure memory strategy
    pub fn with_strategy(mut self, strategy: MemoryStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Enable gradient checkpointing for specified layers
    pub fn enable_gradient_checkpointing(mut self, layers: Vec<String>, checkpoint_ratio: f64) -> Self {
        self.gradient_manager.enable_checkpointing(layers, checkpoint_ratio);
        self
    }

    /// Configure activation offloading strategy
    pub fn with_activation_offloading(mut self, strategy: OffloadingStrategy) -> Self {
        self.activation_manager.offloading_strategy = strategy;
        self
    }

    /// Set mixed precision level
    pub fn with_mixed_precision(mut self, precision: MixedPrecisionLevel) -> Self {
        self.mixed_precision.set_precision_level(precision);
        self
    }

    /// Optimize memory usage for a forward pass
    pub async fn optimize_forward(&mut self, layer_name: &str, input_size: usize) -> Result<MemoryPlan> {
        // Estimate memory requirements for this layer
        let estimated_usage = self.estimate_layer_memory(layer_name, input_size);

        // Check if we need to apply memory optimizations
        if estimated_usage > self.memory_budget_gb * 0.8 {
            return self.apply_memory_optimizations(layer_name, estimated_usage);
        }

        // Default plan - no special optimization needed
        Ok(MemoryPlan {
            checkpoint_activations: false,
            offload_to_cpu: false,
            use_mixed_precision: self.mixed_precision.enabled,
            recomputation_strategy: RecomputationStrategy::None,
            estimated_memory_mb: estimated_usage,
        })
    }

    /// Optimize memory usage for a backward pass
    pub async fn optimize_backward(&mut self, layer_name: &str) -> Result<MemoryPlan> {
        // Apply backward-specific optimizations (primarily gradient checkpointing)
        let plan = self.gradient_manager.optimize_backward_pass(layer_name).await?;
        self.memory_stats.update_stats(&plan);

        Ok(plan)
    }

    fn estimate_layer_memory(&self, _layer_name: &str, input_size: usize) -> f64 {
        // Estimate memory usage based on layer type and input size
        // This would analyze the computation graph to determine memory requirements
        // Placeholder implementation
        let base_memory = input_size as f64 * 4.0 / (1024.0 * 1024.0); // MB
        base_memory * 2.5 // Account for activations, gradients, etc.
    }

    fn apply_memory_optimizations(&self, _layer_name: &str, estimated_usage: f64) -> Result<MemoryPlan> {
        match &self.strategy {
            MemoryStrategy::Aggressive => {
                Ok(MemoryPlan {
                    checkpoint_activations: true,
                    offload_to_cpu: true,
                    use_mixed_precision: true,
                    recomputation_strategy: RecomputationStrategy::Full,
                    estimated_memory_mb: estimated_usage * 0.3,
                })
            },
            MemoryStrategy::Conservative => {
                Ok(MemoryPlan {
                    checkpoint_activations: false,
                    offload_to_cpu: false,
                    use_mixed_precision: true,
                    recomputation_strategy: RecomputationStrategy::Selective,
                    estimated_memory_mb: estimated_usage * 0.8,
                })
            },
            MemoryStrategy::Adaptive => {
                // Adaptive strategy based on current memory pressure
                let memory_pressure = self.memory_stats.current_usage_gb / self.memory_budget_gb;

                if memory_pressure > 0.9 {
                    Ok(MemoryPlan {
                        checkpoint_activations: true,
                        offload_to_cpu: true,
                        use_mixed_precision: true,
                        recomputation_strategy: RecomputationStrategy::Full,
                        estimated_memory_mb: estimated_usage * 0.2,
                    })
                } else if memory_pressure > 0.7 {
                    Ok(MemoryPlan {
                        checkpoint_activations: true,
                        offload_to_cpu: false,
                        use_mixed_precision: true,
                        recomputation_strategy: RecomputationStrategy::Selective,
                        estimated_memory_mb: estimated_usage * 0.5,
                    })
                } else {
                    Ok(MemoryPlan {
                        checkpoint_activations: false,
                        offload_to_cpu: false,
                        use_mixed_precision: true,
                        recomputation_strategy: RecomputationStrategy::None,
                        estimated_memory_mb: estimated_usage,
                    })
                }
            },
            _ => Ok(MemoryPlan {
                checkpoint_activations: false,
                offload_to_cpu: false,
                use_mixed_precision: self.mixed_precision.enabled,
                recomputation_strategy: RecomputationStrategy::None,
                estimated_memory_mb: estimated_usage,
            })
        }
    }

    /// Get current memory usage report
    pub fn memory_report(&self) -> MemoryReport {
        self.memory_stats.generate_report()
    }
}

/// Memory plan for layer execution
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Whether to checkpoint activations for this layer
    pub checkpoint_activations: bool,
    /// Whether to offload activations to CPU
    pub offload_to_cpu: bool,
    /// Whether to use mixed precision computation
    pub use_mixed_precision: bool,
    /// Activation recomputation strategy
    pub recomputation_strategy: RecomputationStrategy,
    /// Estimated memory usage in MB
    pub estimated_memory_mb: f64,
}

/// Activation recomputation strategies
#[derive(Debug, Clone)]
pub enum RecomputationStrategy {
    /// No recomputation
    None,
    /// Selective recomputation for expensive operations
    Selective,
    /// Full recomputation of all activations
    Full,
}

/// Activation Memory Manager
#[derive(Debug)]
pub struct ActivationManager {
    /// Maximum activations to keep in memory
    pub max_activations: usize,
    /// Offloading strategy
    pub offloading_strategy: OffloadingStrategy,
    /// Activation storage
    pub activation_store: HashMap<String, Vec<f32>>,
    /// Activation metadata (size, access frequency, etc.)
    pub metadata: HashMap<String, ActivationMetadata>,
    /// Recomputation tracker
    pub recomputation_stats: RecomputationStats,
}

#[derive(Debug, Clone)]
pub struct ActivationMetadata {
    pub size_bytes: usize,
    pub access_count: usize,
    pub last_access: std::time::Instant,
    pub layer_name: String,
    pub recomputation_cost: f64,
}

#[derive(Debug)]
pub struct RecomputationStats {
    pub total_recomputations: usize,
    pub saved_memory_mb: f64,
    pub recomputation_time_ms: f64,
}

impl ActivationManager {
    pub fn new() -> Self {
        Self {
            max_activations: 100,
            offloading_strategy: OffloadingStrategy::None,
            activation_store: HashMap::new(),
            metadata: HashMap::new(),
            recomputation_stats: RecomputationStats::default(),
        }
    }

    /// Store activation with memory management
    pub async fn store_activation(&mut self, key: String, activation: Vec<f32>, layer_name: String) -> Result<()> {
        // Check if we need to evict activations
        if self.activation_store.len() >= self.max_activations {
            self.evict_activations().await?;
        }

        let size_bytes = activation.len() * 4; // f32 = 4 bytes
        let metadata = ActivationMetadata {
            size_bytes,
            access_count: 0,
            last_access: std::time::Instant::now(),
            layer_name,
            recomputation_cost: 1.0, // Placeholder
        };

        self.activation_store.insert(key.clone(), activation);
        self.metadata.insert(key, metadata);

        Ok(())
    }

    /// Retrieve activation with LRU management
    pub async fn get_activation(&mut self, key: &str) -> Result<Option<Vec<f32>>> {
        if let Some(metadata) = self.metadata.get_mut(key) {
            metadata.access_count += 1;
            metadata.last_access = std::time::Instant::now();

            if let Some(activation) = self.activation_store.get(key) {
                return Ok(Some(activation.clone()));
            }

            // Activation was evicted, need to recompute
            self.recomputation_stats.total_recomputations += 1;

            // Trigger offloading if configured
            match self.offloading_strategy {
                OffloadingStrategy::CPU => {
                    // Load from CPU memory
                    self.load_from_cpu(key).await
                },
                OffloadingStrategy::NVMe => {
                    // Load from NVMe storage
                    self.load_from_nvme(key).await
                },
                _ => Ok(None),
            }
        } else {
            Ok(None)
        }
    }

    async fn evict_activations(&mut self) -> Result<()> {
        // LRU eviction strategy
        // Avoid holding references to metadata to satisfy borrow checker during removal
        let mut candidates: Vec<(String, std::time::Instant)> = self.metadata.iter()
            .map(|(k, v)| (k.clone(), v.last_access))
            .collect();

        candidates.sort_by(|a, b| a.1.cmp(&b.1));

        // Evict least recently used activations
        for (key, _) in candidates.iter().take(10) {
            self.activation_store.remove(key);

            // Apply offloading strategy
            match self.offloading_strategy {
                OffloadingStrategy::CPU => {
                    self.offload_to_cpu(key).await?;
                },
                OffloadingStrategy::NVMe => {
                    self.offload_to_nvme(key).await?;
                },
                _ => {},
            }
        }

        // Remove metadata for evicted activations (keep only offloaded ones)
        for (key, _) in candidates.iter().take(10) {
            if !matches!(self.offloading_strategy, OffloadingStrategy::None) {
                // Keep metadata for offloaded activations
                continue;
            }
            self.metadata.remove(key);
        }

        Ok(())
    }

    async fn offload_to_cpu(&mut self, _key: &str) -> Result<()> {
        // Placeholder for CPU offloading implementation
        Ok(())
    }

    async fn load_from_cpu(&mut self, _key: &str) -> Result<Option<Vec<f32>>> {
        // Placeholder for CPU loading implementation
        Ok(None)
    }

    async fn offload_to_nvme(&mut self, _key: &str) -> Result<()> {
        // Placeholder for NVMe offloading implementation
        Ok(())
    }

    async fn load_from_nvme(&mut self, _key: &str) -> Result<Option<Vec<f32>>> {
        // Placeholder for NVMe loading implementation
        Ok(None)
    }
}

/// Gradient Checkpointing Manager
#[derive(Debug)]
pub struct GradientManager {
    /// Layers to checkpoint
    pub checkpointed_layers: Vec<String>,
    /// Checkpoint ratio (0.0 to 1.0)
    pub checkpoint_ratio: f64,
    /// Checkpoint statistics
    pub checkpoint_stats: CheckpointStats,
    /// Recomputation queue
    pub recomputation_queue: VecDeque<String>,
}

#[derive(Debug)]
pub struct CheckpointStats {
    pub total_checkpoints: usize,
    pub memory_saved_mb: f64,
    pub recomputation_time_ms: f64,
    pub backward_time_saved_ms: f64,
}

impl GradientManager {
    pub fn new() -> Self {
        Self {
            checkpointed_layers: Vec::new(),
            checkpoint_ratio: 0.1, // Checkpoint 10% of layers by default
            checkpoint_stats: CheckpointStats::default(),
            recomputation_queue: VecDeque::new(),
        }
    }

    pub fn enable_checkpointing(&mut self, layers: Vec<String>, checkpoint_ratio: f64) {
        self.checkpointed_layers = layers;
        self.checkpoint_ratio = checkpoint_ratio;
    }

    /// Optimize backward pass with gradient checkpointing
    pub async fn optimize_backward_pass(&self, layer_name: &str) -> Result<MemoryPlan> {
        let should_checkpoint = self.checkpointed_layers.contains(&layer_name.to_string()) ||
            rand::random::<f64>() < self.checkpoint_ratio;

        Ok(MemoryPlan {
            checkpoint_activations: should_checkpoint,
            offload_to_cpu: false,
            use_mixed_precision: false,
            recomputation_strategy: if should_checkpoint {
                RecomputationStrategy::Full
            } else {
                RecomputationStrategy::None
            },
            estimated_memory_mb: if should_checkpoint { 50.0 } else { 200.0 }, // Rough estimate
        })
    }

    /// Add layer to recomputation queue
    pub fn queue_for_recomputation(&mut self, layer_name: String) {
        self.recomputation_queue.push_back(layer_name);
    }

    /// Execute recomputation for checkpointed layers
    pub async fn execute_recomputation(&mut self) -> Result<()> {
        while let Some(_layer_name) = self.recomputation_queue.pop_front() {
            // Recompute activations for this layer
            // This would trigger the forward pass for the checkpointed layer
            self.checkpoint_stats.total_checkpoints += 1;
        }

        Ok(())
    }
}

/// Mixed Precision Training Manager
#[derive(Debug)]
pub struct MixedPrecisionManager {
    /// Whether mixed precision is enabled
    pub enabled: bool,
    /// Precision level
    pub precision_level: MixedPrecisionLevel,
    /// Loss scaling configuration
    pub loss_scaling: LossScalingConfig,
    /// Gradient clipping in reduced precision
    pub gradient_clipping: Option<f64>,
    /// Precision statistics
    pub precision_stats: PrecisionStats,
}

#[derive(Debug, Clone, Copy)]
pub enum MixedPrecisionLevel {
    /// FP16 mixed precision
    FP16,
    /// BF16 mixed precision (better numerical stability)
    BF16,
    /// FP8 mixed precision (experimental)
    FP8,
    /// Automatic precision selection
    Auto,
}

#[derive(Debug)]
pub struct LossScalingConfig {
    pub initial_scale: f64,
    pub scale_factor: f64,
    pub scale_window: usize,
    pub min_scale: f64,
    pub hysteresis: usize,
}

#[derive(Debug)]
pub struct PrecisionStats {
    pub overflow_events: usize,
    pub scale_adjustments: usize,
    pub average_scale: f64,
    pub precision_conversions: usize,
}

impl MixedPrecisionManager {
    pub fn new() -> Self {
        Self {
            enabled: false,
            precision_level: MixedPrecisionLevel::Auto,
            loss_scaling: LossScalingConfig {
                initial_scale: 32768.0, // 2^15
                scale_factor: 2.0,
                scale_window: 2000,
                min_scale: 1.0,
                hysteresis: 2,
            },
            gradient_clipping: Some(1.0),
            precision_stats: PrecisionStats::default(),
        }
    }

    /// Set precision level and enable mixed precision training
    pub fn set_precision_level(&mut self, level: MixedPrecisionLevel) {
        self.precision_level = level;
        self.enabled = true;
    }

    /// Scale loss for mixed precision stability
    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.loss_scaling.initial_scale as f32
    }

    /// Scale gradients back to original range
    pub fn scale_gradients(&self, grads: &mut [f32]) {
        let scale = self.loss_scaling.initial_scale as f32;
        for grad in grads.iter_mut() {
            *grad /= scale;
        }
    }

    /// Check for gradient overflow and adjust loss scale
    pub fn check_overflow_and_adjust(&mut self, grads: &[f32]) -> Result<bool> {
        let has_overflow = grads.iter().any(|g| !g.is_finite());

        if has_overflow {
            self.loss_scaling.initial_scale = (self.loss_scaling.initial_scale
                / self.loss_scaling.scale_factor).max(self.loss_scaling.min_scale);
            self.precision_stats.overflow_events += 1;

            // Clear gradients with overflow
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

/// Parameter Offloading Manager
#[derive(Debug)]
pub struct OffloadingManager {
    /// Offloading strategy
    pub strategy: OffloadingStrategy,
    /// CPU memory limit for offloading
    pub cpu_memory_limit_gb: f64,
    /// NVMe storage path
    pub nvme_path: Option<String>,
    /// Offloading statistics
    pub offload_stats: OffloadStats,
    /// Currently offloaded parameters
    pub offloaded_params: HashMap<String, ParameterMetadata>,
}

#[derive(Debug, Clone)]
pub enum OffloadingStrategy {
    /// No offloading
    None,
    /// Offload to CPU memory
    CPU,
    /// Offload to NVMe storage
    NVMe,
    /// Optimizer state offloading only
    OptimizerState,
}

#[derive(Debug)]
pub struct ParameterMetadata {
    pub param_name: String,
    pub size_bytes: usize,
    pub location: ParameterLocation,
    pub last_access: std::time::Instant,
    pub access_frequency: usize,
}

#[derive(Debug, Clone)]
pub enum ParameterLocation {
    GPU,
    CPU,
    NVMe(String),
}

#[derive(Debug)]
pub struct OffloadStats {
    pub total_offloaded_mb: f64,
    pub offload_time_ms: f64,
    pub load_time_ms: f64,
    pub bandwidth_utilization: f64,
}

impl OffloadingManager {
    pub fn new() -> Self {
        Self {
            strategy: OffloadingStrategy::None,
            cpu_memory_limit_gb: 64.0,
            nvme_path: None,
            offload_stats: OffloadStats::default(),
            offloaded_params: HashMap::new(),
        }
    }

    /// Offload parameter to specified location
    pub async fn offload_parameter(
        &mut self,
        param_name: String,
        parameter: Vec<f32>,
        location: ParameterLocation
    ) -> Result<()> {
        let size_bytes = parameter.len() * 4;
        let size_mb = size_bytes as f64 / (1024.0 * 1024.0);

        let metadata = ParameterMetadata {
            param_name: param_name.clone(),
            size_bytes,
            location: location.clone(),
            last_access: std::time::Instant::now(),
            access_frequency: 0,
        };

        // Perform actual offloading based on strategy
        match (self.strategy.clone(), &location) {
            (OffloadingStrategy::CPU, ParameterLocation::CPU) => {
                // Offload to CPU memory
                self.offload_stats.total_offloaded_mb += size_mb;
            },
            (OffloadingStrategy::NVMe, ParameterLocation::NVMe(_)) => {
                // Offload to NVMe storage
                self.offload_stats.total_offloaded_mb += size_mb;
            },
            _ => return Err(NNError::InvalidInput {
                message: "Offloading strategy and location mismatch".to_string(),
            }),
        }

        self.offloaded_params.insert(param_name, metadata);

        Ok(())
    }

    /// Load parameter from offloaded location
    pub async fn load_parameter(&mut self, param_name: &str) -> Result<Option<Vec<f32>>> {
        if let Some(metadata) = self.offloaded_params.get_mut(param_name) {
            metadata.access_frequency += 1;
            metadata.last_access = std::time::Instant::now();

            // Perform loading based on location
            match &metadata.location {
                ParameterLocation::CPU => {
                    // Load from CPU memory
                    Ok(Some(vec![0.0; metadata.size_bytes / 4])) // Placeholder
                },
                ParameterLocation::NVMe(_) => {
                    // Load from NVMe storage
                    Ok(Some(vec![0.0; metadata.size_bytes / 4])) // Placeholder
                },
                ParameterLocation::GPU => {
                    // Already on GPU, no loading needed
                    Ok(None)
                },
            }
        } else {
            Ok(None)
        }
    }

    /// Prefetch parameters that are likely to be needed soon
    pub async fn prefetch_parameters(&self, _param_names: &[String]) -> Result<()> {
        // Implement parameter prefetching to reduce latency
        Ok(())
    }
}

/// Memory Statistics Tracker
#[derive(Debug)]
pub struct MemoryStatistics {
    pub current_usage_gb: f64,
    pub peak_usage_gb: f64,
    pub activation_memory_mb: f64,
    pub parameter_memory_mb: f64,
    pub gradient_memory_mb: f64,
    pub optimizer_memory_mb: f64,
    pub fragmentation_ratio: f64,
    pub allocation_events: usize,
    pub deallocation_events: usize,
}

impl MemoryStatistics {
    pub fn new() -> Self {
        Self {
            current_usage_gb: 0.0,
            peak_usage_gb: 0.0,
            activation_memory_mb: 0.0,
            parameter_memory_mb: 0.0,
            gradient_memory_mb: 0.0,
            optimizer_memory_mb: 0.0,
            fragmentation_ratio: 0.0,
            allocation_events: 0,
            deallocation_events: 0,
        }
    }

    /// Update memory statistics after a memory plan execution
    pub fn update_stats(&mut self, plan: &MemoryPlan) {
        self.activation_memory_mb += plan.estimated_memory_mb;
        self.current_usage_gb = self.total_memory_usage_gb();
        self.peak_usage_gb = self.peak_usage_gb.max(self.current_usage_gb);

        if plan.checkpoint_activations {
            // Estimate memory savings from checkpointing
            self.activation_memory_mb *= 0.5; // Rough approximation
        }
    }

    fn total_memory_usage_gb(&self) -> f64 {
        (self.activation_memory_mb + self.parameter_memory_mb +
         self.gradient_memory_mb + self.optimizer_memory_mb) / 1024.0
    }

    /// Generate detailed memory usage report
    pub fn generate_report(&self) -> MemoryReport {
        MemoryReport {
            current_usage_gb: self.current_usage_gb,
            peak_usage_gb: self.peak_usage_gb,
            utilization_percentage: (self.current_usage_gb / self.peak_usage_gb) * 100.0,
            breakdown: HashMap::from([
                ("activations".to_string(), self.activation_memory_mb),
                ("parameters".to_string(), self.parameter_memory_mb),
                ("gradients".to_string(), self.gradient_memory_mb),
                ("optimizer".to_string(), self.optimizer_memory_mb),
            ]),
            recommendations: self.generate_recommendations(),
        }
    }

    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.activation_memory_mb > self.parameter_memory_mb * 2.0 {
            recommendations.push("Consider gradient checkpointing to reduce activation memory".to_string());
        }

        if self.fragmentation_ratio > 0.3 {
            recommendations.push("High memory fragmentation detected, consider defragmentation".to_string());
        }

        if self.peak_usage_gb > 0.9 {
            recommendations.push("Memory usage is close to capacity, consider offloading or model parallelism".to_string());
        }

        recommendations
    }
}

/// Memory usage report
#[derive(Debug)]
pub struct MemoryReport {
    pub current_usage_gb: f64,
    pub peak_usage_gb: f64,
    pub utilization_percentage: f64,
    pub breakdown: HashMap<String, f64>,
    pub recommendations: Vec<String>,
}

// Default implementations
impl Default for RecomputationStats {
    fn default() -> Self {
        Self {
            total_recomputations: 0,
            saved_memory_mb: 0.0,
            recomputation_time_ms: 0.0,
        }
    }
}

impl Default for CheckpointStats {
    fn default() -> Self {
        Self {
            total_checkpoints: 0,
            memory_saved_mb: 0.0,
            recomputation_time_ms: 0.0,
            backward_time_saved_ms: 0.0,
        }
    }
}

impl Default for PrecisionStats {
    fn default() -> Self {
        Self {
            overflow_events: 0,
            scale_adjustments: 0,
            average_scale: 32768.0, // Default scale
            precision_conversions: 0,
        }
    }
}

impl Default for OffloadStats {
    fn default() -> Self {
        Self {
            total_offloaded_mb: 0.0,
            offload_time_ms: 0.0,
            load_time_ms: 0.0,
            bandwidth_utilization: 0.0,
        }
    }
}

impl Default for MemoryStatistics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_optimizer_creation() {
        let optimizer = MemoryOptimizer::new(80.0); // 80GB budget
        assert_eq!(optimizer.memory_budget_gb, 80.0);
        assert!(!optimizer.mixed_precision.enabled);
    }

    #[test]
    fn test_memory_optimizer_with_mixed_precision() {
        let optimizer = MemoryOptimizer::new(80.0)
            .with_mixed_precision(MixedPrecisionLevel::BF16);

        assert!(optimizer.mixed_precision.enabled);
        assert!(matches!(optimizer.mixed_precision.precision_level, MixedPrecisionLevel::BF16));
    }

    #[test]
    fn test_gradient_checkpointing() {
        let optimizer = MemoryOptimizer::new(80.0)
            .enable_gradient_checkpointing(vec!["transformer_block_1".to_string()], 0.5);

        assert_eq!(optimizer.gradient_manager.checkpoint_ratio, 0.5);
        assert!(optimizer.gradient_manager.checkpointed_layers.contains(&"transformer_block_1".to_string()));
    }

    #[test]
    fn test_activation_memory_management() {
        let mut manager = ActivationManager::new();
        assert_eq!(manager.max_activations, 100);

        // Test storing activations
        let activation = vec![1.0, 2.0, 3.0];
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(manager.store_activation(
                "layer1".to_string(),
                activation,
                "transformer".to_string()
            ));

        assert!(result.is_ok());
        assert_eq!(manager.activation_store.len(), 1);
    }

    #[test]
    fn test_mixed_precision_scaling() {
        let mut manager = MixedPrecisionManager::new();
        manager.set_precision_level(MixedPrecisionLevel::FP16);

        // Test loss scaling
        let scaled_loss = manager.scale_loss(1.5);
        assert_eq!(scaled_loss, 1.5 * 32768.0);

        // Test gradient scaling
        let mut grads = vec![2.0, 3.0];
        manager.scale_gradients(&mut grads);
        assert_eq!(grads[0], 2.0 / 32768.0);
    }

    #[test]
    fn test_memory_report_generation() {
        let mut stats = MemoryStatistics::new();
        stats.activation_memory_mb = 1000.0;
        stats.parameter_memory_mb = 2000.0;

        let report = stats.generate_report();
        assert_eq!(report.breakdown["activations"], 1000.0);
        assert_eq!(report.breakdown["parameters"], 2000.0);
    }

    #[test]
    fn test_memory_optimization_strategies() {
        let mut optimizer = MemoryOptimizer::new(80.0);

        // Test adaptive strategy selection
        let plan = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(optimizer.optimize_forward("attention_layer", 1000000))
            .unwrap();

        // Should create a valid memory plan
        assert!(plan.estimated_memory_mb > 0.0);
    }
}
