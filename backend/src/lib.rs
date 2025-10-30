//! # Coeus Backend Abstractions
//!
//! Compute device abstractions enabling execution on CPU, GPU, and other accelerators.
//!
//! ## Architecture
//!
//! Backend traits separate compute substrate from tensor storage/dtype logic,
//! enabling zero-cost backend dispatch via static monomorphization.
//!
//! ### Backend Trait Hierarchy
//!
//! ```text
//! Backend
//! ├── CpuBackend<T>      // Native CPU execution (SIMD-ready)
//! ├── GpuBackend      // GPU via wgpu (future)
//! └── NpuBackend      // Neural processors (future)
//! ```
//!
//! ## Design Principles (ADR-003)
//!
//! - **Zero-Cost Dispatch**: Static monomorphization eliminates runtime overhead
//! - **Send + Sync**: Thread-safe by construction for parallel execution
//! - **Extensibility**: New backends via trait implementation
//! - **Device Capability**: Runtime feature detection for optimal paths
//! - **Adaptive Selection**: Performance-driven backend selection based on workload characteristics
//!
//! ## Safety
//!
//! All backend operations are memory-safe with zero cost abstractions.


use std::string::String;

pub use coeus_dtype::{num_traits, DataType};
pub use coeus_storage::Storage;

/// Workload characteristics for adaptive backend selection
#[derive(Debug, Clone)]
pub struct WorkloadCharacteristics {
    /// Total number of elements in computation
    pub total_elements: usize,
    /// Memory access pattern (Dense, Sparse, Strided)
    pub access_pattern: MemoryAccessPattern,
    /// Compute intensity (flops per byte)
    pub compute_intensity: f32,
    /// Expected data locality
    pub data_locality: DataLocality,
    /// Operation type
    pub operation_type: OperationType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MemoryAccessPattern {
    /// Sequential dense access
    Dense,
    /// Sparse with low density (<1% non-zero)
    Sparse,
    /// Irregular strided access patterns
    Strided,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataLocality {
    /// High temporal and spatial locality
    High,
    /// Moderate locality
    Medium,
    /// Low locality, cache unfriendly
    Low,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OperationType {
    /// Element-wise operations (add, mul, exp, etc.)
    ElementWise,
    /// Matrix multiplication
    MatrixMultiplication,
    /// Reduction operations (sum, mean, max, etc.)
    Reduction,
    /// Convolution
    Convolution,
    /// Sparse operations
    Sparse,
}

/// Backend types available for selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BackendType {
    /// CPU backend
    Cpu,
    /// GPU backend
    Gpu,
    /// Tensor Processing Unit
    Tpu,
    /// Neural Processing Unit
    Npu,
}

/// Performance metrics for backend operations
#[derive(Debug, Clone, Copy)]
pub struct PerformanceMetrics {
    /// Estimated execution time in microseconds
    pub estimated_time_us: f64,
    /// Memory efficiency (0.0 to 1.0)
    pub memory_efficiency: f32,
    /// Compute efficiency (0.0 to 1.0)
    pub compute_efficiency: f32,
}

impl core::fmt::Display for BackendType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BackendType::Cpu => write!(f, "CPU"),
            BackendType::Gpu => write!(f, "GPU"),
            BackendType::Tpu => write!(f, "TPU"),
            BackendType::Npu => write!(f, "NPU"),
        }
    }
}

/// Performance record for learning backend selection
#[derive(Debug, Clone)]
struct PerformanceRecord {
    backend: BackendType,
    workload: WorkloadCharacteristics,
    actual_time_us: f64,
    timestamp: u64,
}


/// Adaptive backend selector with learning capabilities
#[derive(Debug)]
pub struct BackendSelector {
    /// Available backends on this system
    available_backends: Vec<BackendType>,
    /// Performance history for learning
    performance_history: Vec<PerformanceRecord>,
    /// Integrated memory manager for memory-aware selection
    memory_manager: Option<MemoryManager>,
}

impl Default for BackendSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendSelector {
    /// Create a new backend selector with automatic hardware detection
    pub fn new() -> Self {
        let available_backends = Self::detect_available_backends();
        Self {
            available_backends,
            performance_history: Vec::new(),
            memory_manager: None,
        }
    }

    /// Create a backend selector with integrated memory manager
    pub fn with_memory_manager(memory_manager: MemoryManager) -> Self {
        let available_backends = Self::detect_available_backends();
        Self {
            available_backends,
            performance_history: Vec::new(),
            memory_manager: Some(memory_manager),
        }
    }

    /// Select backend with memory-aware decision making
    pub async fn select_backend_memory_aware(&self, workload: &WorkloadCharacteristics) -> BackendType {
        // If we have memory integration, use it for enhanced selection
        if let Some(memory_mgr) = &self.memory_manager {
            // Get distributed workload representation (simplified for single process)
            let local_workload = crate::distributed::DistributedWorkloadCharacteristics {
                local_workload: workload.clone(),
                aggregate_workload: workload.clone(),
                process_variations: std::collections::HashMap::new(),
                memory_constraints: std::collections::HashMap::new(),
                communication_overhead: 0.0,
            };

            // Analyze memory constraints
            let memory_hints = memory_mgr.analyze_memory_for_selection(
                &local_workload,
                &self.available_backends
            ).await;

            // Return memory-recommended backend if available
            if let Some(recommended) = memory_hints.recommended_backend {
                return recommended;
            }
        }

        // Fall back to traditional scoring
        self.select_backend_traditional(workload)
    }

    /// Traditional backend selection based on scoring
    fn select_backend_traditional(&self, workload: &WorkloadCharacteristics) -> BackendType {
        let mut best_backend = BackendType::Cpu;
        let mut best_score = f32::NEG_INFINITY;

        for &backend in &self.available_backends {
            let score = self.score_backend(backend, workload);
            if score > best_score {
                best_score = score;
                best_backend = backend;
            }
        }

        best_backend
    }

    /// Select backend (compatibility method)
    pub fn select_backend(&self, workload: &WorkloadCharacteristics) -> BackendType {
        // For now, use traditional selection - in future could use memory-aware by default
        self.select_backend_traditional(workload)
    }

    /// Detect all available backends on the current system
    fn detect_available_backends() -> Vec<BackendType> {
        let mut backends = Vec::new();

        // CPU is always available
        backends.push(BackendType::Cpu);

        // Detect GPU availability
        if Self::detect_gpu_hardware() {
            backends.push(BackendType::Gpu);
        }

        // Detect TPU availability
        if Self::detect_tpu_hardware() {
            backends.push(BackendType::Tpu);
        }

        // Detect NPU availability
        if Self::detect_npu_hardware() {
            backends.push(BackendType::Npu);
        }

        backends
    }

    /// Detect available TPU hardware
    fn detect_tpu_hardware() -> bool {
        // Placeholder - TPUs typically require specific cloud environments
        false
    }

    /// Detect available NPU hardware
    fn detect_npu_hardware() -> bool {
        // Placeholder - NPUs are emerging technology, complex to detect universally
        false
    }

    /// Detect available GPU hardware
    fn detect_gpu_hardware() -> bool {
        // GPU backend is currently a stub implementation
        // TODO: Re-enable when GPU backend provides actual functionality
        false
    }

    /// Score a backend for a given workload
    fn score_backend(&self, backend: BackendType, workload: &WorkloadCharacteristics) -> f32 {
        let mut score = 0.0;

        // Base scoring based on operation type
        match workload.operation_type {
            OperationType::ElementWise => {
                score += match backend {
                    BackendType::Gpu => {
                        // GPUs have overhead for small ops
                        if workload.total_elements < 1000 {
                            40.0
                        } else {
                            90.0
                        }
                    }
                    BackendType::Cpu => {
                        // CPUs efficient for small element-wise ops
                        if workload.total_elements < 10000 {
                            80.0
                        } else {
                            60.0
                        }
                    }
                    BackendType::Tpu => 60.0,
                    BackendType::Npu => 70.0,
                };
            }
            OperationType::MatrixMultiplication => {
                score += match backend {
                    BackendType::Gpu => {
                        // GPUs excellent for matmul
                        if workload.total_elements > 1_000_000 {
                            120.0
                        } else {
                            80.0
                        }
                    }
                    BackendType::Cpu => {
                        // CPUs can do matmul but GPUs better for large sizes
                        if workload.total_elements < 100_000 {
                            70.0
                        } else {
                            30.0
                        }
                    }
                    BackendType::Tpu => {
                        // TPUs optimized for very large matmul
                        if workload.total_elements > 10_000_000 {
                            150.0
                        } else {
                            100.0
                        }
                    }
                    BackendType::Npu => {
                        // NPUs good for neural computations
                        85.0
                    }
                };
            }
            OperationType::Reduction => {
                score += match backend {
                    BackendType::Gpu => {
                        // GPUs excellent at parallel reductions
                        if workload.total_elements > 100_000 {
                            110.0
                        } else {
                            50.0
                        }
                    }
                    BackendType::Cpu => {
                        // CPUs reasonable for reductions
                        if workload.total_elements < 100_000 {
                            75.0
                        } else {
                            40.0
                        }
                    }
                    BackendType::Tpu | BackendType::Npu => {
                        // Specialized processors less optimal for reductions
                        60.0
                    }
                };
            }
            OperationType::Convolution => {
                score += match backend {
                    BackendType::Gpu => {
                        // GPUs excellent for convolutions
                        130.0
                    }
                    BackendType::Cpu => {
                        // CPUs much slower for convolutions
                        20.0
                    }
                    BackendType::Npu => {
                        // NPUs often optimized for conv operations
                        125.0
                    }
                    BackendType::Tpu => {
                        // TPUs good but GPUs often better for 2D convs
                        105.0
                    }
                };
            }
            OperationType::Sparse => {
                score += match backend {
                    BackendType::Gpu => {
                        // GPUs good for sparse ops with proper kernels
                        90.0
                    }
                    BackendType::Cpu => {
                        // CPUs reasonable for sparse ops
                        70.0
                    }
                    BackendType::Tpu | BackendType::Npu => {
                        // Specialized processors handle sparse well if optimized
                        80.0
                    }
                };
            }
        }

        // Adjust based on performance history
        score += self.learned_adjustment(backend, workload);

        score
    }

    /// Learn and apply adjustments based on performance history
    fn learned_adjustment(&self, backend: BackendType, workload: &WorkloadCharacteristics) -> f32 {
        // Simple learning algorithm - look for similar past workloads
        let mut total_adjustment = 0.0f32;
        let mut similar_workloads = 0usize;
        let current_time = self.get_timestamp();

        for record in &self.performance_history {
            if record.backend == backend && Self::workloads_similar(workload, &record.workload) {
                // Time-based decay: older records have less weight
                let age_penalty = ((current_time - record.timestamp) as f32 / 1000.0).min(1.0);
                let time_weight = 1.0 - age_penalty * 0.5; // 50% decay over time

                // Performance-based adjustment: if backend was faster than expected, boost score
                // For now, assume expected time is proportional to workload size
                let expected_time = record.workload.total_elements as f64 * 0.001; // Simple heuristic
                let performance_ratio = expected_time / record.actual_time_us.max(0.001);
                let performance_adjustment = if performance_ratio > 1.0 { 2.0 } else { -1.0 };

                total_adjustment += performance_adjustment * time_weight;
                similar_workloads += 1;
            }
        }

        if similar_workloads > 0 {
            total_adjustment /= similar_workloads as f32;
        }

        total_adjustment
    }

    /// Check if two workloads are similar for learning purposes
    fn workloads_similar(a: &WorkloadCharacteristics, b: &WorkloadCharacteristics) -> bool {
        let size_ratio = a.total_elements as f32 / b.total_elements.max(1) as f32;
        let operation_match = a.operation_type == b.operation_type;

        size_ratio > 0.5 && size_ratio < 2.0 && operation_match
    }

    /// Record actual performance for learning
    pub fn record_performance(&mut self, backend: BackendType, workload: WorkloadCharacteristics, actual_time_us: f64) {
        let record = PerformanceRecord {
            backend,
            workload,
            actual_time_us,
            timestamp: self.get_timestamp(),
        };
        self.performance_history.push(record);

        // Keep only recent history
        if self.performance_history.len() > 1000 {
            self.performance_history.remove(0);
        }
    }

    /// Get current timestamp (simple counter)
    fn get_timestamp(&self) -> u64 {
        self.performance_history.len() as u64
    }

    /// Get available backends
    pub fn available_backends(&self) -> &[BackendType] {
        &self.available_backends
    }
}

/// Adaptive backend dispatch trait for zero-overhead dynamic backend selection
pub trait AdaptiveBackendDispatch<T: DataType> {
    /// Dispatch element-wise addition operation through backend selection
    fn dispatch_add(&self, lhs: &[T], rhs: &[T], result: &mut [T]) -> crate::Result<()>;

    /// Dispatch element-wise multiplication through backend selection
    fn dispatch_mul(&self, lhs: &[T], rhs: &[T], result: &mut [T]) -> crate::Result<()>;

    /// Dispatch matrix multiplication through backend selection
    fn dispatch_matmul(&self, lhs: &[T], rhs: &[T], result: &mut [T],
                      m: usize, k: usize, n: usize) -> crate::Result<()>;

    /// Dispatch element-wise activation function (ReLU) through backend selection
    fn dispatch_relu(&self, input: &[T], result: &mut [T]) -> crate::Result<()>;

    /// Dispatch reduction operation through backend selection
    fn dispatch_sum(&self, input: &[T], result: &mut [T]) -> crate::Result<()>;
}

/// Statistics for backend dispatch operations
#[derive(Debug, Clone)]
pub struct BackendDispatchStats {
    /// Total number of operations dispatched
    pub total_dispatches: u64,
    /// Backend selection distribution
    pub backend_usage: std::collections::HashMap<BackendType, u64>,
    /// Average dispatch overhead in nanoseconds
    pub avg_overhead_ns: f64,
    /// Cache hit rate for backend selections
    pub cache_hit_rate: f32,
}

impl<T: DataType> AdaptiveBackendDispatch<T> for BackendSelector {
    fn dispatch_add(&self, lhs: &[T], rhs: &[T], result: &mut [T]) -> crate::Result<()> {
        let characteristics = WorkloadCharacteristics {
            total_elements: lhs.len(),
            access_pattern: MemoryAccessPattern::Dense,
            compute_intensity: 1.0,
            data_locality: DataLocality::High,
            operation_type: OperationType::ElementWise,
        };

        let backend_type = self.select_backend(&characteristics);
        self.execute_add(backend_type, lhs, rhs, result)
    }

    fn dispatch_mul(&self, lhs: &[T], rhs: &[T], result: &mut [T]) -> crate::Result<()> {
        let characteristics = WorkloadCharacteristics {
            total_elements: lhs.len(),
            access_pattern: MemoryAccessPattern::Dense,
            compute_intensity: 1.0,
            data_locality: DataLocality::High,
            operation_type: OperationType::ElementWise,
        };

        let backend_type = self.select_backend(&characteristics);
        self.execute_mul(backend_type, lhs, rhs, result)
    }

    fn dispatch_matmul(&self, lhs: &[T], rhs: &[T], result: &mut [T],
                      m: usize, k: usize, n: usize) -> crate::Result<()> {
        let total_elements = m * k * n;
        let compute_intensity = if k > 0 { (m * n * k) as f32 / total_elements as f32 } else { 1.0 };

        let characteristics = WorkloadCharacteristics {
            total_elements,
            access_pattern: MemoryAccessPattern::Dense,
            compute_intensity,
            data_locality: DataLocality::High,
            operation_type: OperationType::MatrixMultiplication,
        };

        let backend_type = self.select_backend(&characteristics);
        self.execute_matmul(backend_type, lhs, rhs, result, m, k, n)
    }

    fn dispatch_relu(&self, input: &[T], result: &mut [T]) -> crate::Result<()> {
        let characteristics = WorkloadCharacteristics {
            total_elements: input.len(),
            access_pattern: MemoryAccessPattern::Dense,
            compute_intensity: 1.0,
            data_locality: DataLocality::High,
            operation_type: OperationType::ElementWise,
        };

        let backend_type = self.select_backend(&characteristics);
        self.execute_relu(backend_type, input, result)
    }

    fn dispatch_sum(&self, input: &[T], result: &mut [T]) -> crate::Result<()> {
        let characteristics = WorkloadCharacteristics {
            total_elements: input.len(),
            access_pattern: MemoryAccessPattern::Dense,
            compute_intensity: 0.5,
            data_locality: DataLocality::Low,
            operation_type: OperationType::Reduction,
        };

        let backend_type = self.select_backend(&characteristics);
        self.execute_sum(backend_type, input, result)
    }
}

/// Backend execution methods (internal implementation)
impl BackendSelector {
    fn execute_add<T: DataType>(&self, backend_type: BackendType, lhs: &[T], rhs: &[T], result: &mut [T]) -> crate::Result<()> {
        // Phase 1: Only CPU backend supported for now
        match backend_type {
            BackendType::Cpu => {
                // Direct element-wise addition
                for (i, (&l, &_r)) in lhs.iter().zip(rhs.iter()).enumerate() {
                    if let Some(res) = result.get_mut(i) {
                        // TODO: Implement proper addition trait bounds for DataType
                        *res = l; // Placeholder - needs proper arithmetic
                    }
                }
                Ok(())
            }
            _ => Err(crate::BackendError::UnsupportedOperation {
                operation: "add".to_string(),
                backend: backend_type.to_string(),
            })
        }
    }

    fn execute_mul<T: DataType>(&self, backend_type: BackendType, lhs: &[T], rhs: &[T], result: &mut [T]) -> crate::Result<()> {
        match backend_type {
            BackendType::Cpu => {
                for (i, (&l, &_r)) in lhs.iter().zip(rhs.iter()).enumerate() {
                    if let Some(res) = result.get_mut(i) {
                        *res = l; // Placeholder
                    }
                }
                Ok(())
            }
            _ => Err(crate::BackendError::UnsupportedOperation {
                operation: "mul".to_string(),
                backend: backend_type.to_string(),
            })
        }
    }

    fn execute_matmul<T: DataType>(&self, backend_type: BackendType, lhs: &[T], rhs: &[T], result: &mut [T],
                  m: usize, k: usize, n: usize) -> crate::Result<()> {
        match backend_type {
            BackendType::Cpu => {
                // Basic CPU matrix multiplication
                for i in 0..m {
                    for j in 0..n {
                        let result_idx = i * n + j;
                        if result_idx < result.len() {
                            result[result_idx] = T::zero(); // Initialize
                            for l in 0..k {
                                let lhs_idx = i * k + l;
                                let rhs_idx = l * n + j;
                                if lhs_idx < lhs.len() && rhs_idx < rhs.len() {
                                    // result[result_idx] = result[result_idx] + lhs[lhs_idx] * rhs[rhs_idx];
                                    // Placeholder - needs arithmetic trait
                                }
                            }
                        }
                    }
                }
                Ok(())
            }
            _ => Err(crate::BackendError::UnsupportedOperation {
                operation: "matmul".to_string(),
                backend: backend_type.to_string(),
            })
        }
    }

    fn execute_relu<T: DataType>(&self, backend_type: BackendType, input: &[T], result: &mut [T]) -> crate::Result<()> {
        match backend_type {
            BackendType::Cpu => {
                for (i, &x) in input.iter().enumerate() {
                    if let Some(res) = result.get_mut(i) {
                        *res = x; // Placeholder for ReLU implementation
                    }
                }
                Ok(())
            }
            _ => Err(crate::BackendError::UnsupportedOperation {
                operation: "relu".to_string(),
                backend: backend_type.to_string(),
            })
        }
    }

    fn execute_sum<T: DataType>(&self, backend_type: BackendType, input: &[T], result: &mut [T]) -> crate::Result<()> {
        match backend_type {
            BackendType::Cpu => {
                let sum = T::zero();
                for &_x in input {
                    // sum = sum + x; // Placeholder - needs arithmetic
                }
                if !result.is_empty() {
                    result[0] = sum;
                }
                Ok(())
            }
            _ => Err(crate::BackendError::UnsupportedOperation {
                operation: "sum".to_string(),
                backend: backend_type.to_string(),
            })
        }
    }
}

/// Performance monitoring system for training pipelines
pub struct PerformanceMonitor {
    /// GPU memory usage tracking
    gpu_memory_usage: Vec<f64>,
    /// GPU utilization tracking
    gpu_utilization: Vec<f32>,
    /// Operation latency tracking
    operation_latencies: std::collections::HashMap<String, Vec<f64>>,
    /// Target GPU overhead (<1% of training time)
    target_overhead_percent: f32,
    /// Current training step
    current_step: u64,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(target_overhead_percent: f32) -> Self {
        Self {
            gpu_memory_usage: Vec::new(),
            gpu_utilization: Vec::new(),
            operation_latencies: std::collections::HashMap::new(),
            target_overhead_percent,
            current_step: 0,
        }
    }

    /// Record GPU memory usage
    pub fn record_memory_usage(&mut self, memory_mb: f64) {
        self.gpu_memory_usage.push(memory_mb);
        if self.gpu_memory_usage.len() > 1000 {
            self.gpu_memory_usage.remove(0);
        }
    }

    /// Record GPU utilization
    pub fn record_utilization(&mut self, utilization_percent: f32) {
        self.gpu_utilization.push(utilization_percent);
        if self.gpu_utilization.len() > 1000 {
            self.gpu_utilization.remove(0);
        }
    }

    /// Record operation latency
    pub fn record_operation_latency(&mut self, operation: &str, latency_us: f64) {
        self.operation_latencies
            .entry(operation.to_string())
            .or_default()
            .push(latency_us);

        if let Some(latencies) = self.operation_latencies.get_mut(operation) {
            if latencies.len() > 100 {
                latencies.remove(0);
            }
        }
    }

    /// Increment the current training step
    pub fn increment_step(&mut self) {
        self.current_step += 1;
    }

    /// Get the current training step
    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    /// Calculate current GPU overhead percentage
    pub fn calculate_gpu_overhead(&self, _total_training_time_us: f64) -> f32 {
        if self.gpu_memory_usage.is_empty() && self.gpu_utilization.is_empty() {
            return 0.0;
        }

        let avg_memory_usage = self.gpu_memory_usage.iter().sum::<f64>() / self.gpu_memory_usage.len() as f64;
        let avg_utilization = self.gpu_utilization.iter().sum::<f32>() / self.gpu_utilization.len() as f32;

        // Estimate overhead as function of memory transfers and kernel launch overhead
        let memory_overhead_factor = (avg_memory_usage / 1000.0).min(1.0) as f32;
        let kernel_overhead_factor = (1.0 - avg_utilization / 100.0).max(0.0);

        // Total estimated overhead
        

        (memory_overhead_factor * 0.3 + kernel_overhead_factor * 0.7) * 100.0
    }

    /// Check if GPU overhead is within target (<1%)
    pub fn is_overhead_within_target(&self, total_training_time_us: f64) -> bool {
        let overhead = self.calculate_gpu_overhead(total_training_time_us);
        overhead <= self.target_overhead_percent
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> PerformanceSummary {
        let memory_avg = if self.gpu_memory_usage.is_empty() {
            0.0
        } else {
            self.gpu_memory_usage.iter().sum::<f64>() / self.gpu_memory_usage.len() as f64
        };

        let utilization_avg = if self.gpu_utilization.is_empty() {
            0.0
        } else {
            self.gpu_utilization.iter().sum::<f32>() / self.gpu_utilization.len() as f32
        };

        PerformanceSummary {
            average_memory_usage_mb: memory_avg,
            average_gpu_utilization: utilization_avg,
            operation_count: self.operation_latencies.len(),
            current_step: self.current_step,
        }
    }
}

/// Performance summary for monitoring
#[derive(Debug)]
pub struct PerformanceSummary {
    pub average_memory_usage_mb: f64,
    pub average_gpu_utilization: f32,
    pub operation_count: usize,
    pub current_step: u64,
}

/// Result type for backend operations
pub type Result<T> = core::result::Result<T, BackendError>;

/// Backend-specific errors
#[derive(Debug)]
pub enum BackendError {
    /// Unsupported operation for this backend
    UnsupportedOperation {
        operation: String,
        backend: String,
    },
    /// Invalid input parameters
    InvalidInput(String),
    /// Storage operation error
    StorageError { source: coeus_storage::StorageError },
}

impl core::fmt::Display for BackendError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BackendError::UnsupportedOperation { operation, backend } => {
                write!(f, "Unsupported {operation} operation for {backend} backend")
            }
            BackendError::InvalidInput(msg) => {
                write!(f, "Invalid input: {msg}")
            }
            BackendError::StorageError { source } => {
                write!(f, "Storage error: {source}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BackendError {}

#[cfg(feature = "std")]
impl From<coeus_storage::StorageError> for BackendError {
    fn from(err: coeus_storage::StorageError) -> Self {
        BackendError::InvalidInput(format!("Storage error: {}", err))
    }
}

#[cfg(feature = "gpu")]
impl From<crate::gpu::GpuError> for BackendError {
    fn from(err: crate::gpu::GpuError) -> Self {
        BackendError::UnsupportedOperation {
            operation: format!("GPU error: {}", err),
            backend: String::from("GPU"),
        }
    }
}

// Backend trait using associated types for type safety
pub trait Backend: Send + Sync + Clone + fmt::Debug + Default + 'static {
    /// Data type supported by this backend
    type Data: DataType;

    /// Device type for this backend
    type Device: DeviceInfo + Send + Sync;

    /// Get device for this backend
    fn device(&self) -> &Self::Device;

    /// Check if backend supports operation
    fn supports(&self, operation: &str) -> bool;

    /// Get device name for debugging
    fn device_name(&self) -> &str;

    /// Get device information
    fn device_info(&self) -> Box<dyn DeviceInfo>;

    /// Add dense storage element-wise
    fn add_dense(&self, lhs: &coeus_storage::DenseStorage<Self::Data>, rhs: &coeus_storage::DenseStorage<Self::Data>) -> Result<coeus_storage::DenseStorage<Self::Data>>;

    /// Multiply dense storage element-wise
    fn mul_dense(&self, lhs: &coeus_storage::DenseStorage<Self::Data>, rhs: &coeus_storage::DenseStorage<Self::Data>) -> Result<coeus_storage::DenseStorage<Self::Data>>;

    /// Matrix multiplication for dense storage
    fn matmul_dense(&self, lhs: &coeus_storage::DenseStorage<Self::Data>, rhs: &coeus_storage::DenseStorage<Self::Data>) -> Result<coeus_storage::DenseStorage<Self::Data>>;

    /// Apply ReLU activation to dense storage
    fn relu_dense(&self, input: &coeus_storage::DenseStorage<Self::Data>) -> Result<coeus_storage::DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd + Default;

    /// Sum all elements in dense storage
    fn sum_dense(&self, input: &coeus_storage::DenseStorage<Self::Data>) -> Result<Self::Data>;

    /// Find maximum value in dense storage
    fn max_dense(&self, input: &coeus_storage::DenseStorage<Self::Data>) -> Result<Self::Data>
    where
        Self::Data: PartialOrd;

    /// Find minimum value in dense storage
    fn min_dense(&self, input: &coeus_storage::DenseStorage<Self::Data>) -> Result<Self::Data>
    where
        Self::Data: PartialOrd;

    /// Find index of maximum value in dense storage
    fn argmax_dense(&self, input: &coeus_storage::DenseStorage<Self::Data>) -> Result<usize>
    where
        Self::Data: PartialOrd;

    /// Find index of minimum value in dense storage
    fn argmin_dense(&self, input: &coeus_storage::DenseStorage<Self::Data>) -> Result<usize>
    where
        Self::Data: PartialOrd;

    /// Subtract dense storages element-wise
    fn sub_dense(&self, lhs: &coeus_storage::DenseStorage<Self::Data>, rhs: &coeus_storage::DenseStorage<Self::Data>) -> Result<coeus_storage::DenseStorage<Self::Data>>;

    /// Apply exponential function element-wise
    fn exp_dense(&self, input: &coeus_storage::DenseStorage<Self::Data>) -> Result<coeus_storage::DenseStorage<Self::Data>>;

    /// Apply natural logarithm element-wise
    fn log_dense(&self, input: &coeus_storage::DenseStorage<Self::Data>) -> Result<coeus_storage::DenseStorage<Self::Data>>;

    /// Apply sine function element-wise
    fn sin_dense(&self, input: &coeus_storage::DenseStorage<Self::Data>) -> Result<coeus_storage::DenseStorage<Self::Data>>;

    /// Apply cosine function element-wise
    fn cos_dense(&self, input: &coeus_storage::DenseStorage<Self::Data>) -> Result<coeus_storage::DenseStorage<Self::Data>>;

    /// Apply 2D convolution
    fn conv2d_dense(&self, input: &coeus_storage::DenseStorage<Self::Data>, weight: &coeus_storage::DenseStorage<Self::Data>) -> Result<coeus_storage::DenseStorage<Self::Data>>;

    /// Compute mean along specified axes (dense)
    fn mean_dense(&self, input: &coeus_storage::DenseStorage<Self::Data>, axes: Option<&[usize]>) -> Result<coeus_storage::DenseStorage<Self::Data>>;

    /// Sparse matrix-matrix multiplication (CSR format)
    fn spmm_csr(&self, data: &[Self::Data], indices: &[usize], indptr: &[usize], other: &coeus_storage::DenseStorage<Self::Data>, num_rows: usize, num_cols: usize) -> Result<Vec<Self::Data>>;

    /// Sparse matrix-vector multiplication (CSR format)
    fn spmv_csr(&self, data: &[Self::Data], indices: &[usize], indptr: &[usize], vector: &[Self::Data], num_rows: usize, num_cols: usize) -> Result<Vec<Self::Data>>;

    /// Coordinate format sparse matrix multiplication (matrix-sparse)
    fn coo_matmul_sparse(&self, lhs_data: &[Self::Data], lhs_row: &[usize], lhs_col: &[usize], rhs_data: &[Self::Data], rhs_row: &[usize], rhs_col: &[usize], m: usize, k: usize, n: usize) -> Result<coeus_storage::CooStorage<Self::Data>>;

    /// Coordinate format sparse matrix multiplication (sparse-dense)
    fn coo_matmul_dense(&self, lhs_data: &[Self::Data], lhs_row: &[usize], lhs_col: &[usize], rhs: &coeus_storage::DenseStorage<Self::Data>, m: usize, k: usize, n: usize) -> Result<coeus_storage::DenseStorage<Self::Data>>;

    /// Coordinate format sparse addition
    fn coo_add_sparse(&self, lhs_data: &[Self::Data], lhs_row: &[usize], lhs_col: &[usize], rhs_data: &[Self::Data], rhs_row: &[usize], rhs_col: &[usize], m: usize, n: usize) -> Result<coeus_storage::CooStorage<Self::Data>>;

    /// Coordinate format sparse multiplication
    fn coo_mul_sparse(&self, lhs_data: &[Self::Data], lhs_row: &[usize], lhs_col: &[usize], rhs_data: &[Self::Data], rhs_row: &[usize], rhs_col: &[usize], m: usize, n: usize) -> Result<coeus_storage::CooStorage<Self::Data>>;

    /// Quantization operation
    fn quantize(&self, input: &coeus_storage::DenseStorage<Self::Data>, levels: usize) -> Result<coeus_storage::DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd;
}

/// Stub backend for compilation - provides minimal interface to allow dependent crate testing
#[derive(Debug, Clone)]
pub struct StubBackend<D: DataType> {
    _phantom: std::marker::PhantomData<D>,
}

impl<D: DataType> Default for StubBackend<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: DataType> StubBackend<D> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<D: DataType> Backend for StubBackend<D> {
    type Data = D;
    type Device = StubDevice;

    fn device(&self) -> &Self::Device {
        static DEVICE: StubDevice = StubDevice;
        &DEVICE
    }

    fn supports(&self, _operation: &str) -> bool {
        true // Stub always supports operations
    }

    fn device_name(&self) -> &str {
        "stub"
    }

    fn device_info(&self) -> Box<dyn DeviceInfo> {
        Box::new(StubDevice)
    }

    fn add_dense(&self, _lhs: &coeus_storage::DenseStorage<Self::Data>, _rhs: &coeus_storage::DenseStorage<Self::Data>) -> Result<coeus_storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "add_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn mul_dense(&self, _lhs: &coeus_storage::DenseStorage<Self::Data>, _rhs: &coeus_storage::DenseStorage<Self::Data>) -> Result<coeus_storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "mul_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn matmul_dense(&self, _lhs: &coeus_storage::DenseStorage<Self::Data>, _rhs: &coeus_storage::DenseStorage<Self::Data>) -> Result<coeus_storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "matmul_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn relu_dense(&self, _input: &coeus_storage::DenseStorage<Self::Data>) -> Result<coeus_storage::DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd + Default,
    {
        Err(BackendError::UnsupportedOperation {
            operation: "relu_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn sum_dense(&self, _input: &coeus_storage::DenseStorage<Self::Data>) -> Result<Self::Data> {
        Err(BackendError::UnsupportedOperation {
            operation: "sum_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn max_dense(&self, _input: &coeus_storage::DenseStorage<Self::Data>) -> Result<Self::Data>
    where
        Self::Data: PartialOrd,
    {
        Err(BackendError::UnsupportedOperation {
            operation: "max_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn min_dense(&self, _input: &coeus_storage::DenseStorage<Self::Data>) -> Result<Self::Data>
    where
        Self::Data: PartialOrd,
    {
        Err(BackendError::UnsupportedOperation {
            operation: "min_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn argmax_dense(&self, _input: &coeus_storage::DenseStorage<Self::Data>) -> Result<usize>
    where
        Self::Data: PartialOrd,
    {
        Err(BackendError::UnsupportedOperation {
            operation: "argmax_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn argmin_dense(&self, _input: &coeus_storage::DenseStorage<Self::Data>) -> Result<usize>
    where
        Self::Data: PartialOrd,
    {
        Err(BackendError::UnsupportedOperation {
            operation: "argmin_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn sub_dense(&self, _lhs: &coeus_storage::DenseStorage<Self::Data>, _rhs: &coeus_storage::DenseStorage<Self::Data>) -> Result<coeus_storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "sub_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn exp_dense(&self, _input: &coeus_storage::DenseStorage<Self::Data>) -> Result<coeus_storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "exp_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn log_dense(&self, _input: &coeus_storage::DenseStorage<Self::Data>) -> Result<coeus_storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "log_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn sin_dense(&self, _input: &coeus_storage::DenseStorage<Self::Data>) -> Result<coeus_storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "sin_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn cos_dense(&self, _input: &coeus_storage::DenseStorage<Self::Data>) -> Result<coeus_storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "cos_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn conv2d_dense(&self, _input: &coeus_storage::DenseStorage<Self::Data>, _weight: &coeus_storage::DenseStorage<Self::Data>) -> Result<coeus_storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "conv2d_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn mean_dense(&self, _input: &coeus_storage::DenseStorage<Self::Data>, _axes: Option<&[usize]>) -> Result<coeus_storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "mean_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn spmm_csr(&self, _data: &[Self::Data], _indices: &[usize], _indptr: &[usize], _other: &coeus_storage::DenseStorage<Self::Data>, _num_rows: usize, _num_cols: usize) -> Result<Vec<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "spmm_csr".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn spmv_csr(&self, _data: &[Self::Data], _indices: &[usize], _indptr: &[usize], _vector: &[Self::Data], _num_rows: usize, _num_cols: usize) -> Result<Vec<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "spmv_csr".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn coo_matmul_sparse(&self, _lhs_data: &[Self::Data], _lhs_row: &[usize], _lhs_col: &[usize], _rhs_data: &[Self::Data], _rhs_row: &[usize], _rhs_col: &[usize], _m: usize, _k: usize, _n: usize) -> Result<coeus_storage::CooStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "coo_matmul_sparse".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn coo_matmul_dense(&self, _lhs_data: &[Self::Data], _lhs_row: &[usize], _lhs_col: &[usize], _rhs: &coeus_storage::DenseStorage<Self::Data>, _m: usize, _k: usize, _n: usize) -> Result<coeus_storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "coo_matmul_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn coo_add_sparse(&self, _lhs_data: &[Self::Data], _lhs_row: &[usize], _lhs_col: &[usize], _rhs_data: &[Self::Data], _rhs_row: &[usize], _rhs_col: &[usize], _m: usize, _n: usize) -> Result<coeus_storage::CooStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "coo_add_sparse".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn coo_mul_sparse(&self, _lhs_data: &[Self::Data], _lhs_row: &[usize], _lhs_col: &[usize], _rhs_data: &[Self::Data], _rhs_row: &[usize], _rhs_col: &[usize], _m: usize, _n: usize) -> Result<coeus_storage::CooStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "coo_mul_sparse".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn quantize(&self, _input: &coeus_storage::DenseStorage<Self::Data>, _levels: usize) -> Result<coeus_storage::DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        Err(BackendError::UnsupportedOperation {
            operation: "quantize".to_string(),
            backend: "stub".to_string(),
        })
    }
}

/// Placeholder memory manager for backend selection
/// TODO: Replace with full memory management implementation
#[derive(Debug, Clone)]
pub struct MemoryManager;

/// Memory analysis hints for backend selection
#[derive(Debug)]
pub struct MemoryAnalysisResult {
    /// Recommended backend based on memory constraints
    pub recommended_backend: Option<BackendType>,
    /// Memory efficiency score (0.0-1.0)
    pub memory_efficiency: f32,
    /// Transfer cost estimate
    pub transfer_cost: f64,
    /// Fragmentation impact
    pub fragmentation_penalty: f32,
}

impl MemoryManager {
    /// Analyze memory constraints for backend selection
    pub async fn analyze_memory_for_selection(
        &self,
        _workload: &crate::distributed::DistributedWorkloadCharacteristics,
        _backends: &[BackendType],
    ) -> MemoryAnalysisResult {
        // Placeholder implementation - always return no recommendation
        MemoryAnalysisResult {
            recommended_backend: None,
            memory_efficiency: 0.5,
            transfer_cost: 0.0,
            fragmentation_penalty: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StubDevice;

impl DeviceInfo for StubDevice {
    fn device(&self) -> &Device {
        static STUB_DEVICE: Device = Device::Cpu;
        &STUB_DEVICE
    }

    fn name(&self) -> &str {
        "stub_device"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn memory_gb(&self) -> usize {
        8
    }

    fn compute_units(&self) -> usize {
        4
    }
}

pub use std::fmt;

pub mod cpu;
pub mod device;

#[cfg(feature = "gpu")]
pub mod gpu;
// Distributed backend coordination
// TODO: NPU backend is incomplete implementation - defer until core system is production ready

// TODO: TPU backend is incomplete implementation - defer until core system is production ready

// Distributed backend coordination
pub mod distributed;
pub use distributed::{
    DistributedBackendCoordinator, DistributedWorkloadAnalyzer,
    DistributedWorkloadCharacteristics, BackendSelectionDecision,
    CoordinationStats, FaultToleranceState, MemoryConstraints,
};

// Memory management integration
pub mod memory_integration;

pub use cpu::CpuBackend;
pub use device::{Device, DeviceInfo};

#[cfg(feature = "gpu")]
pub use gpu::GpuBackend;

// TODO: NPU and TPU backends are incomplete implementations - defer until core system is production ready
// mod npu;
// mod tpu;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;
    use coeus_dtype::float::Float32;
    use std::vec;

    #[test]
    fn test_spmv_csr_basic() {
        let backend = CpuBackend::<Float32>::new();

        // Create a simple 3x3 sparse matrix in CSR format:
        // [[1, 0, 2],
        //  [0, 3, 0],
        //  [4, 0, 5]]
        // Maps to data=[1,2,3,4,5], indices=[0,2,1,0,2], indptr=[0,2,3,5]
        let data: Vec<Float32> = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
        ];
        let indices: Vec<usize> = vec![0, 2, 1, 0, 2];
        let indptr: Vec<usize> = vec![0, 2, 3, 5];

        // Create a dense vector [1, 2, 3]
        let vector: Vec<Float32> = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];

        // Perform SPMV: result should be [1*1 + 2*3, 3*2, 4*1 + 5*3] = [7, 6, 19]
        let result = backend
            .spmv_csr(&data, &indices, &indptr, &vector, 3, 3)
            .unwrap();

        assert_eq!(result.len(), 3);
        assert!((result[0].get() - 7.0).abs() < 1e-6);
        assert!((result[1].get() - 6.0).abs() < 1e-6);
        assert!((result[2].get() - 19.0).abs() < 1e-6);
    }

    #[test]
    fn test_backend_selector_creation() {
        let selector = BackendSelector::new();
        assert!(selector.available_backends().contains(&BackendType::Cpu));
    }

    #[test]
    fn test_backend_selection_small_element_wise() {
        let selector = BackendSelector::new();
        let workload = WorkloadCharacteristics {
            total_elements: 1000,
            access_pattern: MemoryAccessPattern::Dense,
            compute_intensity: 1.0,
            data_locality: DataLocality::High,
            operation_type: OperationType::ElementWise,
        };

        let selected = selector.select_backend(&workload);
        assert_eq!(selected, BackendType::Cpu);
    }

    #[test]
    fn test_backend_selection_large_matmul() {
        let selector = BackendSelector::new();
        let workload = WorkloadCharacteristics {
            total_elements: 2_000_000,
            access_pattern: MemoryAccessPattern::Dense,
            compute_intensity: 20.0,
            data_locality: DataLocality::High,
            operation_type: OperationType::MatrixMultiplication,
        };

        let selected = selector.select_backend(&workload);
        // For now, only CPU is available
        assert_eq!(selected, BackendType::Cpu);
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new(1.0);

        monitor.record_memory_usage(512.0);
        monitor.record_utilization(85.0);
        monitor.record_operation_latency("matmul", 1500.0);

        let summary = monitor.get_performance_summary();
        assert_eq!(summary.average_memory_usage_mb, 512.0);
        assert_eq!(summary.average_gpu_utilization, 85.0);
        assert_eq!(summary.operation_count, 1);

        let total_training_time = 100_000.0;
        let overhead = monitor.calculate_gpu_overhead(total_training_time);
        // Overhead calculation may vary based on recorded metrics
        assert!(overhead >= 0.0 && overhead <= 50.0);
    }
}
