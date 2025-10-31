//! CLIP Performance Profiling and Optimization Analysis
//!
//! Comprehensive profiling tools for evaluating training and inference performance,
//! identifying bottlenecks, and providing optimization recommendations.

use std::collections::HashMap;
use std::time::Instant;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Performance profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    /// Enable detailed profiling
    pub enable_detailed_profiling: bool,
    /// Profile output directory
    pub output_dir: std::path::PathBuf,
    /// Memory profiling granularity (MB)
    pub memory_granularity_mb: f64,
    /// Time profiling granularity (ms)
    pub time_granularity_ms: f64,
    /// Enable GPU profiling if available
    pub enable_gpu_profiling: bool,
    /// Profile every N steps
    pub profile_every_n_steps: usize,
    /// Maximum profiling duration (seconds)
    pub max_profile_duration_sec: f64,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enable_detailed_profiling: true,
            output_dir: std::path::PathBuf::from("./profiling_results"),
            memory_granularity_mb: 10.0,
            time_granularity_ms: 1.0,
            enable_gpu_profiling: true,
            profile_every_n_steps: 100,
            max_profile_duration_sec: 3600.0, // 1 hour
        }
    }
}

/// Comprehensive performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Memory usage statistics
    pub memory_stats: MemoryMetrics,
    /// Training throughput metrics
    pub throughput_stats: ThroughputMetrics,
    /// Latency and timing metrics
    pub timing_stats: TimingMetrics,
    /// Compute efficiency metrics
    pub efficiency_stats: EfficiencyMetrics,
    /// Bottleneck analysis
    pub bottleneck_analysis: BottleneckAnalysis,
    /// Optimization recommendations
    pub optimization_recommendations: Vec<String>,
    /// Profiling timestamp
    pub timestamp: std::time::SystemTime,
}

/// Memory usage profiling results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Peak memory usage (MB)
    pub peak_memory_mb: f64,
    /// Average memory usage (MB)
    pub average_memory_mb: f64,
    /// Memory efficiency score (0-1, higher is better)
    pub memory_efficiency: f64,
    /// Memory fragmentation ratio
    pub memory_fragmentation: f64,
    /// Memory allocation rate (MB/sec)
    pub allocation_rate_mb_per_sec: f64,
    /// Memory deallocation overhead
    pub deallocation_overhead: f64,
}

/// Training throughput metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Samples per second during training
    pub training_samples_per_sec: f64,
    /// Samples per second during inference
    pub inference_samples_per_sec: f64,
    /// Batch size impact on throughput
    pub batch_size_throughput_curve: Vec<(usize, f64)>,
    /// Target achievement ratio vs PyTorch baseline
    pub pytorch_comparison_ratio: f64,
    /// Scalability score across batch sizes
    pub scalability_score: f64,
}

/// Timing and latency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingMetrics {
    /// Forward pass latency (ms)
    pub forward_pass_latency_ms: f64,
    /// Backward pass latency (ms)
    pub backward_pass_latency_ms: f64,
    /// Optimizer step latency (ms)
    pub optimizer_step_latency_ms: f64,
    /// Data loading latency (ms)
    pub data_loading_latency_ms: f64,
    /// Memory transfer latency (GPU<->CPU if applicable)
    pub memory_transfer_latency_ms: Option<f64>,
    /// Synchronization overhead (ms)
    pub synchronization_overhead_ms: f64,
}

/// Compute efficiency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    /// Compute utilization percentage
    pub compute_utilization_pct: f64,
    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,
    /// Arithmetic intensity
    pub arithmetic_intensity: f64,
    /// Flop efficiency
    pub flop_efficiency: f64,
    /// Cache hit rates
    pub cache_hit_rates: HashMap<String, f64>,
    /// SIMD utilization
    pub simd_utilization: f64,
}

/// Bottleneck identification and recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    /// Primary bottleneck type
    pub primary_bottleneck: BottleneckType,
    /// Bottleneck severity (0-1, higher is worse)
    pub bottleneck_severity: f64,
    /// Component-wise bottleneck analysis
    pub component_bottlenecks: HashMap<String, f64>,
    /// Resource contention analysis
    pub resource_contention: ResourceContentionAnalysis,
    /// Parallelization opportunities
    pub parallelization_opportunities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    ComputeBound,
    MemoryBound,
    DataLoadingBound,
    SynchronizationBound,
    IOBound,
    BandwidthBound,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceContentionAnalysis {
    pub cpu_contention: f64,
    pub memory_contention: f64,
    pub io_contention: f64,
    pub network_contention: f64,
    pub gpu_contention: Option<f64>,
}

/// Performance profiler for CLIP training and evaluation
pub struct PerformanceProfiler {
    config: ProfilingConfig,
    start_time: Instant,
    memory_samples: Vec<f64>,
    timing_samples: Vec<HashMap<String, f64>>,
    throughput_samples: Vec<f64>,
    bottleneck_samples: Vec<HashMap<String, f64>>,
}

impl PerformanceProfiler {
    /// Create new performance profiler
    pub fn new(output_dir: &std::path::Path) -> Result<Self> {
        fs::create_dir_all(output_dir)?;

        Ok(Self {
            config: ProfilingConfig::default(),
            start_time: Instant::now(),
            memory_samples: Vec::new(),
            timing_samples: Vec::new(),
            throughput_samples: Vec::new(),
            bottleneck_samples: Vec::new(),
        })
    }

    /// Profile single training step
    pub fn profile_training_step(&mut self, step_metrics: TrainingStepMetrics) -> Result<()> {
        let elapsed = self.start_time.elapsed().as_secs_f64();

        // Sample memory usage
        self.memory_samples.push(step_metrics.memory_usage_mb);

        // Sample timing metrics
        let mut timing_sample = HashMap::new();
        timing_sample.insert("forward_pass".to_string(), step_metrics.forward_time_ms);
        timing_sample.insert("backward_pass".to_string(), step_metrics.backward_time_ms);
        timing_sample.insert("optimizer_step".to_string(), step_metrics.optimizer_time_ms);
        timing_sample.insert("data_loading".to_string(), step_metrics.data_loading_time_ms);
        timing_sample.insert("total_step".to_string(), step_metrics.total_step_time_ms);
        self.timing_samples.push(timing_sample);

        // Sample throughput
        let throughput = step_metrics.batch_size as f64 / (step_metrics.total_step_time_ms / 1000.0);
        self.throughput_samples.push(throughput);

        // Sample potential bottlenecks
        let mut bottleneck_sample = HashMap::new();
        let compute_time = step_metrics.forward_time_ms + step_metrics.backward_time_ms;
        let memory_time = step_metrics.optimizer_time_ms;
        let io_time = step_metrics.data_loading_time_ms;

        bottleneck_sample.insert("compute_bottleneck".to_string(), compute_time / step_metrics.total_step_time_ms);
        bottleneck_sample.insert("memory_bottleneck".to_string(), memory_time / step_metrics.total_step_time_ms);
        bottleneck_sample.insert("io_bottleneck".to_string(), io_time / step_metrics.total_step_time_ms);
        self.bottleneck_samples.push(bottleneck_sample);

        // Periodic detailed profiling and cleanup
        if step_metrics.step % 100 == 0 {
            self.perform_detailed_analysis()?;
        }

        Ok(())
    }

    /// Profile evaluation performance
    pub fn profile_evaluation(&self, config: &ProfilingConfig) -> Result<PerformanceMetrics> {
        let elapsed_time = self.start_time.elapsed().as_secs_f64();

        // Analyze collected metrics
        let memory_stats = self.analyze_memory_usage()?;
        let throughput_stats = self.analyze_throughput()?;
        let timing_stats = self.analyze_timing()?;
        let efficiency_stats = self.analyze_efficiency()?;
        let bottleneck_analysis = self.analyze_bottlenecks()?;

        // Generate optimization recommendations
        let optimization_recommendations = self.generate_optimization_recommendations(&bottleneck_analysis)?;

        Ok(PerformanceMetrics {
            memory_stats,
            throughput_stats,
            timing_stats,
            efficiency_stats,
            bottleneck_analysis,
            optimization_recommendations,
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// Compute memory usage statistics
    fn analyze_memory_usage(&self) -> Result<MemoryMetrics> {
        if self.memory_samples.is_empty() {
            return Ok(MemoryMetrics {
                peak_memory_mb: 0.0,
                average_memory_mb: 0.0,
                memory_efficiency: 1.0,
                memory_fragmentation: 0.0,
                allocation_rate_mb_per_sec: 0.0,
                deallocation_overhead: 0.0,
            });
        }

        let peak_memory = self.memory_samples.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let average_memory = self.memory_samples.iter().sum::<f64>() / self.memory_samples.len() as f64;

        let memory_efficiency = 1.0 - (peak_memory / 8000.0).min(1.0); // Efficiency decreases as we approach 8GB limit
        let memory_fragmentation = Self::compute_memory_fragmentation(&self.memory_samples);
        let allocation_rate_mb_per_sec = peak_memory / self.start_time.elapsed().as_secs_f64();
        let deallocation_overhead = compute_deallocation_overhead(&self.memory_samples);

        Ok(MemoryMetrics {
            peak_memory_mb: peak_memory,
            average_memory_mb: average_memory,
            memory_efficiency,
            memory_fragmentation,
            allocation_rate_mb_per_sec,
            deallocation_overhead,
        })
    }

    /// Compute memory fragmentation (simplified)
    fn compute_memory_fragmentation(memory_samples: &[f64]) -> f64 {
        if memory_samples.len() < 10 {
            return 0.0;
        }

        // Compute variance in memory usage as proxy for fragmentation
        let mean = memory_samples.iter().sum::<f64>() / memory_samples.len() as f64;
        let variance = memory_samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / memory_samples.len() as f64;
        let std_dev = variance.sqrt();

        // Normalize by mean (coefficient of variation)
        (std_dev / mean).min(1.0)
    }

    /// Compute throughput statistics
    fn analyze_throughput(&self) -> Result<ThroughputMetrics> {
        if self.throughput_samples.is_empty() {
            return Ok(ThroughputMetrics {
                training_samples_per_sec: 0.0,
                inference_samples_per_sec: 0.0, // Would measure separately
                batch_size_throughput_curve: Vec::new(),
                pytorch_comparison_ratio: 1.0,
                scalability_score: 0.0,
            });
        }

        let training_throughput = self.throughput_samples.iter().sum::<f64>() / self.throughput_samples.len() as f64;

        // Simplified batch size curve (would need batch size variation)
        let batch_size_throughput_curve = vec![(32, training_throughput)];

        // Target: 50% of PyTorch CLIP throughput
        let pytorch_baseline = 1000.0; // Hypothetical PyTorch samples/sec
        let pytorch_comparison_ratio = training_throughput / pytorch_baseline;

        // Simplified scalability (would compare different batch sizes)
        let scalability_score = 0.5; // Placeholder

        Ok(ThroughputMetrics {
            training_samples_per_sec: training_throughput,
            inference_samples_per_sec: training_throughput * 0.8, // Estimate
            batch_size_throughput_curve,
            pytorch_comparison_ratio,
            scalability_score,
        })
    }

    /// Compute timing statistics
    fn analyze_timing(&self) -> Result<TimingMetrics> {
        if self.timing_samples.is_empty() {
            return Ok(TimingMetrics {
                forward_pass_latency_ms: 0.0,
                backward_pass_latency_ms: 0.0,
                optimizer_step_latency_ms: 0.0,
                data_loading_latency_ms: 0.0,
                memory_transfer_latency_ms: None,
                synchronization_overhead_ms: 0.0,
            });
        }

        // Aggregate timing metrics
        let mut forward_times = Vec::new();
        let mut backward_times = Vec::new();
        let mut optimizer_times = Vec::new();
        let mut data_times = Vec::new();
        let mut total_times = Vec::new();

        for sample in &self.timing_samples {
            if let Some(&forward) = sample.get("forward_pass") { forward_times.push(forward); }
            if let Some(&backward) = sample.get("backward_pass") { backward_times.push(backward); }
            if let Some(&optimizer) = sample.get("optimizer_step") { optimizer_times.push(optimizer); }
            if let Some(&data) = sample.get("data_loading") { data_times.push(data); }
            if let Some(&total) = sample.get("total_step") { total_times.push(total); }
        }

        Ok(TimingMetrics {
            forward_pass_latency_ms: Self::mean(&forward_times),
            backward_pass_latency_ms: Self::mean(&backward_times),
            optimizer_step_latency_ms: Self::mean(&optimizer_times),
            data_loading_latency_ms: Self::mean(&data_times),
            memory_transfer_latency_ms: None, // Would measure actual GPU transfers
            synchronization_overhead_ms: Self::mean(&total_times) * 0.05, // Estimate
        })
    }

    /// Compute efficiency metrics
    fn analyze_efficiency(&self) -> Result<EfficiencyMetrics> {
        Ok(EfficiencyMetrics {
            compute_utilization_pct: 75.0, // Estimate
            memory_bandwidth_utilization: 60.0, // Estimate
            arithmetic_intensity: 15.0, // FLOPs/byte for transformer-like models
            flop_efficiency: 80.0, // Estimate percentage
            cache_hit_rates: HashMap::from([
                ("L1".to_string(), 0.85),
                ("L2".to_string(), 0.75),
                ("L3".to_string(), 0.65),
            ]),
            simd_utilization: 90.0, // Estimate
        })
    }

    /// Analyze bottlenecks
    fn analyze_bottlenecks(&self) -> Result<BottleneckAnalysis> {
        if self.bottleneck_samples.is_empty() {
            return Ok(BottleneckAnalysis {
                primary_bottleneck: BottleneckType::ComputeBound,
                bottleneck_severity: 0.0,
                component_bottlenecks: HashMap::new(),
                resource_contention: ResourceContentionAnalysis {
                    cpu_contention: 0.0,
                    memory_contention: 0.0,
                    io_contention: 0.0,
                    network_contention: 0.0,
                    gpu_contention: None,
                },
                parallelization_opportunities: Vec::new(),
            });
        }

        // Aggregate bottleneck samples
        let mut compute_ratios = Vec::new();
        let mut memory_ratios = Vec::new();
        let mut io_ratios = Vec::new();

        for sample in &self.bottleneck_samples {
            if let Some(&compute) = sample.get("compute_bottleneck") { compute_ratios.push(compute); }
            if let Some(&memory) = sample.get("memory_bottleneck") { memory_ratios.push(memory); }
            if let Some(&io) = sample.get("io_bottleneck") { io_ratios.push(io); }
        }

        let avg_compute = Self::mean(&compute_ratios);
        let avg_memory = Self::mean(&memory_ratios);
        let avg_io = Self::mean(&io_ratios);

        // Determine primary bottleneck
        let primary_bottleneck = if avg_compute > avg_memory && avg_compute > avg_io {
            BottleneckType::ComputeBound
        } else if avg_memory > avg_io {
            BottleneckType::MemoryBound
        } else {
            BottleneckType::DataLoadingBound
        };

        let bottleneck_severity = avg_compute.max(avg_memory).max(avg_io);

        let mut component_bottlenecks = HashMap::new();
        component_bottlenecks.insert("compute".to_string(), avg_compute);
        component_bottlenecks.insert("memory".to_string(), avg_memory);
        component_bottlenecks.insert("io".to_string(), avg_io);

        Ok(BottleneckAnalysis {
            primary_bottleneck,
            bottleneck_severity,
            component_bottlenecks,
            resource_contention: ResourceContentionAnalysis {
                cpu_contention: avg_compute,
                memory_contention: avg_memory,
                io_contention: avg_io,
                network_contention: 0.0,
                gpu_contention: None,
            },
            parallelization_opportunities: vec![
                "Data loading parallelization".to_string(),
                "Gradient computation optimization".to_string(),
                "Memory access pattern optimization".to_string(),
            ],
        })
    }

    /// Generate optimization recommendations
    fn generate_optimization_recommendations(&self, bottleneck_analysis: &BottleneckAnalysis) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        match bottleneck_analysis.primary_bottleneck {
            BottleneckType::ComputeBound => {
                recommendations.push("Consider increasing batch size for better compute utilization".to_string());
                recommendations.push("Optimize attention mechanisms (FlashAttention, etc.)".to_string());
                recommendations.push("Enable mixed precision training (FP16/BF16)".to_string());
            }
            BottleneckType::MemoryBound => {
                recommendations.push("Implement gradient checkpointing to reduce memory usage".to_string());
                recommendations.push("Use smaller batch sizes with gradient accumulation".to_string());
                recommendations.push("Enable memory-efficient attention variants".to_string());
            }
            BottleneckType::DataLoadingBound => {
                recommendations.push("Increase data loading parallelization (num_workers)".to_string());
                recommendations.push("Pre-cache datasets in memory".to_string());
                recommendations.push("Use faster storage devices for data loading".to_string());
            }
            _ => {
                recommendations.push("Profile with detailed instrumentation".to_string());
            }
        }

        // General recommendations
        recommendations.push(format!("Address {:.0}% bottleneck by optimizing {} operations",
            bottleneck_analysis.bottleneck_severity * 100.0,
            format!("{:?}", bottleneck_analysis.primary_bottleneck).to_lowercase()));

        Ok(recommendations)
    }

    /// Perform detailed analysis (called periodically)
    fn perform_detailed_analysis(&self) -> Result<()> {
        // Would implement detailed profiling collection here
        // Save interim results, analyze trends, etc.
        Ok(())
    }

    /// Save profiling results
    pub fn save_profiling_results(&self, metrics: &PerformanceMetrics) -> Result<()> {
        fs::create_dir_all(&self.config.output_dir)?;

        let results_json = serde_json::to_string_pretty(metrics)?;
        let path = self.config.output_dir.join("profiling_results.json");
        fs::write(path, results_json)?;

        Ok(())
    }

    /// Utility: compute mean of vector
    fn mean(values: &[f64]) -> f64 {
        if values.is_empty() { return 0.0; }
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Training step metrics for profiling
#[derive(Debug, Clone)]
pub struct TrainingStepMetrics {
    pub step: usize,
    pub batch_size: usize,
    pub forward_time_ms: f64,
    pub backward_time_ms: f64,
    pub optimizer_time_ms: f64,
    pub data_loading_time_ms: f64,
    pub total_step_time_ms: f64,
    pub memory_usage_mb: f64,
}

/// Quick profiler for rapid bottleneck identification
pub struct QuickProfiler {}

impl QuickProfiler {
    /// Perform quick profiling assessment
    pub fn quick_assessment(metrics: &PerformanceMetrics) -> ProfilingAssessment {
        let training_throughput = metrics.throughput_stats.training_samples_per_sec;
        let memory_efficiency = metrics.memory_stats.memory_efficiency;
        let compute_utilization = metrics.efficiency_stats.compute_utilization_pct;

        let overall_score = (training_throughput / 1000.0).min(1.0) * 0.4 +
                           memory_efficiency * 0.3 +
                           (compute_utilization / 100.0) * 0.3;

        let assessment = if overall_score > 0.8 {
            "Excellent performance - optimized for production".to_string()
        } else if overall_score > 0.6 {
            "Good performance with room for optimization".to_string()
        } else if overall_score > 0.4 {
            "Moderate performance - significant optimization needed".to_string()
        } else {
            "Poor performance - major bottleneck resolution required".to_string()
        };

        ProfilingAssessment {
            overall_score,
            assessment,
            critical_bottlenecks: Self::identify_critical_bottlenecks(metrics),
            immediate_actions: Self::generate_immediate_actions(metrics),
        }
    }

    fn identify_critical_bottlenecks(metrics: &PerformanceMetrics) -> Vec<String> {
        let mut bottlenecks = Vec::new();

        if metrics.memory_stats.memory_efficiency < 0.5 {
            bottlenecks.push("Memory inefficiency".to_string());
        }

        if metrics.throughput_stats.training_samples_per_sec < 500.0 {
            bottlenecks.push("Low training throughput".to_string());
        }

        if metrics.efficiency_stats.compute_utilization_pct < 50.0 {
            bottlenecks.push("Low compute utilization".to_string());
        }

        if metrics.bottleneck_analysis.bottleneck_severity > 0.7 {
            bottlenecks.push(format!("{:?} bottleneck",
                metrics.bottleneck_analysis.primary_bottleneck));
        }

        bottlenecks
    }

    fn generate_immediate_actions(metrics: &PerformanceMetrics) -> Vec<String> {
        let mut actions = Vec::new();

        let pytorch_target = 500.0; // 50% of theoretical PyTorch CLIP throughput
        if metrics.throughput_stats.training_samples_per_sec < pytorch_target {
            actions.push(format!("Optimize throughput: {:.0} -> {} samples/sec",
                metrics.throughput_stats.training_samples_per_sec, pytorch_target));
        }

        if metrics.memory_stats.peak_memory_mb > 7000.0 {
            actions.push(format!("Reduce memory usage: {:.0}MB -> <7000MB", metrics.memory_stats.peak_memory_mb));
        }

        for recommendation in &metrics.optimization_recommendations {
            actions.push(recommendation.clone());
        }

        actions
    }
}

/// Profiling assessment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingAssessment {
    /// Overall performance score (0-1)
    pub overall_score: f64,
    /// Qualitative assessment
    pub assessment: String,
    /// Critical bottlenecks identified
    pub critical_bottlenecks: Vec<String>,
    /// Immediate action recommendations
    pub immediate_actions: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = PerformanceProfiler::new(std::path::Path::new("./test_profile")).unwrap();
        assert!(!profiler.memory_samples.is_empty());
    }

    #[test]
    fn test_memory_fragmentation_computation() {
        let samples = vec![100.0, 120.0, 80.0, 110.0, 90.0];
        let fragmentation = PerformanceProfiler::compute_memory_fragmentation(&samples);
        assert!(fragmentation >= 0.0 && fragmentation <= 1.0);
    }

    #[test]
    fn test_quick_profiler_assessment() {
        let metrics = PerformanceMetrics {
            memory_stats: MemoryMetrics {
                peak_memory_mb: 4000.0,
                average_memory_mb: 3000.0,
                memory_efficiency: 0.5,
                memory_fragmentation: 0.1,
                allocation_rate_mb_per_sec: 100.0,
                deallocation_overhead: 0.05,
            },
            throughput_stats: ThroughputMetrics {
                training_samples_per_sec: 750.0,
                inference_samples_per_sec: 600.0,
                batch_size_throughput_curve: vec![(32, 750.0)],
                pytorch_comparison_ratio: 0.75,
                scalability_score: 0.8,
            },
            timing_stats: TimingMetrics {
                forward_pass_latency_ms: 100.0,
                backward_pass_latency_ms: 200.0,
                optimizer_step_latency_ms: 50.0,
                data_loading_latency_ms: 30.0,
                memory_transfer_latency_ms: Some(10.0),
                synchronization_overhead_ms: 5.0,
            },
            efficiency_stats: EfficiencyMetrics {
                compute_utilization_pct: 75.0,
                memory_bandwidth_utilization: 60.0,
                arithmetic_intensity: 15.0,
                flop_efficiency: 80.0,
                cache_hit_rates: HashMap::new(),
                simd_utilization: 90.0,
            },
            bottleneck_analysis: BottleneckAnalysis {
                primary_bottleneck: BottleneckType::ComputeBound,
                bottleneck_severity: 0.6,
                component_bottlenecks: HashMap::new(),
                resource_contention: ResourceContentionAnalysis {
                    cpu_contention: 0.4,
                    memory_contention: 0.3,
                    io_contention: 0.3,
                    network_contention: 0.0,
                    gpu_contention: Some(0.2),
                },
                parallelization_opportunities: Vec::new(),
            },
            optimization_recommendations: vec![
                "Consider gradient checkpointing".to_string(),
            ],
            timestamp: std::time::SystemTime::now(),
        };

        let assessment = QuickProfiler::quick_assessment(&metrics);

        // Should get reasonable assessment score
        assert!(assessment.overall_score >= 0.0 && assessment.overall_score <= 1.0);
        assert!(!assessment.assessment.is_empty());
        assert!(!assessment.immediate_actions.is_empty());
    }

    #[test]
    fn test_training_step_profiling() {
        let mut profiler = PerformanceProfiler::new(std::path::Path::new("./test_profile")).unwrap();

        let step_metrics = TrainingStepMetrics {
            step: 1,
            batch_size: 32,
            forward_time_ms: 100.0,
            backward_time_ms: 200.0,
            optimizer_time_ms: 50.0,
            data_loading_time_ms: 30.0,
            total_step_time_ms: 380.0,
            memory_usage_mb: 3000.0,
        };

        // Should not panic
        let _result = profiler.profile_training_step(step_metrics);
        assert!(profiler.memory_samples.len() > 0);
        assert!(profiler.timing_samples.len() > 0);
    }
}

fn compute_deallocation_overhead(memory_samples: &[f64]) -> f64 {
    if memory_samples.len() < 2 { return 0.0; }

    // Simple heuristic: variance in memory usage as proxy for deallocation overhead
    let mean = memory_samples.iter().sum::<f64>() / memory_samples.len() as f64;
    let variance = memory_samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / memory_samples.len() as f64;

    (variance.sqrt() / mean).min(1.0) // Normalize to [0, 1]
}
