//! Communication performance profiling for distributed training

use std::collections::HashMap;

use crate::{format, Duration, Instant, String, Vec};

/// Communication profiler for distributed training performance analysis
#[derive(Debug)]
pub struct CommunicationProfiler {
    /// Recorded communication operations
    operations: Vec<CommunicationOperation>,
    /// Communication statistics
    stats: CommunicationStats,
}

/// Communication operation record
#[derive(Debug, Clone)]
pub struct CommunicationOperation {
    /// Operation name (e.g., "all_reduce", "all_gather")
    pub name: String,
    /// Operation duration
    pub duration: Duration,
    /// Data size in bytes
    pub data_size_bytes: usize,
    /// Timestamp when operation completed
    pub timestamp: Instant,
}

/// Communication statistics for performance analysis
#[derive(Debug, Clone)]
pub struct CommunicationStats {
    /// Total operations performed
    pub total_operations: usize,
    /// Total data transferred (bytes)
    pub total_data_bytes: u64,
    /// Total communication time
    pub total_time: Duration,
    /// Average bandwidth (GB/s)
    pub avg_bandwidth_gbps: f64,
    /// Peak bandwidth (GB/s)
    pub peak_bandwidth_gbps: f64,
    /// Average latency (microseconds)
    pub avg_latency_us: f64,
    /// Communication efficiency (data/time)
    pub efficiency_gbps: f64,
}

impl CommunicationProfiler {
    /// Create a new communication profiler
    #[must_use]
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            stats: CommunicationStats {
                total_operations: 0,
                total_data_bytes: 0,
                total_time: Duration::ZERO,
                avg_bandwidth_gbps: 0.0,
                peak_bandwidth_gbps: 0.0,
                avg_latency_us: 0.0,
                efficiency_gbps: 0.0,
            },
        }
    }

    /// Record a communication operation
    pub fn record_operation(&mut self, name: String, duration: Duration, data_size_bytes: usize) {
        let operation = CommunicationOperation {
            name,
            duration,
            data_size_bytes,
            timestamp: Instant::now(),
        };

        self.operations.push(operation);
        self.update_stats();
    }

    /// Get communication statistics
    #[must_use]
    pub fn statistics(&self) -> &CommunicationStats {
        &self.stats
    }

    /// Get recent operations (last N)
    #[must_use]
    pub fn recent_operations(&self, count: usize) -> &[CommunicationOperation] {
        let start = self.operations.len().saturating_sub(count);
        &self.operations[start..]
    }

    /// Generate communication performance report
    #[must_use]
    pub fn generate_report(&self) -> CommunicationReport {
        CommunicationReport::from_operations(&self.operations)
    }

    fn update_stats(&mut self) {
        if self.operations.is_empty() {
            return;
        }

        self.stats.total_operations = self.operations.len();
        self.stats.total_data_bytes = self
            .operations
            .iter()
            .map(|op| op.data_size_bytes as u64)
            .sum();
        self.stats.total_time = self.operations.iter().map(|op| op.duration).sum();

        // Calculate bandwidths
        if self.stats.total_time > Duration::ZERO {
            let total_data_gb = self.stats.total_data_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let total_time_sec = self.stats.total_time.as_secs_f64();
            self.stats.avg_bandwidth_gbps = total_data_gb / total_time_sec;

            // Peak bandwidth (fastest single operation)
            self.stats.peak_bandwidth_gbps = self
                .operations
                .iter()
                .filter_map(|op| {
                    if op.duration > Duration::ZERO {
                        let data_gb = op.data_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                        let time_sec = op.duration.as_secs_f64();
                        Some(data_gb / time_sec)
                    } else {
                        None
                    }
                })
                .fold(0.0, f64::max);
        }

        // Average latency
        let total_latency_us: f64 = self
            .operations
            .iter()
            .map(|op| op.duration.as_secs_f64() * 1_000_000.0)
            .sum();
        #[allow(clippy::cast_precision_loss)]
        let denom = self.operations.len() as f64;
        self.stats.avg_latency_us = total_latency_us / denom;

        // Efficiency (overall bandwidth)
        self.stats.efficiency_gbps = self.stats.avg_bandwidth_gbps;
    }
}

impl Default for CommunicationProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Communication performance report
#[derive(Debug, Clone)]
pub struct CommunicationReport {
    /// Total operations by type
    pub operations_by_type: HashMap<String, usize>,
    /// Bandwidth statistics by operation type
    pub bandwidth_by_type: HashMap<String, f64>,
    /// Performance bottlenecks (slowest operations)
    pub bottlenecks: Vec<(String, Duration)>,
    /// Recommendations for optimization
    pub recommendations: Vec<String>,
}

impl CommunicationReport {
    /// Generate report from communication operations
    #[must_use]
    pub fn from_operations(operations: &[CommunicationOperation]) -> Self {
        let mut operations_by_type = HashMap::new();
        let mut bandwidth_by_type = HashMap::new();
        let mut bottlenecks = Vec::new();

        for op in operations {
            // Count operations by type
            *operations_by_type.entry(op.name.clone()).or_insert(0) += 1;

            // Calculate bandwidth for this operation
            if op.duration > Duration::ZERO {
                #[allow(clippy::cast_precision_loss)]
                let data_gb = op.data_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                let time_sec = op.duration.as_secs_f64();
                let bandwidth = data_gb / time_sec;
                let entry = bandwidth_by_type.entry(op.name.clone()).or_insert(0.0);
                *entry = (*entry + bandwidth) * 0.5;
            }

            // Track bottlenecks (operations taking > 10ms)
            if op.duration > Duration::from_millis(10) {
                bottlenecks.push((op.name.clone(), op.duration));
            }
        }

        // Sort bottlenecks by duration (slowest first)
        bottlenecks.sort_by(|a, b| b.1.cmp(&a.1));
        bottlenecks.truncate(10); // Top 10 bottlenecks

        // Generate recommendations
        let recommendations = Self::generate_recommendations(&operations_by_type, &bottlenecks);

        Self {
            operations_by_type,
            bandwidth_by_type,
            bottlenecks,
            recommendations,
        }
    }

    fn generate_recommendations(
        operations_by_type: &HashMap<String, usize>,
        bottlenecks: &[(String, Duration)],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check for frequent all_reduce operations
        if let Some(count) = operations_by_type.get("all_reduce") {
            if *count > 100 {
                recommendations.push(
                    String::from("High frequency all_reduce operations detected. Consider gradient accumulation to reduce communication overhead.")
                );
            }
        }

        // Check for slow operations
        if !bottlenecks.is_empty() {
            recommendations.push(
                format!("{} slow communication operations detected. Consider optimizing data layout or using larger message sizes.", bottlenecks.len())
            );
        }

        // General recommendations
        if operations_by_type.values().sum::<usize>() > 1000 {
            recommendations.push(
                String::from("High communication frequency detected. Consider model/data parallelism adjustments.")
            );
        }

        recommendations
    }

    /// Generate human-readable report
    #[must_use]
    pub fn summary(&self) -> String {
        use core::fmt::Write;

        let mut summary = String::from("# Communication Performance Report\n\n");

        summary.push_str("## Operations by Type\n");
        for (op_type, count) in &self.operations_by_type {
            if let Some(bandwidth) = self.bandwidth_by_type.get(op_type) {
                let _ = writeln!(
                    &mut summary,
                    "- {op_type}: {count} operations, {bandwidth:.2} GB/s avg bandwidth"
                );
            } else {
                let _ = writeln!(&mut summary, "- {op_type}: {count} operations");
            }
        }

        if !self.bottlenecks.is_empty() {
            summary.push_str("\n## Performance Bottlenecks\n");
            for (op_name, duration) in self.bottlenecks.iter().take(5) {
                let _ = writeln!(&mut summary, "- {op_name}: {:.2}ms", duration.as_millis());
            }
        }

        if !self.recommendations.is_empty() {
            summary.push_str("\n## Optimization Recommendations\n");
            for recommendation in &self.recommendations {
                let _ = writeln!(&mut summary, "- {recommendation}");
            }
        }

        summary
    }
}
