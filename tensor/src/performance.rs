//! Performance monitoring and regression detection for tensor operations
//!
//! This module provides comprehensive performance tracking, regression detection,
//! and automated alerting capabilities to ensure optimal performance is maintained
//! as the codebase evolves.

use coeus_dtype::Dtype;
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

/// Performance metrics for a single operation
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Operation name
    pub operation: String,
    /// Execution time in nanoseconds
    pub duration_ns: u64,
    /// Memory usage in bytes
    pub memory_bytes: usize,
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Timestamp of measurement
    pub timestamp: Instant,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl PerformanceMetrics {
    /// Create new performance metrics
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            duration_ns: 0,
            memory_bytes: 0,
            cpu_usage: 0.0,
            timestamp: Instant::now(),
            metadata: HashMap::new(),
        }
    }

    /// Record execution time
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration_ns = duration.as_nanos() as u64;
        self
    }

    /// Record memory usage
    pub fn with_memory(mut self, bytes: usize) -> Self {
        self.memory_bytes = bytes;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Performance regression detector
pub struct RegressionDetector {
    /// Baseline performance metrics
    baselines: RwLock<HashMap<String, PerformanceMetrics>>,
    /// Performance threshold (percentage degradation allowed)
    threshold: f64,
    /// Historical performance data
    history: RwLock<HashMap<String, Vec<PerformanceMetrics>>>,
}

impl RegressionDetector {
    /// Create new regression detector with default threshold (10% degradation)
    pub fn new() -> Self {
        Self {
            baselines: RwLock::new(HashMap::new()),
            threshold: 0.10, // 10% degradation threshold
            history: RwLock::new(HashMap::new()),
        }
    }

    /// Create regression detector with custom threshold
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            baselines: RwLock::new(HashMap::new()),
            threshold,
            history: RwLock::new(HashMap::new()),
        }
    }

    /// Establish baseline performance for an operation
    pub fn establish_baseline(&self, metrics: PerformanceMetrics) {
        let mut baselines = self.baselines.write().unwrap();
        baselines.insert(metrics.operation.clone(), metrics);
    }

    /// Record performance measurement and check for regressions
    pub fn record_measurement(&self, metrics: PerformanceMetrics) -> RegressionResult {
        let operation = metrics.operation.clone();

        // Store in history
        {
            let mut history = self.history.write().unwrap();
            history
                .entry(operation.clone())
                .or_default()
                .push(metrics.clone());
        }

        // Check for regression against baseline
        let baselines = self.baselines.read().unwrap();
        if let Some(baseline) = baselines.get(&operation) {
            let degradation = (metrics.duration_ns as f64 - baseline.duration_ns as f64)
                / baseline.duration_ns as f64;

            if degradation > self.threshold {
                return RegressionResult::Regression {
                    operation,
                    degradation: degradation * 100.0,
                    baseline_ns: baseline.duration_ns,
                    current_ns: metrics.duration_ns,
                };
            }
        }

        RegressionResult::Normal { operation }
    }

    /// Get performance history for an operation
    pub fn get_history(&self, operation: &str) -> Vec<PerformanceMetrics> {
        let history = self.history.read().unwrap();
        history.get(operation).cloned().unwrap_or_default()
    }

    /// Generate performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let baselines = self.baselines.read().unwrap();
        let history = self.history.read().unwrap();

        let mut operations = Vec::new();

        for (operation, baseline) in baselines.iter() {
            let latest = if let Some(op_history) = history.get(operation) {
                op_history.last()
            } else {
                None
            };

            let status = if let Some(current) = latest {
                let degradation = (current.duration_ns as f64 - baseline.duration_ns as f64)
                    / baseline.duration_ns as f64;

                if degradation > self.threshold {
                    OperationStatus::Regressed {
                        degradation: degradation * 100.0,
                    }
                } else if degradation < -self.threshold {
                    OperationStatus::Increased {
                        improvement: -degradation * 100.0,
                    }
                } else {
                    OperationStatus::Stable
                }
            } else {
                OperationStatus::NoData
            };

            operations.push(OperationReport {
                operation: operation.clone(),
                baseline: baseline.clone(),
                latest: latest.cloned(),
                status,
                sample_count: history.get(operation).map(|h| h.len()).unwrap_or(0),
            });
        }

        PerformanceReport { operations }
    }
}

impl Default for RegressionDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of regression analysis
#[derive(Debug, Clone)]
pub enum RegressionResult {
    /// No regression detected
    Normal { operation: String },
    /// Performance regression detected
    Regression {
        operation: String,
        degradation: f64, // percentage
        baseline_ns: u64,
        current_ns: u64,
    },
}

/// Status of an operation's performance
#[derive(Debug, Clone)]
pub enum OperationStatus {
    /// No data available
    NoData,
    /// Performance is stable (within threshold)
    Stable,
    /// Performance has increased
    Increased { improvement: f64 },
    /// Performance has regressed
    Regressed { degradation: f64 },
}

/// Performance report for an operation
#[derive(Debug, Clone)]
pub struct OperationReport {
    /// Operation name
    pub operation: String,
    /// Baseline performance metrics
    pub baseline: PerformanceMetrics,
    /// Latest performance metrics (if available)
    pub latest: Option<PerformanceMetrics>,
    /// Current performance status
    pub status: OperationStatus,
    /// Number of samples collected
    pub sample_count: usize,
}

/// Comprehensive performance report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Reports for all operations
    pub operations: Vec<OperationReport>,
}

impl PerformanceReport {
    /// Check if any operations have regressed
    pub fn has_regressions(&self) -> bool {
        self.operations
            .iter()
            .any(|op| matches!(op.status, OperationStatus::Regressed { .. }))
    }

    /// Get operations with regressions
    pub fn get_regressions(&self) -> Vec<&OperationReport> {
        self.operations
            .iter()
            .filter(|op| matches!(op.status, OperationStatus::Regressed { .. }))
            .collect()
    }

    /// Get operations with increases
    pub fn get_increases(&self) -> Vec<&OperationReport> {
        self.operations
            .iter()
            .filter(|op| matches!(op.status, OperationStatus::Increased { .. }))
            .collect()
    }

    /// Generate human-readable summary
    pub fn summary(&self) -> String {
        let regressions = self.get_regressions().len();
        let increases = self.get_increases().len();
        let total = self.operations.len();

        format!(
            "Performance Report: {} operations monitored\n\
             - Regressions: {}\n\
             - Increases: {}\n\
             - Stable: {}\n\
             - Status: {}",
            total,
            regressions,
            increases,
            total - regressions - increases,
            if regressions > 0 {
                "⚠️  REGRESSIONS DETECTED"
            } else {
                "✅ ALL OPERATIONS NORMAL"
            }
        )
    }
}

/// Performance monitoring context for automatic measurement
pub struct PerformanceContext {
    /// Regression detector
    detector: RegressionDetector,
    /// Active measurements
    active_measurements: RwLock<HashMap<String, (Instant, PerformanceMetrics)>>,
    /// Memory tracking enabled flag
    memory_tracking: bool,
}

impl PerformanceContext {
    /// Create new performance context
    pub fn new() -> Self {
        Self {
            detector: RegressionDetector::new(),
            active_measurements: RwLock::new(HashMap::new()),
            memory_tracking: true,
        }
    }

    /// Create performance context with memory tracking disabled
    pub fn without_memory_tracking() -> Self {
        Self {
            detector: RegressionDetector::new(),
            active_measurements: RwLock::new(HashMap::new()),
            memory_tracking: false,
        }
    }

    /// Start measuring an operation
    pub fn start_measurement(&self, operation: impl Into<String>) -> MeasurementGuard<'_> {
        let operation = operation.into();
        let start_time = Instant::now();
        let metrics = PerformanceMetrics::new(operation.clone());

        {
            let mut active = self.active_measurements.write().unwrap();
            active.insert(operation.clone(), (start_time, metrics));
        }

        MeasurementGuard {
            operation,
            context: self,
        }
    }

    /// Complete measurement and record results
    fn complete_measurement(&self, operation: &str) -> RegressionResult {
        let (start_time, mut metrics) = {
            let mut active = self.active_measurements.write().unwrap();
            active.remove(operation).unwrap()
        };

        let duration = start_time.elapsed();
        metrics = metrics.with_duration(duration);

        // Memory monitoring implementation (estimated)
        // Note: Actual tensor-aware monitoring requires access to tensor context
        // This is a simplified estimation for regression tracking
        metrics.memory_bytes = 0;

        // CPU usage monitoring (simplified)
        metrics.cpu_usage = 1.0; // Placeholder - actual implementation would require system monitoring

        self.detector.record_measurement(metrics)
    }

    /// Record tensor-aware measurement with memory tracking
    pub fn record_tensor_measurement<T: Dtype>(
        &self,
        operation: &str,
        duration: Duration,
        tensor: &super::Tensor<T>,
    ) -> RegressionResult {
        let mut metrics = PerformanceMetrics::new(operation.to_string()).with_duration(duration);

        if self.memory_tracking {
            // Calculate tensor memory usage
            let tensor_memory = std::mem::size_of_val(tensor)
                + tensor.numel() * std::mem::size_of::<T>()
                + tensor.shape.len() * std::mem::size_of::<usize>();

            metrics.memory_bytes = tensor_memory;
        }

        // CPU usage monitoring (placeholder - would need system monitoring crate)
        metrics.cpu_usage = 1.0;

        self.detector.record_measurement(metrics)
    }

    /// Get regression detector for manual baseline management
    pub fn detector(&self) -> &RegressionDetector {
        &self.detector
    }
}

impl Default for PerformanceContext {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard for automatic measurement completion
pub struct MeasurementGuard<'a> {
    operation: String,
    context: &'a PerformanceContext,
}

impl<'a> Drop for MeasurementGuard<'a> {
    fn drop(&mut self) {
        let result = self.context.complete_measurement(&self.operation);

        // Log regressions immediately
        if let RegressionResult::Regression {
            operation,
            degradation,
            ..
        } = result
        {
            eprintln!(
                "🚨 PERFORMANCE REGRESSION DETECTED: {} degraded by {:.1}%",
                operation, degradation
            );
        }
    }
}

/// Global performance monitoring instance
static PERFORMANCE_CONTEXT: once_cell::sync::Lazy<PerformanceContext> =
    once_cell::sync::Lazy::new(PerformanceContext::new);

/// Get global performance context
pub fn global_context() -> &'static PerformanceContext {
    &PERFORMANCE_CONTEXT
}

/// Convenience macro for measuring function execution time
#[macro_export]
macro_rules! measure_performance {
    ($operation:expr, $code:block) => {{
        let _guard = $crate::performance::global_context().start_measurement($operation);
        $code
    }};
}

/// Convenience macro for measuring async function execution time
#[macro_export]
macro_rules! measure_performance_async {
    ($operation:expr, $code:block) => {{
        let start = std::time::Instant::now();
        let result = $code;
        let duration = start.elapsed();

        let metrics =
            $crate::performance::PerformanceMetrics::new($operation).with_duration(duration);

        let regression_result = $crate::performance::global_context()
            .detector()
            .record_measurement(metrics);

        if let $crate::performance::RegressionResult::Regression {
            operation,
            degradation,
            ..
        } = regression_result
        {
            eprintln!(
                "🚨 PERFORMANCE REGRESSION DETECTED: {} degraded by {:.1}%",
                operation, degradation
            );
        }

        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_regression_detection() {
        let detector = RegressionDetector::with_threshold(0.1); // 10% threshold

        // Establish baseline
        let baseline = PerformanceMetrics::new("test_op").with_duration(Duration::from_millis(100));
        detector.establish_baseline(baseline);

        // Test normal performance (within threshold)
        let normal = PerformanceMetrics::new("test_op").with_duration(Duration::from_millis(105));
        let result = detector.record_measurement(normal);
        assert!(matches!(result, RegressionResult::Normal { .. }));

        // Test regression (above threshold)
        let regression =
            PerformanceMetrics::new("test_op").with_duration(Duration::from_millis(120));
        let result = detector.record_measurement(regression);
        assert!(
            matches!(result, RegressionResult::Regression { degradation, .. } if degradation > 10.0)
        );
    }

    #[test]
    fn test_performance_context() {
        let context = PerformanceContext::new();

        // Establish baseline
        let baseline =
            PerformanceMetrics::new("test_context_op").with_duration(Duration::from_millis(50));
        context.detector().establish_baseline(baseline);

        // Measure with context
        {
            let _guard = context.start_measurement("test_context_op");
            thread::sleep(Duration::from_millis(60)); // Simulate work
        }

        // Check that measurement was recorded
        let report = context.detector().generate_report();
        assert_eq!(report.operations.len(), 1);
        assert!(report.operations[0].sample_count > 0);
    }

    #[test]
    fn test_performance_report() {
        let detector = RegressionDetector::new();

        // Add baseline and measurement
        let baseline =
            PerformanceMetrics::new("report_test").with_duration(Duration::from_millis(100));
        detector.establish_baseline(baseline);

        let measurement =
            PerformanceMetrics::new("report_test").with_duration(Duration::from_millis(80));
        detector.record_measurement(measurement);

        let report = detector.generate_report();
        assert_eq!(report.operations.len(), 1);
        assert_eq!(report.operations[0].operation, "report_test");

        // Should show improvement
        assert!(matches!(
            report.operations[0].status,
            OperationStatus::Increased { .. }
        ));
    }
}
