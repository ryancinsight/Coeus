//! # Coeus Profiling
//!
//! Comprehensive performance profiling, training monitoring, and analytics for Coeus deep learning framework.
//!
//! ## Features
//!
//! - **Training Metrics**: Loss curves, learning rates, gradient statistics, validation metrics
//! - **Memory Profiling**: GPU/CPU memory usage tracking and leak detection
//! - **Communication Monitoring**: NCCL/Gloo performance diagnostics and bandwidth analysis
//! - **Real-time Dashboards**: Training progress visualization and alerting
//! - **Performance Analytics**: Bottleneck identification and optimization insights
//! - **Distributed Training Profiling**: Multi-GPU/multi-node performance analysis
//!
//! ## Training Monitoring
//!
//! ```rust
//! use coeus_profiling::{TrainingMonitor, TrainingMetrics};
//!
//! let mut monitor = TrainingMonitor::new();
//!
//! // During training loop
//! for epoch in 0..num_epochs {
//!     for batch in training_data {
//!         // Forward pass and loss computation
//!         let loss = model.forward(&batch.input)?;
//!
//!         // Record training metrics
//!         monitor.record_metrics(TrainingMetrics {
//!             epoch,
//!             step: batch.step,
//!             loss: loss.item(),
//!             learning_rate: optimizer.learning_rate(),
//!             gradient_norm: compute_gradient_norm(&model.gradients()),
//!             ..Default::default()
//!         });
//!
//!         // Backward pass and optimization
//!         loss.backward()?;
//!         optimizer.step()?;
//!     }
//! }
//!
//! // Generate training report
//! let report = monitor.generate_report();
//! println!("{}", report.summary());
//! ```
//!
//! ## Performance Profiling
//!
//! ```rust
//! use coeus_profiling::{Timer, Profiler};
//!
//! // Time a tensor operation
//! let timer = Timer::start();
//! let result = model.forward(&input)?;
//! let elapsed = timer.elapsed();
//! println!("Forward pass took: {:?}", elapsed);
//!
//! // Profile with automatic timing and memory tracking
//! let profiler = Profiler::new();
//! let profile = profiler.profile_comprehensive(|| {
//!     model.forward(&input)
//! });
//! println!("Memory delta: {} KB", profile.memory_delta.unwrap().physical_delta / 1024);
//! ```
//!
//! ## Communication Monitoring
//!
//! ```rust
//! use coeus_profiling::CommunicationProfiler;
//!
//! let comm_profiler = CommunicationProfiler::new();
//!
//! // During distributed training
//! let start = std::time::Instant::now();
//! process_group.all_reduce(&mut gradients)?;
//! let duration = start.elapsed();
//!
//! comm_profiler.record_operation("all_reduce", duration, gradients.len() * 4);
//!
//! // Get communication statistics
//! let stats = comm_profiler.statistics();
//! println!("Communication bandwidth: {:.2} GB/s", stats.bandwidth_gbps);
//! ```

#![no_std]
#![warn(missing_docs, clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
use std::{
    boxed::Box,
    collections::HashMap,
    format,
    string::{String, ToString},
    sync::Mutex,
    time::{Duration, Instant},
    vec::Vec,
};

#[cfg(not(feature = "std"))]
use core::time::Duration;

use instant::Instant;

#[cfg(all(feature = "memory_profiling", feature = "std"))]
use memory_stats::memory_stats;

/// High-precision timer for measuring operation execution time
#[derive(Debug, Clone)]
pub struct Timer {
    /// Start time of the timer
    start: Instant,
}

impl Timer {
    /// Create a new timer and start measuring time immediately
    #[must_use]
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Get the elapsed time since the timer was started
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Reset the timer to start measuring from now
    pub fn reset(&mut self) {
        self.start = Instant::now();
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::start()
    }
}

/// Profiling statistics for a series of operations
#[derive(Debug, Clone)]
pub struct ProfileStats {
    /// Number of operations profiled
    pub count: usize,
    /// Total time spent across all operations
    pub total_time: Duration,
    /// Mean time per operation
    pub mean_time: Duration,
    /// Minimum operation time
    pub min_time: Duration,
    /// Maximum operation time
    pub max_time: Duration,
    /// Standard deviation of operation times
    pub std_dev: Duration,
}

impl ProfileStats {
    /// Create new profile statistics from a vector of durations
    #[must_use]
    pub fn from_durations(durations: &[Duration]) -> Option<Self> {
        if durations.is_empty() {
            return None;
        }

        let count = durations.len();
        let total_time = durations.iter().sum();
        let mean_time = total_time / u32::try_from(count).unwrap_or(u32::MAX);

        let min_time = durations.iter().min().copied().unwrap_or(Duration::ZERO);
        let max_time = durations.iter().max().copied().unwrap_or(Duration::ZERO);

        // Calculate standard deviation
        #[allow(clippy::cast_precision_loss)]
        let variance = durations
            .iter()
            .map(|&d| {
                let diff = if d > mean_time {
                    (d - mean_time).as_nanos() as f64
                } else {
                    (mean_time - d).as_nanos() as f64
                };
                diff * diff
            })
            .sum::<f64>()
            / f64::from(u32::try_from(count).unwrap_or(u32::MAX));

        let std_dev_nanos = variance.sqrt();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let std_dev = Duration::from_nanos(std_dev_nanos as u64);

        Some(Self {
            count,
            total_time,
            mean_time,
            min_time,
            max_time,
            std_dev,
        })
    }
}

/// Memory usage statistics
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Current physical memory usage in bytes
    pub physical_mem: usize,
    /// Current virtual memory usage in bytes
    pub virtual_mem: usize,
}

#[cfg(feature = "std")]
impl MemoryStats {
    /// Get current memory usage statistics
    ///
    /// Returns None if memory profiling is not available or supported
    #[must_use]
    pub fn current() -> Option<Self> {
        #[cfg(feature = "memory_profiling")]
        {
            memory_stats().map(|usage| Self {
                physical_mem: usage.physical_mem,
                virtual_mem: usage.virtual_mem,
            })
        }

        #[cfg(not(feature = "memory_profiling"))]
        {
            None
        }
    }
}

/// Combined performance and memory profiling results
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Timing statistics
    pub timing: ProfileStats,
    /// Memory usage statistics (if available)
    pub memory: Option<MemoryStats>,
    /// Memory usage before operation
    pub memory_before: Option<MemoryStats>,
    /// Memory usage after operation
    pub memory_after: Option<MemoryStats>,
    /// Memory delta (increase during operation)
    pub memory_delta: Option<MemoryDelta>,
}

/// Memory usage change during an operation
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct MemoryDelta {
    /// Change in physical memory (positive = increase)
    pub physical_delta: i64,
    /// Change in virtual memory (positive = increase)
    pub virtual_delta: i64,
}

#[cfg(feature = "std")]
impl MemoryDelta {
    /// Calculate memory delta from before and after measurements
    #[must_use]
    pub fn from_stats(before: &MemoryStats, after: &MemoryStats) -> Self {
        Self {
            #[allow(clippy::cast_possible_wrap)]
            physical_delta: after.physical_mem as i64 - before.physical_mem as i64,
            #[allow(clippy::cast_possible_wrap)]
            virtual_delta: after.virtual_mem as i64 - before.virtual_mem as i64,
        }
    }
}

/// Performance profiler for benchmarking operations
#[derive(Debug)]
pub struct Profiler {
    /// Number of warm-up iterations
    warmup_iterations: usize,
    /// Number of measurement iterations
    measurement_iterations: usize,
}

impl Profiler {
    /// Create a new profiler with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            warmup_iterations: 10,
            measurement_iterations: 100,
        }
    }

    /// Set the number of warm-up iterations
    #[must_use]
    pub fn with_warmup_iterations(mut self, iterations: usize) -> Self {
        self.warmup_iterations = iterations;
        self
    }

    /// Set the number of measurement iterations
    #[must_use]
    pub fn with_measurement_iterations(mut self, iterations: usize) -> Self {
        self.measurement_iterations = iterations;
        self
    }

    /// Profile a function by running it multiple times
    ///
    /// # Arguments
    /// * `operation` - The function to profile
    ///
    /// # Returns
    /// Statistics about the operation's performance
    #[must_use]
    pub fn profile<F>(&self, mut operation: F) -> ProfileStats
    where
        F: FnMut(),
    {
        // Warm-up phase
        for _ in 0..self.warmup_iterations {
            operation();
        }

        // Measurement phase
        let mut durations = Vec::with_capacity(self.measurement_iterations);
        for _ in 0..self.measurement_iterations {
            let timer = Timer::start();
            operation();
            durations.push(timer.elapsed());
        }

        ProfileStats::from_durations(&durations).unwrap_or(ProfileStats {
            count: 0,
            total_time: Duration::ZERO,
            mean_time: Duration::ZERO,
            min_time: Duration::ZERO,
            max_time: Duration::ZERO,
            std_dev: Duration::ZERO,
        })
    }

    /// Profile a function once and return the execution time
    #[must_use]
    pub fn time<F>(mut operation: F) -> Duration
    where
        F: FnMut(),
    {
        let timer = Timer::start();
        operation();
        timer.elapsed()
    }

    /// Profile a function with comprehensive performance and memory analysis
    ///
    /// This method measures both timing and memory usage during operation execution.
    /// Memory profiling requires the `memory_profiling` feature to be enabled.
    ///
    /// # Arguments
    /// * `operation` - The function to profile
    ///
    /// # Returns
    /// Comprehensive profiling results including timing and memory statistics
    #[cfg(feature = "std")]
    #[must_use]
    pub fn profile_comprehensive<F>(&self, mut operation: F) -> PerformanceProfile
    where
        F: FnMut(),
    {
        // Capture memory before operation
        let memory_before = MemoryStats::current();

        // Profile timing
        let mut durations = Vec::with_capacity(self.measurement_iterations);

        // Warm-up
        for _ in 0..self.warmup_iterations {
            operation();
        }

        // Measurement phase
        for _ in 0..self.measurement_iterations {
            let timer = Timer::start();
            operation();
            durations.push(timer.elapsed());
        }

        let timing_stats = ProfileStats::from_durations(&durations).unwrap_or(ProfileStats {
            count: 0,
            total_time: Duration::ZERO,
            mean_time: Duration::ZERO,
            min_time: Duration::ZERO,
            max_time: Duration::ZERO,
            std_dev: Duration::ZERO,
        });

        // Capture memory after operation
        let memory_after = MemoryStats::current();

        // Calculate memory delta
        let memory_delta = match (&memory_before, &memory_after) {
            (Some(before), Some(after)) => Some(MemoryDelta::from_stats(before, after)),
            _ => None,
        };

        PerformanceProfile {
            timing: timing_stats,
            memory: memory_after.clone(),
            memory_before,
            memory_after,
            memory_delta,
        }
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Scoped timer that automatically logs when dropped
#[cfg(feature = "std")]
pub struct ScopedTimer {
    /// Timer name for logging
    name: String,
    /// Internal timer
    timer: Timer,
}

#[cfg(feature = "std")]
impl ScopedTimer {
    /// Create a new scoped timer with the given name
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: String::from(name),
            timer: Timer::start(),
        }
    }
}

#[cfg(feature = "std")]
impl Drop for ScopedTimer {
    fn drop(&mut self) {
        let elapsed = self.timer.elapsed();
        tracing::info!("{} completed in {:?}", self.name, elapsed);
    }
}

/// Tracing-compatible span timer
#[cfg(feature = "std")]
#[macro_export]
macro_rules! time_span {
    ($name:expr) => {
        let _timer = $crate::ScopedTimer::new($name);
    };
    ($name:expr, $($fields:tt)*) => {
        tracing::info_span!($name, $($fields)*).entered();
        let _timer = $crate::ScopedTimer::new($name);
    };
}

/// Profile an operation with tracing integration
#[cfg(feature = "std")]
#[macro_export]
macro_rules! profile_span {
    ($name:expr, $operation:expr) => {{
        let span = tracing::info_span!($name);
        let _enter = span.enter();
        let timer = $crate::Timer::start();
        let result = $operation;
        let elapsed = timer.elapsed();
        tracing::info!("Operation completed in {:?}", elapsed);
        result
    }};
    ($name:expr, $operation:expr, $($fields:tt)*) => {{
        let span = tracing::info_span!($name, $($fields)*);
        let _enter = span.enter();
        let timer = $crate::Timer::start();
        let result = $operation;
        let elapsed = timer.elapsed();
        tracing::info!("Operation completed in {:?}", elapsed);
        result
    }};
}

/// Performance event logging for profiling
#[cfg(feature = "std")]
#[derive(Clone)]
pub struct PerformanceEvent {
    /// Event name
    pub name: String,
    /// Execution duration
    pub duration: Duration,
    /// Memory usage before (if available)
    pub memory_before: Option<usize>,
    /// Memory usage after (if available)
    pub memory_after: Option<usize>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

#[cfg(feature = "std")]
impl PerformanceEvent {
    /// Create a new performance event
    #[must_use]
    pub fn new(name: &str, duration: Duration) -> Self {
        Self {
            name: String::from(name),
            duration,
            memory_before: MemoryStats::current().map(|m| m.physical_mem),
            memory_after: MemoryStats::current().map(|m| m.physical_mem),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add metadata to the event
    #[must_use]
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(String::from(key), String::from(value));
        self
    }

    /// Log the event using tracing
    pub fn log(&self) {
        let mut event = tracing::info_span!("performance_event", name = %self.name, duration_ms = %self.duration.as_millis());

        if let (Some(before), Some(after)) = (self.memory_before, self.memory_after) {
            #[allow(clippy::cast_possible_wrap)]
            let delta = after as i64 - before as i64;
            event = tracing::info_span!("performance_event",
                name = %self.name,
                duration_ms = %self.duration.as_millis(),
                memory_delta_kb = %(delta / 1024)
            );
        }

        let _enter = event.enter();
        tracing::info!("Performance event: {} took {:?}", self.name, self.duration);
    }
}

/// Tracing subscriber for performance monitoring
#[cfg(feature = "std")]
pub struct PerformanceSubscriber {
    events: Mutex<Vec<PerformanceEvent>>,
}

#[cfg(feature = "std")]
impl PerformanceSubscriber {
    /// Create a new performance subscriber
    #[must_use]
    pub fn new() -> Self {
        Self {
            events: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Record a performance event
    pub fn record_event(&self, event: PerformanceEvent) {
        if let Ok(mut events) = self.events.lock() {
            events.push(event);
        }
    }

    /// Get all recorded events
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn events(&self) -> Vec<PerformanceEvent> {
        self.events.lock().unwrap().clone()
    }

    /// Generate a performance report
    #[must_use]
    pub fn generate_report(&self) -> PerformanceReport {
        let events = self.events();
        PerformanceReport::from_events(&events)
    }
}

#[cfg(feature = "std")]
impl Default for PerformanceSubscriber {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance report summarizing profiling data
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Total number of events
    pub total_events: usize,
    /// Total execution time across all events
    pub total_time: Duration,
    /// Average event duration
    pub avg_duration: Duration,
    /// Longest event duration
    pub max_duration: Duration,
    /// Shortest event duration
    pub min_duration: Duration,
    /// Events sorted by duration (longest first)
    pub slowest_events: Vec<(String, Duration)>,
    /// Memory usage summary (if available)
    pub memory_summary: Option<MemoryUsageSummary>,
}

/// Memory usage summary for performance profiling
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct MemoryUsageSummary {
    /// Average memory delta per event (in KB)
    pub avg_memory_delta_kb: f64,
    /// Maximum memory delta (in KB)
    pub max_memory_delta_kb: i64,
    /// Total memory allocated during profiling (in KB)
    pub total_memory_delta_kb: i64,
}

#[cfg(feature = "std")]
impl PerformanceReport {
    /// Generate report from a list of events
    ///
    /// # Panics
    ///
    /// Panics if events is empty (this is a bug since we check for empty events).
    #[must_use]
    pub fn from_events(events: &[PerformanceEvent]) -> Self {
        if events.is_empty() {
            return Self {
                total_events: 0,
                total_time: Duration::ZERO,
                avg_duration: Duration::ZERO,
                max_duration: Duration::ZERO,
                min_duration: Duration::ZERO,
                slowest_events: Vec::new(),
                memory_summary: None,
            };
        }

        let total_events = events.len();
        let total_time = events.iter().map(|e| e.duration).sum();
        let avg_duration = total_time / u32::try_from(total_events).unwrap_or(u32::MAX);

        let max_duration = events.iter().map(|e| e.duration).max().unwrap();
        let min_duration = events.iter().map(|e| e.duration).min().unwrap();

        let mut slowest_events: Vec<_> = events
            .iter()
            .map(|e| (e.name.clone(), e.duration))
            .collect();
        slowest_events.sort_by(|a, b| b.1.cmp(&a.1));
        slowest_events.truncate(10); // Top 10 slowest

        let memory_summary = Self::calculate_memory_summary(events);

        Self {
            total_events,
            total_time,
            avg_duration,
            max_duration,
            min_duration,
            slowest_events,
            memory_summary,
        }
    }

    fn calculate_memory_summary(events: &[PerformanceEvent]) -> Option<MemoryUsageSummary> {
        let mut deltas = Vec::new();
        let mut total_delta = 0i64;

        for event in events {
            if let (Some(before), Some(after)) = (event.memory_before, event.memory_after) {
                #[allow(clippy::cast_possible_wrap)]
                let delta = after as i64 - before as i64;
                deltas.push(delta);
                total_delta += delta;
            }
        }

        if deltas.is_empty() {
            return None;
        }

        #[allow(clippy::cast_precision_loss)]
        let avg_memory_delta_kb = total_delta as f64 / deltas.len() as f64 / 1024.0;
        let max_memory_delta_kb = *deltas.iter().max().unwrap();

        Some(MemoryUsageSummary {
            avg_memory_delta_kb,
            max_memory_delta_kb,
            total_memory_delta_kb: total_delta / 1024,
        })
    }
}

/// Benchmark utility for comparing different implementations
#[cfg(feature = "std")]
pub struct Benchmark {
    implementations: Vec<(String, Box<dyn Fn() + Send + Sync>)>,
    profiler: Profiler,
}

#[cfg(feature = "std")]
impl Benchmark {
    /// Create a new benchmark suite
    #[must_use]
    pub fn new() -> Self {
        Self {
            implementations: Vec::new(),
            profiler: Profiler::new()
                .with_warmup_iterations(5)
                .with_measurement_iterations(50),
        }
    }

    /// Add an implementation to benchmark
    pub fn add_implementation<F>(&mut self, name: &str, implementation: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.implementations
            .push((name.to_string(), Box::new(implementation)));
    }

    /// Run the benchmark and return results for all implementations
    #[must_use]
    pub fn run(&self) -> Vec<BenchmarkResult> {
        self.implementations
            .iter()
            .map(|(name, implementation)| {
                let profile = self.profiler.profile_comprehensive(|| {
                    implementation();
                });

                BenchmarkResult {
                    name: name.clone(),
                    profile,
                }
            })
            .collect()
    }

    /// Run benchmark and return a comparison report
    #[must_use]
    pub fn compare(&self) -> BenchmarkComparison {
        let results = self.run();

        if results.is_empty() {
            return BenchmarkComparison {
                results: Vec::new(),
                fastest: None,
                slowest: None,
                speedup_factors: Vec::new(),
            };
        }

        // Sort by mean time (fastest first)
        let mut sorted_results = results.clone();
        sorted_results.sort_by(|a, b| a.profile.timing.mean_time.cmp(&b.profile.timing.mean_time));

        let fastest = sorted_results.first().cloned();
        let slowest = sorted_results.last().cloned();

        // Calculate speedup factors relative to slowest
        let baseline_time = slowest
            .as_ref()
            .map_or(Duration::ZERO, |r| r.profile.timing.mean_time);

        #[allow(clippy::cast_precision_loss)]
        let speedup_factors = results
            .iter()
            .map(|result| {
                let speedup = if baseline_time > Duration::ZERO
                    && result.profile.timing.mean_time > Duration::ZERO
                {
                    baseline_time.as_nanos() as f64
                        / result.profile.timing.mean_time.as_nanos() as f64
                } else {
                    1.0
                };

                (result.name.clone(), speedup)
            })
            .collect();

        BenchmarkComparison {
            results: sorted_results,
            fastest,
            slowest,
            speedup_factors,
        }
    }
}

#[cfg(feature = "std")]
impl Default for Benchmark {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a single benchmark implementation
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Implementation name
    pub name: String,
    /// Performance profile
    pub profile: PerformanceProfile,
}

/// Comparison of multiple benchmark implementations
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct BenchmarkComparison {
    /// Results sorted by performance (fastest first)
    pub results: Vec<BenchmarkResult>,
    /// Fastest implementation
    pub fastest: Option<BenchmarkResult>,
    /// Slowest implementation
    pub slowest: Option<BenchmarkResult>,
    /// Speedup factors relative to slowest implementation
    pub speedup_factors: Vec<(String, f64)>,
}

#[cfg(feature = "std")]
impl BenchmarkComparison {
    /// Generate a human-readable comparison report
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn report(&self) -> String {
        use std::fmt::Write;

        let mut report = String::from("# Benchmark Comparison Report\n\n");

        if self.results.is_empty() {
            report.push_str("No benchmark results available.\n");
            return report;
        }

        report.push_str("| Implementation | Mean Time | Speedup |\n");
        report.push_str("|----------------|-----------|---------|\n");

        for result in &self.results {
            let speedup = self
                .speedup_factors
                .iter()
                .find(|(name, _)| name == &result.name)
                .map_or_else(|| "-".to_string(), |(_, factor)| format!("{factor:.2}x"));
            #[allow(clippy::cast_precision_loss)]
            let _ = writeln!(
                report,
                "| {} | {:.2}ms | {} |",
                result.name,
                result.profile.timing.mean_time.as_millis() as f64,
                speedup
            );
        }

        report.push('\n');

        if let Some(fastest) = &self.fastest {
            use std::fmt::Write;
            #[allow(clippy::cast_precision_loss)]
            let _ = writeln!(
                report,
                "**Fastest**: {} ({:.2}ms)",
                fastest.name,
                fastest.profile.timing.mean_time.as_millis() as f64
            );
        }

        if let Some(slowest) = &self.slowest {
            use std::fmt::Write;
            #[allow(clippy::cast_precision_loss)]
            let _ = writeln!(
                report,
                "**Slowest**: {} ({:.2}ms)",
                slowest.name,
                slowest.profile.timing.mean_time.as_millis() as f64
            );
        }

        if let (Some(fastest), Some(slowest)) = (&self.fastest, &self.slowest) {
            if fastest.name != slowest.name {
                #[allow(clippy::cast_precision_loss)]
                let ratio = slowest.profile.timing.mean_time.as_nanos() as f64
                    / fastest.profile.timing.mean_time.as_nanos() as f64;
                let _ = writeln!(report, "**Performance Range**: {ratio:.1}x speedup");
            }
        }

        report
    }
}


/// Training monitor for collecting and analyzing training metrics
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct TrainingMonitor {
    /// Collected training metrics
    metrics: Vec<TrainingMetrics>,
    /// Alert thresholds for monitoring
    alert_thresholds: TrainingAlertThresholds,
    /// Whether monitoring is enabled
    enabled: bool,
    /// Maximum number of metrics to keep in memory
    max_history: usize,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct TrainingAlertThresholds {
    /// Maximum acceptable loss value
    pub max_loss: Option<f32>,
    /// Maximum acceptable gradient norm
    pub max_gradient_norm: Option<f32>,
    /// Minimum learning rate threshold
    pub min_learning_rate: Option<f32>,
    /// Maximum step time in milliseconds
    pub max_step_time_ms: Option<f32>,
    /// Maximum memory usage (MB)
    pub max_memory_mb: Option<f32>,
}

#[cfg(feature = "std")]
impl Default for TrainingAlertThresholds {
    fn default() -> Self {
        Self {
            max_loss: None,
            max_gradient_norm: Some(10.0), // Common threshold
            min_learning_rate: Some(1e-8), // Very small learning rate
            max_step_time_ms: None,
            max_memory_mb: None,
        }
    }
}

#[cfg(feature = "std")]
impl TrainingMonitor {
    /// Create a new training monitor
    #[must_use]
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            alert_thresholds: TrainingAlertThresholds::default(),
            enabled: true,
            max_history: 10000, // Keep last 10k metrics
        }
    }

    /// Create monitor with custom alert thresholds
    #[must_use]
    pub fn with_thresholds(thresholds: TrainingAlertThresholds) -> Self {
        Self {
            alert_thresholds: thresholds,
            ..Self::new()
        }
    }

    /// Record training metrics
    pub fn record_metrics(&mut self, metrics: TrainingMetrics) {
        if !self.enabled {
            return;
        }

        // Check for alerts
        self.check_alerts(&metrics);

        // Store metrics
        self.metrics.push(metrics);

        // Maintain history limit
        if self.metrics.len() > self.max_history {
            self.metrics.remove(0); // Remove oldest
        }
    }

    /// Get all recorded metrics
    #[must_use]
    pub fn metrics(&self) -> &[TrainingMetrics] {
        &self.metrics
    }

    /// Get latest metrics
    #[must_use]
    pub fn latest_metrics(&self) -> Option<&TrainingMetrics> {
        self.metrics.last()
    }

    /// Generate training report
    #[must_use]
    pub fn generate_report(&self) -> TrainingReport {
        TrainingReport::from_metrics(&self.metrics)
    }

    /// Check for alert conditions
    fn check_alerts(&self, metrics: &TrainingMetrics) {
        if let Some(max_loss) = self.alert_thresholds.max_loss {
            if metrics.loss > max_loss {
                tracing::warn!("Loss {:.4} exceeds threshold {:.4}", metrics.loss, max_loss);
            }
        }

        if let Some(max_grad_norm) = self.alert_thresholds.max_gradient_norm {
            if metrics.gradient_norm > max_grad_norm {
                tracing::warn!("Gradient norm {:.4} exceeds threshold {:.4}", metrics.gradient_norm, max_grad_norm);
            }
        }

        if let Some(min_lr) = self.alert_thresholds.min_learning_rate {
            if metrics.learning_rate < min_lr {
                tracing::warn!("Learning rate {:.2e} below threshold {:.2e}", metrics.learning_rate, min_lr);
            }
        }

        if let Some(max_time) = self.alert_thresholds.max_step_time_ms {
            if let Some(step_time) = metrics.step_time_ms {
                if step_time > max_time {
                    tracing::warn!("Step time {:.2}ms exceeds threshold {:.2}ms", step_time, max_time);
                }
            }
        }
    }

    /// Enable or disable monitoring
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

#[cfg(feature = "std")]
impl Default for TrainingMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Training report summarizing training progress and performance
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct TrainingReport {
    /// Total training steps
    pub total_steps: usize,
    /// Total epochs completed
    pub total_epochs: usize,
    /// Best loss achieved
    pub best_loss: f32,
    /// Best validation loss (if available)
    pub best_validation_loss: Option<f32>,
    /// Best validation accuracy (if available)
    pub best_validation_accuracy: Option<f32>,
    /// Loss improvement trend (positive = improving)
    pub loss_trend: f32,
    /// Learning rate schedule summary
    pub learning_rate_stats: LearningRateStats,
    /// Gradient norm statistics
    pub gradient_stats: GradientStats,
    /// Performance statistics
    pub performance_stats: PerformanceStats,
    /// Memory usage statistics
    pub memory_stats: MemoryStatsSummary,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct LearningRateStats {
    pub initial_lr: f32,
    pub final_lr: f32,
    pub min_lr: f32,
    pub max_lr: f32,
    pub decay_factor: f32,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct GradientStats {
    pub mean_norm: f32,
    pub max_norm: f32,
    pub min_norm: f32,
    pub norm_std_dev: f32,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub mean_step_time_ms: f32,
    pub max_step_time_ms: f32,
    pub throughput_samples_per_sec: f32,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct MemoryStatsSummary {
    pub peak_gpu_memory_mb: f32,
    pub peak_cpu_memory_mb: f32,
    pub avg_gpu_memory_mb: f32,
    pub avg_cpu_memory_mb: f32,
}

#[cfg(feature = "std")]
impl TrainingReport {
    /// Generate report from training metrics
    #[must_use]
    pub fn from_metrics(metrics: &[TrainingMetrics]) -> Self {
        if metrics.is_empty() {
            return Self::empty();
        }

        let total_steps = metrics.len();
        let total_epochs = metrics.iter().map(|m| m.epoch).max().unwrap_or(0) + 1;

        let losses: Vec<f32> = metrics.iter().map(|m| m.loss).collect();
        let best_loss = losses.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        let best_validation_loss = metrics
            .iter()
            .filter_map(|m| m.validation_loss)
            .fold(None, |min, val| Some(min.map_or(val, |m| m.min(val))));

        let best_validation_accuracy = metrics
            .iter()
            .filter_map(|m| m.validation_accuracy)
            .fold(None, |max, val| Some(max.map_or(val, |m| m.max(val))));

        // Calculate loss trend (recent vs early)
        let early_loss = losses.iter().take(100).sum::<f32>() / 100.0.min(losses.len() as f32);
        let recent_loss = losses.iter().rev().take(100).sum::<f32>() / 100.0.min(losses.len() as f32);
        let loss_trend = early_loss - recent_loss; // Positive = improving

        // Learning rate statistics
        let lr_stats = Self::compute_learning_rate_stats(metrics);

        // Gradient statistics
        let grad_norms: Vec<f32> = metrics.iter().map(|m| m.gradient_norm).collect();
        let gradient_stats = Self::compute_gradient_stats(&grad_norms);

        // Performance statistics
        let step_times: Vec<f32> = metrics.iter().filter_map(|m| m.step_time_ms).collect();
        let performance_stats = Self::compute_performance_stats(&step_times);

        // Memory statistics
        let memory_stats = Self::compute_memory_stats(metrics);

        Self {
            total_steps,
            total_epochs,
            best_loss,
            best_validation_loss,
            best_validation_accuracy,
            loss_trend,
            learning_rate_stats: lr_stats,
            gradient_stats,
            performance_stats,
            memory_stats,
        }
    }

    fn empty() -> Self {
        Self {
            total_steps: 0,
            total_epochs: 0,
            best_loss: 0.0,
            best_validation_loss: None,
            best_validation_accuracy: None,
            loss_trend: 0.0,
            learning_rate_stats: LearningRateStats {
                initial_lr: 0.0,
                final_lr: 0.0,
                min_lr: 0.0,
                max_lr: 0.0,
                decay_factor: 1.0,
            },
            gradient_stats: GradientStats {
                mean_norm: 0.0,
                max_norm: 0.0,
                min_norm: 0.0,
                norm_std_dev: 0.0,
            },
            performance_stats: PerformanceStats {
                mean_step_time_ms: 0.0,
                max_step_time_ms: 0.0,
                throughput_samples_per_sec: 0.0,
            },
            memory_stats: MemoryStatsSummary {
                peak_gpu_memory_mb: 0.0,
                peak_cpu_memory_mb: 0.0,
                avg_gpu_memory_mb: 0.0,
                avg_cpu_memory_mb: 0.0,
            },
        }
    }

    fn compute_learning_rate_stats(metrics: &[TrainingMetrics]) -> LearningRateStats {
        let lrs: Vec<f32> = metrics.iter().map(|m| m.learning_rate).collect();
        if lrs.is_empty() {
            return LearningRateStats {
                initial_lr: 0.0,
                final_lr: 0.0,
                min_lr: 0.0,
                max_lr: 0.0,
                decay_factor: 1.0,
            };
        }

        let initial_lr = lrs[0];
        let final_lr = lrs[lrs.len() - 1];
        let min_lr = lrs.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_lr = lrs.iter().fold(0.0, |a, &b| a.max(b));
        let decay_factor = if initial_lr > 0.0 { final_lr / initial_lr } else { 1.0 };

        LearningRateStats {
            initial_lr,
            final_lr,
            min_lr,
            max_lr,
            decay_factor,
        }
    }

    fn compute_gradient_stats(grad_norms: &[f32]) -> GradientStats {
        if grad_norms.is_empty() {
            return GradientStats {
                mean_norm: 0.0,
                max_norm: 0.0,
                min_norm: 0.0,
                norm_std_dev: 0.0,
            };
        }

        let mean_norm = grad_norms.iter().sum::<f32>() / grad_norms.len() as f32;
        let max_norm = grad_norms.iter().fold(0.0, |a, &b| a.max(b));
        let min_norm = grad_norms.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        let variance = grad_norms
            .iter()
            .map(|&x| (x - mean_norm).powi(2))
            .sum::<f32>() / grad_norms.len() as f32;
        let norm_std_dev = variance.sqrt();

        GradientStats {
            mean_norm,
            max_norm,
            min_norm,
            norm_std_dev,
        }
    }

    fn compute_performance_stats(step_times: &[f32]) -> PerformanceStats {
        if step_times.is_empty() {
            return PerformanceStats {
                mean_step_time_ms: 0.0,
                max_step_time_ms: 0.0,
                throughput_samples_per_sec: 0.0,
            };
        }

        let mean_step_time_ms = step_times.iter().sum::<f32>() / step_times.len() as f32;
        let max_step_time_ms = step_times.iter().fold(0.0, |a, &b| a.max(b));

        // Assume batch size of 32 for throughput calculation
        let batch_size = 32.0;
        let throughput_samples_per_sec = if mean_step_time_ms > 0.0 {
            (batch_size * 1000.0) / mean_step_time_ms
        } else {
            0.0
        };

        PerformanceStats {
            mean_step_time_ms,
            max_step_time_ms,
            throughput_samples_per_sec,
        }
    }

    fn compute_memory_stats(metrics: &[TrainingMetrics]) -> MemoryStatsSummary {
        let gpu_memories: Vec<f32> = metrics.iter().filter_map(|m| m.gpu_memory_mb).collect();
        let cpu_memories: Vec<f32> = metrics.iter().filter_map(|m| m.cpu_memory_mb).collect();

        let peak_gpu_memory_mb = gpu_memories.iter().fold(0.0, |a, &b| a.max(b));
        let peak_cpu_memory_mb = cpu_memories.iter().fold(0.0, |a, &b| a.max(b));

        let avg_gpu_memory_mb = if gpu_memories.is_empty() {
            0.0
        } else {
            gpu_memories.iter().sum::<f32>() / gpu_memories.len() as f32
        };

        let avg_cpu_memory_mb = if cpu_memories.is_empty() {
            0.0
        } else {
            cpu_memories.iter().sum::<f32>() / cpu_memories.len() as f32
        };

        MemoryStatsSummary {
            peak_gpu_memory_mb,
            peak_cpu_memory_mb,
            avg_gpu_memory_mb,
            avg_cpu_memory_mb,
        }
    }

    /// Generate human-readable summary
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Training Report:\n\
             - Total Steps: {}\n\
             - Total Epochs: {}\n\
             - Best Loss: {:.4}\n\
             - Loss Trend: {:.4} (positive = improving)\n\
             - Learning Rate: {:.2e} -> {:.2e} (decay: {:.2f}x)\n\
             - Gradient Norm: mean={:.4}, max={:.4}\n\
             - Performance: {:.2}ms/step, {:.0} samples/sec\n\
             - Memory: peak GPU={:.0}MB, peak CPU={:.0}MB",
            self.total_steps,
            self.total_epochs,
            self.best_loss,
            self.loss_trend,
            self.learning_rate_stats.initial_lr,
            self.learning_rate_stats.final_lr,
            self.learning_rate_stats.decay_factor,
            self.gradient_stats.mean_norm,
            self.gradient_stats.max_norm,
            self.performance_stats.mean_step_time_ms,
            self.performance_stats.throughput_samples_per_sec,
            self.memory_stats.peak_gpu_memory_mb,
            self.memory_stats.peak_cpu_memory_mb,
        )
    }
}

/// Communication profiler for distributed training performance analysis
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct CommunicationProfiler {
    /// Recorded communication operations
    operations: Vec<CommunicationOperation>,
    /// Communication statistics
    stats: CommunicationStats,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct CommunicationOperation {
    /// Operation name (e.g., "all_reduce", "all_gather")
    pub name: String,
    /// Operation duration
    pub duration: Duration,
    /// Data size in bytes
    pub data_size_bytes: usize,
    /// Timestamp when operation completed
    pub timestamp: std::time::Instant,
}

#[cfg(feature = "std")]
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

#[cfg(feature = "std")]
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
            timestamp: std::time::Instant::now(),
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
        self.stats.total_data_bytes = self.operations.iter().map(|op| op.data_size_bytes as u64).sum();
        self.stats.total_time = self.operations.iter().map(|op| op.duration).sum();

        // Calculate bandwidths
        if self.stats.total_time > Duration::ZERO {
            let total_data_gb = self.stats.total_data_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let total_time_sec = self.stats.total_time.as_secs_f64();
            self.stats.avg_bandwidth_gbps = total_data_gb / total_time_sec;

            // Peak bandwidth (fastest single operation)
            self.stats.peak_bandwidth_gbps = self.operations
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
        let total_latency_us: f64 = self.operations.iter()
            .map(|op| op.duration.as_micros() as f64)
            .sum();
        self.stats.avg_latency_us = total_latency_us / self.operations.len() as f64;

        // Efficiency (overall bandwidth)
        self.stats.efficiency_gbps = self.stats.avg_bandwidth_gbps;
    }
}

#[cfg(feature = "std")]
impl Default for CommunicationProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Communication performance report
#[cfg(feature = "std")]
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

#[cfg(feature = "std")]
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
                let data_gb = op.data_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                let time_sec = op.duration.as_secs_f64();
                let bandwidth = data_gb / time_sec;
                let entry = bandwidth_by_type.entry(op.name.clone()).or_insert(0.0);
                *entry = (*entry + bandwidth) / 2.0; // Running average
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
                    "High frequency all_reduce operations detected. Consider gradient accumulation to reduce communication overhead.".to_string()
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
                "High communication frequency detected. Consider model/data parallelism adjustments.".to_string()
            );
        }

        recommendations
    }

    /// Generate human-readable report
    #[must_use]
    pub fn summary(&self) -> String {
        let mut summary = "# Communication Performance Report\n\n".to_string();

        summary.push_str("## Operations by Type\n");
        for (op_type, count) in &self.operations_by_type {
            if let Some(bandwidth) = self.bandwidth_by_type.get(op_type) {
                summary.push_str(&format!("- {}: {} operations, {:.2} GB/s avg bandwidth\n", op_type, count, bandwidth));
            } else {
                summary.push_str(&format!("- {}: {} operations\n", op_type, count));
            }
        }

        if !self.bottlenecks.is_empty() {
            summary.push_str("\n## Performance Bottlenecks\n");
            for (op_name, duration) in self.bottlenecks.iter().take(5) {
                summary.push_str(&format!("- {}: {:.2}ms\n", op_name, duration.as_millis()));
            }
        }

        if !self.recommendations.is_empty() {
            summary.push_str("\n## Optimization Recommendations\n");
            for recommendation in &self.recommendations {
                summary.push_str(&format!("- {}\n", recommendation));
            }
        }

        summary
    }
}

// Re-export training monitoring and communication profiling types
#[cfg(feature = "std")]
pub use self::{
    TrainingMonitor, TrainingMetrics, TrainingReport, TrainingAlertThresholds,
    CommunicationProfiler, CommunicationReport, CommunicationStats,
};

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "std")]
    use std::vec;

    #[test]
    fn test_timer_basic() {
        let timer = Timer::start();
        // Simulate some work
        #[cfg(feature = "std")]
        std::thread::sleep(Duration::from_millis(1));
        let elapsed = timer.elapsed();
        assert!(elapsed > Duration::ZERO);
    }

    #[test]
    fn test_profiler_stats() {
        let durations = vec![
            Duration::from_micros(100),
            Duration::from_micros(150),
            Duration::from_micros(200),
        ];

        let stats = ProfileStats::from_durations(&durations).unwrap();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.min_time, Duration::from_micros(100));
        assert_eq!(stats.max_time, Duration::from_micros(200));
        assert!(stats.mean_time >= Duration::from_micros(150));
    }

    #[test]
    fn test_profiler_profile() {
        let profiler = Profiler::new()
            .with_warmup_iterations(1)
            .with_measurement_iterations(5);

        let mut counter = 0;
        let stats = profiler.profile(|| {
            counter += 1;
            // Simulate work
            #[cfg(feature = "std")]
            std::thread::sleep(Duration::from_micros(10));
        });

        assert_eq!(counter, 6); // 1 warmup + 5 measurements
        assert_eq!(stats.count, 5);
        assert!(stats.mean_time > Duration::ZERO);
    }

    #[test]
    fn test_profiler_time() {
        let duration = Profiler::time(|| {
            // Simulate work
            #[cfg(feature = "std")]
            std::thread::sleep(Duration::from_micros(100));
        });

        assert!(duration >= Duration::from_micros(90)); // Allow some tolerance
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_memory_stats_current() {
        let memory = MemoryStats::current();

        #[cfg(feature = "memory_profiling")]
        {
            // If memory profiling is enabled, we should get some stats
            if let Some(stats) = memory {
                assert!(stats.physical_mem > 0);
                assert!(stats.virtual_mem >= stats.physical_mem);
            }
        }

        #[cfg(not(feature = "memory_profiling"))]
        {
            // If memory profiling is disabled, should return None
            assert!(memory.is_none());
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_memory_delta_calculation() {
        let before = MemoryStats {
            physical_mem: 1000,
            virtual_mem: 2000,
        };

        let after = MemoryStats {
            physical_mem: 1500,
            virtual_mem: 2500,
        };

        let delta = MemoryDelta::from_stats(&before, &after);

        assert_eq!(delta.physical_delta, 500);
        assert_eq!(delta.virtual_delta, 500);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_profiler_comprehensive() {
        let profiler = Profiler::new()
            .with_warmup_iterations(1)
            .with_measurement_iterations(3);

        let profile = profiler.profile_comprehensive(|| {
            // Simulate work
            #[cfg(feature = "std")]
            std::thread::sleep(Duration::from_micros(10));
        });

        // Check timing results
        assert_eq!(profile.timing.count, 3);
        assert!(profile.timing.mean_time > Duration::ZERO);

        // Memory results depend on feature flags
        #[cfg(feature = "memory_profiling")]
        {
            // Should have memory stats if profiling is enabled
            // (but may be None on some systems)
        }

        #[cfg(not(feature = "memory_profiling"))]
        {
            assert!(profile.memory_before.is_none());
            assert!(profile.memory_after.is_none());
            assert!(profile.memory_delta.is_none());
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_performance_event() {
        let duration = Duration::from_millis(100);
        let event = PerformanceEvent::new("test_operation", duration);

        assert_eq!(event.name, "test_operation");
        assert_eq!(event.duration, duration);
        // Memory fields depend on system capabilities
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_performance_subscriber() {
        let subscriber = PerformanceSubscriber::new();
        let event = PerformanceEvent::new("test", Duration::from_millis(50));

        subscriber.record_event(event);
        let events = subscriber.events();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "test");
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_performance_report() {
        let events = vec![
            PerformanceEvent::new("fast", Duration::from_millis(10)),
            PerformanceEvent::new("slow", Duration::from_millis(100)),
            PerformanceEvent::new("medium", Duration::from_millis(50)),
        ];

        let report = PerformanceReport::from_events(&events);

        assert_eq!(report.total_events, 3);
        assert_eq!(report.min_duration, Duration::from_millis(10));
        assert_eq!(report.max_duration, Duration::from_millis(100));
        assert_eq!(report.slowest_events.len(), 3);
        assert_eq!(report.slowest_events[0].0, "slow"); // Should be sorted by duration desc
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_performance_report_empty() {
        let events = vec![];
        let report = PerformanceReport::from_events(&events);

        assert_eq!(report.total_events, 0);
        assert_eq!(report.total_time, Duration::ZERO);
        assert_eq!(report.slowest_events.len(), 0);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_benchmark_comparison() {
        let mut benchmark = Benchmark::new();

        // Add two implementations to compare
        benchmark.add_implementation("impl_a", || {
            // Simulate implementation A (faster)
            #[cfg(feature = "std")]
            std::thread::sleep(Duration::from_micros(50));
        });

        benchmark.add_implementation("impl_b", || {
            // Simulate implementation B (slower)
            #[cfg(feature = "std")]
            std::thread::sleep(Duration::from_micros(500));
        });

        let results = benchmark.run();

        assert_eq!(results.len(), 2);
        // Verify that both implementations were benchmarked
        assert!(results[0].profile.timing.mean_time.as_nanos() > 0);
        assert!(results[1].profile.timing.mean_time.as_nanos() > 0);
        // Verify that the slower implementation takes longer (with tolerance for scheduling)
        #[cfg(feature = "std")]
        assert!(results[1].profile.timing.mean_time >= results[0].profile.timing.mean_time);
    }
}
