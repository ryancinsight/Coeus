//! Training monitoring and metrics collection for deep learning models

use crate::*;
use std::collections::HashMap;

/// Training metrics for monitoring deep learning training progress
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Current training epoch
    pub epoch: usize,
    /// Global training step
    pub step: usize,
    /// Current batch loss value
    pub loss: f32,
    /// Current learning rate
    pub learning_rate: f32,
    /// Gradient norm (L2 norm of all gradients)
    pub gradient_norm: f32,
    /// Validation loss (if available)
    pub validation_loss: Option<f32>,
    /// Validation accuracy (if available)
    pub validation_accuracy: Option<f32>,
    /// Training accuracy (if available)
    pub training_accuracy: Option<f32>,
    /// GPU memory usage (in MB)
    pub gpu_memory_mb: Option<f32>,
    /// CPU memory usage (in MB)
    pub cpu_memory_mb: Option<f32>,
    /// Time per step (in milliseconds)
    pub step_time_ms: Option<f32>,
    /// Additional custom metrics
    pub custom_metrics: HashMap<String, f32>,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self {
            epoch: 0,
            step: 0,
            loss: 0.0,
            learning_rate: 0.0,
            gradient_norm: 0.0,
            validation_loss: None,
            validation_accuracy: None,
            training_accuracy: None,
            gpu_memory_mb: None,
            cpu_memory_mb: None,
            step_time_ms: None,
            custom_metrics: HashMap::new(),
        }
    }
}

/// Training monitor for collecting and analyzing training metrics
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

/// Fault tolerance configuration
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

impl Default for TrainingMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Training report summarizing training progress and performance
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

#[derive(Debug, Clone)]
pub struct LearningRateStats {
    pub initial_lr: f32,
    pub final_lr: f32,
    pub min_lr: f32,
    pub max_lr: f32,
    pub decay_factor: f32,
}

#[derive(Debug, Clone)]
pub struct GradientStats {
    pub mean_norm: f32,
    pub max_norm: f32,
    pub min_norm: f32,
    pub norm_std_dev: f32,
}

#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub mean_step_time_ms: f32,
    pub max_step_time_ms: f32,
    pub throughput_samples_per_sec: f32,
}

#[derive(Debug, Clone)]
pub struct MemoryStatsSummary {
    pub peak_gpu_memory_mb: f32,
    pub peak_cpu_memory_mb: f32,
    pub avg_gpu_memory_mb: f32,
    pub avg_cpu_memory_mb: f32,
}

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
            .fold(None, |min, val| Some(min.map_or(val, |m: f32| m.min(val))));

        let best_validation_accuracy = metrics
            .iter()
            .filter_map(|m| m.validation_accuracy)
            .fold(None, |max, val| Some(max.map_or(val, |m: f32| m.max(val))));

        // Calculate loss trend (recent vs early)
        let early_loss = losses.iter().take(100).sum::<f32>() / 100.0_f32.min(losses.len() as f32);
        let recent_loss = losses.iter().rev().take(100).sum::<f32>() / 100.0_f32.min(losses.len() as f32);
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
        let max_lr = lrs.iter().fold(0.0_f32, |a, &b| a.max(b));
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
        let max_norm = grad_norms.iter().fold(0.0_f32, |a, &b| a.max(b));
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
        let max_step_time_ms = step_times.iter().fold(0.0_f32, |a, &b| a.max(b));

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

        let peak_gpu_memory_mb = gpu_memories.iter().fold(0.0_f32, |a, &b| a.max(b));
        let peak_cpu_memory_mb = cpu_memories.iter().fold(0.0_f32, |a, &b| a.max(b));

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
             - Learning Rate: {:.2e} -> {:.2e} (decay: {:.2}x)\n\
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
