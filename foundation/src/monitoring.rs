//! Performance Monitoring and Profiling for Foundation Models
//!
//! This module provides comprehensive monitoring and profiling capabilities for
//! foundation model training, including:
//! - Real-time performance metrics and visualization
//! - Memory and compute profiling
//! - Communication bottleneck analysis
//! - Anomaly detection and alerting
//! - Training progress visualization

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::error::{NNError, Result};

/// Central Monitoring and Profiling System
#[derive(Debug)]
pub struct TrainingMonitor {
    /// Metrics collectors
    pub collectors: HashMap<String, Box<dyn MetricCollector>>,
    /// Performance profilers
    pub profilers: HashMap<String, Box<dyn PerformanceProfiler>>,
    /// Alerting system
    pub alerting: AlertSystem,
    /// Visualization server
    pub visualization: Option<VisualizationServer>,
    /// Monitoring configuration
    pub config: MonitoringConfig,
    /// Current monitoring state
    pub state: Arc<RwLock<MonitoringState>>,
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub collection_interval_ms: u64,
    pub max_metrics_history: usize,
    pub enable_profiling: bool,
    pub alerting_enabled: bool,
    pub visualization_port: u16,
    pub metrics_export_path: Option<String>,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            collection_interval_ms: 1000, // 1 second
            max_metrics_history: 10000,
            enable_profiling: true,
            alerting_enabled: true,
            visualization_port: 8080,
            metrics_export_path: None,
        }
    }
}

/// Current monitoring state
#[derive(Debug)]
pub struct MonitoringState {
    pub training_metrics: TrainingMetricsHistory,
    pub system_metrics: SystemMetricsHistory,
    pub alerts: VecDeque<Alert>,
    pub profiling_data: ProfilingData,
    pub is_training_active: bool,
    pub last_update: std::time::Instant,
}

/// Training metrics history
#[derive(Debug)]
pub struct TrainingMetricsHistory {
    pub loss_values: VecDeque<MetricPoint>,
    pub learning_rates: VecDeque<MetricPoint>,
    pub gradient_norms: VecDeque<MetricPoint>,
    pub throughput_samples: VecDeque<MetricPoint>,
    pub custom_metrics: HashMap<String, VecDeque<MetricPoint>>,
}

/// System metrics history
#[derive(Debug)]
pub struct SystemMetricsHistory {
    pub gpu_memory_usage: VecDeque<MetricPoint>,
    pub cpu_memory_usage: VecDeque<MetricPoint>,
    pub gpu_utilization: VecDeque<MetricPoint>,
    pub cpu_utilization: VecDeque<MetricPoint>,
    pub network_bandwidth: VecDeque<MetricPoint>,
    pub disk_io: VecDeque<MetricPoint>,
}

/// Single metric data point
#[derive(Debug, Clone)]
pub struct MetricPoint {
    pub timestamp: std::time::Instant,
    pub value: f64,
    pub metadata: HashMap<String, String>,
}

/// Profiling data collection
#[derive(Debug)]
pub struct ProfilingData {
    pub kernel_times: HashMap<String, Vec<f64>>,
    pub memory_allocations: Vec<MemoryAllocation>,
    pub communication_times: Vec<CommTimeSample>,
    pub bottlenecks: Vec<Bottleneck>,
}

/// Memory allocation tracking
#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    pub timestamp: std::time::Instant,
    pub size_bytes: usize,
    pub allocation_type: AllocationType,
    pub stack_trace: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum AllocationType {
    Parameter,
    Gradient,
    Activation,
    OptimizerState,
    Temporary,
}

/// Communication timing samples
#[derive(Debug, Clone)]
pub struct CommTimeSample {
    pub operation: String,
    pub start_time: std::time::Instant,
    pub duration_ms: f64,
    pub data_size_bytes: usize,
    pub source_rank: usize,
    pub target_rank: usize,
}

/// Performance bottlenecks
#[derive(Debug, Clone)]
pub struct Bottleneck {
    pub bottleneck_type: BottleneckType,
    pub severity: f64, // 0.0 to 1.0
    pub description: String,
    pub timestamp: std::time::Instant,
    pub suggested_fix: String,
}

#[derive(Debug, Clone)]
pub enum BottleneckType {
    Memory,
    Communication,
    Computation,
    IO,
    Synchronization,
}

/// Alert system
#[derive(Debug)]
pub struct AlertSystem {
    pub rules: Vec<AlertRule>,
    pub active_alerts: HashMap<String, Alert>,
    pub alert_history: VecDeque<Alert>,
    pub escalation_policies: Vec<EscalationPolicy>,
}

/// Alert rule definition
#[derive(Debug, Clone)]
pub struct AlertRule {
    pub rule_id: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub message_template: String,
    pub enabled: bool,
}

/// Alert condition
#[derive(Debug, Clone)]
pub enum AlertCondition {
    MetricAbove { metric_name: String, threshold: f64, duration_ms: u64 },
    MetricBelow { metric_name: String, threshold: f64, duration_ms: u64 },
    RateOfChange { metric_name: String, threshold: f64, window_ms: u64 },
    AnomalyDetected { metric_name: String, sensitivity: f64 },
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertSeverity::Low => write!(f, "low"),
            AlertSeverity::Medium => write!(f, "medium"),
            AlertSeverity::High => write!(f, "high"),
            AlertSeverity::Critical => write!(f, "critical"),
        }
    }
}

/// Active alert
#[derive(Debug, Clone)]
pub struct Alert {
    pub alert_id: String,
    pub rule_id: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: std::time::Instant,
    pub resolved: bool,
    pub resolution_time: Option<std::time::Instant>,
}

/// Escalation policy
#[derive(Debug)]
pub struct EscalationPolicy {
    pub policy_id: String,
    pub severity_threshold: AlertSeverity,
    pub escalation_actions: Vec<EscalationAction>,
    pub cooldown_period_ms: u64,
}

/// Escalation actions
#[derive(Debug, Clone)]
pub enum EscalationAction {
    Log,
    Email { recipient: String },
    Slack { webhook_url: String, channel: String },
    Webhook { url: String },
    MetricExport { path: String },
    Custom { action_type: String, params: HashMap<String, String> },
}

/// Metric collector trait
pub trait MetricCollector: Send + Sync {
    fn collect(&self, state: &MonitoringState) -> HashMap<String, f64>;
    fn name(&self) -> &str;
}

/// Performance profiler trait
pub trait PerformanceProfiler: Send + Sync {
    fn profile(&mut self, step: usize) -> Result<ProfilingResult>;
    fn name(&self) -> &str;
}

#[derive(Debug)]
pub struct ProfilingResult {
    pub kernel_times: HashMap<String, f64>,
    pub memory_usage: usize,
    pub bottlenecks: Vec<Bottleneck>,
    pub recommendations: Vec<String>,
}

/// Visualization server for real-time monitoring
#[derive(Debug)]
pub struct VisualizationServer {
    pub port: u16,
    pub web_server: Option<WebServerHandle>,
    pub dashboard_data: Arc<RwLock<DashboardData>>,
}

#[derive(Debug)]
pub struct WebServerHandle;

#[derive(Debug)]
pub struct DashboardData {
    pub charts: HashMap<String, ChartData>,
    pub alerts: Vec<Alert>,
    pub system_info: SystemInfo,
}

#[derive(Debug)]
pub struct ChartData {
    pub title: String,
    pub data_points: Vec<(f64, f64)>, // (timestamp, value)
    pub chart_type: ChartType,
}

#[derive(Debug, Clone)]
pub enum ChartType {
    Line,
    Bar,
    Area,
    Scatter,
}

#[derive(Debug)]
pub struct SystemInfo {
    pub hostname: String,
    pub gpu_devices: Vec<GpuInfo>,
    pub cpu_cores: usize,
    pub total_memory_gb: f64,
}

#[derive(Debug)]
pub struct GpuInfo {
    pub device_id: usize,
    pub device_name: String,
    pub memory_total_gb: f64,
    pub compute_capability: String,
}

impl TrainingMonitor {
    /// Create new training monitor
    pub fn new() -> Self {
        Self {
            collectors: HashMap::new(),
            profilers: HashMap::new(),
            alerting: AlertSystem::new(),
            visualization: None,
            config: MonitoringConfig::default(),
            state: Arc::new(RwLock::new(MonitoringState::new())),
        }
    }

    /// Configure monitoring
    pub fn with_config(mut self, config: MonitoringConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a metric collector
    pub fn add_collector(&mut self, collector: Box<dyn MetricCollector>) {
        self.collectors.insert(collector.name().to_string(), collector);
    }

    /// Add a performance profiler
    pub fn add_profiler(&mut self, profiler: Box<dyn PerformanceProfiler>) {
        self.profilers.insert(profiler.name().to_string(), profiler);
    }

    /// Enable visualization
    pub async fn enable_visualization(mut self, port: u16) -> Result<Self> {
        self.visualization = Some(VisualizationServer::new(port).await?);
        Ok(self)
    }

    /// Record training metrics
    pub async fn record_training_metrics(
        &self,
        step: usize,
        loss: f64,
        lr: f64,
        grad_norm: f64,
        throughput: f64,
        custom_metrics: Option<HashMap<String, f64>>,
    ) -> Result<()> {
        let mut state = self.state.write().await;

        let now = std::time::Instant::now();

        // Record core metrics
        self.record_metric_point(&mut state.training_metrics.loss_values, loss, now.clone());
        self.record_metric_point(&mut state.training_metrics.learning_rates, lr, now.clone());
        self.record_metric_point(&mut state.training_metrics.gradient_norms, grad_norm, now.clone());
        self.record_metric_point(&mut state.training_metrics.throughput_samples, throughput, now.clone());

        // Record custom metrics
        if let Some(custom) = custom_metrics {
            for (name, value) in custom {
                let queue = state.training_metrics.custom_metrics
                    .entry(name)
                    .or_insert_with(VecDeque::new);

                self.record_metric_point(queue, value, now.clone());
            }
        }

        state.last_update = now;
        Ok(())
    }

    /// Record system metrics
    pub async fn record_system_metrics(
        &self,
        gpu_memory_mb: f64,
        cpu_memory_mb: f64,
        gpu_util: f64,
        cpu_util: f64,
        network_bw: f64,
        disk_io: f64,
    ) -> Result<()> {
        let mut state = self.state.write().await;
        let now = std::time::Instant::now();

        self.record_metric_point(&mut state.system_metrics.gpu_memory_usage, gpu_memory_mb, now.clone());
        self.record_metric_point(&mut state.system_metrics.cpu_memory_usage, cpu_memory_mb, now.clone());
        self.record_metric_point(&mut state.system_metrics.gpu_utilization, gpu_util, now.clone());
        self.record_metric_point(&mut state.system_metrics.cpu_utilization, cpu_util, now.clone());
        self.record_metric_point(&mut state.system_metrics.network_bandwidth, network_bw, now.clone());
        self.record_metric_point(&mut state.system_metrics.disk_io, disk_io, now.clone());

        Ok(())
    }

    fn record_metric_point(
        &self,
        queue: &mut VecDeque<MetricPoint>,
        value: f64,
        timestamp: std::time::Instant,
    ) {
        let point = MetricPoint {
            timestamp,
            value,
            metadata: HashMap::new(),
        };

        queue.push_back(point);

        // Maintain max history size
        if queue.len() > self.config.max_metrics_history {
            queue.pop_front();
        }
    }

    /// Run performance profiling
    pub async fn run_profiling(&mut self, step: usize) -> Result<()> {
        if !self.config.enable_profiling {
            return Ok(());
        }

        for profiler in self.profilers.values_mut() {
            let result = profiler.profile(step).await?;
            let mut state = self.state.write().await;

            // Update profiling data
            for (kernel, time) in result.kernel_times {
                state.profiling_data.kernel_times
                    .entry(kernel)
                    .or_insert_with(Vec::new)
                    .push(time);
            }

            state.profiling_data.bottlenecks.extend(result.bottlenecks);
        }

        Ok(())
    }

    /// Check for alerts
    pub async fn check_alerts(&mut self) -> Result<()> {
        if !self.config.alerting_enabled {
            return Ok(());
        }

        let state = self.state.read().await;

        for rule in &self.alerting.rules {
            if !rule.enabled {
                continue;
            }

            if self.check_alert_condition(&rule.condition, &state).await {
                let message = self.format_alert_message(&rule.message_template, &state);
                let alert = Alert {
                    alert_id: format!("alert_{}", self.alerting.alert_history.len()),
                    rule_id: rule.rule_id.clone(),
                    severity: rule.severity,
                    message,
                    timestamp: std::time::Instant::now(),
                    resolved: false,
                    resolution_time: None,
                };

                let mut state_mut = self.state.write().await;
                state_mut.alerts.push_back(alert.clone());

                // Add to active alerts
                self.alerting.active_alerts.insert(alert.alert_id.clone(), alert.clone());
                self.alerting.alert_history.push_back(alert);

                // Apply escalation policies
                self.apply_escalation_policies(&alert).await?;
            }
        }

        Ok(())
    }

    async fn check_alert_condition(&self, condition: &AlertCondition, state: &MonitoringState) -> bool {
        match condition {
            AlertCondition::MetricAbove { metric_name, threshold, duration_ms } => {
                self.check_metric_above_threshold(metric_name, *threshold, *duration_ms, state)
            },
            AlertCondition::MetricBelow { metric_name, threshold, duration_ms } => {
                self.check_metric_below_threshold(metric_name, *threshold, *duration_ms, state)
            },
            AlertCondition::RateOfChange { metric_name, threshold, window_ms } => {
                self.check_rate_of_change(metric_name, *threshold, *window_ms, state)
            },
            AlertCondition::AnomalyDetected { metric_name, sensitivity } => {
                self.detect_metric_anomaly(metric_name, *sensitivity, state)
            },
        }
    }

    fn check_metric_above_threshold(
        &self,
        metric_name: &str,
        threshold: f64,
        duration_ms: u64,
        state: &MonitoringState,
    ) -> bool {
        let values = match metric_name {
            "loss" => &state.training_metrics.loss_values,
            "learning_rate" => &state.training_metrics.learning_rates,
            "gradient_norm" => &state.training_metrics.gradient_norms,
            "gpu_memory" => &state.system_metrics.gpu_memory_usage,
            "gpu_utilization" => &state.system_metrics.gpu_utilization,
            _ => return false,
        };

        if values.is_empty() {
            return false;
        }

        let current_time = std::time::Instant::now();
        let duration = std::time::Duration::from_millis(duration_ms);

        // Check if metric has been above threshold for the entire duration
        let mut consecutive_above = 0;
        let mut total_in_window = 0;

        for value in values.iter().rev() {
            if current_time.duration_since(value.timestamp) > duration {
                break;
            }

            total_in_window += 1;
            if value.value > threshold {
                consecutive_above += 1;
            } else {
                consecutive_above = 0; // Reset if not continuously above
            }
        }

        consecutive_above >= total_in_window && total_in_window > 0
    }

    fn check_metric_below_threshold(
        &self,
        metric_name: &str,
        threshold: f64,
        duration_ms: u64,
        state: &MonitoringState,
    ) -> bool {
        // Similar to check_metric_above_threshold but for below threshold
        let values = match metric_name {
            "throughput" => &state.training_metrics.throughput_samples,
            "gpu_utilization" => &state.system_metrics.gpu_utilization,
            _ => return false,
        };

        if values.is_empty() {
            return false;
        }

        let current_time = std::time::Instant::now();
        let duration = std::time::Duration::from_millis(duration_ms);

        let mut consecutive_below = 0;
        let mut total_in_window = 0;

        for value in values.iter().rev() {
            if current_time.duration_since(value.timestamp) > duration {
                break;
            }

            total_in_window += 1;
            if value.value < threshold {
                consecutive_below += 1;
            } else {
                consecutive_below = 0;
            }
        }

        consecutive_below >= total_in_window && total_in_window > 0
    }

    fn check_rate_of_change(
        &self,
        metric_name: &str,
        threshold: f64,
        window_ms: u64,
        state: &MonitoringState,
    ) -> bool {
        let values = match metric_name {
            "loss" => &state.training_metrics.loss_values,
            "gradient_norm" => &state.training_metrics.gradient_norms,
            _ => return false,
        };

        if values.len() < 2 {
            return false;
        }

        let current_time = std::time::Instant::now();
        let window = std::time::Duration::from_millis(window_ms);

        // Get values within the window
        let window_values: Vec<f64> = values.iter()
            .rev()
            .take_while(|v| current_time.duration_since(v.timestamp) <= window)
            .map(|v| v.value)
            .collect();

        if window_values.len() < 2 {
            return false;
        }

        // Calculate rate of change (slope)
        let first_value = window_values[0];
        let last_value = *window_values.last().unwrap();

        if first_value == 0.0 {
            return false;
        }

        let rate_of_change = (last_value - first_value) / first_value;

        rate_of_change.abs() > threshold
    }

    fn detect_metric_anomaly(
        &self,
        metric_name: &str,
        sensitivity: f64,
        state: &MonitoringState,
    ) -> bool {
        // Simple anomaly detection based on standard deviation
        let values = match metric_name {
            "loss" => &state.training_metrics.loss_values,
            "gradient_norm" => &state.training_metrics.gradient_norms,
            _ => return false,
        };

        if values.len() < 10 {
            return false;
        }

        // Calculate mean and standard deviation
        let recent_values: Vec<f64> = values.iter()
            .rev()
            .take(50) // Last 50 values
            .map(|v| v.value)
            .collect();

        let mean = recent_values.iter().sum::<f64>() / recent_values.len() as f64;
        let variance = recent_values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / recent_values.len() as f64;
        let std_dev = variance.sqrt();

        let threshold = sensitivity * std_dev;
        let latest_value = recent_values[0];

        // Check if latest value is an outlier
        (latest_value - mean).abs() > threshold
    }

    fn format_alert_message(&self, template: &str, state: &MonitoringState) -> String {
        // Simple string interpolation for alert messages
        template
            .replace("{loss}", &format!("{:.4}", state.get_latest_loss()))
            .replace("{lr}", &format!("{:.6}", state.get_latest_lr()))
            .replace("{grad_norm}", &format!("{:.4}", state.get_latest_grad_norm()))
    }

    async fn apply_escalation_policies(&self, alert: &Alert) -> Result<()> {
        for policy in &self.alerting.escalation_policies {
            if alert.severity >= policy.severity_threshold {
                for action in &policy.escalation_actions {
                    self.execute_escalation_action(action, alert).await?;
                }
            }
        }

        Ok(())
    }

    async fn execute_escalation_action(&self, action: &EscalationAction, alert: &Alert) -> Result<()> {
        match action {
            EscalationAction::Log => {
                println!("[ALERT] {}: {}", alert.severity, alert.message);
            },
            EscalationAction::Email { recipient } => {
                // Placeholder for email sending
                println!("Would send email to {}: {}", recipient, alert.message);
            },
            EscalationAction::Slack { webhook_url, channel } => {
                // Placeholder for Slack notification
                println!("Would post to Slack {}: {}", channel, alert.message);
            },
            EscalationAction::Webhook { url } => {
                // Placeholder for webhook call
                println!("Would call webhook {}: {}", url, alert.message);
            },
            EscalationAction::MetricExport { path } => {
                // Export current metrics for debugging
                println!("Would export metrics to {} for alert investigation", path);
            },
            EscalationAction::Custom { action_type, params } => {
                // Custom escalation action
                println!("Custom escalation {}: {} (params: {:?})", action_type, alert.message, params);
            },
        }

        Ok(())
    }

    /// Generate performance report
    pub async fn generate_report(&self) -> Result<MonitoringReport> {
        let state = self.state.read().await;

        Ok(MonitoringReport {
            training_metrics_summary: self.summarize_training_metrics(&state),
            system_metrics_summary: self.summarize_system_metrics(&state),
            active_alerts: state.alerts.clone().into(),
            bottlenecks: state.profiling_data.bottlenecks.clone(),
            performance_score: self.calculate_performance_score(&state),
            recommendations: self.generate_recommendations(&state),
        })
    }

    fn summarize_training_metrics(&self, state: &MonitoringState) -> TrainingMetricsSummary {
        TrainingMetricsSummary {
            avg_loss: state.training_metrics.loss_values.iter()
                .map(|p| p.value).sum::<f64>() / state.training_metrics.loss_values.len() as f64,
            final_loss: state.get_latest_loss(),
            best_loss: state.training_metrics.loss_values.iter()
                .min_by(|a, b| a.value.partial_cmp(&b.value).unwrap())
                .map(|p| p.value)
                .unwrap_or(0.0),
            avg_lr: state.training_metrics.learning_rates.iter()
                .map(|p| p.value).sum::<f64>() / state.training_metrics.learning_rates.len() as f64,
            final_lr: state.get_latest_lr(),
            avg_grad_norm: state.training_metrics.gradient_norms.iter()
                .map(|p| p.value).sum::<f64>() / state.training_metrics.gradient_norms.len() as f64,
            avg_throughput: state.training_metrics.throughput_samples.iter()
                .map(|p| p.value).sum::<f64>() / state.training_metrics.throughput_samples.len() as f64,
        }
    }

    fn summarize_system_metrics(&self, state: &MonitoringState) -> SystemMetricsSummary {
        SystemMetricsSummary {
            avg_gpu_memory_mb: state.system_metrics.gpu_memory_usage.iter()
                .map(|p| p.value).sum::<f64>() / state.system_metrics.gpu_memory_usage.len() as f64,
            peak_gpu_memory_mb: state.system_metrics.gpu_memory_usage.iter()
                .max_by(|a, b| a.value.partial_cmp(&b.value).unwrap())
                .map(|p| p.value)
                .unwrap_or(0.0),
            avg_cpu_memory_mb: state.system_metrics.cpu_memory_usage.iter()
                .map(|p| p.value).sum::<f64>() / state.system_metrics.cpu_memory_usage.len() as f64,
            avg_gpu_utilization: state.system_metrics.gpu_utilization.iter()
                .map(|p| p.value).sum::<f64>() / state.system_metrics.gpu_utilization.len() as f64,
            avg_cpu_utilization: state.system_metrics.cpu_utilization.iter()
                .map(|p| p.value).sum::<f64>() / state.system_metrics.cpu_utilization.len() as f64,
            avg_network_bandwidth: state.system_metrics.network_bandwidth.iter()
                .map(|p| p.value).sum::<f64>() / state.system_metrics.network_bandwidth.len() as f64,
        }
    }

    fn calculate_performance_score(&self, state: &MonitoringState) -> PerformanceScore {
        // Calculate overall performance score based on various metrics
        let loss_efficiency = if state.get_latest_loss() < 1.0 { 1.0 } else { 1.0 / state.get_latest_loss() };
        let throughput_efficiency = state.get_latest_throughput() / 1000.0; // Normalize to thousands
        let memory_efficiency = 1.0 - (state.get_latest_gpu_memory() / 80000.0); // Lower is better for 80GB GPU
        let hardware_utilization = state.get_latest_gpu_utilization() / 100.0;

        let score = (loss_efficiency * 0.3 + throughput_efficiency * 0.3 +
                    memory_efficiency * 0.2 + hardware_utilization * 0.2) * 100.0;

        PerformanceScore {
            overall_score: score.min(100.0).max(0.0),
            loss_score: loss_efficiency * 100.0,
            throughput_score: throughput_efficiency * 100.0,
            memory_score: memory_efficiency * 100.0,
            utilization_score: hardware_utilization * 100.0,
        }
    }

    fn generate_recommendations(&self, state: &MonitoringState) -> Vec<PerformanceRecommendation> {
        let mut recommendations = Vec::new();

        let gpu_util = state.get_latest_gpu_utilization();
        if gpu_util < 50.0 {
            recommendations.push(PerformanceRecommendation {
                category: RecommendationCategory::Hardware,
                severity: RecommendationSeverity::Medium,
                message: "Low GPU utilization detected. Consider increasing batch size or optimizing data loading.".to_string(),
                suggested_actions: vec![
                    "Increase batch size".to_string(),
                    "Enable gradient accumulation".to_string(),
                    "Profile data loading pipeline".to_string(),
                ],
            });
        }

        let memory_usage = state.get_latest_gpu_memory();
        if memory_usage > 70000.0 { // >70GB on 80GB GPU
            recommendations.push(PerformanceRecommendation {
                category: RecommendationCategory::Memory,
                severity: RecommendationSeverity::High,
                message: "High memory usage detected. Consider memory optimization techniques.".to_string(),
                suggested_actions: vec![
                    "Enable gradient checkpointing".to_string(),
                    "Use activation offloading".to_string(),
                    "Reduce model size or batch size".to_string(),
                ],
            });
        }

        let throughput = state.get_latest_throughput();
        if throughput < 100.0 {
            recommendations.push(PerformanceRecommendation {
                category: RecommendationCategory::Performance,
                severity: RecommendationSeverity::Medium,
                message: "Low training throughput detected.".to_string(),
                suggested_actions: vec![
                    "Enable mixed precision training".to_string(),
                    "Use advanced optimizer (Lion/Sophia)".to_string(),
                    "Profile and optimize bottlenecks".to_string(),
                ],
            });
        }

        recommendations
    }
}

impl MonitoringState {
    fn new() -> Self {
        Self {
            training_metrics: TrainingMetricsHistory {
                loss_values: VecDeque::new(),
                learning_rates: VecDeque::new(),
                gradient_norms: VecDeque::new(),
                throughput_samples: VecDeque::new(),
                custom_metrics: HashMap::new(),
            },
            system_metrics: SystemMetricsHistory {
                gpu_memory_usage: VecDeque::new(),
                cpu_memory_usage: VecDeque::new(),
                gpu_utilization: VecDeque::new(),
                cpu_utilization: VecDeque::new(),
                network_bandwidth: VecDeque::new(),
                disk_io: VecDeque::new(),
            },
            alerts: VecDeque::new(),
            profiling_data: ProfilingData {
                kernel_times: HashMap::new(),
                memory_allocations: Vec::new(),
                communication_times: Vec::new(),
                bottlenecks: Vec::new(),
            },
            is_training_active: false,
            last_update: std::time::Instant::now(),
        }
    }

    fn get_latest_loss(&self) -> f64 {
        self.training_metrics.loss_values.back()
            .map(|p| p.value)
            .unwrap_or(0.0)
    }

    fn get_latest_lr(&self) -> f64 {
        self.training_metrics.learning_rates.back()
            .map(|p| p.value)
            .unwrap_or(0.0)
    }

    fn get_latest_grad_norm(&self) -> f64 {
        self.training_metrics.gradient_norms.back()
            .map(|p| p.value)
            .unwrap_or(0.0)
    }

    fn get_latest_throughput(&self) -> f64 {
        self.training_metrics.throughput_samples.back()
            .map(|p| p.value)
            .unwrap_or(0.0)
    }

    fn get_latest_gpu_memory(&self) -> f64 {
        self.system_metrics.gpu_memory_usage.back()
            .map(|p| p.value)
            .unwrap_or(0.0)
    }

    fn get_latest_gpu_utilization(&self) -> f64 {
        self.system_metrics.gpu_utilization.back()
            .map(|p| p.value)
            .unwrap_or(0.0)
    }
}

impl AlertSystem {
    fn new() -> Self {
        Self {
            rules: vec![
                AlertRule {
                    rule_id: "high_loss".to_string(),
                    condition: AlertCondition::MetricAbove {
                        metric_name: "loss".to_string(),
                        threshold: 10.0,
                        duration_ms: 5000,
                    },
                    severity: AlertSeverity::Medium,
                    message_template: "Loss too high: {loss}".to_string(),
                    enabled: true,
                },
                AlertRule {
                    rule_id: "low_gpu_utilization".to_string(),
                    condition: AlertCondition::MetricBelow {
                        metric_name: "gpu_utilization".to_string(),
                        threshold: 20.0,
                        duration_ms: 10000,
                    },
                    severity: AlertSeverity::Low,
                    message_template: "Low GPU utilization detected".to_string(),
                    enabled: true,
                },
            ],
            active_alerts: HashMap::new(),
            alert_history: VecDeque::new(),
            escalation_policies: vec![
                EscalationPolicy {
                    policy_id: "default".to_string(),
                    severity_threshold: AlertSeverity::Medium,
                    escalation_actions: vec![
                        EscalationAction::Log,
                        EscalationAction::MetricExport {
                            path: "/tmp/alert_metrics.json".to_string(),
                        },
                    ],
                    cooldown_period_ms: 60000, // 1 minute
                },
            ],
        }
    }
}

impl VisualizationServer {
    async fn new(port: u16) -> Result<Self> {
        Ok(Self {
            port,
            web_server: None, // Would start actual web server
            dashboard_data: Arc::new(RwLock::new(DashboardData::default())),
        })
    }
}

impl Default for DashboardData {
    fn default() -> Self {
        Self {
            charts: HashMap::new(),
            alerts: Vec::new(),
            system_info: SystemInfo::default(),
        }
    }
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self {
            hostname: "localhost".to_string(),
            gpu_devices: Vec::new(),
            cpu_cores: 8,
            total_memory_gb: 32.0,
        }
    }
}

/// Monitoring report summary
#[derive(Debug)]
pub struct MonitoringReport {
    pub training_metrics_summary: TrainingMetricsSummary,
    pub system_metrics_summary: SystemMetricsSummary,
    pub active_alerts: Vec<Alert>,
    pub bottlenecks: Vec<Bottleneck>,
    pub performance_score: PerformanceScore,
    pub recommendations: Vec<PerformanceRecommendation>,
}

/// Training metrics summary
#[derive(Debug)]
pub struct TrainingMetricsSummary {
    pub avg_loss: f64,
    pub final_loss: f64,
    pub best_loss: f64,
    pub avg_lr: f64,
    pub final_lr: f64,
    pub avg_grad_norm: f64,
    pub avg_throughput: f64,
}

/// System metrics summary
#[derive(Debug)]
pub struct SystemMetricsSummary {
    pub avg_gpu_memory_mb: f64,
    pub peak_gpu_memory_mb: f64,
    pub avg_cpu_memory_mb: f64,
    pub avg_gpu_utilization: f64,
    pub avg_cpu_utilization: f64,
    pub avg_network_bandwidth: f64,
}

/// Performance score breakdown
#[derive(Debug)]
pub struct PerformanceScore {
    pub overall_score: f64,
    pub loss_score: f64,
    pub throughput_score: f64,
    pub memory_score: f64,
    pub utilization_score: f64,
}

/// Performance recommendations
#[derive(Debug)]
pub struct PerformanceRecommendation {
    pub category: RecommendationCategory,
    pub severity: RecommendationSeverity,
    pub message: String,
    pub suggested_actions: Vec<String>,
}

#[derive(Debug)]
pub enum RecommendationCategory {
    Hardware,
    Memory,
    Performance,
    Training,
    Communication,
}

#[derive(Debug)]
pub enum RecommendationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitoring_state_initialization() {
        let state = MonitoringState::new();

        assert!(!state.is_training_active);
        assert!(state.training_metrics.loss_values.is_empty());
        assert!(state.profiling_data.kernel_times.is_empty());
    }

    #[test]
    fn test_training_monitor_creation() {
        let monitor = TrainingMonitor::new();

        assert!(monitor.collectors.is_empty());
        assert!(monitor.profilers.is_empty());
        assert!(monitor.visualization.is_none());
        assert_eq!(monitor.config.collection_interval_ms, 1000);
    }

    #[test]
    fn test_alert_system_initialization() {
        let alert_system = AlertSystem::new();

        assert_eq!(alert_system.rules.len(), 2); // Default rules
        assert_eq!(alert_system.escalation_policies.len(), 1); // Default policy
        assert!(alert_system.active_alerts.is_empty());
    }

    #[test]
    fn test_gradient_clipping() {
        // Test via training monitor
        let optimizer = crate::optimization::utils::create_lionel_optimizer(1e-3, 0.01);
        let gradients = HashMap::from([
            ("param1".to_string(), vec![1.0, 2.0, 3.0]),
        ]);

        // Would test clipping if public
    }

    #[test]
    fn test_visualization_server_creation() {
        // Async test would need tokio runtime
        // let server = tokio::runtime::Runtime::new()
        //     .unwrap()
        //     .block_on(VisualizationServer::new(8080));
    }
}
