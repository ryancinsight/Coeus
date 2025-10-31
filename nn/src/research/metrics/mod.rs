//! Advanced Metrics Collection System for Research
//!
//! This module provides comprehensive metrics collection, analysis, and
//! visualization for machine learning research experiments. It supports
//! multi-dimensional metrics, automatic aggregation, statistical analysis,
//! and research publication-ready outputs.

use std::collections::{HashMap, BTreeMap};
use serde::{Serialize, Deserialize};

// Re-export core types
// pub mod collection; // TODO: Implement metrics collection module
// pub mod analysis; // TODO: Implement metrics analysis module
// pub mod aggregation; // TODO: Implement metrics aggregation module
// pub mod export; // TODO: Implement metrics export module

// pub use collection::*;
// pub use analysis::*;
// pub use aggregation::*;
// pub use export::*;

/// Export format options
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
    Plotly,
}

/// Export result types
#[derive(Debug, Clone)]
pub enum ExportResult {
    Json(serde_json::Value),
    Csv(String),
}

/// Aggregation rule for metrics
#[derive(Debug, Clone)]
pub struct AggregationRule {
    pub name: String,
    pub operation: AggregationOperation,
}

/// Aggregation operation types
#[derive(Debug, Clone)]
pub enum AggregationOperation {
    Mean,
    Sum,
    Min,
    Max,
    Count,
}

impl AggregationRule {
    pub fn apply(&self, values: &[f64]) -> crate::error::Result<f64> {
        if values.is_empty() {
            return Err(crate::error::NNError::InvalidInput {
                message: "Cannot aggregate empty values".to_string(),
            });
        }

        match self.operation {
            AggregationOperation::Mean => Ok(values.iter().sum::<f64>() / values.len() as f64),
            AggregationOperation::Sum => Ok(values.iter().sum()),
            AggregationOperation::Min => Ok(values.iter().fold(f64::INFINITY, |a, &b| a.min(b))),
            AggregationOperation::Max => Ok(values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))),
            AggregationOperation::Count => Ok(values.len() as f64),
        }
    }
}

/// Metric alert configuration
#[derive(Debug, Clone)]
pub struct MetricAlert {
    pub metric_name: String,
    pub condition: AlertCondition,
    pub message: String,
    pub severity: AlertSeverity,
}

/// Alert condition types
#[derive(Debug, Clone)]
pub enum AlertCondition {
    AboveThreshold(f64),
    BelowThreshold(f64),
    OutsideRange(f64, f64),
}

impl AlertCondition {
    pub fn evaluate(&self, value: f64) -> bool {
        match self {
            AlertCondition::AboveThreshold(threshold) => value > *threshold,
            AlertCondition::BelowThreshold(threshold) => value < *threshold,
            AlertCondition::OutsideRange(min, max) => value < *min || value > *max,
        }
    }
}

/// Alert severity levels
#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Statistical analysis of metric data
#[derive(Debug, Clone)]
pub struct StatisticalAnalysis {
    pub mean: f64,
    pub median: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub data_points: usize,
    pub outliers: Vec<f64>,
}

impl StatisticalAnalysis {
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                mean: 0.0,
                median: 0.0,
                variance: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                data_points: 0,
                outliers: Vec::new(),
            };
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let min = *sorted.first().unwrap();
        let max = *sorted.last().unwrap();

        // Simple outlier detection using IQR method
        let q1 = sorted[sorted.len() / 4];
        let q3 = sorted[sorted.len() * 3 / 4];
        let iqr = q3 - q1;
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;

        let outliers = values.iter()
            .filter(|&&v| v < lower_bound || v > upper_bound)
            .cloned()
            .collect();

        Self {
            mean,
            median,
            variance,
            std_dev,
            min,
            max,
            data_points: values.len(),
            outliers,
        }
    }
}

/// Metric comparison results
#[derive(Debug, Clone)]
pub struct MetricComparison {
    pub metric1: String,
    pub metric2: String,
    pub series1_count: usize,
    pub series2_count: usize,
    pub correlation: Option<f64>,
    pub divergence_points: Vec<usize>,
}

/// Comprehensive metrics report
#[derive(Debug, Clone)]
pub struct MetricsReport {
    pub title: String,
    pub generated_at: chrono::DateTime<chrono::Utc>,
    pub summary: MetricsSummary,
    pub metrics: HashMap<String, MetricData>,
    pub alerts: Vec<MetricAlert>,
    pub recommendations: Vec<String>,
}

/// Individual metric data
#[derive(Debug, Clone)]
pub struct MetricData {
    pub name: String,
    pub entries: Vec<MetricEntry>,
    pub analysis: Option<StatisticalAnalysis>,
}

/// Metrics summary statistics
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub total_metrics: u64,
    pub active_metrics: usize,
    pub total_collections: u64,
    pub alerts_triggered: usize,
    pub data_points: usize,
}

/// Central research metrics collection and analysis system
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    /// Active metric collectors by category
    collectors: HashMap<String, Box<dyn MetricCollector>>,
    /// Collected metrics data
    metrics_data: BTreeMap<String, Vec<MetricEntry>>,
    /// Automatic aggregation rules
    aggregations: HashMap<String, AggregationRule>,
    /// Alerting thresholds and conditions
    alerts: Vec<MetricAlert>,
    /// Metrics configuration
    config: MetricsConfig,
    /// Statistical analysis results
    analysis_results: HashMap<String, StatisticalAnalysis>,
    /// Collection statistics
    stats: MetricsCollectionStats,
}

impl MetricsCollector {
    /// Create new metrics collector with default configuration
    pub fn new() -> Self {
        Self {
            collectors: HashMap::new(),
            metrics_data: BTreeMap::new(),
            aggregations: HashMap::new(),
            alerts: Vec::new(),
            config: MetricsConfig::default(),
            analysis_results: HashMap::new(),
            stats: MetricsCollectionStats::default(),
        }
    }

    /// Register a custom metric collector
    pub fn register_collector<C: MetricCollector + 'static>(&mut self, name: String, collector: C) {
        self.collectors.insert(name, Box::new(collector));
    }

    /// Collect metrics from registered collectors
    pub fn collect_metrics(&mut self) -> crate::error::Result<()> {
        for (name, collector) in &mut self.collectors {
            let metrics = collector.collect_metrics()?;
            for metric in metrics {
                self.store_metric(metric);
            }
        }

        self.update_aggregations()?;
        self.check_alerts();
        self.update_stats();

        Ok(())
    }

    /// Manually record a metric value
    pub fn record_metric(&mut self, name: String, value: f64, timestamp: Option<chrono::DateTime<chrono::Utc>>, context: HashMap<String, serde_json::Value>) {
        let timestamp = timestamp.unwrap_or_else(|| chrono::Utc::now());

        let entry = MetricEntry {
            name: name.clone(),
            value,
            timestamp,
            context,
            tags: Vec::new(),
            metadata: HashMap::new(),
        };

        self.store_metric(entry);
    }

    /// Get metric values over time
    pub fn get_metric_series(&self, name: &str, start_time: Option<chrono::DateTime<chrono::Utc>>, end_time: Option<chrono::DateTime<chrono::Utc>>) -> Vec<&MetricEntry> {
        if let Some(entries) = self.metrics_data.get(name) {
            entries.iter()
                .filter(|entry| {
                    let after_start = start_time.map_or(true, |start| entry.timestamp >= start);
                    let before_end = end_time.map_or(true, |end| entry.timestamp <= end);
                    after_start && before_end
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Calculate statistical properties of a metric
    pub fn analyze_metric(&mut self, name: &str) -> Option<&StatisticalAnalysis> {
        if let Some(entries) = self.metrics_data.get(name) {
            if entries.is_empty() {
                return None;
            }

            let values: Vec<f64> = entries.iter().map(|e| e.value).collect();
            let analysis = StatisticalAnalysis::from_values(&values);
            self.analysis_results.insert(name.to_string(), analysis);
            self.analysis_results.get(name)
        } else {
            None
        }
    }

    /// Compare two metrics over time
    pub fn compare_metrics(&self, metric1: &str, metric2: &str) -> MetricComparison {
        let series1 = self.get_metric_series(metric1, None, None);
        let series2 = self.get_metric_series(metric2, None, None);

        MetricComparison {
            metric1: metric1.to_string(),
            metric2: metric2.to_string(),
            series1_count: series1.len(),
            series2_count: series2.len(),
            correlation: self.calculate_correlation(&series1, &series2),
            divergence_points: self.find_divergence_points(&series1, &series2),
        }
    }

    /// Generate comprehensive metrics report
    pub fn generate_report(&self, title: String, include_analysis: bool) -> MetricsReport {
        let mut report = MetricsReport {
            title,
            generated_at: chrono::Utc::now(),
            summary: self.generate_summary(),
            metrics: HashMap::new(),
            alerts: self.alerts.clone(),
            recommendations: Vec::new(),
        };

        for (metric_name, entries) in &self.metrics_data {
            let data = MetricData {
                name: metric_name.clone(),
                entries: entries.clone(),
                analysis: if include_analysis {
                    self.analysis_results.get(metric_name).cloned()
                } else {
                    None
                },
            };
            report.metrics.insert(metric_name.clone(), data);
        }

        report.recommendations = self.generate_recommendations();
        report
    }

    /// Set up automatic aggregation for a metric
    pub fn add_aggregation(&mut self, metric_name: String, rule: AggregationRule) {
        self.aggregations.insert(metric_name, rule);
    }

    /// Add metric alert condition
    pub fn add_alert(&mut self, alert: MetricAlert) {
        self.alerts.push(alert);
    }

    /// Export metrics in various formats
    pub fn export(&self, format: ExportFormat) -> ExportResult {
        match format {
            ExportFormat::Json => ExportResult::Json(serde_json::to_value(self).unwrap_or_default()),
            ExportFormat::Csv => ExportResult::Csv(self.export_csv()),
            ExportFormat::Plotly => ExportResult::Json(self.export_plotly()),
        }
    }

    // Private helper methods
    fn store_metric(&mut self, metric: MetricEntry) {
        self.metrics_data.entry(metric.name.clone())
            .or_insert_with(Vec::new)
            .push(metric);
        self.stats.total_metrics += 1;
    }

    fn update_aggregations(&mut self) -> crate::error::Result<()> {
        for (metric_name, rule) in &self.aggregations {
            if let Some(entries) = self.metrics_data.get(metric_name) {
                let values: Vec<f64> = entries.iter().map(|e| e.value).collect();
                let aggregated_value = rule.apply(&values)?;

                let context = HashMap::from([("aggregation".to_string(), serde_json::Value::String(format!("{:?}", rule.operation)))]);
                self.record_metric(format!("{}_{}", metric_name, rule.name), aggregated_value, None, context);
            }
        }
        Ok(())
    }

    fn check_alerts(&self) {
        let latest_values: HashMap<String, f64> = self.metrics_data.iter()
            .filter_map(|(name, entries)| {
                entries.last().map(|entry| (name.clone(), entry.value))
            })
            .collect();

        for alert in &self.alerts {
            if let Some(value) = latest_values.get(&alert.metric_name) {
                if alert.condition.evaluate(*value) {
                    // In a real implementation, this would trigger alerts/notifications
                    println!("ALERT: {} - Value: {:.4}", alert.message, value);
                }
            }
        }
    }

    fn update_stats(&mut self) {
        self.stats.active_metrics = self.metrics_data.len();
        self.stats.total_collections += 1;
    }

    fn calculate_correlation(&self, series1: &[&MetricEntry], series2: &[&MetricEntry]) -> Option<f64> {
        // Simple pearson correlation calculation
        if series1.is_empty() || series2.is_empty() {
            return None;
        }

        let values1: Vec<f64> = series1.iter().map(|e| e.value).collect();
        let values2: Vec<f64> = series2.iter().map(|e| e.value).collect();

        let len = values1.len().min(values2.len());

        let mean1 = values1.iter().sum::<f64>() / len as f64;
        let mean2 = values2.iter().sum::<f64>() / len as f64;

        let numerator: f64 = (0..len).map(|i| (values1[i] - mean1) * (values2[i] - mean2)).sum();
        let denominator1: f64 = values1.iter().map(|v| (v - mean1).powi(2)).sum();
        let denominator2: f64 = values2.iter().map(|v| (v - mean2).powi(2)).sum();

        let denominator = (denominator1 * denominator2).sqrt();

        if denominator == 0.0 {
            None
        } else {
            Some(numerator / denominator)
        }
    }

    fn find_divergence_points(&self, series1: &[&MetricEntry], series2: &[&MetricEntry]) -> Vec<usize> {
        let mut divergences = Vec::new();
        let threshold = 0.1; // 10% relative difference

        let len = series1.len().min(series2.len());
        for i in 0..len {
            let v1 = series1[i].value;
            let v2 = series2[i].value;

            if v1 != 0.0 || v2 != 0.0 {
                let diff = ((v1 - v2) / ((v1 + v2) / 2.0)).abs();
                if diff > threshold {
                    divergences.push(i);
                }
            }
        }

        divergences
    }

    fn generate_summary(&self) -> MetricsSummary {
        MetricsSummary {
            total_metrics: self.stats.total_metrics,
            active_metrics: self.stats.active_metrics,
            total_collections: self.stats.total_collections,
            alerts_triggered: 0, // Would track actual alerts triggered
            data_points: self.metrics_data.values().map(|v| v.len()).sum(),
        }
    }

    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Analyze metric distributions and suggest improvements
        for (name, analysis) in &self.analysis_results {
            if let Some(analysis) = analysis {
                if analysis.variance > analysis.mean * 0.5 {
                    recommendations.push(format!("High variance detected in '{}'. Consider stabilizing the metric.", name));
                }

                if analysis.outliers.len() > analysis.data_points as usize / 10 {
                    recommendations.push(format!("Many outliers detected in '{}'. Consider data cleaning.", name));
                }
            }
        }

        recommendations
    }

    fn export_csv(&self) -> String {
        let mut csv = String::from("metric,timestamp,value\n");

        for (metric_name, entries) in &self.metrics_data {
            for entry in entries {
                csv.push_str(&format!("{},{},{:.6}\n",
                    metric_name,
                    entry.timestamp.to_rfc3339(),
                    entry.value
                ));
            }
        }

        csv
    }

    fn export_plotly(&self) -> serde_json::Value {
        // Basic plotly-compatible format for metrics visualization
        let traces: Vec<serde_json::Value> = self.metrics_data.iter()
            .enumerate()
            .map(|(i, (name, entries))| {
                let x: Vec<String> = entries.iter().map(|e| e.timestamp.to_rfc3339()).collect();
                let y: Vec<f64> = entries.iter().map(|e| e.value).collect();

                serde_json::json!({
                    "name": name,
                    "type": "scatter",
                    "x": x,
                    "y": y,
                    "mode": "lines+markers"
                })
            })
            .collect();

        serde_json::json!({
            "data": traces,
            "layout": {
                "title": "Metrics Visualization",
                "xaxis": {"title": "Time"},
                "yaxis": {"title": "Value"}
            }
        })
    }
}

/// Configuration for metrics collection system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Collection interval in seconds
    pub collection_interval_seconds: u64,
    /// Maximum metrics to store per metric
    pub max_metrics_per_collection: usize,
    /// Maximum total collections to keep
    pub max_collections: usize,
    /// Enable statistical analysis
    pub enable_analysis: bool,
    /// Enable automatic aggregation
    pub enable_aggregation: bool,
    /// Alert check interval (seconds)
    pub alert_check_interval_seconds: u64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            collection_interval_seconds: 60,
            max_metrics_per_collection: 1000,
            max_collections: 10000,
            enable_analysis: true,
            enable_aggregation: true,
            alert_check_interval_seconds: 30,
        }
    }
}

/// Metrics collection statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricsCollectionStats {
    /// Total metrics collected
    pub total_metrics: u64,
    /// Currently active metrics
    pub active_metrics: usize,
    /// Total collection operations performed
    pub total_collections: u64,
    /// Memory used by metrics storage
    pub memory_usage_bytes: u64,
}

/// Trait for custom metric collectors
pub trait MetricCollector {
    /// Collect metrics from this collector
    fn collect_metrics(&mut self) -> crate::error::Result<Vec<MetricEntry>>;

    /// Get collector name
    fn name(&self) -> &str;

    /// Get collector description
    fn description(&self) -> &str {
        ""
    }
}

/// Individual metric entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricEntry {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Timestamp when metric was recorded
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Contextual information
    pub context: HashMap<String, serde_json::Value>,
    /// Metric tags
    pub tags: Vec<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl MetricEntry {
    /// Create new metric entry
    pub fn new(name: String, value: f64) -> Self {
        Self {
            name,
            value,
            timestamp: chrono::Utc::now(),
            context: HashMap::new(),
            tags: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}
