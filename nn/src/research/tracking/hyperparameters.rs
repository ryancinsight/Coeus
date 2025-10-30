//! Hyperparameter Tracking and Versioning System
//!
//! This module provides comprehensive tracking, versioning, and analysis
//! of hyperparameters across experiments, including automatic discovery,
//! validation, and cross-experiment comparison capabilities.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Hyperparameter tracker with versioning and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterTracker {
    /// Current hyperparameters (key -> value with metadata)
    pub parameters: HashMap<String, HyperparameterEntry>,
    /// Parameter history for versioning
    pub history: Vec<HyperparameterVersion>,
    /// Parameter suggestions and search spaces
    pub search_spaces: HashMap<String, ParameterSearchSpace>,
    /// Parameter importance scores
    pub importance_scores: HashMap<String, f64>,
    /// Parameter correlation matrix
    pub correlations: HashMap<String, HashMap<String, f64>>,
    /// Auto-discovered parameters
    pub auto_discovered: Vec<String>,
}

impl HyperparameterTracker {
    /// Create new hyperparameter tracker
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
            history: Vec::new(),
            search_spaces: HashMap::new(),
            importance_scores: HashMap::new(),
            correlations: HashMap::new(),
            auto_discovered: Vec::new(),
        }
    }

    /// Log hyperparameter with automatic versioning
    pub fn log_hyperparameter(&mut self, key: String, value: serde_json::Value, description: Option<String>) -> crate::error::Result<()> {
        let entry = HyperparameterEntry {
            key: key.clone(),
            value: value.clone(),
            description: description.unwrap_or_default(),
            timestamp: chrono::Utc::now(),
            source: ParameterSource::Manual,
            confidence: 1.0,
        };

        // Check if this is an update
        if let Some(existing) = self.parameters.get(&key) {
            if existing.value != value {
                // Create version entry
                self.history.push(HyperparameterVersion {
                    parameter_key: key.clone(),
                    old_value: existing.value.clone(),
                    new_value: value.clone(),
                    timestamp: chrono::Utc::now(),
                    change_reason: "Manual update".to_string(),
                    performance_impact: None,
                });
            }
        }

        self.parameters.insert(key, entry);
        Ok(())
    }

    /// Auto-discover hyperparameter from model configuration
    pub fn auto_discover_parameter(&mut self, key: String, value: serde_json::Value, category: ParameterCategory) -> crate::error::Result<()> {
        let entry = HyperparameterEntry {
            key: key.clone(),
            value,
            description: format!("Auto-discovered {} parameter", category.to_string().to_lowercase()),
            timestamp: chrono::Utc::now(),
            source: ParameterSource::AutoDiscovered,
            confidence: 0.8,
        };

        self.parameters.insert(key.clone(), entry);
        self.auto_discovered.push(key);
        Ok(())
    }

    /// Define search space for hyperparameter optimization
    pub fn define_search_space(&mut self, key: String, search_space: ParameterSearchSpace) {
        self.search_spaces.insert(key, search_space);
    }

    /// Record parameter importance score
    pub fn set_importance(&mut self, key: String, importance: f64) {
        self.importance_scores.insert(key, importance);
    }

    /// Record parameter correlation
    pub fn set_correlation(&mut self, param1: String, param2: String, correlation: f64) {
        self.correlations.entry(param1).or_insert_with(HashMap::new)
                       .insert(param2.clone(), correlation);
        self.correlations.entry(param2).or_insert_with(HashMap::new)
                       .insert(param1, correlation);
    }

    /// Update parameter with performance impact tracking
    pub fn update_with_performance(&mut self, updates: HashMap<String, serde_json::Value>, performance: f64, old_performance: f64) {
        let performance_impact = performance - old_performance;

        for (key, new_value) in updates {
            if let Some(existing) = self.parameters.get(&key) {
                if existing.value != new_value {
                    self.history.push(HyperparameterVersion {
                        parameter_key: key.clone(),
                        old_value: existing.value.clone(),
                        new_value: new_value.clone(),
                        timestamp: chrono::Utc::now(),
                        change_reason: "Performance-guided update".to_string(),
                        performance_impact: Some(performance_impact),
                    });

                    // Update the parameter
                    let updated_entry = HyperparameterEntry {
                        value: new_value,
                        timestamp: chrono::Utc::now(),
                        ..existing.clone()
                    };
                    self.parameters.insert(key, updated_entry);
                }
            }
        }
    }

    /// Get parameter value by key
    pub fn get_parameter(&self, key: &str) -> Option<&HyperparameterEntry> {
        self.parameters.get(key)
    }

    /// Get current hyperparameters as JSON
    pub fn get_current_config(&self) -> serde_json::Value {
        let mut config = serde_json::Map::new();
        for (key, entry) in &self.parameters {
            config.insert(key.clone(), entry.value.clone());
        }
        serde_json::Value::Object(config)
    }

    /// Get parameter evolution history
    pub fn get_parameter_history(&self, key: &str) -> Vec<&HyperparameterVersion> {
        self.history.iter()
            .filter(|v| v.parameter_key == key)
            .collect()
    }

    /// Generate hyperparameter analysis report
    pub fn generate_analysis_report(&self) -> HyperparameterAnalysisReport {
        let mut value_distributions = HashMap::new();
        let mut type_distribution = HashMap::new();

        for entry in self.parameters.values() {
            // Count types
            let type_name = match entry.value {
                serde_json::Value::Number(_) => "numeric",
                serde_json::Value::String(_) => "string",
                serde_json::Value::Bool(_) => "boolean",
                serde_json::Value::Array(_) => "array",
                serde_json::Value::Object(_) => "object",
                serde_json::Value::Null => "null",
            };
            *type_distribution.entry(type_name.to_string()).or_insert(0) += 1;
        }

        // Calculate summary statistics
        let total_parameters = self.parameters.len();
        let manual_parameters = self.parameters.values()
            .filter(|p| matches!(p.source, ParameterSource::Manual))
            .count();
        let auto_discovered = self.auto_discovered.len();
        let with_search_spaces = self.search_spaces.len();
        let with_importance_scores = self.importance_scores.len();

        HyperparameterAnalysisReport {
            total_parameters,
            manual_parameters,
            auto_discovered_parameters: auto_discovered,
            search_space_parameters: with_search_spaces,
            importance_scored_parameters: with_importance_scores,
            parameter_types: type_distribution,
            version_changes: self.history.len(),
            correlations_found: self.correlations.len(),
        }
    }

    /// Export hyperparameters in multiple formats
    pub fn export(&self, format: ExportFormat) -> String {
        match format {
            ExportFormat::Json => serde_json::to_string_pretty(self).unwrap_or_default(),
            ExportFormat::Yaml => {
                // Fallback to JSON for YAML since we don't have yaml crate
                serde_json::to_string_pretty(self).unwrap_or_default()
            },
            ExportFormat::Python => self.export_python_config(),
            ExportFormat::Shell => self.export_shell_config(),
        }
    }

    fn export_python_config(&self) -> String {
        let mut config = String::from("# Hyperparameter Configuration\n# Auto-generated from Coeus Research Framework\n\n");
        config.push_str("hyperparameters = {\n");

        for (key, entry) in &self.parameters {
            let python_value = match &entry.value {
                serde_json::Value::String(s) => format!("'{}'", s),
                serde_json::Value::Number(n) => n.to_string(),
                serde_json::Value::Bool(b) => b.to_string(),
                serde_json::Value::Null => "None".to_string(),
                serde_json::Value::Array(arr) => format!("{:?}", arr),
                serde_json::Value::Object(obj) => format!("{:?}", obj),
            };
            config.push_str(&format!("    '{}': {},\n", key, python_value));
        }

        config.push_str("}\n");
        config
    }

    fn export_shell_config(&self) -> String {
        let mut config = String::from("# Hyperparameter Configuration\n# Auto-generated from Coeus Research Framework\n\n");

        for (key, entry) in &self.parameters {
            match &entry.value {
                serde_json::Value::String(s) => config.push_str(&format!("export {}=\"{}\"\n", key.to_uppercase(), s)),
                serde_json::Value::Number(n) => config.push_str(&format!("export {}={}\n", key.to_uppercase(), n)),
                serde_json::Value::Bool(b) => config.push_str(&format!("export {}={}\n", key.to_uppercase(), if *b { "true" } else { "false" })),
                _ => {} // Skip complex types for shell export
            }
        }

        config
    }
}

/// Individual hyperparameter entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterEntry {
    /// Parameter key/name
    pub key: String,
    /// Parameter value (JSON)
    pub value: serde_json::Value,
    /// Human-readable description
    pub description: String,
    /// When this parameter was set
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// How this parameter was obtained
    pub source: ParameterSource,
    /// Confidence in this parameter value (0.0 to 1.0)
    pub confidence: f64,
}

/// How a parameter was obtained
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ParameterSource {
    /// Manually set by researcher
    Manual,
    /// Auto-discovered from model/code
    AutoDiscovered,
    /// Generated from optimization routine
    Optimization,
    /// Inherited from parent experiment
    Inherited,
    /// Set by automated tuning system
    AutoTuned,
}

/// Hyperparameter version/change tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterVersion {
    /// Parameter that was changed
    pub parameter_key: String,
    /// Previous value
    pub old_value: serde_json::Value,
    /// New value
    pub new_value: serde_json::Value,
    /// When the change occurred
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Reason for the change
    pub change_reason: String,
    /// Performance impact of this change (if known)
    pub performance_impact: Option<f64>,
}

/// Search space definition for parameter optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSearchSpace {
    /// Parameter type
    pub param_type: ParameterType,
    /// Search strategy
    pub search_strategy: SearchStrategy,
    /// Distribution or discrete values
    pub distribution: ParameterDistribution,
    /// Feasibility constraints
    pub constraints: Vec<ParameterConstraint>,
}

impl ParameterSearchSpace {
    /// Create continuous uniform search space
    pub fn continuous_uniform(min: f64, max: f64) -> Self {
        Self {
            param_type: ParameterType::Continuous,
            search_strategy: SearchStrategy::Uniform,
            distribution: ParameterDistribution::Continuous { min, max },
            constraints: Vec::new(),
        }
    }

    /// Create discrete choice search space
    pub fn discrete_choice(values: Vec<serde_json::Value>) -> Self {
        Self {
            param_type: ParameterType::Discrete,
            search_strategy: SearchStrategy::Uniform,
            distribution: ParameterDistribution::Discrete { values },
            constraints: Vec::new(),
        }
    }

    /// Create logarithmic search space
    pub fn logarithmic(min: f64, max: f64, base: f64) -> Self {
        Self {
            param_type: ParameterType::Continuous,
            search_strategy: SearchStrategy::Logarithmic,
            distribution: ParameterDistribution::Logarithmic { min, max, base },
            constraints: Vec::new(),
        }
    }
}

/// Parameter type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    /// Continuous real-valued parameters
    Continuous,
    /// Integer-valued parameters
    Integer,
    /// Discrete/categorical parameters
    Discrete,
    /// Boolean parameters
    Boolean,
}

/// Search strategy for exploring parameter space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Random uniform sampling
    Uniform,
    /// Logarithmic scaling (for ranges spanning orders of magnitude)
    Logarithmic,
    /// Gaussian/normal distribution sampling
    Gaussian,
    /// Grid search over discrete values
    Grid,
}

/// Parameter distribution specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterDistribution {
    /// Continuous uniform distribution
    Continuous { min: f64, max: f64 },
    /// Continuous normal distribution
    Normal { mean: f64, std: f64 },
    /// Logarithmic distribution (base^value)
    Logarithmic { min: f64, max: f64, base: f64 },
    /// Discrete set of values
    Discrete { values: Vec<serde_json::Value> },
    /// Categorical distribution with weights
    Categorical { categories: Vec<String>, weights: Vec<f64> },
}

/// Parameter constraints for feasibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterConstraint {
    /// Minimum value constraint
    MinValue(f64),
    /// Maximum value constraint
    MaxValue(f64),
    /// Must be in discrete set
    InSet(Vec<serde_json::Value>),
    /// Must match regex pattern
    Regex(String),
    /// Custom constraint function (stored as description)
    Custom(String),
}

/// Parameter category for auto-discovery
#[derive(Debug, Clone)]
pub enum ParameterCategory {
    /// Learning rate parameters
    LearningRate,
    /// Batch size parameters
    BatchSize,
    /// Model architecture parameters
    Architecture,
    /// Regularization parameters
    Regularization,
    /// Optimization parameters
    Optimization,
    /// Training schedule parameters
    TrainingSchedule,
    /// Data processing parameters
    DataProcessing,
}

impl std::fmt::Display for ParameterCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LearningRate => write!(f, "Learning Rate"),
            Self::BatchSize => write!(f, "Batch Size"),
            Self::Architecture => write!(f, "Architecture"),
            Self::Regularization => write!(f, "Regularization"),
            Self::Optimization => write!(f, "Optimization"),
            Self::TrainingSchedule => write!(f, "Training Schedule"),
            Self::DataProcessing => write!(f, "Data Processing"),
        }
    }
}

/// Analysis report for hyperparameter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterAnalysisReport {
    /// Total number of parameters
    pub total_parameters: usize,
    /// Number of manually specified parameters
    pub manual_parameters: usize,
    /// Number of auto-discovered parameters
    pub auto_discovered_parameters: usize,
    /// Number of parameters with search spaces defined
    pub search_space_parameters: usize,
    /// Number of parameters with importance scores
    pub importance_scored_parameters: usize,
    /// Distribution of parameter types
    pub parameter_types: HashMap<String, usize>,
    /// Total version changes recorded
    pub version_changes: usize,
    /// Number of parameter correlations discovered
    pub correlations_found: usize,
}

impl std::fmt::Display for HyperparameterAnalysisReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Hyperparameter Analysis:\n\
             ├─ Total Parameters: {}\n\
             ├─ Manual Parameters: {}\n\
             ├─ Auto-Discovered: {}\n\
             ├─ With Search Spaces: {}\n\
             ├─ With Importance Scores: {}\n\
             ├─ Version Changes: {}\n\
             ├─ Correlations Found: {}\n\
             └─ Parameter Types: {}",
            self.total_parameters,
            self.manual_parameters,
            self.auto_discovered_parameters,
            self.search_space_parameters,
            self.importance_scored_parameters,
            self.version_changes,
            self.correlations_found,
            self.parameter_types.iter()
                .map(|(t, c)| format!("{}: {}", t, c))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Export formats for hyperparameters
#[derive(Debug, Clone)]
pub enum ExportFormat {
    /// JSON format
    Json,
    /// YAML format
    Yaml,
    /// Python configuration file
    Python,
    /// Shell environment variables
    Shell,
}
