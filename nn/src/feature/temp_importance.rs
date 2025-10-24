//! Feature Importance.
//!
//! This module implements feature importance analysis techniques including
//! permutation importance, SHAP values, and model-based importance methods.

use crate::error::{NNError, Result};
use std::collections::HashMap;
use rand::Rng;

/// Feature importance result
#[derive(Debug, Clone)]
pub struct ImportanceResult {
    /// Feature importance scores (feature_index -> importance)
    pub feature_importance: HashMap<usize, f64>,
    /// Importance method used
    pub method: ImportanceMethod,
    /// Statistical significance (if available)
    pub significance: Option<HashMap<usize, f64>>,
    /// Computation time
    pub computation_time: f64,
    /// Method-specific metadata
    pub metadata: HashMap<String, String>,
}

/// Feature importance methods
#[derive(Debug, Clone)]
pub enum ImportanceMethod {
    /// Permutation importance
    Permutation { n_repeats: usize },
    /// Mean decrease impurity (for tree-based models)
    MeanDecreaseImpurity,
    /// Mean decrease accuracy (for tree-based models)
    MeanDecreaseAccuracy,
    /// SHAP values (simplified)
    Shap { n_samples: usize },
    /// Gain-based importance
    Gain,
    /// Split-based importance
    Split,
    /// Custom importance method
    Custom(String),
}

/// Feature importance analyzer
#[derive(Debug)]
pub struct FeatureImportance {
    /// Importance method
    pub method: ImportanceMethod,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

}
