//! Feature Selection.
//!
//! This module implements various feature selection algorithms including
//! filter methods (correlation, mutual information), wrapper methods
//! (recursive feature elimination), and embedded methods (LASSO-based).

use crate::error::{NNError, Result};
use std::collections::HashMap;

/// Feature selection result
#[derive(Debug)]
pub struct FeatureSelectionResult {
    /// Selected feature indices
    pub selected_features: Vec<usize>,
    /// Feature importance scores
    pub feature_scores: HashMap<usize, f64>,
    /// Selection method used
    pub method: SelectionMethod,
    /// Number of features originally available
    pub original_count: usize,
    /// Selection statistics
    pub statistics: SelectionStatistics,
}

/// Feature selection statistics
#[derive(Debug, Clone)]
pub struct SelectionStatistics {
    /// Reduction ratio (selected / original)
    pub reduction_ratio: f64,
    /// Selection confidence score
    pub confidence_score: f64,
    /// Computational time
    pub computation_time: f64,
}

/// Feature selection methods
#[derive(Debug, Clone)]
pub enum SelectionMethod {
    /// Filter methods
    Correlation {
        threshold: f64,
    },
    MutualInformation {
        k: usize,
    },
    ChiSquare {
        k: usize,
    },
    /// Wrapper methods
    RecursiveElimination {
        estimator: std::sync::Arc<dyn Estimator>,
        step: usize,
    },
    /// Embedded methods
    Lasso {
        alpha: f64,
    },
    TreeBased {
        max_features: Option<usize>,
    },
    /// Custom method
    Custom(String),
}

/// Estimator trait for wrapper methods
pub trait Estimator: std::fmt::Debug + Send + Sync {
    /// Fit estimator and return score
    fn fit_score(&self, features: &[Vec<f64>], targets: &[f64]) -> Result<f64>;
}

/// Feature selector
#[derive(Debug)]
pub struct FeatureSelector {
    /// Selection method
    pub method: SelectionMethod,
    /// Target number of features (if applicable)
    pub target_features: Option<usize>,
}

impl FeatureSelector {
    /// Create a new feature selector
    pub fn new(method: SelectionMethod) -> Self {
        Self {
            method,
            target_features: None,
        }
    }

    /// Set target number of features
    pub fn with_target_features(mut self, target: usize) -> Self {
        self.target_features = Some(target);
        self
    }

    /// Perform feature selection
    pub fn select(&self, features: &[Vec<f64>], targets: &[f64]) -> Result<FeatureSelectionResult> {
        let start_time = std::time::Instant::now();

        let mut selected_features = Vec::new();
        let mut feature_scores = HashMap::new();

        match &self.method {
            SelectionMethod::Correlation { threshold } => {
                self.select_correlation(
                    features,
                    targets,
                    *threshold,
                    &mut selected_features,
                    &mut feature_scores,
                )?;
            }
            SelectionMethod::MutualInformation { k } => {
                self.select_mutual_information(
                    features,
                    targets,
                    *k,
                    &mut selected_features,
                    &mut feature_scores,
                )?;
            }
            SelectionMethod::ChiSquare { k } => {
                self.select_chi_square(
                    features,
                    targets,
                    *k,
                    &mut selected_features,
                    &mut feature_scores,
                )?;
            }
            SelectionMethod::RecursiveElimination { estimator, step } => {
                self.select_recursive_elimination(
                    features,
                    targets,
                    estimator.as_ref(),
                    *step,
                    &mut selected_features,
                    &mut feature_scores,
                )?;
            }
            SelectionMethod::Lasso { alpha } => {
                self.select_lasso(
                    features,
                    targets,
                    *alpha,
                    &mut selected_features,
                    &mut feature_scores,
                )?;
            }
            SelectionMethod::TreeBased { max_features } => {
                self.select_tree_based(
                    features,
                    targets,
                    *max_features,
                    &mut selected_features,
                    &mut feature_scores,
                )?;
            }
            SelectionMethod::Custom(_) => {
                return Err(NNError::InvalidConfiguration {
                    message: "Custom selection methods not implemented".to_string(),
                });
            }
        }

        let computation_time = start_time.elapsed().as_secs_f64();
        let original_count = features.len();
        let selected_count = selected_features.len();

        let statistics = SelectionStatistics {
            reduction_ratio: selected_count as f64 / original_count as f64,
            confidence_score: self.compute_confidence_score(&feature_scores),
            computation_time,
        };

        Ok(FeatureSelectionResult {
            selected_features,
            feature_scores,
            method: self.method.clone(),
            original_count,
            statistics,
        })
    }

    /// Correlation-based feature selection
    fn select_correlation(
        &self,
        features: &[Vec<f64>],
        targets: &[f64],
        threshold: f64,
        selected: &mut Vec<usize>,
        scores: &mut HashMap<usize, f64>,
    ) -> Result<()> {
        for (i, feature) in features.iter().enumerate() {
            let correlation = self.compute_correlation(feature, targets)?;
            scores.insert(i, correlation.abs());

            if correlation.abs() >= threshold {
                selected.push(i);
            }
        }
        Ok(())
    }

    /// Mutual information-based feature selection
    fn select_mutual_information(
        &self,
        features: &[Vec<f64>],
        targets: &[f64],
        k: usize,
        selected: &mut Vec<usize>,
        scores: &mut HashMap<usize, f64>,
    ) -> Result<()> {
        let mut feature_scores: Vec<(usize, f64)> = Vec::new();

        for (i, feature) in features.iter().enumerate() {
            let mi = self.compute_mutual_information(feature, targets)?;
            feature_scores.push((i, mi));
            scores.insert(i, mi);
        }

        // Sort by mutual information and select top k
        feature_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        selected.extend(feature_scores.iter().take(k).map(|(idx, _)| *idx));

        Ok(())
    }

    /// Chi-square feature selection
    fn select_chi_square(
        &self,
        features: &[Vec<f64>],
        targets: &[f64],
        k: usize,
        selected: &mut Vec<usize>,
        scores: &mut HashMap<usize, f64>,
    ) -> Result<()> {
        let mut feature_scores: Vec<(usize, f64)> = Vec::new();

        for (i, feature) in features.iter().enumerate() {
            let chi2 = self.compute_chi_square(feature, targets)?;
            feature_scores.push((i, chi2));
            scores.insert(i, chi2);
        }

        // Sort by chi-square score and select top k
        feature_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        selected.extend(feature_scores.iter().take(k).map(|(idx, _)| *idx));

        Ok(())
    }

    /// Recursive feature elimination
    fn select_recursive_elimination(
        &self,
        features: &[Vec<f64>],
        targets: &[f64],
        estimator: &dyn Estimator,
        step: usize,
        selected: &mut Vec<usize>,
        scores: &mut HashMap<usize, f64>,
    ) -> Result<()> {
        let mut remaining_features: Vec<usize> = (0..features.len()).collect();
        let target_features = self.target_features.unwrap_or(features.len() / 2);

        while remaining_features.len() > target_features {
            // Fit estimator on remaining features
            let _score = estimator.fit_score(features, targets)?;

            // Compute feature importances (simplified)
            let mut feature_importances: Vec<(usize, f64)> = remaining_features
                .iter()
                .map(|&idx| {
                    (
                        idx,
                        self.compute_feature_importance(&features[idx], targets),
                    )
                })
                .collect();

            // Sort by importance (ascending - least important first)
            feature_importances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Remove least important features
            let to_remove = std::cmp::min(
                step,
                remaining_features.len().saturating_sub(target_features),
            );
            for (idx, _) in feature_importances.iter().take(to_remove) {
                remaining_features.retain(|&x| x != *idx);
                scores.insert(*idx, 0.0); // Mark as eliminated
            }
        }

        selected.extend(remaining_features);
        for &idx in selected.iter() {
            scores.insert(idx, 1.0); // Mark as selected
        }

        Ok(())
    }

    /// LASSO-based feature selection
    fn select_lasso(
        &self,
        features: &[Vec<f64>],
        targets: &[f64],
        alpha: f64,
        selected: &mut Vec<usize>,
        scores: &mut HashMap<usize, f64>,
    ) -> Result<()> {
        // Simplified LASSO implementation
        // In practice, this would use a proper LASSO solver
        for (i, feature) in features.iter().enumerate() {
            let coefficient = self.compute_lasso_coefficient(feature, targets, alpha)?;
            scores.insert(i, coefficient.abs());

            if coefficient.abs() > 1e-6 {
                // Non-zero coefficient
                selected.push(i);
            }
        }
        Ok(())
    }

    /// Tree-based feature selection
    fn select_tree_based(
        &self,
        features: &[Vec<f64>],
        targets: &[f64],
        max_features: Option<usize>,
        selected: &mut Vec<usize>,
        scores: &mut HashMap<usize, f64>,
    ) -> Result<()> {
        // Simplified tree-based feature importance
        for (i, feature) in features.iter().enumerate() {
            let importance = self.compute_feature_importance(feature, targets);
            scores.insert(i, importance);
        }

        // Sort by importance and select top features
        let mut feature_importances: Vec<(usize, f64)> =
            scores.iter().map(|(&idx, &score)| (idx, score)).collect();
        feature_importances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let k = max_features.unwrap_or(features.len());
        selected.extend(feature_importances.iter().take(k).map(|(idx, _)| *idx));

        Ok(())
    }

    /// Compute Pearson correlation coefficient
    fn compute_correlation(&self, feature: &[f64], targets: &[f64]) -> Result<f64> {
        if feature.len() != targets.len() {
            return Err(NNError::InvalidInput {
                message: "Feature and target lengths must match".to_string(),
            });
        }

        let n = feature.len() as f64;
        let mean_x = feature.iter().sum::<f64>() / n;
        let mean_y = targets.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_yy = 0.0;

        for (&x, &y) in feature.iter().zip(targets.iter()) {
            let dx = x - mean_x;
            let dy = y - mean_y;
            numerator += dx * dy;
            sum_xx += dx * dx;
            sum_yy += dy * dy;
        }

        if sum_xx == 0.0 || sum_yy == 0.0 {
            return Ok(0.0);
        }

        Ok(numerator / (sum_xx.sqrt() * sum_yy.sqrt()))
    }

    /// Compute mutual information (simplified)
    fn compute_mutual_information(&self, feature: &[f64], targets: &[f64]) -> Result<f64> {
        // Simplified mutual information computation
        // In practice, this would use proper discretization and entropy calculation
        self.compute_correlation(feature, targets)
            .map(|corr| corr.abs())
    }

    /// Compute chi-square statistic (simplified)
    fn compute_chi_square(&self, feature: &[f64], targets: &[f64]) -> Result<f64> {
        // Simplified chi-square for continuous features
        // In practice, this would discretize features first
        self.compute_correlation(feature, targets)
            .map(|corr| corr.abs() * 100.0)
    }

    /// Compute feature importance (simplified)
    fn compute_feature_importance(&self, feature: &[f64], targets: &[f64]) -> f64 {
        self.compute_correlation(feature, targets)
            .map(|corr| corr.abs())
            .unwrap_or(0.0)
    }

    /// Compute LASSO coefficient (simplified)
    fn compute_lasso_coefficient(
        &self,
        feature: &[f64],
        targets: &[f64],
        alpha: f64,
    ) -> Result<f64> {
        // Simplified LASSO coefficient computation
        let correlation = self.compute_correlation(feature, targets)?;
        if correlation.abs() > alpha {
            Ok(correlation)
        } else {
            Ok(0.0)
        }
    }

    /// Compute confidence score for selection
    fn compute_confidence_score(&self, feature_scores: &HashMap<usize, f64>) -> f64 {
        if feature_scores.is_empty() {
            return 0.0;
        }

        let scores: Vec<f64> = feature_scores.values().cloned().collect();
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance =
            scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;

        if variance == 0.0 {
            1.0 // Perfect confidence if no variance
        } else {
            // Higher confidence when scores are well-separated
            1.0 / (1.0 + variance.sqrt())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_selection() {
        let features = vec![
            vec![1.0, 2.0, 3.0, 4.0], // Correlated with targets
            vec![1.0, 1.0, 1.0, 1.0], // Uncorrelated
        ];
        let targets = vec![1.0, 2.0, 3.0, 4.0];

        let selector = FeatureSelector::new(SelectionMethod::Correlation { threshold: 0.9 });

        let result = selector.select(&features, &targets).unwrap();

        assert_eq!(result.selected_features.len(), 1);
        assert_eq!(result.selected_features[0], 0);
        assert!(result.statistics.reduction_ratio > 0.0);
    }

    #[test]
    fn test_mutual_information_selection() {
        let features = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let selector = FeatureSelector::new(SelectionMethod::MutualInformation { k: 2 });

        let result = selector.select(&features, &targets).unwrap();

        assert_eq!(result.selected_features.len(), 2);
        assert!(result.feature_scores.len() == 3usize);
    }

    #[test]
    fn test_feature_selector_creation() {
        let selector = FeatureSelector::new(SelectionMethod::Correlation { threshold: 0.5 })
            .with_target_features(10);

        match selector.method {
            SelectionMethod::Correlation { threshold } => assert_eq!(threshold, 0.5),
            _ => panic!("Wrong method"),
        }

        assert_eq!(selector.target_features, Some(10));
    }
}
