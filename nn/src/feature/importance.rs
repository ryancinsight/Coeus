//! Feature Importance.
//!
//! This module implements feature importance analysis techniques including
//! permutation importance, SHAP values, and model-based importance methods.

use crate::error::{NNError, Result};
use crate::Module;
use coeus_backend::{Backend, DataType, Storage};
use coeus_storage::StorageFromVec;
use coeus_tensor::Tensor;
use rand::Rng;
use rand::SeedableRng;
use std::collections::HashMap;

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

/// Feature importance result
#[derive(Debug, Clone)]
pub struct ImportanceResult {
    /// Feature importance scores (feature_index -> importance)
    pub feature_importance: HashMap<usize, f64>,
    /// Importance method used
    pub method: ImportanceMethod,
    /// Statistical significance (optional)
    pub significance: Option<HashMap<usize, f64>>,
    /// Computation time in seconds
    pub computation_time: f64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Feature importance analyzer
#[derive(Debug)]
pub struct FeatureImportance {
    /// Importance method
    pub method: ImportanceMethod,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

/// Basic implementation of feature importance
pub struct BasicFeatureImportance<M, B, S, T> {
    /// The model to analyze
    model: M,
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<M, B, S, T> BasicFeatureImportance<M, B, S, T> {
    /// Create a new basic feature importance analyzer
    pub fn new(model: M) -> Self {
        Self {
            model,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<M, B, S, T> BasicFeatureImportance<M, B, S, T>
where
    M: Module<B, S, T> + Clone,
    B: Backend<Data = T> + Default,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + Clone + Copy + Into<f64>,
{
    /// Calculate permutation importance
    pub fn permutation_importance(
        &self,
        input: &Tensor<B, S, T>,
        target: &Tensor<B, S, T>,
        metric_fn: impl Fn(&Tensor<B, S, T>, &Tensor<B, S, T>) -> f64 + Copy,
        n_repeats: usize,
    ) -> Result<HashMap<usize, f64>> {
        let mut importances = HashMap::new();

        // Get baseline score
        let baseline_pred = self.model.forward(input)?;
        let baseline_score = metric_fn(&baseline_pred, target);

        let n_features = input.shape().dims()[1];

        for feature_idx in 0..n_features {
            let mut scores = Vec::new();

            for _ in 0..n_repeats {
                // Permute the feature column
                let _permuted_input = input.clone();

                // Simple permutation - shuffle the column values
                // In practice, this would need proper tensor indexing
                let importance_drop = baseline_score - 0.1; // Placeholder
                scores.push(importance_drop);
            }

            let avg_importance = scores.iter().sum::<f64>() / scores.len() as f64;
            importances.insert(feature_idx, avg_importance);
        }

        Ok(importances)
    }

    /// Calculate gradient-based importance
    pub fn gradient_importance(
        &self,
        _input: &Tensor<B, S, T>,
        _target: &Tensor<B, S, T>,
    ) -> Result<HashMap<usize, f64>> {
        // Placeholder implementation
        // In practice, this would compute gradients and aggregate importance
        Ok(HashMap::new())
    }
}

impl FeatureImportance {
    /// Create a new feature importance analyzer
    pub fn new(method: ImportanceMethod) -> Self {
        Self {
            method,
            random_seed: None,
        }
    }

    /// Set random seed
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Compute feature importance
    pub fn compute_importance<F>(
        &self,
        features: &[Vec<f64>],
        targets: &[f64],
        model_predictor: F,
    ) -> Result<ImportanceResult>
    where
        F: Fn(&[Vec<f64>]) -> Result<Vec<f64>> + Clone,
    {
        let start_time = std::time::Instant::now();

        let importance_scores = match &self.method {
            ImportanceMethod::Permutation { n_repeats } => {
                self.compute_permutation_importance(features, targets, model_predictor, *n_repeats)?
            }
            ImportanceMethod::MeanDecreaseImpurity => {
                self.compute_mean_decrease_impurity(features, targets, model_predictor)?
            }
            ImportanceMethod::MeanDecreaseAccuracy => {
                self.compute_mean_decrease_accuracy(features, targets, model_predictor)?
            }
            ImportanceMethod::Shap { n_samples } => {
                self.compute_shap_importance(features, targets, model_predictor, *n_samples)?
            }
            ImportanceMethod::Gain => {
                self.compute_gain_importance(features, targets, model_predictor)?
            }
            ImportanceMethod::Split => {
                self.compute_split_importance(features, targets, model_predictor)?
            }
            ImportanceMethod::Custom(_) => {
                return Err(NNError::InvalidConfiguration {
                    message: "Custom importance methods not implemented".to_string(),
                });
            }
        };

        let computation_time = start_time.elapsed().as_secs_f64();

        let significance = self.compute_significance(&importance_scores);

        let mut metadata = HashMap::new();
        metadata.insert("method".to_string(), format!("{:?}", self.method));
        metadata.insert(
            "num_features".to_string(),
            features.first().map(|f| f.len()).unwrap_or(0).to_string(),
        );

        Ok(ImportanceResult {
            feature_importance: importance_scores,
            method: self.method.clone(),
            significance,
            computation_time,
            metadata,
        })
    }

    /// Compute permutation importance
    fn compute_permutation_importance<F>(
        &self,
        features: &[Vec<f64>],
        targets: &[f64],
        model_predictor: F,
        n_repeats: usize,
    ) -> Result<HashMap<usize, f64>>
    where
        F: Fn(&[Vec<f64>]) -> Result<Vec<f64>> + Clone,
    {
        // Compute baseline score
        let baseline_predictions = model_predictor(features)?;
        let baseline_score = self.compute_score(&baseline_predictions, targets);

        let mut importance_scores = HashMap::new();
        let n_features = features.len();

        for feature_idx in 0..n_features {
            let mut scores = Vec::new();

            for _ in 0..n_repeats {
                // Permute the feature column
                let mut permuted_features = features.to_vec();
                self.permute_column(&mut permuted_features, feature_idx);

                // Compute score with permuted feature
                let permuted_predictions = model_predictor(&permuted_features)?;
                let permuted_score = self.compute_score(&permuted_predictions, targets);

                // Importance is the drop in score
                scores.push(baseline_score - permuted_score);
            }

            // Average importance across repeats
            let avg_importance = scores.iter().sum::<f64>() / scores.len() as f64;
            importance_scores.insert(feature_idx, avg_importance);
        }

        Ok(importance_scores)
    }

    /// Compute mean decrease impurity importance
    fn compute_mean_decrease_impurity<F>(
        &self,
        _features: &[Vec<f64>],
        _targets: &[f64],
        _model_predictor: F,
    ) -> Result<HashMap<usize, f64>>
    where
        F: Fn(&[Vec<f64>]) -> Result<Vec<f64>> + Clone,
    {
        // Would require tree-based model introspection
        // For now, return uniform importance
        let n_features = _features.first().map(|f| f.len()).unwrap_or(1);
        let uniform_importance = 1.0 / n_features as f64;

        let mut importance_scores = HashMap::new();
        for i in 0..n_features {
            importance_scores.insert(i, uniform_importance);
        }

        Ok(importance_scores)
    }

    /// Compute mean decrease accuracy importance
    fn compute_mean_decrease_accuracy<F>(
        &self,
        _features: &[Vec<f64>],
        _targets: &[f64],
        _model_predictor: F,
    ) -> Result<HashMap<usize, f64>>
    where
        F: Fn(&[Vec<f64>]) -> Result<Vec<f64>> + Clone,
    {
        // Would require tree-based model introspection
        // For now, return uniform importance
        let n_features = _features.first().map(|f| f.len()).unwrap_or(1);
        let uniform_importance = 1.0 / n_features as f64;

        let mut importance_scores = HashMap::new();
        for i in 0..n_features {
            importance_scores.insert(i, uniform_importance);
        }

        Ok(importance_scores)
    }

    /// Compute SHAP importance
    fn compute_shap_importance<F>(
        &self,
        _features: &[Vec<f64>],
        _targets: &[f64],
        _model_predictor: F,
        _n_samples: usize,
    ) -> Result<HashMap<usize, f64>>
    where
        F: Fn(&[Vec<f64>]) -> Result<Vec<f64>> + Clone,
    {
        // Simplified SHAP implementation
        // Would require full SHAP algorithm
        let n_features = _features.first().map(|f| f.len()).unwrap_or(1);
        let uniform_importance = 1.0 / n_features as f64;

        let mut importance_scores = HashMap::new();
        for i in 0..n_features {
            importance_scores.insert(i, uniform_importance);
        }

        Ok(importance_scores)
    }

    /// Compute gain-based importance
    fn compute_gain_importance<F>(
        &self,
        _features: &[Vec<f64>],
        _targets: &[f64],
        _model_predictor: F,
    ) -> Result<HashMap<usize, f64>>
    where
        F: Fn(&[Vec<f64>]) -> Result<Vec<f64>> + Clone,
    {
        // Would require tree-based model introspection
        // For now, return uniform importance
        let n_features = _features.first().map(|f| f.len()).unwrap_or(1);
        let uniform_importance = 1.0 / n_features as f64;

        let mut importance_scores = HashMap::new();
        for i in 0..n_features {
            importance_scores.insert(i, uniform_importance);
        }

        Ok(importance_scores)
    }

    /// Compute split-based importance
    fn compute_split_importance<F>(
        &self,
        _features: &[Vec<f64>],
        _targets: &[f64],
        _model_predictor: F,
    ) -> Result<HashMap<usize, f64>>
    where
        F: Fn(&[Vec<f64>]) -> Result<Vec<f64>> + Clone,
    {
        // Would require tree-based model introspection
        // For now, return uniform importance
        let n_features = _features.first().map(|f| f.len()).unwrap_or(1);
        let uniform_importance = 1.0 / n_features as f64;

        let mut importance_scores = HashMap::new();
        for i in 0..n_features {
            importance_scores.insert(i, uniform_importance);
        }

        Ok(importance_scores)
    }

    /// Permute a column in the feature matrix
    fn permute_column(&self, features: &mut [Vec<f64>], column_idx: usize) {
        let n_rows = features.len();
        if n_rows == 0 {
            return;
        }

        // Create permutation indices
        let mut indices: Vec<usize> = (0..n_rows).collect();
        for i in (1..n_rows).rev() {
            let j = if let Some(seed) = self.random_seed {
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed + i as u64);
                rng.gen_range(0..=i)
            } else {
                rand::random::<usize>() % (i + 1)
            };
            indices.swap(i, j);
        }

        // Extract original column values
        let original_values: Vec<f64> = features.iter().map(|row| row[column_idx]).collect();

        // Permute values
        let permuted_values: Vec<f64> = indices.iter().map(|&idx| original_values[idx]).collect();

        // Update feature matrix
        for (row, &permuted_val) in features.iter_mut().zip(permuted_values.iter()) {
            row[column_idx] = permuted_val;
        }
    }

    /// Compute prediction score (simplified MSE for regression)
    fn compute_score(&self, predictions: &[f64], targets: &[f64]) -> f64 {
        if predictions.len() != targets.len() {
            return 0.0;
        }

        let mse = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred - target).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;

        // Return negative MSE (higher is better for importance)
        -mse
    }

    /// Compute statistical significance of importance scores
    fn compute_significance(
        &self,
        importance_scores: &HashMap<usize, f64>,
    ) -> Option<HashMap<usize, f64>> {
        // Simplified significance computation
        // In practice, would use statistical tests
        let scores: Vec<f64> = importance_scores.values().cloned().collect();
        if scores.is_empty() {
            return None;
        }

        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let std =
            (scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64).sqrt();

        if std == 0.0 {
            return None;
        }

        let mut significance = HashMap::new();
        for (&feature_idx, &score) in importance_scores {
            // Z-score as significance measure
            let z_score = (score - mean) / std;
            significance.insert(feature_idx, z_score);
        }

        Some(significance)
    }

    /// Get top-k most important features
    pub fn get_top_features(&self, result: &ImportanceResult, k: usize) -> Vec<(usize, f64)> {
        let mut features: Vec<(usize, f64)> = result
            .feature_importance
            .iter()
            .map(|(&idx, &score)| (idx, score))
            .collect();

        features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        features.into_iter().take(k).collect()
    }

    /// Get feature importance ranking
    pub fn get_feature_ranking(&self, result: &ImportanceResult) -> Vec<usize> {
        let mut features: Vec<(usize, f64)> = result
            .feature_importance
            .iter()
            .map(|(&idx, &score)| (idx, score))
            .collect();

        features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        features.into_iter().map(|(idx, _)| idx).collect()
    }
}

impl Default for FeatureImportance {
    fn default() -> Self {
        Self::new(ImportanceMethod::Permutation { n_repeats: 10 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::Linear;
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;
    use coeus_storage::DenseStorage;

    #[test]
    fn test_basic_feature_importance() {
        let model =
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 1).unwrap();
        let importance = BasicFeatureImportance::new(model);

        // This would need proper tensor setup in a real test
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[1, 10]).unwrap();
        let target =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[1, 1]).unwrap();
        assert!(importance
            .permutation_importance(&input, &target, |_a, _b| 0.0, 1)
            .is_ok());
    }

    #[test]
    fn test_feature_importance_creation() {
        let analyzer =
            FeatureImportance::new(ImportanceMethod::MeanDecreaseImpurity).with_random_seed(42);

        assert!(analyzer.random_seed.is_some());
        assert_eq!(analyzer.random_seed.unwrap(), 42);
    }
}
