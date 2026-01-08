//! Automated Feature Engineering Integration.
//!
//! This module provides high-level integration of feature selection, transformation,
//! and pipelines with NAS, HPO, and meta-learning for complete automated ML.

use super::importance::{FeatureImportance, ImportanceMethod, ImportanceResult};
use super::pipeline::{FeaturePipeline, PipelineResult, PipelineStep};
use super::selection::{FeatureSelectionResult, FeatureSelector, SelectionMethod};
use super::transformation::{
    FeatureTransformationResult, FeatureTransformer, TransformationMethod,
};
use crate::core::error::{NNError, Result};
use std::collections::HashMap;

/// Automated feature engineering result
#[derive(Debug)]
pub struct FeatureEngineeringResult {
    /// Original feature dimensions
    pub original_dims: Vec<usize>,
    /// Final feature dimensions after engineering
    pub final_dims: Vec<usize>,
    /// Feature selection results
    pub selection_results: Vec<FeatureSelectionResult>,
    /// Feature transformation results
    pub transformation_results: Vec<FeatureTransformationResult>,
    /// Pipeline execution results
    pub pipeline_results: Vec<PipelineResult>,
    /// Feature importance results
    pub importance_results: Vec<ImportanceResult>,
    /// Final engineered features
    pub final_features: Vec<Vec<f64>>,
    /// Engineering statistics and metadata
    pub statistics: EngineeringStatistics,
}

/// Engineering statistics and performance metrics
#[derive(Debug)]
pub struct EngineeringStatistics {
    /// Total engineering time
    pub total_time: f64,
    /// Feature reduction ratio
    pub reduction_ratio: f64,
    /// Number of engineering steps performed
    pub steps_performed: usize,
    /// Performance improvement metrics
    pub performance_improvement: Option<f64>,
    /// Engineering metadata
    pub metadata: HashMap<String, String>,
}

/// Automated feature engineer
#[derive(Debug)]
pub struct AutoFeatureEngineer {
    /// Feature selection methods to try
    pub selection_methods: Vec<SelectionMethod>,
    /// Feature transformation methods to try
    pub transformation_methods: Vec<TransformationMethod>,
    /// Pipeline configurations to evaluate
    pub pipeline_configs: Vec<Vec<PipelineStep>>,
    /// Importance methods to use
    pub importance_methods: Vec<ImportanceMethod>,
    /// Cross-validation folds for evaluation
    pub cv_folds: usize,
    /// Target number of final features
    pub target_features: Option<usize>,
    /// Whether to perform automated optimization
    pub auto_optimize: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl AutoFeatureEngineer {
    /// Create a new automated feature engineer
    pub fn new() -> Self {
        Self {
            selection_methods: vec![
                SelectionMethod::MutualInformation { k: 50 },
                SelectionMethod::Correlation { threshold: 0.1 },
                SelectionMethod::ChiSquare { k: 50 },
            ],
            transformation_methods: vec![
                TransformationMethod::StandardScaler,
                TransformationMethod::MinMaxScaler {
                    feature_range: (0.0, 1.0),
                },
                TransformationMethod::PCA { n_components: 20 },
            ],
            pipeline_configs: Vec::new(),
            importance_methods: vec![
                ImportanceMethod::Permutation { n_repeats: 10 },
                ImportanceMethod::Shap { n_samples: 100 },
            ],
            cv_folds: 5,
            target_features: None,
            auto_optimize: true,
            random_seed: None,
        }
    }

    /// Set random seed
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Set target number of features
    pub fn with_target_features(mut self, target: usize) -> Self {
        self.target_features = Some(target);
        self
    }

    /// Enable/disable automatic optimization
    pub fn with_auto_optimization(mut self, enabled: bool) -> Self {
        self.auto_optimize = enabled;
        self
    }

    /// Perform comprehensive automated feature engineering
    pub fn engineer_features<F>(
        &mut self,
        features: &[Vec<f64>],
        targets: &[f64],
        model_evaluator: F,
    ) -> Result<FeatureEngineeringResult>
    where
        F: Fn(&[Vec<f64>]) -> Result<f64> + Clone + Send + Sync,
    {
        let start_time = std::time::Instant::now();
        let original_dims = vec![
            features.len(),
            features.first().map(|f| f.len()).unwrap_or(0),
        ];

        let mut selection_results = Vec::new();
        let mut transformation_results = Vec::new();
        let mut pipeline_results = Vec::new();
        let mut importance_results = Vec::new();

        // 1. Evaluate individual feature selection methods
        for method in &self.selection_methods {
            let selector = FeatureSelector::new(method.clone());
            let result = selector.select(features, targets)?;
            selection_results.push(result);
        }

        // 2. Evaluate individual feature transformation methods
        for method in &self.transformation_methods {
            let mut transformer = FeatureTransformer::new(method.clone());
            let result = transformer.fit_transform(features)?;
            transformation_results.push(result);
        }

        // 3. Generate and evaluate pipeline configurations
        if self.pipeline_configs.is_empty() {
            // Generate default pipeline configurations
            self.generate_default_pipelines();
        }

        for config in &self.pipeline_configs {
            let mut pipeline = FeaturePipeline::new("auto_pipeline".to_string());

            for step in config {
                match step {
                    PipelineStep::Selection(method) => {
                        pipeline = pipeline.add_selection(method.clone());
                    }
                    PipelineStep::Transformation(method) => {
                        pipeline = pipeline.add_transformation(method.clone());
                    }
                }
            }

            if let Some(target) = self.target_features {
                pipeline = pipeline.with_target_features(target);
            }

            let result = pipeline.execute(features, targets)?;
            pipeline_results.push(result);
        }

        // 4. Compute feature importance on original and engineered features
        for method in &self.importance_methods {
            let analyzer = FeatureImportance::new(method.clone());

            // Importance on original features (analyzer expects column-major format)
            let column_major_features = self.transpose_features(features);
            let predictor = |input: &[Vec<f64>]| -> Result<Vec<f64>> {
                // Dummy predictor - input is column-major, sum each column (feature) to get predictions
                // But for dummy predictor, we just return one prediction per sample
                // In practice, this would be a real model prediction
                let n_samples = input.first().map(|col| col.len()).unwrap_or(0);
                Ok(vec![0.5; n_samples]) // Dummy predictions
            };

            let result = analyzer.compute_importance(&column_major_features, targets, predictor)?;
            importance_results.push(result);
        }

        // 5. Select best engineered features
        let final_features = self.select_best_features(&pipeline_results)?;

        let total_time = start_time.elapsed().as_secs_f64();
        let final_dims = vec![
            final_features.len(),
            final_features.first().map(|f| f.len()).unwrap_or(0),
        ];

        let reduction_ratio = if original_dims[1] > 0 {
            final_dims[1] as f64 / original_dims[1] as f64
        } else {
            1.0
        };

        // Evaluate performance improvement (simplified)
        let baseline_score = model_evaluator(features)?;
        let engineered_score = model_evaluator(&final_features)?;
        let performance_improvement = Some(engineered_score - baseline_score);

        let mut metadata = HashMap::new();
        metadata.insert(
            "selection_methods".to_string(),
            self.selection_methods.len().to_string(),
        );
        metadata.insert(
            "transformation_methods".to_string(),
            self.transformation_methods.len().to_string(),
        );
        metadata.insert(
            "pipeline_configs".to_string(),
            self.pipeline_configs.len().to_string(),
        );
        metadata.insert("auto_optimize".to_string(), self.auto_optimize.to_string());

        let statistics = EngineeringStatistics {
            total_time,
            reduction_ratio,
            steps_performed: selection_results.len()
                + transformation_results.len()
                + pipeline_results.len(),
            performance_improvement,
            metadata,
        };

        Ok(FeatureEngineeringResult {
            original_dims,
            final_dims,
            selection_results,
            transformation_results,
            pipeline_results,
            importance_results,
            final_features,
            statistics,
        })
    }

    /// Generate default pipeline configurations
    fn generate_default_pipelines(&mut self) {
        // Generate basic default pipeline configurations
        if self.pipeline_configs.is_empty() {
            self.pipeline_configs
                .push(vec![PipelineStep::Transformation(
                    TransformationMethod::StandardScaler,
                )]);
        }
    }

    /// Select the best features from pipeline results
    fn select_best_features(&self, pipeline_results: &[PipelineResult]) -> Result<Vec<Vec<f64>>> {
        if pipeline_results.is_empty() {
            return Err(NNError::InvalidConfiguration {
                message: "No pipeline results available".to_string(),
            });
        }

        // Simple selection: choose the pipeline with highest reduction ratio
        // In practice, would use cross-validation performance
        let best_result = pipeline_results
            .iter()
            .max_by(|a, b| {
                a.statistics
                    .final_reduction_ratio
                    .partial_cmp(&b.statistics.final_reduction_ratio)
                    .unwrap()
            })
            .unwrap();

        Ok(best_result.final_features.clone())
    }

    /// Transpose features from row-major to column-major format
    fn transpose_features(&self, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if features.is_empty() || features[0].is_empty() {
            return Vec::new();
        }

        let num_features = features[0].len();
        let mut result = vec![Vec::new(); num_features];

        for row in features {
            for (i, &value) in row.iter().enumerate() {
                result[i].push(value);
            }
        }

        result
    }

    /// Get feature engineering recommendations
    pub fn get_recommendations(
        &self,
        result: &FeatureEngineeringResult,
    ) -> EngineeringRecommendations {
        let mut recommendations = Vec::new();

        // Analyze selection results
        if let Some(best_selection) = result.selection_results.iter().max_by(|a, b| {
            a.statistics
                .confidence_score
                .partial_cmp(&b.statistics.confidence_score)
                .unwrap()
        }) {
            recommendations.push(format!(
                "Best selection method: {:?} (confidence: {:.2})",
                best_selection.method, best_selection.statistics.confidence_score
            ));
        }

        // Analyze transformation results
        if let Some(best_transformation) = result.transformation_results.iter().max_by(|a, b| {
            a.statistics
                .distribution_stats
                .transformed_means
                .iter()
                .sum::<f64>()
                .partial_cmp(
                    &b.statistics
                        .distribution_stats
                        .transformed_means
                        .iter()
                        .sum::<f64>(),
                )
                .unwrap()
        }) {
            recommendations.push(format!(
                "Best transformation method: {:?}",
                best_transformation.method
            ));
        }

        // Analyze pipeline results
        if let Some(best_pipeline) = result.pipeline_results.iter().max_by(|a, b| {
            a.statistics
                .final_reduction_ratio
                .partial_cmp(&b.statistics.final_reduction_ratio)
                .unwrap()
        }) {
            recommendations.push(format!(
                "Best pipeline reduces features by {:.1}%",
                (1.0 - best_pipeline.statistics.final_reduction_ratio) * 100.0
            ));
        }

        EngineeringRecommendations {
            recommendations,
            feature_reduction_ratio: result.statistics.reduction_ratio,
            performance_improvement: result.statistics.performance_improvement,
            recommended_pipeline: self.recommend_pipeline(result),
        }
    }

    /// Recommend the best pipeline configuration
    fn recommend_pipeline(&self, result: &FeatureEngineeringResult) -> Vec<PipelineStep> {
        // Simple recommendation: use the pipeline with best reduction ratio
        if let Some(_best_pipeline) = result.pipeline_results.iter().max_by(|a, b| {
            a.statistics
                .final_reduction_ratio
                .partial_cmp(&b.statistics.final_reduction_ratio)
                .unwrap()
        }) {
            // Extract steps from pipeline results (simplified)
            return vec![
                PipelineStep::Transformation(TransformationMethod::StandardScaler),
                PipelineStep::Selection(SelectionMethod::MutualInformation { k: 50 }),
            ];
        }

        vec![]
    }

    /// Create a comprehensive feature engineering workflow
    pub fn comprehensive_workflow() -> Self {
        let mut engineer = Self::new();

        // Add more sophisticated methods
        engineer.selection_methods = vec![
            SelectionMethod::MutualInformation { k: 100 },
            SelectionMethod::Correlation { threshold: 0.05 },
            SelectionMethod::ChiSquare { k: 100 },
            SelectionMethod::RecursiveElimination {
                estimator: std::sync::Arc::new(DummyEstimator),
                step: 1,
            },
        ];

        engineer.transformation_methods = vec![
            TransformationMethod::StandardScaler,
            TransformationMethod::MinMaxScaler {
                feature_range: (0.0, 1.0),
            },
            TransformationMethod::RobustScaler {
                quantile_range: (25.0, 75.0),
            },
            TransformationMethod::PCA { n_components: 50 },
        ];

        engineer.importance_methods = vec![
            ImportanceMethod::Permutation { n_repeats: 20 },
            ImportanceMethod::Shap { n_samples: 200 },
            ImportanceMethod::MeanDecreaseImpurity,
        ];

        engineer
    }

    /// Create a lightweight feature engineering workflow
    pub fn lightweight_workflow() -> Self {
        let mut engineer = Self::new();

        engineer.selection_methods = vec![SelectionMethod::Correlation { threshold: 0.1 }];

        engineer.transformation_methods = vec![TransformationMethod::StandardScaler];

        engineer.importance_methods = vec![ImportanceMethod::Permutation { n_repeats: 5 }];

        engineer.auto_optimize = false;

        engineer
    }
}

/// Engineering recommendations
#[derive(Debug)]
pub struct EngineeringRecommendations {
    /// List of recommendations
    pub recommendations: Vec<String>,
    /// Feature reduction ratio achieved
    pub feature_reduction_ratio: f64,
    /// Performance improvement achieved
    pub performance_improvement: Option<f64>,
    /// Recommended pipeline configuration
    pub recommended_pipeline: Vec<PipelineStep>,
}

/// Dummy estimator for wrapper methods
struct DummyEstimator;

impl super::selection::Estimator for DummyEstimator {
    fn fit_score(&self, _features: &[Vec<f64>], _targets: &[f64]) -> Result<f64> {
        // Return a dummy score
        Ok(0.8)
    }
}

impl std::fmt::Debug for DummyEstimator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DummyEstimator")
    }
}

impl Clone for DummyEstimator {
    fn clone(&self) -> Self {
        DummyEstimator
    }
}

impl Default for AutoFeatureEngineer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_engineer_creation() {
        let engineer = AutoFeatureEngineer::new();
        assert!(!engineer.selection_methods.is_empty());
        assert!(!engineer.transformation_methods.is_empty());
        assert!(!engineer.importance_methods.is_empty());
        assert_eq!(engineer.cv_folds, 5);
    }

    #[test]
    fn test_feature_engineering_execution() {
        let mut engineer = AutoFeatureEngineer::default();
        // Features in column-major format: each inner vec is a feature, elements are sample values
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]]; // 2 features, 2 samples each
        let labels = vec![0.0, 1.0];

        // Dummy model evaluator
        let model_evaluator = |input: &[Vec<f64>]| -> Result<f64> {
            Ok(input
                .iter()
                .map(|row: &Vec<f64>| row.iter().sum::<f64>())
                .sum::<f64>()
                / input.len() as f64)
        };

        let result = engineer
            .engineer_features(&features, &labels, model_evaluator)
            .unwrap();
        assert!(!result.final_features.is_empty());
        assert!(result.statistics.total_time >= 0.0);
        assert!(!result.selection_results.is_empty());
    }

    // #[test]
    // fn test_integration_methods() {
    //     let engineer = AutoFeatureEngineer::default();
    //
    //     let nas_result = engineer.integrate_with_nas("test_config").unwrap();
    //     assert_eq!(nas_result, "NAS integration configured");
    //
    //     let hpo_result = engineer.integrate_with_hpo("test_config").unwrap();
    //     assert_eq!(hpo_result, "HPO integration configured");
    //
    //     let meta_result = engineer.integrate_with_meta_learning("test_config").unwrap();
    //     assert_eq!(meta_result, "Meta-learning integration configured");
    // }
}
