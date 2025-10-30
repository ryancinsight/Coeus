//! Feature Pipeline.
//!
//! This module implements automated feature engineering pipelines that
//! combine multiple feature selection and transformation steps.

use super::selection::{FeatureSelectionResult, FeatureSelector, SelectionMethod};
use super::transformation::{
    FeatureTransformationResult, FeatureTransformer, TransformationMethod,
};
use crate::error::Result;

/// Pipeline step
#[derive(Debug, Clone)]
pub enum PipelineStep {
    /// Feature selection step
    Selection(SelectionMethod),
    /// Feature transformation step
    Transformation(TransformationMethod),
}

/// Pipeline result
#[derive(Debug)]
pub struct PipelineResult {
    /// Final transformed features
    pub final_features: Vec<Vec<f64>>,
    /// Intermediate results from each step
    pub step_results: Vec<StepResult>,
    /// Pipeline execution statistics
    pub statistics: PipelineStatistics,
    /// Original feature dimensions
    pub original_dims: Vec<usize>,
    /// Final feature dimensions
    pub final_dims: Vec<usize>,
}

/// Result from a single pipeline step
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum StepResult {
    /// Feature selection result
    Selection(FeatureSelectionResult),
    /// Feature transformation result
    Transformation(FeatureTransformationResult),
}

/// Pipeline execution statistics
#[derive(Debug)]
pub struct PipelineStatistics {
    /// Total execution time
    pub total_time: f64,
    /// Number of steps executed
    pub steps_executed: usize,
    /// Final feature reduction ratio
    pub final_reduction_ratio: f64,
    /// Step-by-step timing
    pub step_times: Vec<f64>,
}

/// Feature engineering pipeline
#[derive(Debug)]
pub struct FeaturePipeline {
    /// Pipeline steps in execution order
    pub steps: Vec<PipelineStep>,
    /// Target number of final features (optional)
    pub target_features: Option<usize>,
    /// Whether to perform cross-validation for step selection
    pub cross_validate: bool,
    /// Number of CV folds (if cross_validate is true)
    pub cv_folds: usize,
    /// Pipeline name
    pub name: String,
}

impl FeaturePipeline {
    /// Create a new feature pipeline
    pub fn new(name: String) -> Self {
        Self {
            steps: Vec::new(),
            target_features: None,
            cross_validate: false,
            cv_folds: 5,
            name,
        }
    }

    /// Add a selection step to the pipeline
    pub fn add_selection(mut self, method: SelectionMethod) -> Self {
        self.steps.push(PipelineStep::Selection(method));
        self
    }

    /// Add a transformation step to the pipeline
    pub fn add_transformation(mut self, method: TransformationMethod) -> Self {
        self.steps.push(PipelineStep::Transformation(method));
        self
    }

    /// Set target number of final features
    pub fn with_target_features(mut self, target: usize) -> Self {
        self.target_features = Some(target);
        self
    }

    /// Enable cross-validation for step selection
    pub fn with_cross_validation(mut self, folds: usize) -> Self {
        self.cross_validate = true;
        self.cv_folds = folds;
        self
    }

    /// Execute the pipeline
    pub fn execute(&self, features: &[Vec<f64>], targets: &[f64]) -> Result<PipelineResult> {
        let start_time = std::time::Instant::now();
        let original_dims = vec![
            features.len(),
            features.first().map(|f| f.len()).unwrap_or(0),
        ];

        let mut current_features = features.to_vec();
        let mut step_results = Vec::new();
        let mut step_times = Vec::new();

        for step in &self.steps {
            let step_start = std::time::Instant::now();

            match step {
                PipelineStep::Selection(method) => {
                    let mut selector = FeatureSelector::new(method.clone());
                    if let Some(target) = self.target_features {
                        selector = selector.with_target_features(target);
                    }

                    // Transpose to column-major format for selection
                    let column_major_features = self.transpose(&current_features);
                    let result = selector.select(&column_major_features, targets)?;
                    current_features =
                        self.apply_selection(&current_features, &result.selected_features);

                    step_results.push(StepResult::Selection(result));
                }
                PipelineStep::Transformation(method) => {
                    let mut transformer = FeatureTransformer::new(method.clone());
                    let result = transformer.fit_transform(&current_features)?;
                    current_features = result.transformed_features.clone();

                    step_results.push(StepResult::Transformation(result));
                }
            }

            let step_time = step_start.elapsed().as_secs_f64();
            step_times.push(step_time);
        }

        let total_time = start_time.elapsed().as_secs_f64();
        let final_dims = vec![
            current_features.len(),
            current_features.first().map(|f| f.len()).unwrap_or(0),
        ];

        let final_reduction_ratio = if original_dims[1] > 0 {
            final_dims[1] as f64 / original_dims[1] as f64
        } else {
            1.0
        };

        let statistics = PipelineStatistics {
            total_time,
            steps_executed: self.steps.len(),
            final_reduction_ratio,
            step_times,
        };

        Ok(PipelineResult {
            final_features: current_features,
            step_results,
            statistics,
            original_dims,
            final_dims,
        })
    }

    /// Apply feature selection to feature matrix
    fn apply_selection(&self, features: &[Vec<f64>], selected_indices: &[usize]) -> Vec<Vec<f64>> {
        features
            .iter()
            .map(|row| selected_indices.iter().map(|&idx| row[idx]).collect())
            .collect()
    }

    /// Transpose feature matrix from row-major to column-major format
    fn transpose(&self, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
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
}

impl Default for FeaturePipeline {
    fn default() -> Self {
        Self::new("default_pipeline".to_string())
    }
}

/// Predefined pipeline templates
pub struct PipelineTemplates;

impl PipelineTemplates {
    /// Basic preprocessing pipeline
    pub fn basic_preprocessing() -> FeaturePipeline {
        FeaturePipeline::new("basic_preprocessing".to_string())
            .add_transformation(TransformationMethod::StandardScaler)
    }

    /// Feature selection pipeline
    pub fn feature_selection(k_features: usize) -> FeaturePipeline {
        FeaturePipeline::new("feature_selection".to_string())
            .add_selection(SelectionMethod::MutualInformation { k: k_features })
    }

    /// Comprehensive feature engineering pipeline
    pub fn comprehensive(target_features: usize) -> FeaturePipeline {
        FeaturePipeline::new("comprehensive".to_string())
            .add_transformation(TransformationMethod::StandardScaler)
            .add_selection(SelectionMethod::MutualInformation {
                k: target_features * 2,
            })
            .add_transformation(TransformationMethod::PCA {
                n_components: target_features,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_pipeline() {
        let pipeline = FeaturePipeline::new("test_pipeline".to_string())
            .add_transformation(TransformationMethod::StandardScaler);

        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let targets = vec![1.0, 2.0, 3.0];

        let result = pipeline.execute(&features, &targets).unwrap();

        assert_eq!(result.step_results.len(), 1);
        assert_eq!(result.final_features.len(), 3);
        assert_eq!(result.final_dims[1], 2); // Same number of features
    }

    #[test]
    fn test_selection_pipeline() {
        let pipeline = FeaturePipeline::new("selection_pipeline".to_string())
            .add_selection(SelectionMethod::Correlation { threshold: 0.9 });

        let features = vec![
            vec![1.0, 1.0], // Correlated with targets
            vec![1.0, 1.0], // Constant (should be removed)
        ];
        let targets = vec![1.0, 2.0];

        let result = pipeline.execute(&features, &targets).unwrap();

        assert_eq!(result.step_results.len(), 1);
        assert_eq!(result.final_features.len(), 2);
    }

    #[test]
    fn test_pipeline_templates() {
        let basic = PipelineTemplates::basic_preprocessing();
        assert_eq!(basic.steps.len(), 1);

        let comprehensive = PipelineTemplates::comprehensive(5);
        assert_eq!(comprehensive.steps.len(), 3);
    }
}
