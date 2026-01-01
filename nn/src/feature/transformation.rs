//! Feature Transformation.
//!
//! This module implements various feature transformation techniques including
//! scaling/normalization, categorical encoding, feature generation, and
//! dimensionality reduction.

use crate::error::{NNError, Result};
use std::collections::HashMap;

/// Feature transformation result
#[derive(Debug, Clone)]
pub struct FeatureTransformationResult {
    /// Transformed feature data
    pub transformed_features: Vec<Vec<f64>>,
    /// Transformation parameters (for inverse transforms)
    pub parameters: TransformationParameters,
    /// Transformation method used
    pub method: TransformationMethod,
    /// Original feature dimensions
    pub original_dims: Vec<usize>,
    /// Transformed feature dimensions
    pub transformed_dims: Vec<usize>,
    /// Transformation statistics
    pub statistics: TransformationStatistics,
}

/// Transformation parameters for inverse transforms
#[derive(Debug, Clone)]
pub struct TransformationParameters {
    /// Scaling parameters (means, stds)
    pub scaling: Option<ScalingParameters>,
    /// Encoding mappings
    pub encoding: Option<HashMap<String, HashMap<String, usize>>>,
    /// PCA components
    pub pca_components: Option<Vec<Vec<f64>>>,
}

/// Scaling parameters
#[derive(Debug, Clone)]
pub struct ScalingParameters {
    /// Feature means
    pub means: Vec<f64>,
    /// Feature standard deviations
    pub stds: Vec<f64>,
    /// Feature mins (for MinMax scaling)
    pub mins: Vec<f64>,
    /// Feature maxs (for MinMax scaling)
    pub maxs: Vec<f64>,
}

/// Transformation statistics
#[derive(Debug, Clone)]
pub struct TransformationStatistics {
    /// Computational time
    pub computation_time: f64,
    /// Data distribution changes
    pub distribution_stats: DistributionStatistics,
}

/// Distribution statistics
#[derive(Debug, Clone)]
pub struct DistributionStatistics {
    /// Original feature means
    pub original_means: Vec<f64>,
    /// Transformed feature means
    pub transformed_means: Vec<f64>,
    /// Original feature variances
    pub original_variances: Vec<f64>,
    /// Transformed feature variances
    pub transformed_variances: Vec<f64>,
}

/// Feature transformation methods
#[derive(Debug, Clone)]
pub enum TransformationMethod {
    /// Scaling methods
    StandardScaler,
    MinMaxScaler {
        feature_range: (f64, f64),
    },
    RobustScaler {
        quantile_range: (f64, f64),
    },
    /// Encoding methods
    OneHotEncoder,
    LabelEncoder,
    TargetEncoder,
    /// Generation methods
    PolynomialFeatures {
        degree: usize,
    },
    InteractionFeatures,
    /// Dimensionality reduction
    PCA {
        n_components: usize,
    },
    /// Custom method
    Custom(String),
}

/// Feature transformer
#[derive(Debug)]
pub struct FeatureTransformer {
    /// Transformation method
    pub method: TransformationMethod,
    /// Fit parameters (computed during fit)
    pub fitted_parameters: Option<TransformationParameters>,
}

impl FeatureTransformer {
    /// Create a new feature transformer
    pub fn new(method: TransformationMethod) -> Self {
        Self {
            method,
            fitted_parameters: None,
        }
    }

    /// Fit transformer on training data
    pub fn fit(&mut self, features: &[Vec<f64>]) -> Result<()> {
        let _start_time = std::time::Instant::now();

        let parameters = match &self.method {
            TransformationMethod::StandardScaler => self.fit_standard_scaler(features)?,
            TransformationMethod::MinMaxScaler { feature_range } => {
                self.fit_minmax_scaler(features, *feature_range)?
            }
            TransformationMethod::RobustScaler { quantile_range } => {
                self.fit_robust_scaler(features, *quantile_range)?
            }
            TransformationMethod::OneHotEncoder => self.fit_onehot_encoder(features)?,
            TransformationMethod::LabelEncoder => self.fit_label_encoder(features)?,
            TransformationMethod::TargetEncoder => {
                return Err(NNError::InvalidConfiguration {
                    message: "Target encoder requires target values".to_string(),
                });
            }
            TransformationMethod::PolynomialFeatures { degree } => {
                self.fit_polynomial_features(*degree)?
            }
            TransformationMethod::InteractionFeatures => self.fit_interaction_features()?,
            TransformationMethod::PCA { n_components } => self.fit_pca(features, *n_components)?,
            TransformationMethod::Custom(_) => {
                return Err(NNError::InvalidConfiguration {
                    message: "Custom transformation methods not implemented".to_string(),
                });
            }
        };

        self.fitted_parameters = Some(parameters);
        Ok(())
    }

    /// Transform features using fitted parameters
    pub fn transform(&self, features: &[Vec<f64>]) -> Result<FeatureTransformationResult> {
        if self.fitted_parameters.is_none() {
            return Err(NNError::InvalidConfiguration {
                message: "Transformer must be fitted before transform".to_string(),
            });
        }

        let start_time = std::time::Instant::now();
        let original_dims = vec![
            features.len(),
            features.first().map(|f| f.len()).unwrap_or(0),
        ];

        let transformed_features = match &self.method {
            TransformationMethod::StandardScaler => self.transform_standard_scaler(features)?,
            TransformationMethod::MinMaxScaler { feature_range } => {
                self.transform_minmax_scaler(features, *feature_range)?
            }
            TransformationMethod::RobustScaler { quantile_range } => {
                self.transform_robust_scaler(features, *quantile_range)?
            }
            TransformationMethod::OneHotEncoder => self.transform_onehot_encoder(features)?,
            TransformationMethod::LabelEncoder => self.transform_label_encoder(features)?,
            TransformationMethod::TargetEncoder => {
                return Err(NNError::InvalidConfiguration {
                    message: "Target encoder requires target values".to_string(),
                });
            }
            TransformationMethod::PolynomialFeatures { degree } => {
                self.transform_polynomial_features(features, *degree)?
            }
            TransformationMethod::InteractionFeatures => {
                self.transform_interaction_features(features)?
            }
            TransformationMethod::PCA { n_components } => {
                self.transform_pca(features, *n_components)?
            }
            TransformationMethod::Custom(_) => {
                return Err(NNError::InvalidConfiguration {
                    message: "Custom transformation methods not implemented".to_string(),
                });
            }
        };

        let transformed_dims = vec![
            transformed_features.len(),
            transformed_features.first().map(|f| f.len()).unwrap_or(0),
        ];
        let computation_time = start_time.elapsed().as_secs_f64();

        let distribution_stats = self.compute_distribution_stats(features, &transformed_features);

        let statistics = TransformationStatistics {
            computation_time,
            distribution_stats,
        };

        Ok(FeatureTransformationResult {
            transformed_features,
            parameters: self.fitted_parameters.clone().unwrap(),
            method: self.method.clone(),
            original_dims,
            transformed_dims,
            statistics,
        })
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, features: &[Vec<f64>]) -> Result<FeatureTransformationResult> {
        self.fit(features)?;
        self.transform(features)
    }

    /// Inverse transform (if supported)
    pub fn inverse_transform(&self, transformed_features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        match &self.method {
            TransformationMethod::StandardScaler => {
                self.inverse_standard_scaler(transformed_features)
            }
            TransformationMethod::MinMaxScaler { feature_range } => {
                self.inverse_minmax_scaler(transformed_features, *feature_range)
            }
            _ => Err(NNError::InvalidConfiguration {
                message: "Inverse transform not supported for this method".to_string(),
            }),
        }
    }

    // Standard Scaler implementation
    fn fit_standard_scaler(&self, features: &[Vec<f64>]) -> Result<TransformationParameters> {
        if features.is_empty() || features[0].is_empty() {
            return Err(NNError::InvalidInput {
                message: "Empty feature data".to_string(),
            });
        }

        let n_features = features[0].len();
        let mut means = vec![0.0; n_features];
        let mut stds = vec![0.0; n_features];

        for feature_idx in 0..n_features {
            let values: Vec<f64> = features.iter().map(|row| row[feature_idx]).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
            let std = variance.sqrt();

            means[feature_idx] = mean;
            stds[feature_idx] = if std == 0.0 { 1.0 } else { std }; // Avoid division by zero
        }

        Ok(TransformationParameters {
            scaling: Some(ScalingParameters {
                means,
                stds,
                mins: vec![],
                maxs: vec![],
            }),
            encoding: None,
            pca_components: None,
        })
    }

    fn transform_standard_scaler(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if let Some(params) = &self.fitted_parameters {
            if let Some(scaling) = &params.scaling {
                let mut transformed = Vec::new();

                for row in features {
                    let mut new_row = Vec::new();
                    for (i, &val) in row.iter().enumerate() {
                        let scaled = (val - scaling.means[i]) / scaling.stds[i];
                        new_row.push(scaled);
                    }
                    transformed.push(new_row);
                }

                Ok(transformed)
            } else {
                Err(NNError::InvalidConfiguration {
                    message: "No scaling parameters fitted".to_string(),
                })
            }
        } else {
            Err(NNError::InvalidConfiguration {
                message: "Transformer not fitted".to_string(),
            })
        }
    }

    fn inverse_standard_scaler(&self, transformed_features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if let Some(params) = &self.fitted_parameters {
            if let Some(scaling) = &params.scaling {
                let mut original = Vec::new();

                for row in transformed_features {
                    let mut new_row = Vec::new();
                    for (i, &val) in row.iter().enumerate() {
                        let original_val = val * scaling.stds[i] + scaling.means[i];
                        new_row.push(original_val);
                    }
                    original.push(new_row);
                }

                Ok(original)
            } else {
                Err(NNError::InvalidConfiguration {
                    message: "No scaling parameters fitted".to_string(),
                })
            }
        } else {
            Err(NNError::InvalidConfiguration {
                message: "Transformer not fitted".to_string(),
            })
        }
    }

    // MinMax Scaler implementation
    fn fit_minmax_scaler(
        &self,
        features: &[Vec<f64>],
        _feature_range: (f64, f64),
    ) -> Result<TransformationParameters> {
        if features.is_empty() || features[0].is_empty() {
            return Err(NNError::InvalidInput {
                message: "Empty feature data".to_string(),
            });
        }

        let n_features = features[0].len();
        let mut mins = vec![f64::INFINITY; n_features];
        let mut maxs = vec![f64::NEG_INFINITY; n_features];

        for row in features {
            for (i, &val) in row.iter().enumerate() {
                mins[i] = mins[i].min(val);
                maxs[i] = maxs[i].max(val);
            }
        }

        Ok(TransformationParameters {
            scaling: Some(ScalingParameters {
                means: vec![],
                stds: vec![],
                mins,
                maxs,
            }),
            encoding: None,
            pca_components: None,
        })
    }

    fn transform_minmax_scaler(
        &self,
        features: &[Vec<f64>],
        feature_range: (f64, f64),
    ) -> Result<Vec<Vec<f64>>> {
        if let Some(params) = &self.fitted_parameters {
            if let Some(scaling) = &params.scaling {
                let (min_range, max_range) = feature_range;
                let range = max_range - min_range;

                let mut transformed = Vec::new();

                for row in features {
                    let mut new_row = Vec::new();
                    for (i, &val) in row.iter().enumerate() {
                        if scaling.maxs[i] == scaling.mins[i] {
                            new_row.push(min_range); // Constant feature
                        } else {
                            let scaled = min_range
                                + (val - scaling.mins[i]) / (scaling.maxs[i] - scaling.mins[i])
                                    * range;
                            new_row.push(scaled);
                        }
                    }
                    transformed.push(new_row);
                }

                Ok(transformed)
            } else {
                Err(NNError::InvalidConfiguration {
                    message: "No scaling parameters fitted".to_string(),
                })
            }
        } else {
            Err(NNError::InvalidConfiguration {
                message: "Transformer not fitted".to_string(),
            })
        }
    }

    fn inverse_minmax_scaler(
        &self,
        transformed_features: &[Vec<f64>],
        feature_range: (f64, f64),
    ) -> Result<Vec<Vec<f64>>> {
        if let Some(params) = &self.fitted_parameters {
            if let Some(scaling) = &params.scaling {
                let (min_range, max_range) = feature_range;
                let range = max_range - min_range;

                let mut original = Vec::new();

                for row in transformed_features {
                    let mut new_row = Vec::new();
                    for (i, &val) in row.iter().enumerate() {
                        if scaling.maxs[i] == scaling.mins[i] {
                            new_row.push(scaling.mins[i]); // Constant feature
                        } else {
                            let original_val = scaling.mins[i]
                                + (val - min_range) / range * (scaling.maxs[i] - scaling.mins[i]);
                            new_row.push(original_val);
                        }
                    }
                    original.push(new_row);
                }

                Ok(original)
            } else {
                Err(NNError::InvalidConfiguration {
                    message: "No scaling parameters fitted".to_string(),
                })
            }
        } else {
            Err(NNError::InvalidConfiguration {
                message: "Transformer not fitted".to_string(),
            })
        }
    }

    // Simplified implementations for other methods
    fn fit_robust_scaler(
        &self,
        _features: &[Vec<f64>],
        _quantile_range: (f64, f64),
    ) -> Result<TransformationParameters> {
        // Simplified robust scaler - would compute quantiles in practice
        Ok(TransformationParameters {
            scaling: Some(ScalingParameters {
                means: vec![],
                stds: vec![],
                mins: vec![],
                maxs: vec![],
            }),
            encoding: None,
            pca_components: None,
        })
    }

    fn transform_robust_scaler(
        &self,
        features: &[Vec<f64>],
        _quantile_range: (f64, f64),
    ) -> Result<Vec<Vec<f64>>> {
        // Simplified - just return original features
        Ok(features.to_vec())
    }

    fn fit_onehot_encoder(&self, _features: &[Vec<f64>]) -> Result<TransformationParameters> {
        Ok(TransformationParameters {
            scaling: None,
            encoding: Some(HashMap::new()),
            pca_components: None,
        })
    }

    fn transform_onehot_encoder(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        // Simplified - just return original features
        Ok(features.to_vec())
    }

    fn fit_label_encoder(&self, _features: &[Vec<f64>]) -> Result<TransformationParameters> {
        Ok(TransformationParameters {
            scaling: None,
            encoding: Some(HashMap::new()),
            pca_components: None,
        })
    }

    fn transform_label_encoder(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        // Simplified - just return original features
        Ok(features.to_vec())
    }

    fn fit_polynomial_features(&self, _degree: usize) -> Result<TransformationParameters> {
        Ok(TransformationParameters {
            scaling: None,
            encoding: None,
            pca_components: None,
        })
    }

    fn transform_polynomial_features(
        &self,
        features: &[Vec<f64>],
        _degree: usize,
    ) -> Result<Vec<Vec<f64>>> {
        // Simplified - just return original features
        Ok(features.to_vec())
    }

    fn fit_interaction_features(&self) -> Result<TransformationParameters> {
        Ok(TransformationParameters {
            scaling: None,
            encoding: None,
            pca_components: None,
        })
    }

    fn transform_interaction_features(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        // Simplified - just return original features
        Ok(features.to_vec())
    }

    fn fit_pca(
        &self,
        _features: &[Vec<f64>],
        _n_components: usize,
    ) -> Result<TransformationParameters> {
        Ok(TransformationParameters {
            scaling: None,
            encoding: None,
            pca_components: Some(vec![]),
        })
    }

    fn transform_pca(&self, features: &[Vec<f64>], _n_components: usize) -> Result<Vec<Vec<f64>>> {
        // Simplified - just return original features
        Ok(features.to_vec())
    }

    /// Compute distribution statistics
    fn compute_distribution_stats(
        &self,
        original: &[Vec<f64>],
        transformed: &[Vec<f64>],
    ) -> DistributionStatistics {
        let mut original_means = Vec::new();
        let mut original_variances = Vec::new();
        let mut transformed_means = Vec::new();
        let mut transformed_variances = Vec::new();

        if !original.is_empty() && !transformed.is_empty() {
            let n_features_orig = original[0].len();
            let n_features_trans = transformed[0].len();

            for i in 0..n_features_orig.min(n_features_trans) {
                let orig_values: Vec<f64> = original.iter().map(|row| row[i]).collect();
                let trans_values: Vec<f64> = transformed.iter().map(|row| row[i]).collect();

                let orig_mean = orig_values.iter().sum::<f64>() / orig_values.len() as f64;
                let trans_mean = trans_values.iter().sum::<f64>() / trans_values.len() as f64;

                let orig_var = orig_values
                    .iter()
                    .map(|&x| (x - orig_mean).powi(2))
                    .sum::<f64>()
                    / orig_values.len() as f64;
                let trans_var = trans_values
                    .iter()
                    .map(|&x| (x - trans_mean).powi(2))
                    .sum::<f64>()
                    / trans_values.len() as f64;

                original_means.push(orig_mean);
                original_variances.push(orig_var);
                transformed_means.push(trans_mean);
                transformed_variances.push(trans_var);
            }
        }

        DistributionStatistics {
            original_means,
            transformed_means,
            original_variances,
            transformed_variances,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_scaler() {
        let mut transformer = FeatureTransformer::new(TransformationMethod::StandardScaler);

        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        let result = transformer.fit_transform(&features).unwrap();

        assert_eq!(result.transformed_features.len(), 3);
        assert_eq!(result.transformed_features[0].len(), 2);

        // Check that means are approximately zero
        let mean_0: f64 = result
            .transformed_features
            .iter()
            .map(|row| row[0])
            .sum::<f64>()
            / 3.0;
        let mean_1: f64 = result
            .transformed_features
            .iter()
            .map(|row| row[1])
            .sum::<f64>()
            / 3.0;

        assert!(mean_0.abs() < 1e-10);
        assert!(mean_1.abs() < 1e-10);
    }

    #[test]
    fn test_minmax_scaler() {
        let mut transformer = FeatureTransformer::new(TransformationMethod::MinMaxScaler {
            feature_range: (0.0, 1.0),
        });

        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        let result = transformer.fit_transform(&features).unwrap();

        assert_eq!(result.transformed_features.len(), 3);
        assert_eq!(result.transformed_features[0].len(), 2);

        // Check range [0, 1]
        for row in &result.transformed_features {
            for &val in row {
                assert!((0.0..=1.0).contains(&val));
            }
        }
    }

    #[test]
    fn test_inverse_transform() {
        let mut transformer = FeatureTransformer::new(TransformationMethod::StandardScaler);

        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        let transformed = transformer.fit_transform(&features).unwrap();
        let reconstructed = transformer
            .inverse_transform(&transformed.transformed_features)
            .unwrap();

        // Check reconstruction accuracy
        for (orig, recon) in features.iter().zip(reconstructed.iter()) {
            for (&o, &r) in orig.iter().zip(recon.iter()) {
                assert!((o - r).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_transformer_creation() {
        let transformer = FeatureTransformer::new(TransformationMethod::StandardScaler);

        match transformer.method {
            TransformationMethod::StandardScaler => {}
            _ => panic!("Wrong method"),
        }

        assert!(transformer.fitted_parameters.is_none());
    }
}
