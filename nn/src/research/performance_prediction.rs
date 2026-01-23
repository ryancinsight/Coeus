//! Performance Prediction for NAS and AutoML
//!
//! This module provides advanced performance prediction models for neural architectures
//! and hyperparameter configurations, enabling efficient search by predicting performance
//! without expensive training evaluations.

use std::collections::HashMap;

use crate::core::error::{NNError, Result};
use crate::hpo::{HyperparameterConfig, HyperparameterValue};
use crate::nas::search_space::LayerSpec;
use crate::nas::Architecture;
use crate::research::performance_prediction::builders::{
    BoostedTreesBuilder, GNNNASBuilder, SurrogateHPOBuilder,
};
use crate::research::performance_prediction::predictors::{
    BoostedTreesPredictor, GNNNASPredictor, SurrogateHPOPredictor,
};

/// Performance Predictor Trait
/// Interface for all performance prediction models
pub trait PerformancePredictor: Send + Sync + std::fmt::Debug {
    /// Predict performance for an architecture or configuration
    fn predict(&self, input: &PredictionInput) -> Result<PredictionOutput>;

    /// Train or update the predictor with new data
    fn train(&mut self, training_data: &[TrainingExample]) -> Result<()>;

    /// Get prediction confidence/intervals
    fn confidence(&self, input: &PredictionInput) -> Result<PredictionConfidence>;

    /// Get predictor name
    fn name(&self) -> &str;

    /// Get supported prediction types
    fn supported_types(&self) -> Vec<PredictionType>;
}

/// Input for performance prediction
#[derive(Debug, Clone)]
pub enum PredictionInput {
    /// Neural architecture prediction
    Architecture(ArchitecturePredictionInput),
    /// Hyperparameter configuration prediction
    Hyperparameters(HyperparameterPredictionInput),
    /// Joint architecture and hyperparameters prediction
    Joint(JointPredictionInput),
}

/// Neural architecture prediction input
#[derive(Debug, Clone)]
pub struct ArchitecturePredictionInput {
    pub architecture: Architecture,
    pub dataset_info: DatasetInfo,
    pub task_info: TaskInfo,
    pub hardware_info: Option<HardwareInfo>,
}

/// Hyperparameter prediction input
#[derive(Debug, Clone)]
pub struct HyperparameterPredictionInput {
    pub architecture: Option<Architecture>,
    pub config: HyperparameterConfig,
    pub dataset_info: DatasetInfo,
    pub task_info: TaskInfo,
}

/// Joint architecture and hyperparameter prediction
#[derive(Debug, Clone)]
pub struct JointPredictionInput {
    pub architecture: Architecture,
    pub config: HyperparameterConfig,
    pub dataset_info: DatasetInfo,
    pub task_info: TaskInfo,
}

/// Dataset information
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    pub name: String,
    pub size: usize,
    pub input_shape: Vec<usize>,
    pub output_classes: usize,
    pub complexity_score: f64, // 0.0-1.0
}

/// Task information
#[derive(Debug, Clone)]
pub struct TaskInfo {
    pub task_type: String, // "classification", "regression", etc.
    pub metric: String,    // "accuracy", "mse", etc.
    pub domain: String,    // "vision", "nlp", "tabular"
}

/// Hardware information for prediction
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub device_type: String, // "gpu", "cpu", "tpu"
    pub memory_gb: f64,
    pub bandwidth_gbps: f64,
    pub compute_units: usize,
}

/// Prediction output
#[derive(Debug, Clone)]
pub struct PredictionOutput {
    pub predicted_performance: f64,
    pub auxiliary_predictions: HashMap<String, f64>, // latency, memory, etc.
    pub prediction_type: PredictionType,
    pub metadata: HashMap<String, String>,
}

/// Types of predictions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PredictionType {
    Accuracy,
    Latency,
    Memory,
    Energy,
    Throughput,
    Convergence,
    MultiObjective(Vec<String>),
}

/// Prediction confidence information
#[derive(Debug, Clone)]
pub struct PredictionConfidence {
    pub confidence_score: f64, // 0.0-1.0
    pub uncertainty: f64,
    pub confidence_interval: Option<(f64, f64)>,
    pub reliability_score: f64,
}

/// Training example for predictor
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub input: PredictionInput,
    pub actual_output: PredictionOutput,
    pub training_metadata: HashMap<String, String>,
}

/// Performance Prediction Framework
/// Manages multiple prediction models and orchestrates predictions
#[derive(Debug)]
pub struct PerformancePredictionFramework {
    /// Registered predictors
    predictors: HashMap<String, Box<dyn PerformancePredictor>>,
    /// Training data storage
    training_data: Vec<TrainingExample>,
    /// Active predictor cache
    active_predictors: HashMap<PredictionType, String>, // type -> predictor name
    /// Prediction model factory
    model_factory: PredictionModelFactory,
}

/// Factory for creating prediction models
#[derive(Debug)]
pub struct PredictionModelFactory {
    /// Available model builders
    builders: HashMap<String, Box<dyn ModelBuilder>>,
}

/// Model builder trait
pub trait ModelBuilder: Send + Sync + std::fmt::Debug {
    /// Build a new predictor model
    fn build(&self, config: &PredictionConfig) -> Result<Box<dyn PerformancePredictor>>;

    /// Get supported prediction types
    fn supported_types(&self) -> Vec<PredictionType>;

    /// Get model name
    fn name(&self) -> &str;
}

/// Prediction model configuration
#[derive(Debug, Clone)]
pub struct PredictionConfig {
    pub model_type: String,
    pub hyperparameters: HashMap<String, f64>,
    pub feature_engineering: FeatureEngineeringConfig,
    pub training_params: TrainingParameters,
}

/// Feature engineering configuration
#[derive(Debug, Clone)]
pub struct FeatureEngineeringConfig {
    pub architecture_features: Vec<String>,
    pub hyperparameter_features: Vec<String>,
    pub dataset_features: Vec<String>,
    pub hardware_features: Vec<String>,
}

/// Training parameters
#[derive(Debug, Clone)]
pub struct TrainingParameters {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub validation_split: f64,
    pub early_stopping: bool,
}

impl Default for PerformancePredictionFramework {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformancePredictionFramework {
    /// Create new performance prediction framework
    pub fn new() -> Self {
        let mut framework = Self {
            predictors: HashMap::new(),
            training_data: Vec::new(),
            active_predictors: HashMap::new(),
            model_factory: PredictionModelFactory::new(),
        };

        // Initialize with default predictors
        framework.initialize_default_predictors();

        framework
    }

    /// Initialize default predictors
    fn initialize_default_predictors(&mut self) {
        // Register GNN-based NAS predictor
        self.register_predictor(
            "gnn_nas_predictor".to_string(),
            Box::new(GNNNASPredictor::new()),
        );

        // Register surrogate model for HPO
        self.register_predictor(
            "surrogate_hpo_predictor".to_string(),
            Box::new(SurrogateHPOPredictor::new()),
        );

        // Register boosted trees predictor
        self.register_predictor(
            "boosted_trees_predictor".to_string(),
            Box::new(BoostedTreesPredictor::new()),
        );

        // Set default active predictors
        self.active_predictors
            .insert(PredictionType::Accuracy, "gnn_nas_predictor".to_string());
        self.active_predictors.insert(
            PredictionType::Latency,
            "surrogate_hpo_predictor".to_string(),
        );
        self.active_predictors.insert(
            PredictionType::Memory,
            "boosted_trees_predictor".to_string(),
        );
    }

    /// Predict performance for given input
    pub fn predict(&self, input: &PredictionInput) -> Result<PredictionOutput> {
        let prediction_type = match input {
            PredictionInput::Architecture(_) => PredictionType::Accuracy,
            PredictionInput::Hyperparameters(_) => PredictionType::Accuracy,
            PredictionInput::Joint(_) => PredictionType::Accuracy,
        };

        if let Some(predictor_name) = self.active_predictors.get(&prediction_type) {
            if let Some(predictor) = self.predictors.get(predictor_name) {
                predictor.predict(input)
            } else {
                Err(NNError::InvalidConfiguration {
                    message: format!("Active predictor '{}' not found", predictor_name),
                })
            }
        } else {
            Err(NNError::InvalidConfiguration {
                message: format!("No active predictor for type {:?}", prediction_type),
            })
        }
    }

    /// Train predictors with new data
    pub fn train_predictors(&mut self, data: &[TrainingExample]) -> Result<()> {
        // Add to training data
        self.training_data.extend_from_slice(data);

        // Train each predictor
        for predictor in self.predictors.values_mut() {
            predictor.train(&self.training_data)?;
        }

        Ok(())
    }

    /// Register a predictor
    pub fn register_predictor(&mut self, name: String, predictor: Box<dyn PerformancePredictor>) {
        self.predictors.insert(name, predictor);
    }

    /// Set active predictor for a prediction type
    pub fn set_active_predictor(
        &mut self,
        prediction_type: PredictionType,
        predictor_name: String,
    ) -> Result<()> {
        if self.predictors.contains_key(&predictor_name) {
            self.active_predictors
                .insert(prediction_type, predictor_name);
            Ok(())
        } else {
            Err(NNError::InvalidConfiguration {
                message: format!("Predictor '{}' not found", predictor_name),
            })
        }
    }

    /// Get prediction confidence
    pub fn get_confidence(&self, input: &PredictionInput) -> Result<f64> {
        let prediction_type = match input {
            PredictionInput::Architecture(_) => PredictionType::Accuracy,
            PredictionInput::Hyperparameters(_) => PredictionType::Accuracy,
            PredictionInput::Joint(_) => PredictionType::Accuracy,
        };

        if let Some(predictor_name) = self.active_predictors.get(&prediction_type) {
            if let Some(predictor) = self.predictors.get(predictor_name) {
                let confidence = predictor.confidence(input)?;
                Ok(confidence.confidence_score)
            } else {
                Ok(0.0) // No predictor available
            }
        } else {
            Ok(0.0)
        }
    }

    /// Create and train a new predictor
    pub fn create_predictor(&mut self, name: String, config: PredictionConfig) -> Result<()> {
        if let Some(builder) = self.model_factory.builders.get(&config.model_type) {
            let predictor = builder.build(&config)?;
            self.register_predictor(name, predictor);
            Ok(())
        } else {
            Err(NNError::InvalidConfiguration {
                message: format!("Model builder '{}' not found", config.model_type),
            })
        }
    }

    /// Generate training features from architectures/hyperparameters
    pub fn generate_features(&self, input: &PredictionInput) -> Result<Vec<f64>> {
        match input {
            PredictionInput::Architecture(arch_input) => {
                self.generate_architecture_features(&arch_input.architecture)
            }
            PredictionInput::Hyperparameters(hp_input) => {
                self.generate_hyperparameter_features(&hp_input.config)
            }
            PredictionInput::Joint(joint_input) => {
                let mut arch_features =
                    self.generate_architecture_features(&joint_input.architecture)?;
                let mut hp_features = self.generate_hyperparameter_features(&joint_input.config)?;
                arch_features.append(&mut hp_features);
                Ok(arch_features)
            }
        }
    }

    /// Generate features from architecture
    fn generate_architecture_features(&self, architecture: &Architecture) -> Result<Vec<f64>> {
        let mut features = Vec::new();

        // Basic architectural features
        features.push(architecture.layers.len() as f64); // number of layers
        features.push(architecture.num_parameters() as f64); // parameter count
        features.push(architecture.connections.len() as f64); // number of connections

        // Layer type distribution (one-hot encoded features)
        let mut conv_count = 0.0;
        let mut linear_count = 0.0;
        let mut attn_count = 0.0;
        let mut pool_count = 0.0;

        for layer in &architecture.layers {
            match layer {
                LayerSpec::Conv2D { .. } => conv_count += 1.0,
                LayerSpec::Linear { .. } => linear_count += 1.0,
                LayerSpec::Attention { .. } => attn_count += 1.0,
                LayerSpec::Pooling { .. } => pool_count += 1.0,
                _ => {}
            }
        }

        features.push(conv_count);
        features.push(linear_count);
        features.push(attn_count);
        features.push(pool_count);

        // Architecture depth and width statistics
        let total_params = architecture.num_parameters() as f64;
        features.push(total_params / architecture.layers.len() as f64); // avg params per layer

        Ok(features)
    }

    /// Generate features from hyperparameter configuration
    fn generate_hyperparameter_features(&self, config: &HyperparameterConfig) -> Result<Vec<f64>> {
        let mut features = Vec::new();

        // Extract numerical hyperparameters
        if let Some(HyperparameterValue::Float(lr)) = config.get("learning_rate") {
            features.push(*lr);
        }
        if let Some(HyperparameterValue::Float(batch_size)) = config.get("batch_size") {
            features.push(*batch_size);
        }
        if let Some(HyperparameterValue::Float(dropout)) = config.get("dropout") {
            features.push(*dropout);
        }
        if let Some(HyperparameterValue::Float(weight_decay)) = config.get("weight_decay") {
            features.push(*weight_decay);
        }

        // Add categorical hyperparameters as numerical codes
        if let Some(HyperparameterValue::Categorical(optimizer)) = config.get("optimizer") {
            let opt_code = match optimizer.as_str() {
                "adam" => 1.0,
                "sgd" => 2.0,
                "rmsprop" => 3.0,
                _ => 0.0,
            };
            features.push(opt_code);
        }

        Ok(features)
    }

    /// Export predictor metadata for analysis
    pub fn export_metadata(&self) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();

        metadata.insert("total_predictors".to_string(), self.predictors.len().into());
        metadata.insert(
            "active_predictors".to_string(),
            self.active_predictors.len().into(),
        );
        metadata.insert(
            "training_examples".to_string(),
            self.training_data.len().into(),
        );

        let predictor_names: Vec<String> = self.predictors.keys().cloned().collect();
        metadata.insert(
            "predictor_names".to_string(),
            serde_json::to_value(predictor_names).unwrap(),
        );

        metadata
    }
}

impl Default for PredictionModelFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl PredictionModelFactory {
    pub fn new() -> Self {
        let mut factory = Self {
            builders: HashMap::new(),
        };

        // Register default builders
        factory
            .builders
            .insert("gnn_nas".to_string(), Box::new(GNNNASBuilder::new()));
        factory.builders.insert(
            "surrogate_hpo".to_string(),
            Box::new(SurrogateHPOBuilder::new()),
        );
        factory.builders.insert(
            "boosted_trees".to_string(),
            Box::new(BoostedTreesBuilder::new()),
        );

        factory
    }
}

/// Predictor implementations
pub mod predictors {
    use super::*;

    /// GNN-based NAS predictor (uses Graph Neural Networks for architecture prediction)
    #[derive(Debug)]
    pub struct GNNNASPredictor {
        trained: bool,
        // In real implementation, would contain GNN model
    }

    impl Default for GNNNASPredictor {
        fn default() -> Self {
            Self::new()
        }
    }

    impl GNNNASPredictor {
        pub fn new() -> Self {
            Self { trained: false }
        }
    }

    impl PerformancePredictor for GNNNASPredictor {
        fn predict(&self, input: &PredictionInput) -> Result<PredictionOutput> {
            match input {
                PredictionInput::Architecture(arch_input) => {
                    if !self.trained {
                        return Err(NNError::NotInitialized {
                            component: "GNN predictor not trained".to_string(),
                        });
                    }

                    // Placeholder prediction logic
                    let base_score = 0.7;
                    let complexity_penalty =
                        arch_input.architecture.num_parameters() as f64 * 0.000001;
                    let predicted_accuracy = (base_score - complexity_penalty).clamp(0.0, 1.0);

                    Ok(PredictionOutput {
                        predicted_performance: predicted_accuracy,
                        auxiliary_predictions: HashMap::from([
                            ("latency".to_string(), 100.0),
                            ("memory".to_string(), 1024.0),
                        ]),
                        prediction_type: PredictionType::Accuracy,
                        metadata: HashMap::from([
                            ("model".to_string(), "gnn_nas".to_string()),
                            (
                                "architecture_complexity".to_string(),
                                arch_input.architecture.num_parameters().to_string(),
                            ),
                        ]),
                    })
                }
                _ => Err(NNError::InvalidInput {
                    message: "GNN predictor only supports architecture input".to_string(),
                }),
            }
        }

        fn train(&mut self, _training_data: &[TrainingExample]) -> Result<()> {
            // Placeholder training logic
            self.trained = true;
            Ok(())
        }

        fn confidence(&self, _input: &PredictionInput) -> Result<PredictionConfidence> {
            if !self.trained {
                return Ok(PredictionConfidence {
                    confidence_score: 0.0,
                    uncertainty: 1.0,
                    confidence_interval: None,
                    reliability_score: 0.0,
                });
            }

            Ok(PredictionConfidence {
                confidence_score: 0.8,
                uncertainty: 0.1,
                confidence_interval: Some((0.65, 0.75)),
                reliability_score: 0.85,
            })
        }

        fn name(&self) -> &str {
            "GNN NAS Predictor"
        }

        fn supported_types(&self) -> Vec<PredictionType> {
            vec![
                PredictionType::Accuracy,
                PredictionType::Memory,
                PredictionType::Latency,
            ]
        }
    }

    /// Surrogate model for HPO prediction
    #[derive(Debug)]
    pub struct SurrogateHPOPredictor {
        trained: bool,
        // In real implementation, would contain surrogate model (GP, Random Forest, etc.)
    }

    impl Default for SurrogateHPOPredictor {
        fn default() -> Self {
            Self::new()
        }
    }

    impl SurrogateHPOPredictor {
        pub fn new() -> Self {
            Self { trained: false }
        }
    }

    impl PerformancePredictor for SurrogateHPOPredictor {
        fn predict(&self, input: &PredictionInput) -> Result<PredictionOutput> {
            match input {
                PredictionInput::Hyperparameters(hp_input) => {
                    if !self.trained {
                        return Err(NNError::NotInitialized {
                            component: "Surrogate predictor not trained".to_string(),
                        });
                    }

                    // Simple rule-based prediction for demonstration
                    let mut score: f64 = 0.5;

                    if let Some(HyperparameterValue::Float(lr)) =
                        hp_input.config.get("learning_rate")
                    {
                        if *lr > 0.0001 && *lr < 0.1 {
                            score += 0.2;
                        }
                    }

                    if let Some(HyperparameterValue::Float(batch_size)) =
                        hp_input.config.get("batch_size")
                    {
                        if *batch_size >= 16.0 && *batch_size <= 128.0 {
                            score += 0.1;
                        }
                    }

                    Ok(PredictionOutput {
                        predicted_performance: score.clamp(0.0, 1.0),
                        auxiliary_predictions: HashMap::new(),
                        prediction_type: PredictionType::Accuracy,
                        metadata: HashMap::from([(
                            "model".to_string(),
                            "surrogate_hpo".to_string(),
                        )]),
                    })
                }
                _ => Err(NNError::InvalidInput {
                    message: "Surrogate predictor only supports hyperparameter input".to_string(),
                }),
            }
        }

        fn train(&mut self, _training_data: &[TrainingExample]) -> Result<()> {
            self.trained = true;
            Ok(())
        }

        fn confidence(&self, _input: &PredictionInput) -> Result<PredictionConfidence> {
            Ok(PredictionConfidence {
                confidence_score: 0.75,
                uncertainty: 0.15,
                confidence_interval: Some((0.5, 0.8)),
                reliability_score: 0.8,
            })
        }

        fn name(&self) -> &str {
            "Surrogate HPO Predictor"
        }

        fn supported_types(&self) -> Vec<PredictionType> {
            vec![PredictionType::Accuracy]
        }
    }

    /// Boosted trees predictor for general use
    #[derive(Debug)]
    pub struct BoostedTreesPredictor {
        trained: bool,
    }

    impl Default for BoostedTreesPredictor {
        fn default() -> Self {
            Self::new()
        }
    }

    impl BoostedTreesPredictor {
        pub fn new() -> Self {
            Self { trained: false }
        }
    }

    impl PerformancePredictor for BoostedTreesPredictor {
        fn predict(&self, _input: &PredictionInput) -> Result<PredictionOutput> {
            if !self.trained {
                return Err(NNError::NotInitialized {
                    component: "Boosted trees predictor not trained".to_string(),
                });
            }

            Ok(PredictionOutput {
                predicted_performance: 0.65,
                auxiliary_predictions: HashMap::from([("throughput".to_string(), 1000.0)]),
                prediction_type: PredictionType::Throughput,
                metadata: HashMap::from([("model".to_string(), "boosted_trees".to_string())]),
            })
        }

        fn train(&mut self, _training_data: &[TrainingExample]) -> Result<()> {
            self.trained = true;
            Ok(())
        }

        fn confidence(&self, _input: &PredictionInput) -> Result<PredictionConfidence> {
            Ok(PredictionConfidence {
                confidence_score: 0.7,
                uncertainty: 0.2,
                confidence_interval: Some((0.6, 0.8)),
                reliability_score: 0.75,
            })
        }

        fn name(&self) -> &str {
            "Boosted Trees Predictor"
        }

        fn supported_types(&self) -> Vec<PredictionType> {
            vec![
                PredictionType::Latency,
                PredictionType::Throughput,
                PredictionType::Memory,
            ]
        }
    }
}

/// Model builders
pub mod builders {
    use super::*;

    #[derive(Debug)]
    pub struct GNNNASBuilder;

    impl Default for GNNNASBuilder {
        fn default() -> Self {
            Self::new()
        }
    }

    impl GNNNASBuilder {
        pub fn new() -> Self {
            Self
        }
    }

    impl ModelBuilder for GNNNASBuilder {
        fn build(&self, _config: &PredictionConfig) -> Result<Box<dyn PerformancePredictor>> {
            Ok(Box::new(super::predictors::GNNNASPredictor::new()))
        }

        fn supported_types(&self) -> Vec<PredictionType> {
            vec![
                PredictionType::Accuracy,
                PredictionType::Memory,
                PredictionType::Latency,
            ]
        }

        fn name(&self) -> &str {
            "GNN NAS Builder"
        }
    }

    #[derive(Debug)]
    pub struct SurrogateHPOBuilder;

    impl Default for SurrogateHPOBuilder {
        fn default() -> Self {
            Self::new()
        }
    }

    impl SurrogateHPOBuilder {
        pub fn new() -> Self {
            Self
        }
    }

    impl ModelBuilder for SurrogateHPOBuilder {
        fn build(&self, _config: &PredictionConfig) -> Result<Box<dyn PerformancePredictor>> {
            Ok(Box::new(super::predictors::SurrogateHPOPredictor::new()))
        }

        fn supported_types(&self) -> Vec<PredictionType> {
            vec![PredictionType::Accuracy]
        }

        fn name(&self) -> &str {
            "Surrogate HPO Builder"
        }
    }

    #[derive(Debug)]
    pub struct BoostedTreesBuilder;

    impl Default for BoostedTreesBuilder {
        fn default() -> Self {
            Self::new()
        }
    }

    impl BoostedTreesBuilder {
        pub fn new() -> Self {
            Self
        }
    }

    impl ModelBuilder for BoostedTreesBuilder {
        fn build(&self, _config: &PredictionConfig) -> Result<Box<dyn PerformancePredictor>> {
            Ok(Box::new(super::predictors::BoostedTreesPredictor::new()))
        }

        fn supported_types(&self) -> Vec<PredictionType> {
            vec![
                PredictionType::Latency,
                PredictionType::Throughput,
                PredictionType::Memory,
            ]
        }

        fn name(&self) -> &str {
            "Boosted Trees Builder"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas::search_space::{ArchitectureSpace, ArchitectureType};

    #[test]
    fn test_performance_prediction_framework() {
        let framework = PerformancePredictionFramework::new();
        assert!(!framework.predictors.is_empty());
        assert!(!framework.active_predictors.is_empty());
    }

    #[test]
    fn test_architecture_prediction_input() {
        let space = ArchitectureSpace::new(ArchitectureType::CNN);
        let architecture = space.sample_random(3).unwrap();

        let input = ArchitecturePredictionInput {
            architecture: architecture.clone(),
            dataset_info: DatasetInfo {
                name: "cifar10".to_string(),
                size: 50000,
                input_shape: vec![32, 32, 3],
                output_classes: 10,
                complexity_score: 0.5,
            },
            task_info: TaskInfo {
                task_type: "classification".to_string(),
                metric: "accuracy".to_string(),
                domain: "vision".to_string(),
            },
            hardware_info: Some(HardwareInfo {
                device_type: "gpu".to_string(),
                memory_gb: 8.0,
                bandwidth_gbps: 50.0,
                compute_units: 1,
            }),
        };

        assert_eq!(input.dataset_info.name, "cifar10");
        assert_eq!(input.task_info.task_type, "classification");
    }

    #[test]
    fn test_feature_generation() {
        let framework = PerformancePredictionFramework::new();

        // Create test architecture
        let space = ArchitectureSpace::new(ArchitectureType::CNN);
        let architecture = space.sample_random(2).unwrap();

        let input = PredictionInput::Architecture(ArchitecturePredictionInput {
            architecture: architecture.clone(),
            dataset_info: DatasetInfo {
                name: "test".to_string(),
                size: 1000,
                input_shape: vec![28, 28, 1],
                output_classes: 10,
                complexity_score: 0.5,
            },
            task_info: TaskInfo {
                task_type: "classification".to_string(),
                metric: "accuracy".to_string(),
                domain: "vision".to_string(),
            },
            hardware_info: None,
        });

        let features = framework.generate_features(&input).unwrap();
        assert!(!features.is_empty());

        // Should have basic architectural features
        assert!(features.len() >= 3); // At least layers, params, connections
    }

    #[test]
    fn test_prediction_confidence() {
        let framework = PerformancePredictionFramework::new();

        let space = ArchitectureSpace::new(ArchitectureType::CNN);
        let architecture = space.sample_random(2).unwrap();

        let input = PredictionInput::Architecture(ArchitecturePredictionInput {
            architecture,
            dataset_info: DatasetInfo {
                name: "test".to_string(),
                size: 1000,
                input_shape: vec![28, 28, 1],
                output_classes: 10,
                complexity_score: 0.5,
            },
            task_info: TaskInfo {
                task_type: "classification".to_string(),
                metric: "accuracy".to_string(),
                domain: "vision".to_string(),
            },
            hardware_info: None,
        });

        // Confidence should be 0.0 when no predictor is trained
        let confidence = framework.get_confidence(&input).unwrap();
        assert_eq!(confidence, 0.0);
    }
}
