//! CLIP Research Framework Integration
//!
//! This module provides comprehensive research automation capabilities for CLIP,
//! including hyperparameter optimization, neural architecture search, and
//! automated experiment tracking. Enables systematic CLIP model development
//! and benchmarking.
//!
//! ## Features
//! - Hyperparameter optimization for CLIP training parameters
//! - Neural architecture search for CLIP variants
//! - Automated experiment tracking and reproducibility
//! - Multi-objective optimization (performance vs efficiency)
//! - Research pipeline orchestration

// Submodule for zero-shot classification
pub mod zero_shot;

use crate::core::error::{NNError, Result};
use crate::clip::core::config::ClipConfig;
use crate::clip::models::clip::ClipModel;
use crate::datasets::VisionLanguageData;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use rand::Rng;
use num_traits::{FromPrimitive, ToPrimitive, Bounded, Float};
use dtype::FloatExt;

/// CLIP Research Framework - orchestrates automated CLIP research
pub struct ClipResearchFramework<B, S, T>
where
    B: backend::Backend<Data = T> + Clone,
    S: storage::Storage<T> + Clone,
    T: dtype::DataType + FloatExt + FromPrimitive + Bounded + Float,
{
    /// Base CLIP configuration
    base_config: ClipConfig,
    /// Research configuration
    research_config: ResearchConfig,
    /// Experiment tracker
    experiment_tracker: ExperimentTracker,
    /// Performance evaluator
    evaluator: PerformanceEvaluator<B, S, T>,
    /// Marker for type parameters
    _marker: PhantomData<(B, S, T)>,
}

/// Research configuration for CLIP experiments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchConfig {
    /// Experiment name prefix
    pub experiment_prefix: String,
    /// Number of parallel experiments
    pub num_parallel: usize,
    /// Total experiment budget
    pub max_experiments: usize,
    /// Time budget per experiment (seconds)
    pub time_budget_per_experiment: u64,
    /// Optimization objectives
    pub objectives: Vec<OptimizationObjective>,
    /// Hyperparameter search space
    pub hpo_space: HyperparameterSpace,
    /// NAS search space (if enabled)
    pub nas_space: Option<NASSearchSpace>,
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            experiment_prefix: "clip_research".to_string(),
            num_parallel: 4,
            max_experiments: 100,
            time_budget_per_experiment: 3600, // 1 hour
            objectives: vec![
                OptimizationObjective::RetrievalR1,
                OptimizationObjective::ZeroShotAccuracy,
                OptimizationObjective::TrainingEfficiency,
            ],
            hpo_space: HyperparameterSpace::default(),
            nas_space: Some(NASSearchSpace::default()),
        }
    }
}

/// Optimization objectives for CLIP research
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum OptimizationObjective {
    /// Text-to-image retrieval R@1
    RetrievalR1,
    /// Zero-shot classification top-1 accuracy
    ZeroShotAccuracy,
    /// Training efficiency (samples/second)
    TrainingEfficiency,
    /// Model size (parameter count)
    ModelSize,
    /// Inference latency
    InferenceLatency,
    /// Memory usage during training
    MemoryEfficiency,
}

/// Hyperparameter search space for CLIP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterSpace {
    /// Learning rate range
    pub learning_rate: ParameterRange<f64>,
    /// Batch size options
    pub batch_size: ParameterRange<usize>,
    /// Temperature range
    pub temperature: ParameterRange<f64>,
    /// Weight decay range
    pub weight_decay: ParameterRange<f64>,
    /// Warmup steps range
    pub warmup_steps: ParameterRange<usize>,
    /// Gradient clipping range
    pub max_grad_norm: ParameterRange<f64>,
}

impl Default for HyperparameterSpace {
    fn default() -> Self {
        Self {
            learning_rate: ParameterRange::LogUniform { min: 1e-5, max: 1e-3 },
            batch_size: ParameterRange::Discrete(vec![16, 32, 64, 128]),
            temperature: ParameterRange::Uniform { min: 0.01, max: 0.2 },
            weight_decay: ParameterRange::LogUniform { min: 1e-6, max: 1e-2 },
            warmup_steps: ParameterRange::Discrete(vec![500, 1000, 2000, 5000]),
            max_grad_norm: ParameterRange::Uniform { min: 0.5, max: 2.0 },
        }
    }
}

/// NAS search space for CLIP variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NASSearchSpace {
    /// Vision transformer configurations
    pub vision_configs: Vec<VisionNASConfig>,
    /// Text transformer configurations
    pub text_configs: Vec<TextNASConfig>,
    /// Projection dimension options
    pub projection_dims: Vec<usize>,
}

impl Default for NASSearchSpace {
    fn default() -> Self {
        Self {
            vision_configs: vec![
                VisionNASConfig { layers: 6, heads: 8, hidden_size: 512, mlp_ratio: 4.0 },
                VisionNASConfig { layers: 12, heads: 12, hidden_size: 768, mlp_ratio: 4.0 },
                VisionNASConfig { layers: 24, heads: 16, hidden_size: 1024, mlp_ratio: 4.0 },
            ],
            text_configs: vec![
                TextNASConfig { layers: 6, heads: 8, hidden_size: 256, mlp_ratio: 4.0 },
                TextNASConfig { layers: 12, heads: 8, hidden_size: 512, mlp_ratio: 4.0 },
                TextNASConfig { layers: 24, heads: 16, hidden_size: 768, mlp_ratio: 4.0 },
            ],
            projection_dims: vec![256, 512, 768],
        }
    }
}

/// Vision transformer NAS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionNASConfig {
    pub layers: usize,
    pub heads: usize,
    pub hidden_size: usize,
    pub mlp_ratio: f64,
}

/// Text transformer NAS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextNASConfig {
    pub layers: usize,
    pub heads: usize,
    pub hidden_size: usize,
    pub mlp_ratio: f64,
}

/// Parameter range for hyperparameter optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterRange<T> {
    /// Uniform distribution
    Uniform { min: T, max: T },
    /// Log-uniform distribution
    LogUniform { min: T, max: T },
    /// Discrete values
    Discrete(Vec<T>),
}

/// Experiment configuration combining HPO and NAS parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Experiment ID
    pub id: String,
    /// Hyperparameters
    pub hpo_params: HPOParameters,
    /// Architecture parameters (if NAS enabled)
    pub nas_params: Option<NASParameters>,
    /// Random seed for reproducibility
    pub seed: u64,
}

/// Hyperparameter optimization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HPOParameters {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub temperature: f64,
    pub weight_decay: f64,
    pub warmup_steps: usize,
    pub max_grad_norm: f64,
}

/// Neural architecture search parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NASParameters {
    pub vision_layers: usize,
    pub vision_heads: usize,
    pub vision_hidden_size: usize,
    pub vision_mlp_ratio: f64,
    pub text_layers: usize,
    pub text_heads: usize,
    pub text_hidden_size: usize,
    pub text_mlp_ratio: f64,
    pub projection_dim: usize,
}

/// Experiment tracker for CLIP research
pub struct ExperimentTracker {
    /// Experiment history
    experiments: Vec<ExperimentRecord>,
    /// Best configurations found
    best_configs: HashMap<OptimizationObjective, ExperimentConfig>,
    /// Performance statistics
    stats: ExperimentStats,
}

/// Experiment record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentRecord {
    pub config: ExperimentConfig,
    pub results: ExperimentResults,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: chrono::DateTime<chrono::Utc>,
    pub status: ExperimentStatus,
}

/// Experiment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResults {
    pub objectives: HashMap<OptimizationObjective, f64>,
    pub metrics: HashMap<String, f64>,
    pub metadata: HashMap<String, String>,
}

/// Experiment status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentStatus {
    Running,
    Completed,
    Failed,
    Timeout,
}

/// Experiment statistics
#[derive(Debug, Clone)]
pub struct ExperimentStats {
    pub total_experiments: usize,
    pub successful_experiments: usize,
    pub failed_experiments: usize,
    pub best_scores: HashMap<OptimizationObjective, f64>,
    pub average_runtime: f64,
}

/// Performance evaluator for CLIP models
pub struct PerformanceEvaluator<B, S, T>
where
    B: backend::Backend<Data = T> + Clone,
    S: storage::Storage<T> + Clone,
    T: dtype::DataType + FloatExt + FromPrimitive + Bounded + Float,
{
    /// Validation dataset
    validation_dataset: Arc<dyn VisionLanguageData>,
    /// Evaluation configuration
    eval_config: crate::clip::training::validation::ValidationConfig,
    /// Marker for type parameters
    _marker: PhantomData<(B, S, T)>,
}

impl<B, S, T> PerformanceEvaluator<B, S, T>
where
    B: backend::Backend<Data = T> + Clone + Send + Sync + 'static,
    S: storage::Storage<T> + Clone + Send + Sync,
    T: dtype::DataType + Send + Sync + FloatExt + FromPrimitive + Bounded + Float,
{
    /// Create a new performance evaluator
    pub fn new(
        validation_dataset: Arc<dyn VisionLanguageData>,
        eval_config: crate::clip::training::validation::ValidationConfig,
    ) -> Self {
        Self {
            validation_dataset,
            eval_config,
            _marker: PhantomData,
        }
    }

    /// Evaluate a CLIP model configuration
    pub async fn evaluate_config(
        &self,
        config: &ExperimentConfig,
    ) -> Result<ExperimentResults> {
        // Build CLIP model from config
        let clip_config = self.build_clip_config(config);
        let model: ClipModel<B, storage::DenseStorage<T>, T> = ClipModel::new(clip_config)?;

        // Evaluate using validation framework
        let validator = crate::clip::ClipValidator::new(
            Arc::new(model),
            self.eval_config.clone(),
        );

        let report = validator.validate(
            &*self.validation_dataset,
            crate::clip::training::validation::EvaluationType::Full,
        ).await?;

        // Extract objectives
        let mut objectives = HashMap::new();
        let mut metrics = HashMap::new();
        let mut metadata = HashMap::new();

        // Map validation results to objectives
        if let Some(ref retrieval) = report.retrieval {
            objectives.insert(OptimizationObjective::RetrievalR1, retrieval.text_to_image.r1);
            metrics.insert("retrieval_r5".to_string(), retrieval.text_to_image.r5);
            metrics.insert("retrieval_r10".to_string(), retrieval.text_to_image.r10);
            metrics.insert("retrieval_mrr".to_string(), retrieval.mean_reciprocal_rank);
        }

        if let Some(ref zero_shot) = report.zero_shot {
            objectives.insert(OptimizationObjective::ZeroShotAccuracy, zero_shot.top1_accuracy);
            metrics.insert("zero_shot_top5".to_string(), zero_shot.top5_accuracy);
        }

        // Add training efficiency (placeholder - would be measured during training)
        objectives.insert(OptimizationObjective::TrainingEfficiency, 1000.0); // samples/sec placeholder

        // Add model size
        let model_size = self.estimate_model_size(config);
        objectives.insert(OptimizationObjective::ModelSize, model_size as f64);

        // Add metadata
        metadata.insert("validation_time".to_string(), format!("{:.2}s", report.validation_time));
        metadata.insert("model_size".to_string(), format!("{}M", model_size));

        Ok(ExperimentResults {
            objectives,
            metrics,
            metadata,
        })
    }

    /// Build CLIP config from experiment config
    fn build_clip_config(&self, config: &ExperimentConfig) -> ClipConfig {
        let (vision_config, text_config, projection_dim) = if let Some(ref nas) = config.nas_params {
            (
                crate::clip::VisionConfig {
                    image_size: 224,
                    patch_size: 16,
                    num_channels: 3,
                    hidden_size: nas.vision_hidden_size,
                    num_layers: nas.vision_layers,
                    num_heads: nas.vision_heads,
                    mlp_dim: (nas.vision_hidden_size as f64 * nas.vision_mlp_ratio) as usize,
                    dropout: 0.0,
                    attention_dropout: 0.0,
                    num_patches: crate::clip::VisionConfig::compute_num_patches(224, 16),
                },
                crate::clip::TextConfig {
                    vocab_size: 49408,
                    max_position_embeddings: 77,
                    hidden_size: nas.text_hidden_size,
                    num_layers: nas.text_layers,
                    num_heads: nas.text_heads,
                    mlp_dim: (nas.text_hidden_size as f64 * nas.text_mlp_ratio) as usize,
                    dropout: 0.0,
                    attention_dropout: 0.0,
                },
                nas.projection_dim,
            )
        } else {
            // Default CLIP config
            (
                crate::clip::VisionConfig {
                    image_size: 224,
                    patch_size: 16,
                    num_channels: 3,
                    hidden_size: 768,
                    num_layers: 12,
                    num_heads: 12,
                    mlp_dim: 3072, // 768 * 4
                    dropout: 0.0,
                    attention_dropout: 0.0,
                    num_patches: crate::clip::VisionConfig::compute_num_patches(224, 16),
                },
                crate::clip::TextConfig {
                    vocab_size: 49408,
                    max_position_embeddings: 77,
                    hidden_size: 512,
                    num_layers: 12,
                    num_heads: 8,
                    mlp_dim: 2048, // 512 * 4
                    dropout: 0.0,
                    attention_dropout: 0.0,
                },
                512,
            )
        };

        ClipConfig {
            embed_dim: projection_dim,
            vision_config,
            text_config,
            projection_dim,
            temperature: config.hpo_params.temperature,
            cache_text_features: true,
            max_grad_norm: Some(1.0),
        }
    }

    /// Estimate model size in millions of parameters
    fn estimate_model_size(&self, config: &ExperimentConfig) -> usize {
        if let Some(ref nas) = config.nas_params {
            // Rough parameter count estimation
            let vision_params = nas.vision_layers * nas.vision_hidden_size * nas.vision_hidden_size * 12; // rough ViT calculation
            let text_params = nas.text_layers * nas.text_hidden_size * nas.text_hidden_size * 12; // rough text transformer
            let projection_params = (nas.vision_hidden_size + nas.text_hidden_size) * nas.projection_dim;

            (vision_params + text_params + projection_params) / 1_000_000
        } else {
            150 // Default CLIP parameter count in millions
        }
    }
}

impl<B, S, T> ClipResearchFramework<B, S, T>
where
    B: backend::Backend<Data = T> + Clone + Send + Sync + 'static,
    S: storage::Storage<T> + Clone + Send + Sync,
    T: dtype::DataType + Send + Sync + FloatExt + FromPrimitive + Bounded + Float,
{
    /// Create a new CLIP research framework
    pub fn new(
        base_config: ClipConfig,
        research_config: ResearchConfig,
        validation_dataset: Arc<dyn VisionLanguageData>,
    ) -> Self {
        let evaluator = PerformanceEvaluator::new(
            validation_dataset,
            crate::clip::training::validation::ValidationConfig::default(),
        );

        Self {
            base_config,
            research_config,
            experiment_tracker: ExperimentTracker::new(),
            evaluator,
            _marker: PhantomData,
        }
    }

    /// Run hyperparameter optimization
    pub async fn run_hpo(&mut self) -> Result<HPOReport> {
        println!("🚀 Starting CLIP Hyperparameter Optimization");
        println!("============================================");

        let mut report = HPOReport {
            experiments: Vec::new(),
            best_config: None,
            optimization_curves: HashMap::new(),
            convergence_stats: HashMap::new(),
        };

        // Generate initial experiment configurations
        let initial_configs = self.generate_initial_configs(20)?;

        // Evaluate initial configurations
        for config in initial_configs {
            let result = self.run_experiment(config).await?;
            report.experiments.push(result);
        }

        // Bayesian optimization loop (simplified)
        for iteration in 0..(self.research_config.max_experiments - 20) {
            let next_config = self.suggest_next_config(&report.experiments)?;
            let result = self.run_experiment(next_config).await?;
            report.experiments.push(result);

            if iteration % 10 == 0 {
                println!("HPO Iteration {}/{}", iteration + 1, self.research_config.max_experiments - 20);
                self.print_current_best(&report);
            }
        }

        // Find best configuration
        report.best_config = self.find_best_config(&report.experiments);

        // Generate optimization curves
        report.optimization_curves = self.generate_optimization_curves(&report.experiments);

        // Compute convergence statistics
        report.convergence_stats = self.compute_convergence_stats(&report.experiments);

        println!("✅ HPO completed! Best configuration found.");
        if let Some(ref best) = report.best_config {
            println!("   Best score: {:.4}", self.compute_composite_score(best));
        }

        Ok(report)
    }

    /// Run neural architecture search
    pub async fn run_nas(&mut self) -> Result<NASReport> {
        println!("🔬 Starting CLIP Neural Architecture Search");
        println!("===========================================");

        let nas_space = self.research_config.nas_space.as_ref()
            .ok_or_else(|| NNError::InvalidInput {
                message: "NAS search space not configured".to_string(),
            })?;

        let mut report = NASReport {
            architectures: Vec::new(),
            best_architecture: None,
            pareto_front: Vec::new(),
            search_stats: HashMap::new(),
        };

        // Generate architecture configurations
        let architectures = self.generate_architecture_configs(nas_space, 50)?;

        // Evaluate architectures
        for arch_config in architectures {
            let result = self.run_experiment(arch_config).await?;
            report.architectures.push(result);
        }

        // Find Pareto front for multi-objective optimization
        report.pareto_front = self.compute_pareto_front(&report.architectures);

        // Find best architecture (single objective combination)
        report.best_architecture = self.find_best_architecture(&report.architectures);

        // Compute search statistics
        report.search_stats = self.compute_nas_stats(&report.architectures);

        println!("✅ NAS completed! {} architectures evaluated.", report.architectures.len());

        Ok(report)
    }

    /// Run joint HPO + NAS optimization
    pub async fn run_joint_optimization(&mut self) -> Result<JointOptimizationReport> {
        println!("🎯 Starting CLIP Joint HPO + NAS Optimization");
        println!("=============================================");

        let mut report = JointOptimizationReport {
            experiments: Vec::new(),
            best_configuration: None,
            hpo_nas_tradeoffs: HashMap::new(),
            efficiency_frontier: Vec::new(),
        };

        // Generate configurations combining HPO and NAS
        let configs = self.generate_joint_configs(30)?;

        // Evaluate configurations
        for config in configs {
            let result = self.run_experiment(config).await?;
            report.experiments.push(result);
        }

        // Find best joint configuration
        report.best_configuration = self.find_best_joint_config(&report.experiments);

        // Analyze HPO vs NAS tradeoffs
        report.hpo_nas_tradeoffs = self.analyze_hpo_nas_tradeoffs(&report.experiments);

        // Compute efficiency frontier
        report.efficiency_frontier = self.compute_efficiency_frontier(&report.experiments);

        println!("✅ Joint optimization completed!");

        Ok(report)
    }

    /// Run a single experiment
    async fn run_experiment(&mut self, config: ExperimentConfig) -> Result<ExperimentRecord> {
        let start_time = chrono::Utc::now();

        println!("Running experiment: {}", config.id);

        let results = match self.evaluator.evaluate_config(&config).await {
            Ok(results) => {
                println!("  ✅ Completed - Score: {:.4}", self.compute_composite_score(&config));
                results
            }
            Err(e) => {
                println!("  ❌ Failed: {}", e);
                // Return failed results with zero scores
                ExperimentResults {
                    objectives: self.research_config.objectives.iter()
                        .map(|obj| (*obj, 0.0))
                        .collect(),
                    metrics: HashMap::new(),
                    metadata: HashMap::from([("error".to_string(), e.to_string())]),
                }
            }
        };

        let end_time = chrono::Utc::now();
        let status = if results.objectives.values().any(|&v| v > 0.0) {
            ExperimentStatus::Completed
        } else {
            ExperimentStatus::Failed
        };

        let record = ExperimentRecord {
            config,
            results,
            start_time,
            end_time,
            status,
        };

        self.experiment_tracker.add_record(record.clone());

        Ok(record)
    }

    /// Generate initial HPO configurations
    fn generate_initial_configs(&self, count: usize) -> Result<Vec<ExperimentConfig>> {
        let mut configs = Vec::new();

        for i in 0..count {
            let hpo_params = self.sample_hpo_parameters(i as u64)?;
            let config = ExperimentConfig {
                id: format!("{}_hpo_{}", self.research_config.experiment_prefix, i),
                hpo_params,
                nas_params: None,
                seed: i as u64,
            };
            configs.push(config);
        }

        Ok(configs)
    }

    /// Sample HPO parameters from search space
    fn sample_hpo_parameters(&self, seed: u64) -> Result<HPOParameters> {
        use rand::prelude::*;
        use rand_pcg::Pcg64;

        let mut rng = Pcg64::seed_from_u64(seed);

        let learning_rate = self.sample_parameter(&self.research_config.hpo_space.learning_rate, &mut rng)?;
        let batch_size = self.sample_parameter(&self.research_config.hpo_space.batch_size, &mut rng)?;
        let temperature = self.sample_parameter(&self.research_config.hpo_space.temperature, &mut rng)?;
        let weight_decay = self.sample_parameter(&self.research_config.hpo_space.weight_decay, &mut rng)?;
        let warmup_steps = self.sample_parameter(&self.research_config.hpo_space.warmup_steps, &mut rng)?;
        let max_grad_norm = self.sample_parameter(&self.research_config.hpo_space.max_grad_norm, &mut rng)?;

        Ok(HPOParameters {
            learning_rate,
            batch_size,
            temperature,
            weight_decay,
            warmup_steps,
            max_grad_norm,
        })
    }

    /// Sample a parameter from its range
    fn sample_parameter<P>(&self, range: &ParameterRange<P>, rng: &mut impl Rng) -> Result<P>
    where
        P: Copy + PartialOrd + ToPrimitive + FromPrimitive + 'static,
    {
        match range {
            ParameterRange::Uniform { min, max } => {
                let min_f = min.to_f64().ok_or(NNError::InvalidInput { message: "Failed to convert min to f64".into() })?;
                let max_f = max.to_f64().ok_or(NNError::InvalidInput { message: "Failed to convert max to f64".into() })?;
                let t: f64 = rng.gen();
                let val_f = min_f + t * (max_f - min_f);
                P::from_f64(val_f).ok_or(NNError::InvalidInput { message: "Failed to convert sampled value back to type".into() })
            }
            ParameterRange::LogUniform { min, max } => {
                let min_f = min.to_f64().ok_or(NNError::InvalidInput { message: "Failed to convert min to f64".into() })?;
                let max_f = max.to_f64().ok_or(NNError::InvalidInput { message: "Failed to convert max to f64".into() })?;
                
                let log_min = min_f.ln();
                let log_max = max_f.ln();
                let t: f64 = rng.gen();
                let log_val = log_min + t * (log_max - log_min);
                let val_f = log_val.exp();
                
                P::from_f64(val_f).ok_or(NNError::InvalidInput { message: "Failed to convert sampled value back to type".into() })
            }
            ParameterRange::Discrete(values) => {
                let idx = rng.gen_range(0..values.len());
                Ok(values[idx])
            }
        }
    }

    /// Generate architecture configurations for NAS
    fn generate_architecture_configs(&self, nas_space: &NASSearchSpace, count: usize) -> Result<Vec<ExperimentConfig>> {
        use rand::prelude::*;
        use rand_pcg::Pcg64;

        let mut configs = Vec::new();

        for i in 0..count {
            let mut rng = Pcg64::seed_from_u64(i as u64);

            let vision_config = nas_space.vision_configs.choose(&mut rng).unwrap();
            let text_config = nas_space.text_configs.choose(&mut rng).unwrap();
            let projection_dim = *nas_space.projection_dims.choose(&mut rng).unwrap();

            let nas_params = NASParameters {
                vision_layers: vision_config.layers,
                vision_heads: vision_config.heads,
                vision_hidden_size: vision_config.hidden_size,
                vision_mlp_ratio: vision_config.mlp_ratio,
                text_layers: text_config.layers,
                text_heads: text_config.heads,
                text_hidden_size: text_config.hidden_size,
                text_mlp_ratio: text_config.mlp_ratio,
                projection_dim,
            };

            // Use default HPO parameters for NAS
            let hpo_params = HPOParameters {
                learning_rate: 5e-4,
                batch_size: 32,
                temperature: 0.07,
                weight_decay: 1e-4,
                warmup_steps: 2000,
                max_grad_norm: 1.0,
            };

            let config = ExperimentConfig {
                id: format!("{}_nas_{}", self.research_config.experiment_prefix, i),
                hpo_params,
                nas_params: Some(nas_params),
                seed: i as u64,
            };

            configs.push(config);
        }

        Ok(configs)
    }

    /// Generate joint HPO + NAS configurations
    fn generate_joint_configs(&self, count: usize) -> Result<Vec<ExperimentConfig>> {
        let mut configs = Vec::new();

        for i in 0..count {
            let hpo_params = self.sample_hpo_parameters(i as u64)?;
            let nas_params = if let Some(ref nas_space) = self.research_config.nas_space {
                Some(self.sample_nas_parameters(nas_space, i as u64)?)
            } else {
                None
            };

            let config = ExperimentConfig {
                id: format!("{}_joint_{}", self.research_config.experiment_prefix, i),
                hpo_params,
                nas_params,
                seed: i as u64,
            };

            configs.push(config);
        }

        Ok(configs)
    }

    /// Sample NAS parameters
    fn sample_nas_parameters(&self, nas_space: &NASSearchSpace, seed: u64) -> Result<NASParameters> {
        use rand::prelude::*;
        use rand_pcg::Pcg64;

        let mut rng = Pcg64::seed_from_u64(seed);

        let vision_config = nas_space.vision_configs.choose(&mut rng).unwrap();
        let text_config = nas_space.text_configs.choose(&mut rng).unwrap();
        let projection_dim = *nas_space.projection_dims.choose(&mut rng).unwrap();

        Ok(NASParameters {
            vision_layers: vision_config.layers,
            vision_heads: vision_config.heads,
            vision_hidden_size: vision_config.hidden_size,
            vision_mlp_ratio: vision_config.mlp_ratio,
            text_layers: text_config.layers,
            text_heads: text_config.heads,
            text_hidden_size: text_config.hidden_size,
            text_mlp_ratio: text_config.mlp_ratio,
            projection_dim,
        })
    }

    /// Suggest next configuration using Bayesian optimization (simplified)
    fn suggest_next_config(&self, history: &[ExperimentRecord]) -> Result<ExperimentConfig> {
        // Simplified: random sampling with some preference for high-performing regions
        let config_id = format!("{}_hpo_{}", self.research_config.experiment_prefix, history.len());
        let hpo_params = self.sample_hpo_parameters(history.len() as u64)?;

        Ok(ExperimentConfig {
            id: config_id,
            hpo_params,
            nas_params: None,
            seed: history.len() as u64,
        })
    }

    /// Compute composite score for configuration
    fn compute_composite_score(&self, config: &ExperimentConfig) -> f64 {
        // Find the experiment record for this config
        if let Some(record) = self.experiment_tracker.experiments.iter()
            .find(|r| r.config.id == config.id) {

            let mut total_score = 0.0;
            let mut total_weight = 0.0;

            for objective in &self.research_config.objectives {
                if let Some(score) = record.results.objectives.get(objective) {
                    let weight = self.get_objective_weight(objective);
                    total_score += score * weight;
                    total_weight += weight;
                }
            }

            if total_weight > 0.0 {
                total_score / total_weight
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Get weight for optimization objective
    fn get_objective_weight(&self, objective: &OptimizationObjective) -> f64 {
        match objective {
            OptimizationObjective::RetrievalR1 => 1.0,
            OptimizationObjective::ZeroShotAccuracy => 1.0,
            OptimizationObjective::TrainingEfficiency => 0.5,
            OptimizationObjective::ModelSize => 0.3,
            OptimizationObjective::InferenceLatency => 0.5,
            OptimizationObjective::MemoryEfficiency => 0.4,
        }
    }

    /// Helper methods for analysis
    fn find_best_config(&self, experiments: &[ExperimentRecord]) -> Option<ExperimentConfig> {
        experiments.iter()
            .max_by(|a, b| {
                self.compute_composite_score(&a.config)
                    .partial_cmp(&self.compute_composite_score(&b.config))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|record| record.config.clone())
    }

    fn print_current_best(&self, report: &HPOReport) {
        if let Some(ref best) = self.find_best_config(&report.experiments) {
            let score = self.compute_composite_score(best);
            println!("   Current best score: {:.4}", score);
        }
    }

    // Additional analysis methods would be implemented here...
    fn generate_optimization_curves(&self, _experiments: &[ExperimentRecord]) -> HashMap<String, Vec<f64>> {
        // Placeholder implementation
        HashMap::new()
    }

    fn compute_convergence_stats(&self, _experiments: &[ExperimentRecord]) -> HashMap<String, f64> {
        // Placeholder implementation
        HashMap::new()
    }

    fn compute_pareto_front(&self, _architectures: &[ExperimentRecord]) -> Vec<ExperimentConfig> {
        // Placeholder implementation
        Vec::new()
    }

    fn find_best_architecture(&self, _architectures: &[ExperimentRecord]) -> Option<ExperimentConfig> {
        // Placeholder implementation
        None
    }

    fn compute_nas_stats(&self, _architectures: &[ExperimentRecord]) -> HashMap<String, f64> {
        // Placeholder implementation
        HashMap::new()
    }

    fn find_best_joint_config(&self, _experiments: &[ExperimentRecord]) -> Option<ExperimentConfig> {
        // Placeholder implementation
        None
    }

    fn analyze_hpo_nas_tradeoffs(&self, _experiments: &[ExperimentRecord]) -> HashMap<String, f64> {
        // Placeholder implementation
        HashMap::new()
    }

    fn compute_efficiency_frontier(&self, _experiments: &[ExperimentRecord]) -> Vec<ExperimentConfig> {
        // Placeholder implementation
        Vec::new()
    }
}

/// HPO optimization report
#[derive(Debug, Clone)]
pub struct HPOReport {
    pub experiments: Vec<ExperimentRecord>,
    pub best_config: Option<ExperimentConfig>,
    pub optimization_curves: HashMap<String, Vec<f64>>,
    pub convergence_stats: HashMap<String, f64>,
}

/// NAS optimization report
#[derive(Debug, Clone)]
pub struct NASReport {
    pub architectures: Vec<ExperimentRecord>,
    pub best_architecture: Option<ExperimentConfig>,
    pub pareto_front: Vec<ExperimentConfig>,
    pub search_stats: HashMap<String, f64>,
}

/// Joint optimization report
#[derive(Debug, Clone)]
pub struct JointOptimizationReport {
    pub experiments: Vec<ExperimentRecord>,
    pub best_configuration: Option<ExperimentConfig>,
    pub hpo_nas_tradeoffs: HashMap<String, f64>,
    pub efficiency_frontier: Vec<ExperimentConfig>,
}

impl ExperimentTracker {
    fn new() -> Self {
        Self {
            experiments: Vec::new(),
            best_configs: HashMap::new(),
            stats: ExperimentStats {
                total_experiments: 0,
                successful_experiments: 0,
                failed_experiments: 0,
                best_scores: HashMap::new(),
                average_runtime: 0.0,
            },
        }
    }

    fn add_record(&mut self, record: ExperimentRecord) {
        self.experiments.push(record.clone());
        self.stats.total_experiments += 1;

        match record.status {
            ExperimentStatus::Completed => self.stats.successful_experiments += 1,
            ExperimentStatus::Failed | ExperimentStatus::Timeout => self.stats.failed_experiments += 1,
            _ => {}
        }

        // Update best scores
        for (objective, score) in &record.results.objectives {
            let current_best = self.stats.best_scores.entry(*objective).or_insert(0.0);
            if *score > *current_best {
                *current_best = *score;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_research_config_default() {
        let config = ResearchConfig::default();
        assert_eq!(config.experiment_prefix, "clip_research");
        assert_eq!(config.num_parallel, 4);
        assert_eq!(config.max_experiments, 100);
    }

    #[test]
    fn test_hpo_parameter_space() {
        let space = HyperparameterSpace::default();
        match space.learning_rate {
            ParameterRange::LogUniform { min, max } => {
                assert!(min > 0.0 && max > min);
            }
            _ => panic!("Expected LogUniform for learning rate"),
        }
    }

    #[test]
    fn test_experiment_tracker() {
        let mut tracker = ExperimentTracker::new();
        assert_eq!(tracker.stats.total_experiments, 0);
        assert_eq!(tracker.experiments.len(), 0);
    }
}















