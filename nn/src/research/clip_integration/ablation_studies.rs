//! CLIP Ablation Studies
//!
//! Systematic ablation studies for CLIP components including
//! architecture variants, training strategies, and data augmentations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Ablation study configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationStudy {
    /// Study name
    pub name: String,
    /// Study description
    pub description: String,
    /// Base configuration to ablate from
    pub base_config: AblationConfig,
    /// Components to ablate
    pub ablations: Vec<AblationComponent>,
    /// Metrics to track
    pub metrics: Vec<String>,
}

/// Ablation configuration (subset of training config)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AblationConfig {
    /// Vision encoder configuration
    pub vision_config: VisionAblationConfig,
    /// Text encoder configuration
    pub text_config: TextAblationConfig,
    /// Training configuration
    pub training_config: TrainingAblationConfig,
}

/// Vision encoder ablation options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionAblationConfig {
    /// Model architecture variant
    pub architecture: VisionArchitecture,
    /// Patch size for ViT
    pub patch_size: Option<usize>,
    /// Hidden dimension
    pub hidden_dim: Option<usize>,
    /// Number of layers
    pub num_layers: Option<usize>,
    /// Number of attention heads
    pub num_heads: Option<usize>,
}

/// Text encoder ablation options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextAblationConfig {
    /// Model architecture variant
    pub architecture: TextArchitecture,
    /// Vocabulary size
    pub vocab_size: Option<usize>,
    /// Hidden dimension
    pub hidden_dim: Option<usize>,
    /// Number of layers
    pub num_layers: Option<usize>,
    /// Maximum sequence length
    pub max_seq_length: Option<usize>,
}

/// Training ablation options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingAblationConfig {
    /// Learning rate schedule
    pub lr_schedule: Option<LrSchedule>,
    /// Temperature parameter
    pub temperature: Option<f64>,
    /// Batch size
    pub batch_size: Option<usize>,
    /// Gradient clipping
    pub grad_clip: Option<f64>,
    /// Weight decay
    pub weight_decay: Option<f64>,
}

/// Vision architecture variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisionArchitecture {
    /// Vision Transformer
    ViT,
    /// ResNet
    ResNet,
    /// Convolutional Neural Network
    CNN,
    /// Remove vision encoder (text-only)
    None,
}

/// Text architecture variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextArchitecture {
    /// Transformer
    Transformer,
    /// LSTM
    LSTM,
    /// Bag of Words
    BoW,
    /// Remove text encoder (vision-only)
    None,
}

/// Learning rate schedules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LrSchedule {
    /// Constant learning rate
    Constant,
    /// Cosine annealing
    Cosine,
    /// Linear decay
    Linear,
    /// Exponential decay
    Exponential,
    /// Warmup + constant
    WarmupConstant,
    /// Warmup + cosine
    WarmupCosine,
}

/// Individual ablation component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationComponent {
    /// Component name
    pub name: String,
    /// Component description
    pub description: String,
    /// Ablation type
    pub ablation_type: AblationType,
    /// Parameters to modify
    pub parameters: HashMap<String, AblationValue>,
}

/// Ablation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AblationType {
    /// Remove component entirely
    Remove,
    /// Modify parameter value
    Modify,
    /// Replace with alternative implementation
    Replace,
    /// Add new component/feature
    Add,
}

/// Ablation parameter values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AblationValue {
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// String value
    String(String),
    /// Boolean value
    Bool(bool),
    /// Null/None value
    Null,
}

impl AblationStudy {
    /// Create standard CLIP ablation study
    pub fn standard_clip_ablation() -> Self {
        Self {
            name: "CLIP_Standard_Ablation".to_string(),
            description: "Standard ablation study for CLIP components".to_string(),
            base_config: AblationConfig::default(),
            ablations: vec![
                AblationComponent {
                    name: "vision_encoder".to_string(),
                    description: "Remove vision encoder for text-only CLIP".to_string(),
                    ablation_type: AblationType::Remove,
                    parameters: HashMap::from([(
                        "vision_config.architecture".to_string(),
                        AblationValue::String("None".to_string()),
                    )]),
                },
                AblationComponent {
                    name: "text_encoder".to_string(),
                    description: "Remove text encoder for vision-only CLIP".to_string(),
                    ablation_type: AblationType::Remove,
                    parameters: HashMap::from([(
                        "text_config.architecture".to_string(),
                        AblationValue::String("None".to_string()),
                    )]),
                },
                AblationComponent {
                    name: "temperature".to_string(),
                    description: "Ablate temperature parameter".to_string(),
                    ablation_type: AblationType::Modify,
                    parameters: HashMap::from([
                        (
                            "training_config.temperature".to_string(),
                            AblationValue::Float(1.0),
                        ),
                        (
                            "training_config.temperature".to_string(),
                            AblationValue::Float(0.1),
                        ),
                        (
                            "training_config.temperature".to_string(),
                            AblationValue::Float(10.0),
                        ),
                    ]),
                },
                AblationComponent {
                    name: "lr_schedule".to_string(),
                    description: "Compare different learning rate schedules".to_string(),
                    ablation_type: AblationType::Modify,
                    parameters: HashMap::from([
                        (
                            "training_config.lr_schedule".to_string(),
                            AblationValue::String("Constant".to_string()),
                        ),
                        (
                            "training_config.lr_schedule".to_string(),
                            AblationValue::String("Cosine".to_string()),
                        ),
                        (
                            "training_config.lr_schedule".to_string(),
                            AblationValue::String("WarmupCosine".to_string()),
                        ),
                    ]),
                },
            ],
            metrics: vec![
                "validation_loss".to_string(),
                "zero_shot_accuracy".to_string(),
                "retrieval_r@1".to_string(),
                "retrieval_r@5".to_string(),
                "retrieval_r@10".to_string(),
            ],
        }
    }

    /// Create architecture ablation study
    pub fn architecture_ablation() -> Self {
        Self {
            name: "CLIP_Architecture_Ablation".to_string(),
            description: "Ablation study for different CLIP architectures".to_string(),
            base_config: AblationConfig::default(),
            ablations: vec![
                AblationComponent {
                    name: "vit_patch_size".to_string(),
                    description: "Different ViT patch sizes".to_string(),
                    ablation_type: AblationType::Modify,
                    parameters: HashMap::from([
                        (
                            "vision_config.patch_size".to_string(),
                            AblationValue::Int(16),
                        ),
                        (
                            "vision_config.patch_size".to_string(),
                            AblationValue::Int(32),
                        ),
                    ]),
                },
                AblationComponent {
                    name: "vit_layers".to_string(),
                    description: "Different numbers of ViT layers".to_string(),
                    ablation_type: AblationType::Modify,
                    parameters: HashMap::from([
                        (
                            "vision_config.num_layers".to_string(),
                            AblationValue::Int(6),
                        ),
                        (
                            "vision_config.num_layers".to_string(),
                            AblationValue::Int(12),
                        ),
                        (
                            "vision_config.num_layers".to_string(),
                            AblationValue::Int(24),
                        ),
                    ]),
                },
                AblationComponent {
                    name: "text_model_size".to_string(),
                    description: "Different text model sizes".to_string(),
                    ablation_type: AblationType::Modify,
                    parameters: HashMap::from([
                        (
                            "text_config.hidden_dim".to_string(),
                            AblationValue::Int(256),
                        ),
                        (
                            "text_config.hidden_dim".to_string(),
                            AblationValue::Int(512),
                        ),
                        (
                            "text_config.hidden_dim".to_string(),
                            AblationValue::Int(768),
                        ),
                    ]),
                },
            ],
            metrics: vec![
                "validation_loss".to_string(),
                "zero_shot_accuracy".to_string(),
                "training_time".to_string(),
                "memory_usage".to_string(),
            ],
        }
    }
}

impl Default for VisionAblationConfig {
    fn default() -> Self {
        Self {
            architecture: VisionArchitecture::ViT,
            patch_size: Some(16),
            hidden_dim: Some(768),
            num_layers: Some(12),
            num_heads: Some(12),
        }
    }
}

impl Default for TextAblationConfig {
    fn default() -> Self {
        Self {
            architecture: TextArchitecture::Transformer,
            vocab_size: Some(49408),
            hidden_dim: Some(512),
            num_layers: Some(12),
            max_seq_length: Some(77),
        }
    }
}

impl Default for TrainingAblationConfig {
    fn default() -> Self {
        Self {
            lr_schedule: Some(LrSchedule::WarmupCosine),
            temperature: Some(0.07),
            batch_size: Some(32),
            grad_clip: Some(1.0),
            weight_decay: Some(0.2),
        }
    }
}

/// Ablation study runner
pub struct AblationRunner {
    study: AblationStudy,
}

impl AblationRunner {
    /// Create ablation runner
    pub fn new(study: AblationStudy) -> Self {
        Self { study }
    }

    /// Get all ablation configurations
    pub fn get_ablation_configs(&self) -> Vec<AblationConfig> {
        let mut configs = vec![self.study.base_config.clone()];

        for ablation in &self.study.ablations {
            let mut modified_config = self.study.base_config.clone();
            self.apply_ablation(&mut modified_config, ablation);
            configs.push(modified_config);
        }

        configs
    }

    /// Apply ablation to configuration
    fn apply_ablation(&self, config: &mut AblationConfig, ablation: &AblationComponent) {
        for (param_path, value) in &ablation.parameters {
            match param_path.as_str() {
                "vision_config.architecture" => {
                    if let AblationValue::String(arch) = value {
                        config.vision_config.architecture = match arch.as_str() {
                            "ViT" => VisionArchitecture::ViT,
                            "ResNet" => VisionArchitecture::ResNet,
                            "CNN" => VisionArchitecture::CNN,
                            "None" => VisionArchitecture::None,
                            _ => VisionArchitecture::ViT,
                        };
                    }
                }
                "text_config.architecture" => {
                    if let AblationValue::String(arch) = value {
                        config.text_config.architecture = match arch.as_str() {
                            "Transformer" => TextArchitecture::Transformer,
                            "LSTM" => TextArchitecture::LSTM,
                            "BoW" => TextArchitecture::BoW,
                            "None" => TextArchitecture::None,
                            _ => TextArchitecture::Transformer,
                        };
                    }
                }
                "training_config.temperature" => {
                    if let AblationValue::Float(temp) = value {
                        config.training_config.temperature = Some(*temp);
                    }
                }
                "vision_config.patch_size" => {
                    if let AblationValue::Int(size) = value {
                        config.vision_config.patch_size = Some(*size as usize);
                    }
                }
                "vision_config.num_layers" => {
                    if let AblationValue::Int(layers) = value {
                        config.vision_config.num_layers = Some(*layers as usize);
                    }
                }
                "text_config.hidden_dim" => {
                    if let AblationValue::Int(dim) = value {
                        config.text_config.hidden_dim = Some(*dim as usize);
                    }
                }
                _ => {} // Ignore unknown parameters
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_ablation_study() {
        let study = AblationStudy::standard_clip_ablation();
        assert_eq!(study.name, "CLIP_Standard_Ablation");
        assert!(!study.ablations.is_empty());
        assert!(!study.metrics.is_empty());
    }

    #[test]
    fn test_architecture_ablation_study() {
        let study = AblationStudy::architecture_ablation();
        assert_eq!(study.name, "CLIP_Architecture_Ablation");
        assert!(!study.ablations.is_empty());
    }

    #[test]
    fn test_ablation_runner_configs() {
        let study = AblationStudy::standard_clip_ablation();
        let runner = AblationRunner::new(study);
        let configs = runner.get_ablation_configs();

        // Should have base config + ablation configs
        assert!(!configs.is_empty());
        assert!(configs.len() > 1);
    }

    #[test]
    fn test_ablation_config_defaults() {
        let config = AblationConfig::default();
        assert!(matches!(
            config.vision_config.architecture,
            VisionArchitecture::ViT
        ));
        assert!(matches!(
            config.text_config.architecture,
            TextArchitecture::Transformer
        ));
        assert_eq!(config.training_config.temperature, Some(0.07));
    }
}
