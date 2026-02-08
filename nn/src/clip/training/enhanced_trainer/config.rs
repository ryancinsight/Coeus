use std::path::PathBuf;

/// Enhanced CLIP training configuration with optimizer settings
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnhancedClipTrainingConfig {
    /// Base CLIP training configuration
    pub base_config: crate::clip::training::trainer::ClipTrainingConfig,
    /// Accumulation steps for gradient accumulation
    pub gradient_accumulation_steps: usize,
    /// Early stopping patience (0 = disabled)
    pub early_stopping_patience: usize,
    /// Minimum learning rate (stop if LR drops below this)
    pub min_learning_rate: f64,
    /// Save best model only
    pub save_best_only: bool,
    /// Checkpoint directory
    pub checkpoint_dir: PathBuf,
    /// Resume from checkpoint path
    pub resume_from: Option<PathBuf>,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f64,
    /// Log gradients and loss scaling
    pub log_gradients: bool,
}

impl Default for EnhancedClipTrainingConfig {
    fn default() -> Self {
        Self {
            base_config: Default::default(),
            gradient_accumulation_steps: 1,
            early_stopping_patience: 10,
            min_learning_rate: 1e-7,
            save_best_only: true,
            checkpoint_dir: PathBuf::from("./checkpoints"),
            resume_from: None,
            max_grad_norm: 1.0,
            log_gradients: false,
        }
    }
}
