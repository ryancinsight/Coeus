/// Training state for checkpointing and resuming
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TrainingState {
    /// Current epoch
    pub epoch: usize,
    /// Current step
    pub step: usize,
    /// Best validation loss
    pub best_val_loss: f64,
    /// Steps since best validation loss (for early stopping)
    pub steps_since_best: usize,
    /// Total training samples processed
    pub total_samples: usize,
    /// Learning rate history
    pub lr_history: Vec<f64>,
    /// Loss history
    pub loss_history: Vec<f64>,
    /// Validation loss history
    pub val_loss_history: Vec<f64>,
    /// Model temperature history
    pub temperature_history: Vec<f64>,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            epoch: 0,
            step: 0,
            best_val_loss: f64::INFINITY,
            steps_since_best: 0,
            total_samples: 0,
            lr_history: Vec::new(),
            loss_history: Vec::new(),
            val_loss_history: Vec::new(),
            temperature_history: Vec::new(),
        }
    }
}
