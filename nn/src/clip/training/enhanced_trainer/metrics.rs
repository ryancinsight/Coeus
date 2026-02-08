/// Training metrics structures
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    pub loss: f64,
    pub steps: usize,
    pub batches: usize,
}

#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    pub loss: f64,
    pub batches: usize,
}

#[derive(Debug, Clone)]
pub struct EnhancedTrainingReport {
    pub total_epochs: usize,
    pub total_steps: usize,
    pub best_validation_loss: f64,
    pub final_learning_rate: f64,
    pub total_training_time: f64,
    pub training_samples: usize,
    pub final_temperature: f64,
}
