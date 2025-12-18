//! Training Orchestration and Monitoring for Foundation Models
//!
//! This module provides comprehensive training orchestration including:
//! - Learning rate schedules and curriculum learning
//! - Training phases and multi-stage optimization
//! - Real-time monitoring and performance profiling
//! - Automated convergence detection and early stopping
//! - Checkpoint management and recovery

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::Arc;
use crate::Result;
use crate::monitoring::{TrainingMonitor, MonitoringReport};
use crate::distributed::DistributedCoordinator;

/// Training Orchestrator - coordinates the entire training process
#[derive(Debug)]
pub struct TrainingOrchestrator {
    /// Current training phase
    pub current_phase: TrainingPhase,
    /// Learning rate scheduler
    pub lr_scheduler: LearningRateScheduler,
    /// Curriculum learning manager
    pub curriculum: CurriculumLearningManager,
    /// Performance monitor
    pub monitor: TrainingMonitor,
    /// Early stopping detector
    pub early_stopping: EarlyStopping,
    /// Checkpoint manager
    pub checkpoint_manager: CheckpointManager,
    /// Training configuration
    pub config: TrainingConfig,
    /// Last step timestamp for throughput calculation
    last_step_time: Instant,
    /// Distributed coordinator (optional)
    pub distributed_coordinator: Option<Arc<DistributedCoordinator>>,
}

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub total_steps: usize,
    pub evaluation_steps: usize,
    pub save_steps: usize,
    pub log_steps: usize,
    pub max_grad_norm: Option<f64>,
    pub warmup_steps: usize,
    pub cooldown_steps: usize,
}

impl TrainingOrchestrator {
    /// Create new training orchestrator
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            current_phase: TrainingPhase::Warmup,
            lr_scheduler: LearningRateScheduler::new(),
            curriculum: CurriculumLearningManager::new(),
            monitor: TrainingMonitor::new(),
            early_stopping: EarlyStopping::new(),
            checkpoint_manager: CheckpointManager::new(),
            config,
            last_step_time: Instant::now(),
            distributed_coordinator: None,
        }
    }

    /// Execute a complete training step
    pub async fn training_step(
        &mut self,
        step: usize,
        loss: f64,
        metrics: &HashMap<String, f64>,
        gradients: &[f32]
    ) -> Result<TrainingAction> {
        // Calculate step metrics
        let step_duration = self.last_step_time.elapsed().as_secs_f64();
        let throughput = if step_duration > 0.0 { 1.0 / step_duration } else { 0.0 };
        self.last_step_time = Instant::now();

        // Update learning rate (needed for monitoring)
        let lr = self.lr_scheduler.get_lr(step);

        // Calculate gradient norm
        let grad_norm = gradients.iter().map(|x| (x * x) as f64).sum::<f64>().sqrt();

        // Update monitor
        let mut combined_metrics = metrics.clone();
        
        // Add distributed metrics if available
        if let Some(coordinator) = &self.distributed_coordinator {
            let state = coordinator.state.read().await;
            combined_metrics.insert("dist_load_balance".to_string(), state.sync_stats.load_balance_score());
            combined_metrics.insert("dist_comm_overhead".to_string(), state.sync_stats.communication_overhead);
            combined_metrics.insert("dist_avg_sync_ms".to_string(), state.sync_stats.average_sync_time_ms);
        }

        self.monitor.record_training_metrics(
            step,
            loss,
            lr,
            grad_norm,
            throughput,
            Some(combined_metrics)
        ).await?;

        // Update curriculum
        self.curriculum.update(step);
        self.curriculum.set_difficulty_level(step);

        // Check for phase transitions
        self.update_phase(step)?;

        // Apply gradient clipping if configured
        if let Some(max_norm) = self.config.max_grad_norm {
            self.apply_gradient_clipping(gradients, max_norm)?;
        }

        // Check early stopping
        if self.early_stopping.should_stop(loss, step) {
            return Ok(TrainingAction::Stop);
        }

        // Handle checkpoints
        if step % self.config.save_steps == 0 {
            self.checkpoint_manager.save_checkpoint(step, loss, &metrics).await?;
        }

        // Log progress
        if step % self.config.log_steps == 0 {
            self.log_progress(step, loss, lr, &metrics);
        }

        Ok(TrainingAction::Continue)
    }

    fn update_phase(&mut self, step: usize) -> Result<()> {
        match self.current_phase {
            TrainingPhase::Warmup => {
                if step >= self.config.warmup_steps {
                    self.current_phase = TrainingPhase::Training;
                    self.lr_scheduler.set_peak_lr();
                }
            },
            TrainingPhase::Training => {
                if step >= self.config.total_steps - self.config.cooldown_steps {
                    self.current_phase = TrainingPhase::Cooldown;
                }
            },
            TrainingPhase::Cooldown => {
                if step >= self.config.total_steps {
                    self.current_phase = TrainingPhase::Finished;
                }
            },
            TrainingPhase::Finished => {},
        }
        Ok(())
    }

    fn apply_gradient_clipping(&self, gradients: &[f32], max_norm: f64) -> Result<()> {
        // Calculate global norm
        let global_norm: f64 = gradients.iter()
            .map(|g| (g * g) as f64)
            .sum::<f64>()
            .sqrt();

        if global_norm > max_norm {
            // Clip gradients
            let _scale_factor = max_norm / global_norm;
            // Apply scaling in-place (would need mutable access in real implementation)
        }

        Ok(())
    }

    fn log_progress(&self, step: usize, loss: f64, lr: f64, metrics: &HashMap<String, f64>) {
        println!("Step {}/{} | Loss: {:.4} | LR: {:.6} | Phase: {:?}",
                 step, self.config.total_steps, loss, lr, self.current_phase);

        for (name, value) in metrics {
            println!("  {}: {:.4}", name, value);
        }
    }

    /// Get training report
    pub async fn get_training_report(&self) -> Result<TrainingReport> {
        let monitoring_report = self.monitor.generate_report().await?;
        let stats = &monitoring_report.training_metrics_summary;
        let sys_stats = &monitoring_report.system_metrics_summary;

        // Calculate training statistics from monitoring report
        let performance_stats = TrainingStatistics {
            total_steps: self.monitor.state.read().await.training_metrics.loss_values.len(),
            average_step_time: if stats.avg_throughput > 0.0 { 1.0 / stats.avg_throughput } else { 0.0 },
            total_training_time: self.monitor.state.read().await.last_update.elapsed(), // This is approximation, ideal would be start time
            peak_memory_usage: (sys_stats.peak_gpu_memory_mb * 1024.0 * 1024.0) as u64,
            throughput: stats.avg_throughput,
        };

        Ok(TrainingReport {
            best_loss: self.early_stopping.best_loss,
            best_step: self.early_stopping.best_step,
            total_steps: performance_stats.total_steps,
            convergence_rate: self.calculate_convergence_rate().await,
            final_metrics: self.get_final_metrics(&monitoring_report),
            early_stopped: self.early_stopping.triggered,
            performance_stats,
        })
    }

    async fn calculate_convergence_rate(&self) -> f64 {
        let state = self.monitor.state.read().await;
        if state.training_metrics.loss_values.len() < 2 {
            return 0.0;
        }

        let initial_loss = state.training_metrics.loss_values.front().map(|p| p.value).unwrap_or(0.0);
        let final_loss = state.training_metrics.loss_values.back().map(|p| p.value).unwrap_or(0.0);

        if initial_loss == 0.0 {
            return 0.0;
        }

        (initial_loss - final_loss) / initial_loss
    }

    fn get_final_metrics(&self, report: &MonitoringReport) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("loss".to_string(), report.training_metrics_summary.final_loss);
        metrics.insert("lr".to_string(), report.training_metrics_summary.final_lr);
        metrics.insert("grad_norm".to_string(), report.training_metrics_summary.avg_grad_norm); // Using avg as final might not be available in summary
        metrics
    }
}

/// Possible actions after a training step
#[derive(Debug)]
pub enum TrainingAction {
    Continue,
    Stop,
    SaveCheckpoint,
    Evaluate,
}

/// Training phases
#[derive(Debug, Clone)]
pub enum TrainingPhase {
    Warmup,
    Training,
    Cooldown,
    Finished,
}

/// Learning Rate Scheduler
#[derive(Debug)]
pub struct LearningRateScheduler {
    scheduler_type: LRSchedulerType,
    peak_lr: f64,
    min_lr: f64,
    total_steps: usize,
    warmup_steps: usize,
    current_lr: f64,
}

#[derive(Debug)]
pub enum LRSchedulerType {
    Cosine,
    Linear,
    Polynomial { power: f64 },
    Exponential { gamma: f64 },
    Constant,
}

impl LearningRateScheduler {
    pub fn new() -> Self {
        Self {
            scheduler_type: LRSchedulerType::Cosine,
            peak_lr: 1e-3,
            min_lr: 1e-6,
            total_steps: 100000,
            warmup_steps: 1000,
            current_lr: 0.0,
        }
    }

    pub fn configure(&mut self, scheduler_type: LRSchedulerType, peak_lr: f64, min_lr: f64, total_steps: usize, warmup_steps: usize) {
        self.scheduler_type = scheduler_type;
        self.peak_lr = peak_lr;
        self.min_lr = min_lr;
        self.total_steps = total_steps;
        self.warmup_steps = warmup_steps;
    }

    pub fn set_peak_lr(&mut self) {
        self.current_lr = self.peak_lr;
    }

    pub fn get_lr(&mut self, step: usize) -> f64 {
        self.current_lr = match &self.scheduler_type {
            LRSchedulerType::Cosine => self.cosine_schedule(step),
            LRSchedulerType::Linear => self.linear_schedule(step),
            LRSchedulerType::Polynomial { power } => self.polynomial_schedule(step, *power),
            LRSchedulerType::Exponential { gamma } => self.exponential_schedule(step, *gamma),
            LRSchedulerType::Constant => self.peak_lr,
        };
        self.current_lr
    }

    fn cosine_schedule(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            self.peak_lr * (step as f64 / self.warmup_steps as f64)
        } else {
            // Cosine decay
            let progress = (step - self.warmup_steps) as f64 / (self.total_steps - self.warmup_steps) as f64;
            self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + (progress * std::f64::consts::PI).cos())
        }
    }

    fn linear_schedule(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            self.peak_lr * (step as f64 / self.warmup_steps as f64)
        } else {
            let progress = (step - self.warmup_steps) as f64 / (self.total_steps - self.warmup_steps) as f64;
            self.peak_lr - (self.peak_lr - self.min_lr) * progress
        }
    }

    fn polynomial_schedule(&self, step: usize, power: f64) -> f64 {
        if step < self.warmup_steps {
            self.peak_lr * (step as f64 / self.warmup_steps as f64)
        } else {
            let progress = (step - self.warmup_steps) as f64 / (self.total_steps - self.warmup_steps) as f64;
            self.min_lr + (self.peak_lr - self.min_lr) * (1.0 - progress).powf(power)
        }
    }

    fn exponential_schedule(&self, step: usize, gamma: f64) -> f64 {
        if step < self.warmup_steps {
            self.peak_lr * (step as f64 / self.warmup_steps as f64)
        } else {
            // Exponential decay: lr = lr_0 * gamma^step
            self.peak_lr * gamma.powi(step as i32)
        }
    }
}

/// Curriculum Learning Manager
#[derive(Debug)]
pub struct CurriculumLearningManager {
    /// Sequence length schedule
    pub seq_len_schedule: Vec<(usize, usize)>,
    /// Task difficulty schedule
    pub difficulty_schedule: Vec<(usize, f64)>,
    /// Current sequence length
    pub current_seq_len: usize,
    /// Current difficulty level
    pub current_difficulty: f64,
    /// Curriculum strategy
    pub strategy: CurriculumStrategy,
}

#[derive(Debug, Clone)]
pub enum CurriculumStrategy {
    Linear,
    Exponential,
    Sudden,
    Custom,
}

impl CurriculumLearningManager {
    pub fn new() -> Self {
        Self {
            seq_len_schedule: vec![(0, 128), (1000, 256), (5000, 512), (10000, 1024), (50000, 2048)],
            difficulty_schedule: vec![(0, 0.1), (2000, 0.5), (8000, 0.8), (15000, 1.0)],
            current_seq_len: 128,
            current_difficulty: 0.1,
            strategy: CurriculumStrategy::Linear,
        }
    }

    pub fn set_sequence_schedule(&mut self, schedule: Vec<(usize, usize)>) {
        self.seq_len_schedule = schedule;
    }

    pub fn update(&mut self, step: usize) {
        // Update sequence length based on schedule
        for (target_step, seq_len) in &self.seq_len_schedule {
            if step >= *target_step {
                self.current_seq_len = *seq_len;
            }
        }

        // Update difficulty level
        for (target_step, difficulty) in &self.difficulty_schedule {
            if step >= *target_step {
                self.current_difficulty = *difficulty;
            }
        }
    }

    pub fn set_difficulty_level(&mut self, _step: usize) {
        // Could implement more sophisticated curriculum scheduling
    }

    pub fn get_current_config(&self) -> CurriculumConfig {
        CurriculumConfig {
            sequence_length: self.current_seq_len,
            difficulty_level: self.current_difficulty,
        }
    }
}

#[derive(Debug)]
pub struct CurriculumConfig {
    pub sequence_length: usize,
    pub difficulty_level: f64,
}

/// Early Stopping Detector
#[derive(Debug)]
pub struct EarlyStopping {
    pub patience: usize,
    pub min_delta: f64,
    pub best_loss: f64,
    pub best_step: usize,
    pub wait_count: usize,
    pub triggered: bool,
    pub restore_best_weights: bool,
}

impl EarlyStopping {
    pub fn new() -> Self {
        Self {
            patience: 1000,
            min_delta: 1e-4,
            best_loss: f64::INFINITY,
            best_step: 0,
            wait_count: 0,
            triggered: false,
            restore_best_weights: true,
        }
    }

    pub fn configure(&mut self, patience: usize, min_delta: f64) {
        self.patience = patience;
        self.min_delta = min_delta;
    }

    pub fn should_stop(&mut self, current_loss: f64, current_step: usize) -> bool {
        if current_loss < self.best_loss - self.min_delta {
            self.best_loss = current_loss;
            self.best_step = current_step;
            self.wait_count = 0;
            self.triggered = false;
        } else {
            self.wait_count += 1;
            if self.wait_count >= self.patience {
                self.triggered = true;
                return true;
            }
        }
        false
    }
}

#[derive(Debug)]
pub struct TrainingStatistics {
    pub total_steps: usize,
    pub average_step_time: f64,
    pub total_training_time: Duration,
    pub peak_memory_usage: u64,
    pub throughput: f64,
}

/// Checkpoint Manager
#[derive(Debug)]
pub struct CheckpointManager {
    pub checkpoint_dir: String,
    pub max_checkpoints: usize,
    pub save_optimizer_state: bool,
    pub save_scheduler_state: bool,
    pub checkpoints: Vec<CheckpointInfo>,
}

#[derive(Debug, Clone)]
pub struct CheckpointInfo {
    pub step: usize,
    pub loss: f64,
    pub metrics: HashMap<String, f64>,
    pub timestamp: Instant,
    pub path: String,
}

impl CheckpointManager {
    pub fn new() -> Self {
        Self {
            checkpoint_dir: "checkpoints".to_string(),
            max_checkpoints: 5,
            save_optimizer_state: true,
            save_scheduler_state: true,
            checkpoints: Vec::new(),
        }
    }

    pub async fn save_checkpoint(&mut self, step: usize, loss: f64, metrics: &HashMap<String, f64>) -> Result<()> {
        // Create checkpoint directory if needed
        std::fs::create_dir_all(&self.checkpoint_dir)?;

        let checkpoint_path = format!("{}/checkpoint_step_{}.ckpt", self.checkpoint_dir, step);

        // Save checkpoint (placeholder - would serialize model, optimizer, scheduler state)
        // let checkpoint = Checkpoint {
        //     step,
        //     loss,
        //     metrics: metrics.clone(),
        //     model_state: model_state,
        //     optimizer_state: optimizer_state,
        //     scheduler_state: scheduler_state,
        // };
        // save_to_file(&checkpoint_path, &checkpoint)?;

        let checkpoint_info = CheckpointInfo {
            step,
            loss,
            metrics: metrics.clone(),
            timestamp: Instant::now(),
            path: checkpoint_path,
        };

        self.checkpoints.push(checkpoint_info);

        // Clean up old checkpoints
        if self.checkpoints.len() > self.max_checkpoints {
            self.cleanup_old_checkpoints()?;
        }

        Ok(())
    }

    pub async fn load_checkpoint(&self, step: usize) -> Result<Option<CheckpointInfo>> {
        for checkpoint in &self.checkpoints {
            if checkpoint.step == step {
                return Ok(Some(checkpoint.clone()));
            }
        }
        Ok(None)
    }

    pub async fn load_latest_checkpoint(&self) -> Result<Option<CheckpointInfo>> {
        self.checkpoints.last().cloned().map_or(Ok(None), |c| Ok(Some(c)))
    }

    fn cleanup_old_checkpoints(&mut self) -> Result<()> {
        if self.checkpoints.len() > self.max_checkpoints {
            // Sort by step number and keep the most recent ones
            self.checkpoints.sort_by(|a, b| b.step.cmp(&a.step));
            self.checkpoints.truncate(self.max_checkpoints);

            // Remove old checkpoint files (placeholder)
            // for old_checkpoint in old_checkpoints {
            //     std::fs::remove_file(&old_checkpoint.path)?;
            // }
        }
        Ok(())
    }
}

/// Training Report
#[derive(Debug)]
pub struct TrainingReport {
    pub best_loss: f64,
    pub best_step: usize,
    pub total_steps: usize,
    pub convergence_rate: f64,
    pub final_metrics: HashMap<String, f64>,
    pub early_stopped: bool,
    pub performance_stats: TrainingStatistics,
}

impl TrainingReport {
    pub fn print_summary(&self) {
        println!("=== Training Summary ===");
        println!("Best Loss: {:.6} (Step {})", self.best_loss, self.best_step);
        println!("Total Steps: {}", self.total_steps);
        println!("Convergence Rate: {:.2}%", self.convergence_rate * 100.0);
        println!("Early Stopped: {}", self.early_stopped);
        println!("Average Throughput: {:.2} steps/sec", self.performance_stats.throughput);
        println!("Total Training Time: {:.2}s", self.performance_stats.total_training_time.as_secs_f64());
        println!("Peak Memory Usage: {} MB", self.performance_stats.peak_memory_usage / (1024 * 1024));

        println!("\nFinal Metrics:");
        for (name, value) in &self.final_metrics {
            println!("  {}: {:.4}", name, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::DistributedCoordinator;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_training_orchestrator_distributed_integration() {
        // Setup
        let config = TrainingConfig {
            total_steps: 10,
            evaluation_steps: 5,
            save_steps: 5,
            log_steps: 1,
            max_grad_norm: Some(1.0),
            warmup_steps: 2,
            cooldown_steps: 2,
        };

        let mut orchestrator = TrainingOrchestrator::new(config);

        // Manually inject DistributedCoordinator
        let coordinator = Arc::new(DistributedCoordinator::new(
            0,
            2,
            "127.0.0.1".to_string(),
            8000
        ));
        orchestrator.distributed_coordinator = Some(coordinator.clone());

        // Record some sync stats to verify they flow through
        coordinator.record_sync(50.0).await; // 50ms sync
        coordinator.record_sync(150.0).await; // 150ms sync
        // Min: 50, Max: 150, Avg: 100, Load Balance: 50/150 = 0.33

        // Perform a training step
        let metrics = HashMap::new();
        let gradients = vec![0.1; 10];
        
        let result = orchestrator.training_step(
            1,
            0.5,
            &metrics,
            &gradients
        ).await;

        assert!(result.is_ok());

        // Verify metrics were recorded in monitor
        let state = orchestrator.monitor.state.read().await;
        // The last recorded metrics should contain distributed stats
        // We can't easily access the exact last metric map from here without public accessors or digging into the deque
        // But we can check if the monitor has data.
        
        assert!(!state.training_metrics.loss_values.is_empty());
        
        // In a real integration test we might expose ways to inspect the last metrics more directly,
        // or check the log output if we could capture it.
        // For now, ensuring it runs without error and correctly accesses the coordinator is a good first step.
        
        // Let's verify via the distributed state directly that it was updated
        let dist_state = coordinator.state.read().await;
        assert_eq!(dist_state.sync_stats.total_sync_ops, 2);
        assert!((dist_state.sync_stats.average_sync_time_ms - 100.0).abs() < 1e-6);
        assert!((dist_state.sync_stats.load_balance_score() - 0.333333).abs() < 1e-4);
    }
}
