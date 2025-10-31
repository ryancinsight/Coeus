//! Learning rate schedulers for neural network training.
//!
//! This module provides PyTorch-compatible learning rate schedulers:
//! - StepLR: Decays learning rate by gamma every step_size epochs
//! - ExponentialLR: Decays learning rate exponentially by gamma every epoch
//! - CosineAnnealingLR: Decays learning rate using cosine annealing schedule
//! - MultiStepLR: Decays learning rate by gamma at specified milestones
//! - ReduceLROnPlateau: Reduces learning rate when a metric has stopped improving
//! - OneCycleLR: 1cycle learning rate policy for super-convergence

/// Core trait for learning rate schedulers.
///
/// All schedulers must implement this trait to provide a consistent interface
/// for learning rate adjustment during training.
pub trait LRScheduler {
    /// Get the current learning rate.
    fn learning_rate(&self) -> f64;

    /// Update the learning rate based on the scheduler's policy.
    ///
    /// This method should be called at the appropriate intervals (e.g., after each epoch).
    /// The scheduler will modify the learning rate according to its internal logic.
    fn step(&mut self);

    /// Get the last epoch/step number.
    fn last_epoch(&self) -> usize;

    /// Manually set the last epoch/step number.
    ///
    /// This is useful for resuming training or adjusting the scheduler state.
    fn set_last_epoch(&mut self, epoch: usize);
}

/// Step learning rate scheduler.
///
/// Decays the learning rate by `gamma` every `step_size` epochs.
/// This is equivalent to PyTorch's `torch.optim.lr_scheduler.StepLR`.
///
/// # Formula
/// ```text
/// lr = base_lr * gamma^(epoch // step_size)
/// ```
///
/// # Examples
/// ```rust
/// use optim::schedulers::{StepLR, LRScheduler};
///
/// let mut scheduler = StepLR::new(0.1, 30, 0.1); // lr=0.1, step every 30 epochs, gamma=0.1
/// assert_eq!(scheduler.learning_rate(), 0.1);
///
/// // After 30 epochs
/// for _ in 0..30 { scheduler.step(); }
/// assert!((scheduler.learning_rate() - 0.01).abs() < 1e-6); // 0.1 * 0.1
///
/// // After another 30 epochs
/// for _ in 0..30 { scheduler.step(); }
/// assert!((scheduler.learning_rate() - 0.001).abs() < 1e-6); // 0.01 * 0.1
/// ```
#[derive(Debug, Clone)]
pub struct StepLR {
    base_lr: f64,
    step_size: usize,
    gamma: f64,
    last_epoch: usize,
}

impl StepLR {
    /// Create a new StepLR scheduler.
    ///
    /// # Arguments
    /// * `base_lr` - Initial learning rate
    /// * `step_size` - Number of epochs between learning rate decays
    /// * `gamma` - Multiplicative factor for learning rate decay
    ///
    /// # Panics
    /// Panics if `base_lr <= 0`, `step_size == 0`, or `gamma <= 0`.
    pub fn new(base_lr: f64, step_size: usize, gamma: f64) -> Self {
        assert!(base_lr > 0.0, "base_lr must be > 0");
        assert!(step_size > 0, "step_size must be > 0");
        assert!(gamma > 0.0, "gamma must be > 0");

        Self {
            base_lr,
            step_size,
            gamma,
            last_epoch: 0,
        }
    }

    /// Compute the learning rate for a given epoch.
    fn lr_at_epoch(&self, epoch: usize) -> f64 {
        let decay_factor = (epoch / self.step_size) as i32;
        self.base_lr * self.gamma.powi(decay_factor)
    }
}

impl LRScheduler for StepLR {
    fn learning_rate(&self) -> f64 {
        self.lr_at_epoch(self.last_epoch)
    }

    fn step(&mut self) {
        self.last_epoch += 1;
    }

    fn last_epoch(&self) -> usize {
        self.last_epoch
    }

    fn set_last_epoch(&mut self, epoch: usize) {
        self.last_epoch = epoch;
    }
}

/// Exponential learning rate scheduler.
///
/// Decays the learning rate exponentially by `gamma` every epoch.
/// This is equivalent to PyTorch's `torch.optim.lr_scheduler.ExponentialLR`.
///
/// # Formula
/// ```text
/// lr = base_lr * gamma^epoch
/// ```
///
/// # Examples
/// ```rust
/// use optim::schedulers::{ExponentialLR, LRScheduler};
///
/// let mut scheduler = ExponentialLR::new(0.1, 0.9); // lr=0.1, gamma=0.9
/// assert_eq!(scheduler.learning_rate(), 0.1);
///
/// scheduler.step(); // epoch 1
/// assert!((scheduler.learning_rate() - 0.09).abs() < 1e-6); // 0.1 * 0.9
///
/// scheduler.step(); // epoch 2
/// assert!((scheduler.learning_rate() - 0.081).abs() < 1e-6); // 0.09 * 0.9 // 0.09 * 0.9
/// ```
#[derive(Debug, Clone)]
pub struct ExponentialLR {
    base_lr: f64,
    gamma: f64,
    last_epoch: usize,
}

impl ExponentialLR {
    /// Create a new ExponentialLR scheduler.
    ///
    /// # Arguments
    /// * `base_lr` - Initial learning rate
    /// * `gamma` - Multiplicative factor for learning rate decay per epoch
    ///
    /// # Panics
    /// Panics if `base_lr <= 0` or `gamma <= 0`.
    pub fn new(base_lr: f64, gamma: f64) -> Self {
        assert!(base_lr > 0.0, "base_lr must be > 0");
        assert!(gamma > 0.0, "gamma must be > 0");

        Self {
            base_lr,
            gamma,
            last_epoch: 0,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn learning_rate(&self) -> f64 {
        self.base_lr * self.gamma.powi(self.last_epoch as i32)
    }

    fn step(&mut self) {
        self.last_epoch += 1;
    }

    fn last_epoch(&self) -> usize {
        self.last_epoch
    }

    fn set_last_epoch(&mut self, epoch: usize) {
        self.last_epoch = epoch;
    }
}

/// Cosine annealing learning rate scheduler.
///
/// Decays the learning rate using a cosine annealing schedule.
/// The learning rate follows a cosine curve from `base_lr` to `min_lr`
/// over `t_max` epochs, then resets.
///
/// This is equivalent to PyTorch's `torch.optim.lr_scheduler.CosineAnnealingLR`.
///
/// # Formula
/// ```text
/// lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * epoch / t_max))
/// ```
///
/// # Examples
/// ```rust
/// use optim::schedulers::{CosineAnnealingLR, LRScheduler};
///
/// let mut scheduler = CosineAnnealingLR::new(0.1, 0.0001, 100); // lr=0.1 to 0.0001 over 100 epochs
/// assert_eq!(scheduler.learning_rate(), 0.1);
///
/// // At epoch 50 (halfway)
/// for _ in 0..50 { scheduler.step(); }
/// let lr_at_50 = scheduler.learning_rate();
/// assert!(lr_at_50 > 0.0001 && lr_at_50 < 0.1); // Between min and max
///
/// // At epoch 100 (end)
/// for _ in 50..100 { scheduler.step(); }
/// assert!((scheduler.learning_rate() - 0.0001).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    base_lr: f64,
    min_lr: f64,
    t_max: usize,
    last_epoch: usize,
}

impl CosineAnnealingLR {
    /// Create a new CosineAnnealingLR scheduler.
    ///
    /// # Arguments
    /// * `base_lr` - Initial learning rate (maximum LR)
    /// * `min_lr` - Minimum learning rate
    /// * `t_max` - Maximum number of epochs for one annealing cycle
    ///
    /// # Panics
    /// Panics if `base_lr <= 0`, `min_lr < 0`, `t_max == 0`, or `base_lr <= min_lr`.
    pub fn new(base_lr: f64, min_lr: f64, t_max: usize) -> Self {
        assert!(base_lr > 0.0, "base_lr must be > 0");
        assert!(min_lr >= 0.0, "min_lr must be >= 0");
        assert!(t_max > 0, "t_max must be > 0");
        assert!(base_lr > min_lr, "base_lr must be > min_lr");

        Self {
            base_lr,
            min_lr,
            t_max,
            last_epoch: 0,
        }
    }

    /// Compute the learning rate for a given epoch using cosine annealing.
    ///
    /// Matches PyTorch's CosineAnnealingLR behavior: cycles every t_max epochs,
    /// reaching min_lr at the end of each cycle.
    fn lr_at_epoch(&self, epoch: usize) -> f64 {
        let effective_epoch = if epoch % self.t_max == 0 && epoch > 0 {
            self.t_max
        } else {
            epoch % self.t_max
        };
        let progress = effective_epoch as f64 / self.t_max as f64;
        let cosine_term = (std::f64::consts::PI * progress).cos();
        self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + cosine_term)
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn learning_rate(&self) -> f64 {
        self.lr_at_epoch(self.last_epoch)
    }

    fn step(&mut self) {
        self.last_epoch += 1;
    }

    fn last_epoch(&self) -> usize {
        self.last_epoch
    }

    fn set_last_epoch(&mut self, epoch: usize) {
        self.last_epoch = epoch;
    }
}

/// Multi-step learning rate scheduler.
///
/// Decays the learning rate by `gamma` at specified milestone epochs.
/// This is equivalent to PyTorch's `torch.optim.lr_scheduler.MultiStepLR`.
///
/// # Formula
/// ```text
/// lr = base_lr * gamma^(number of milestones passed)
/// ```
///
/// # Examples
/// ```rust
/// use optim::schedulers::{MultiStepLR, LRScheduler};
///
/// // Reduce LR by 0.1x at epochs 30, 60, 90
/// let mut scheduler = MultiStepLR::new(0.1, vec![30, 60, 90], 0.1);
/// assert_eq!(scheduler.learning_rate(), 0.1);
///
/// // After 30 epochs
/// for _ in 0..30 { scheduler.step(); }
/// assert!((scheduler.learning_rate() - 0.01).abs() < 1e-6); // 0.1 * 0.1
///
/// // After 60 epochs total
/// for _ in 0..30 { scheduler.step(); }
/// assert!((scheduler.learning_rate() - 0.001).abs() < 1e-6); // 0.1 * 0.1 * 0.1
/// ```
#[derive(Debug, Clone)]
pub struct MultiStepLR {
    /// Base learning rate
    base_lr: f64,
    /// Current learning rate
    current_lr: f64,
    /// Milestone epochs where LR is reduced
    milestones: Vec<usize>,
    /// Multiplicative factor for LR decay
    gamma: f64,
    /// Current epoch/step number
    last_epoch: usize,
}

impl MultiStepLR {
    /// Create a new MultiStepLR scheduler.
    ///
    /// # Arguments
    /// * `base_lr` - Initial learning rate
    /// * `milestones` - List of epoch indices where LR is reduced (must be sorted)
    /// * `gamma` - Multiplicative factor for LR decay (typically 0.1)
    ///
    /// # Panics
    /// Panics if `base_lr <= 0`, `gamma <= 0`, or milestones are not sorted.
    ///
    /// # Examples
    /// ```rust
    /// use optim::schedulers::{MultiStepLR, LRScheduler};
    ///
    /// let scheduler = MultiStepLR::new(0.1, vec![30, 60, 90], 0.1);
    /// ```
    pub fn new(base_lr: f64, milestones: Vec<usize>, gamma: f64) -> Self {
        assert!(base_lr > 0.0, "base_lr must be > 0");
        assert!(gamma > 0.0, "gamma must be > 0");

        // Verify milestones are sorted
        for i in 1..milestones.len() {
            assert!(
                milestones[i] > milestones[i - 1],
                "milestones must be sorted in ascending order"
            );
        }

        Self {
            base_lr,
            current_lr: base_lr,
            milestones,
            gamma,
            last_epoch: 0,
        }
    }

    /// Calculate learning rate for a given epoch.
    fn calculate_lr(&self, epoch: usize) -> f64 {
        // Count how many milestones have been passed
        let num_milestones_passed = self.milestones.iter().filter(|&&m| epoch >= m).count();
        self.base_lr * self.gamma.powi(num_milestones_passed as i32)
    }
}

impl LRScheduler for MultiStepLR {
    fn learning_rate(&self) -> f64 {
        self.current_lr
    }

    fn step(&mut self) {
        self.last_epoch += 1;
        self.current_lr = self.calculate_lr(self.last_epoch);
    }

    fn last_epoch(&self) -> usize {
        self.last_epoch
    }

    fn set_last_epoch(&mut self, epoch: usize) {
        self.last_epoch = epoch;
        self.current_lr = self.calculate_lr(epoch);
    }
}

/// Reduce learning rate on plateau scheduler.
///
/// Reduces the learning rate when a metric (e.g., validation loss) has stopped improving.
/// This is equivalent to PyTorch's `torch.optim.lr_scheduler.ReduceLROnPlateau`.
///
/// # Examples
/// ```rust
/// use optim::schedulers::{ReduceLROnPlateau, ReduceLRMode};
///
/// // For validation loss (minimize)
/// let mut scheduler = ReduceLROnPlateau::new(0.1, ReduceLRMode::Min, 0.1, 10, 1e-4, 0, 1e-8);
///
/// // After each epoch, call step with validation loss
/// scheduler.step(0.5); // loss = 0.5
/// scheduler.step(0.4); // loss improved
/// scheduler.step(0.39); // loss improved slightly
/// scheduler.step(0.39); // no improvement (within threshold)
/// // ... after 10 epochs of no improvement, LR is reduced
/// ```
///
/// # References
/// - Commonly used in practice for adaptive learning rate adjustment
/// - Particularly effective when validation metrics plateau
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceLRMode {
    /// Minimize the metric (e.g., for loss)
    Min,
    /// Maximize the metric (e.g., for accuracy)
    Max,
}

#[derive(Debug, Clone)]
pub struct ReduceLROnPlateau {
    /// Current learning rate
    current_lr: f64,
    /// Mode: minimize or maximize the metric
    mode: ReduceLRMode,
    /// Multiplicative factor for LR reduction
    factor: f64,
    /// Number of epochs with no improvement after which LR is reduced
    patience: usize,
    /// Threshold for measuring improvement
    threshold: f64,
    /// Number of epochs to wait before resuming normal operation after LR reduction
    cooldown: usize,
    /// Minimum learning rate
    min_lr: f64,
    /// Best metric value seen so far
    best: f64,
    /// Number of epochs with no improvement
    num_bad_epochs: usize,
    /// Cooldown counter
    cooldown_counter: usize,
    /// Last epoch number
    last_epoch: usize,
}

impl ReduceLROnPlateau {
    /// Create a new ReduceLROnPlateau scheduler.
    ///
    /// # Arguments
    /// * `initial_lr` - Initial learning rate
    /// * `mode` - Min (for loss) or Max (for accuracy)
    /// * `factor` - Factor by which LR is reduced (new_lr = lr * factor)
    /// * `patience` - Number of epochs with no improvement after which LR is reduced
    /// * `threshold` - Threshold for measuring improvement
    /// * `cooldown` - Number of epochs to wait before resuming after LR reduction
    /// * `min_lr` - Minimum learning rate
    ///
    /// # Panics
    /// Panics if `initial_lr <= 0`, `factor <= 0` or `factor >= 1`, `min_lr < 0`, or `min_lr >= initial_lr`.
    ///
    /// # Examples
    /// ```rust
    /// use optim::schedulers::{ReduceLROnPlateau, ReduceLRMode};
    ///
    /// let scheduler = ReduceLROnPlateau::new(0.1, ReduceLRMode::Min, 0.1, 10, 1e-4, 0, 1e-8);
    /// ```
    pub fn new(
        initial_lr: f64,
        mode: ReduceLRMode,
        factor: f64,
        patience: usize,
        threshold: f64,
        cooldown: usize,
        min_lr: f64,
    ) -> Self {
        assert!(initial_lr > 0.0, "initial_lr must be > 0");
        assert!(factor > 0.0 && factor < 1.0, "factor must be in (0, 1)");
        assert!(min_lr >= 0.0, "min_lr must be >= 0");
        assert!(min_lr < initial_lr, "min_lr must be < initial_lr");

        let best = match mode {
            ReduceLRMode::Min => f64::INFINITY,
            ReduceLRMode::Max => f64::NEG_INFINITY,
        };

        Self {
            current_lr: initial_lr,
            mode,
            factor,
            patience,
            threshold,
            cooldown,
            min_lr,
            best,
            num_bad_epochs: 0,
            cooldown_counter: 0,
            last_epoch: 0,
        }
    }

    /// Step the scheduler with a metric value.
    ///
    /// # Arguments
    /// * `metric` - Current metric value (e.g., validation loss or accuracy)
    ///
    /// # Examples
    /// ```rust
    /// use optim::schedulers::{ReduceLROnPlateau, ReduceLRMode};
    ///
    /// let mut scheduler = ReduceLROnPlateau::new(0.1, ReduceLRMode::Min, 0.1, 10, 1e-4, 0, 1e-8);
    /// scheduler.step(0.5); // validation loss = 0.5
    /// ```
    pub fn step(&mut self, metric: f64) {
        self.last_epoch += 1;

        // Check if in cooldown period
        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            return;
        }

        // Check if metric improved
        let is_better = match self.mode {
            ReduceLRMode::Min => metric < self.best - self.threshold,
            ReduceLRMode::Max => metric > self.best + self.threshold,
        };

        if is_better {
            self.best = metric;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
        }

        // Reduce LR if patience exceeded
        if self.num_bad_epochs >= self.patience {
            let new_lr = (self.current_lr * self.factor).max(self.min_lr);
            if new_lr < self.current_lr {
                self.current_lr = new_lr;
                self.cooldown_counter = self.cooldown;
                self.num_bad_epochs = 0;
            }
        }
    }

    /// Get the current learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.current_lr
    }

    /// Get the last epoch number.
    pub fn last_epoch(&self) -> usize {
        self.last_epoch
    }

    /// Get the best metric value seen so far.
    pub fn best_metric(&self) -> f64 {
        self.best
    }
}

/// OneCycleLR scheduler.
///
/// Sets the learning rate according to the 1cycle learning rate policy.
/// The 1cycle policy anneals the learning rate from an initial learning rate
/// to some maximum learning rate and then from that maximum learning rate to
/// some minimum learning rate much lower than the initial learning rate.
///
/// This policy was initially described in the paper
/// [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120).
///
/// The 1cycle learning rate policy changes the learning rate after every batch.
/// `step` should be called after a batch has been used for training.
///
/// # Three Phases
/// 1. **Warmup phase** (0 to pct_start * total_steps): Linear increase from initial_lr to max_lr
/// 2. **Annealing phase** (pct_start * total_steps to total_steps): Cosine decrease from max_lr to min_lr
/// 3. **Final phase** (after total_steps): Constant at min_lr
///
/// # Formula
/// ```text
/// Phase 1 (Warmup): lr = initial_lr + (max_lr - initial_lr) * (step / warmup_steps)
/// Phase 2 (Annealing): lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(π * progress))
/// Phase 3 (Final): lr = min_lr
/// ```
///
/// # Examples
/// ```rust
/// use optim::schedulers::{OneCycleLR, LRScheduler};
///
/// // Train for 1000 steps with max_lr=0.1
/// let mut scheduler = OneCycleLR::new(0.1, 1000, 0.3, 0.001);
/// assert_eq!(scheduler.learning_rate(), 0.001); // Starts at initial_lr
///
/// // Step through training
/// for _ in 0..1000 {
///     scheduler.step();
///     let lr = scheduler.learning_rate();
///     // Use lr for training
/// }
/// ```
///
/// # References
/// - [Super-Convergence Paper](https://arxiv.org/abs/1708.07120)
/// - [PyTorch OneCycleLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)
pub struct OneCycleLR {
    max_lr: f64,
    total_steps: usize,
    pct_start: f64,
    initial_lr: f64,
    min_lr: f64,
    current_step: usize,
    current_lr: f64,
}

impl OneCycleLR {
    /// Create a new OneCycleLR scheduler.
    ///
    /// # Arguments
    /// * `max_lr` - Maximum learning rate (peak of the cycle)
    /// * `total_steps` - Total number of training steps
    /// * `pct_start` - Percentage of cycle spent in warmup phase (default: 0.3)
    /// * `initial_lr` - Initial learning rate (default: max_lr / 25)
    ///
    /// # Panics
    /// Panics if `max_lr <= 0`, `total_steps == 0`, or `pct_start` not in (0, 1).
    ///
    /// # Examples
    /// ```rust
    /// use optim::schedulers::{OneCycleLR, LRScheduler};
    ///
    /// let scheduler = OneCycleLR::new(0.1, 1000, 0.3, 0.001);
    /// assert_eq!(scheduler.learning_rate(), 0.001);
    /// ```
    pub fn new(max_lr: f64, total_steps: usize, pct_start: f64, initial_lr: f64) -> Self {
        assert!(max_lr > 0.0, "max_lr must be > 0");
        assert!(total_steps > 0, "total_steps must be > 0");
        assert!(
            pct_start > 0.0 && pct_start < 1.0,
            "pct_start must be in (0, 1)"
        );
        assert!(
            initial_lr > 0.0 && initial_lr < max_lr,
            "initial_lr must be in (0, max_lr)"
        );

        let min_lr = initial_lr / 10.0; // min_lr is typically much smaller than initial_lr

        Self {
            max_lr,
            total_steps,
            pct_start,
            initial_lr,
            min_lr,
            current_step: 0,
            current_lr: initial_lr,
        }
    }

    /// Create OneCycleLR with default parameters.
    ///
    /// Uses `initial_lr = max_lr / 25` and `pct_start = 0.3`.
    ///
    /// # Examples
    /// ```rust
    /// use optim::schedulers::{OneCycleLR, LRScheduler};
    ///
    /// let scheduler = OneCycleLR::default(0.1, 1000);
    /// ```
    pub fn default(max_lr: f64, total_steps: usize) -> Self {
        let initial_lr = max_lr / 25.0;
        Self::new(max_lr, total_steps, 0.3, initial_lr)
    }

    /// Compute learning rate for the current step.
    fn compute_lr(&self) -> f64 {
        let warmup_steps = (self.total_steps as f64 * self.pct_start) as usize;

        if self.current_step < warmup_steps {
            // Phase 1: Warmup (linear increase)
            let progress = self.current_step as f64 / warmup_steps as f64;
            self.initial_lr + (self.max_lr - self.initial_lr) * progress
        } else if self.current_step < self.total_steps {
            // Phase 2: Annealing (cosine decrease)
            let annealing_steps = self.total_steps - warmup_steps;
            let progress = (self.current_step - warmup_steps) as f64 / annealing_steps as f64;
            let cosine_term = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            self.min_lr + (self.max_lr - self.min_lr) * cosine_term
        } else {
            // Phase 3: Final (constant at min_lr)
            self.min_lr
        }
    }
}

impl LRScheduler for OneCycleLR {
    fn learning_rate(&self) -> f64 {
        self.current_lr
    }

    fn step(&mut self) {
        self.current_step += 1;
        self.current_lr = self.compute_lr();
    }

    fn last_epoch(&self) -> usize {
        self.current_step
    }

    fn set_last_epoch(&mut self, step: usize) {
        self.current_step = step;
        self.current_lr = self.compute_lr();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_step_lr_creation() {
        let scheduler = StepLR::new(0.1, 30, 0.1);
        assert_eq!(scheduler.learning_rate(), 0.1);
        assert_eq!(scheduler.last_epoch(), 0);
    }

    #[test]
    fn test_step_lr_stepping() {
        let mut scheduler = StepLR::new(0.1, 2, 0.5); // Every 2 epochs, multiply by 0.5

        // Initial
        assert_eq!(scheduler.learning_rate(), 0.1);

        // After 1 step (epoch 1)
        scheduler.step();
        assert_eq!(scheduler.learning_rate(), 0.1);

        // After 2 steps (epoch 2) - decay
        scheduler.step();
        assert_eq!(scheduler.learning_rate(), 0.05);

        // After 3 steps (epoch 3)
        scheduler.step();
        assert_eq!(scheduler.learning_rate(), 0.05);

        // After 4 steps (epoch 4) - decay
        scheduler.step();
        assert_eq!(scheduler.learning_rate(), 0.025);
    }

    #[test]
    fn test_exponential_lr_creation() {
        let scheduler = ExponentialLR::new(0.1, 0.9);
        assert_eq!(scheduler.learning_rate(), 0.1);
        assert_eq!(scheduler.last_epoch(), 0);
    }

    #[test]
    fn test_exponential_lr_stepping() {
        let mut scheduler = ExponentialLR::new(0.1, 0.9);

        // Initial
        assert_eq!(scheduler.learning_rate(), 0.1);

        // After 1 step
        scheduler.step();
        assert_relative_eq!(scheduler.learning_rate(), 0.09, epsilon = 1e-10);

        // After 2 steps
        scheduler.step();
        assert_relative_eq!(scheduler.learning_rate(), 0.081, epsilon = 1e-10);

        // After 3 steps
        scheduler.step();
        assert_relative_eq!(scheduler.learning_rate(), 0.0729, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_annealing_lr_creation() {
        let scheduler = CosineAnnealingLR::new(0.1, 0.01, 100);
        assert_eq!(scheduler.learning_rate(), 0.1);
        assert_eq!(scheduler.last_epoch(), 0);
    }

    #[test]
    fn test_cosine_annealing_lr_boundaries() {
        let mut scheduler = CosineAnnealingLR::new(0.1, 0.01, 10);

        // Start: should be at base_lr
        assert_relative_eq!(scheduler.learning_rate(), 0.1, epsilon = 1e-6);

        // Step to epoch 5 (middle)
        for _ in 0..5 {
            scheduler.step();
        }
        let lr_at_5 = scheduler.learning_rate();
        assert!(lr_at_5 > 0.01 && lr_at_5 < 0.1);

        // Step to epoch 10 (end of cycle): should be at min_lr
        for _ in 5..10 {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.learning_rate(), 0.01, epsilon = 1e-6);

        // Cosine annealing continues beyond T_max, doesn't reset
        // At epoch 11, T_cur = 11 % 10 = 1, progress = 1/10 = 0.1
        // cosine_term = cos(π * 0.1) ≈ cos(0.314) ≈ 0.951
        // lr = 0.01 + 0.5 * (0.1 - 0.01) * (1 + 0.951) ≈ 0.01 + 0.045 * 1.951 ≈ 0.01 + 0.0878 ≈ 0.0978
        scheduler.step();
        let lr_at_11 = scheduler.learning_rate();
        assert!(lr_at_11 > 0.09 && lr_at_11 < 0.11);
    }

    #[test]
    fn test_scheduler_set_last_epoch() {
        let mut scheduler = StepLR::new(0.1, 5, 0.5);

        // Set to epoch 10
        scheduler.set_last_epoch(10);
        assert_eq!(scheduler.last_epoch(), 10);
        assert_eq!(scheduler.learning_rate(), 0.025); // 0.1 * 0.5^(10/5) = 0.1 * 0.5^2 = 0.025
    }

    #[test]
    #[should_panic(expected = "base_lr must be > 0")]
    fn test_step_lr_invalid_base_lr() {
        let _scheduler = StepLR::new(0.0, 30, 0.1);
    }

    #[test]
    #[should_panic(expected = "step_size must be > 0")]
    fn test_step_lr_invalid_step_size() {
        let _scheduler = StepLR::new(0.1, 0, 0.1);
    }

    #[test]
    #[should_panic(expected = "gamma must be > 0")]
    fn test_step_lr_invalid_gamma() {
        let _scheduler = StepLR::new(0.1, 30, 0.0);
    }

    #[test]
    #[should_panic(expected = "base_lr must be > min_lr")]
    fn test_cosine_annealing_lr_invalid_lr_range() {
        let _scheduler = CosineAnnealingLR::new(0.01, 0.1, 100);
    }

    #[test]
    fn test_multistep_lr_basic() {
        let mut scheduler = MultiStepLR::new(0.1, vec![30, 60, 90], 0.1);
        assert_eq!(scheduler.learning_rate(), 0.1);
        assert_eq!(scheduler.last_epoch(), 0);

        // Before first milestone
        for _ in 0..29 {
            scheduler.step();
        }
        assert_eq!(scheduler.learning_rate(), 0.1);

        // At first milestone (epoch 30)
        scheduler.step();
        assert!((scheduler.learning_rate() - 0.01).abs() < 1e-9);

        // Before second milestone
        for _ in 0..29 {
            scheduler.step();
        }
        assert!((scheduler.learning_rate() - 0.01).abs() < 1e-9);

        // At second milestone (epoch 60)
        scheduler.step();
        assert!((scheduler.learning_rate() - 0.001).abs() < 1e-9);

        // Before third milestone
        for _ in 0..29 {
            scheduler.step();
        }
        assert!((scheduler.learning_rate() - 0.001).abs() < 1e-9);

        // At third milestone (epoch 90)
        scheduler.step();
        assert!((scheduler.learning_rate() - 0.0001).abs() < 1e-9);
    }

    #[test]
    fn test_multistep_lr_set_last_epoch() {
        let mut scheduler = MultiStepLR::new(0.1, vec![30, 60, 90], 0.1);

        // Set to epoch 50 (after first milestone, before second)
        scheduler.set_last_epoch(50);
        assert_eq!(scheduler.last_epoch(), 50);
        assert!((scheduler.learning_rate() - 0.01).abs() < 1e-9);

        // Set to epoch 70 (after second milestone, before third)
        scheduler.set_last_epoch(70);
        assert_eq!(scheduler.last_epoch(), 70);
        assert!((scheduler.learning_rate() - 0.001).abs() < 1e-9);
    }

    #[test]
    fn test_multistep_lr_empty_milestones() {
        let mut scheduler = MultiStepLR::new(0.1, vec![], 0.1);
        assert_eq!(scheduler.learning_rate(), 0.1);

        // LR should never change with no milestones
        for _ in 0..100 {
            scheduler.step();
        }
        assert_eq!(scheduler.learning_rate(), 0.1);
    }

    #[test]
    #[should_panic(expected = "base_lr must be > 0")]
    fn test_multistep_lr_invalid_base_lr() {
        let _scheduler = MultiStepLR::new(0.0, vec![30, 60], 0.1);
    }

    #[test]
    #[should_panic(expected = "gamma must be > 0")]
    fn test_multistep_lr_invalid_gamma() {
        let _scheduler = MultiStepLR::new(0.1, vec![30, 60], 0.0);
    }

    #[test]
    #[should_panic(expected = "milestones must be sorted")]
    fn test_multistep_lr_unsorted_milestones() {
        let _scheduler = MultiStepLR::new(0.1, vec![60, 30, 90], 0.1);
    }

    #[test]
    fn test_reduce_lr_on_plateau_min_mode() {
        let mut scheduler = ReduceLROnPlateau::new(0.1, ReduceLRMode::Min, 0.1, 3, 1e-4, 0, 1e-8);
        assert_eq!(scheduler.learning_rate(), 0.1);

        // Improving loss
        scheduler.step(1.0);
        assert_eq!(scheduler.learning_rate(), 0.1);

        scheduler.step(0.5);
        assert_eq!(scheduler.learning_rate(), 0.1);

        // No improvement (within threshold) - patience counter starts
        scheduler.step(0.5); // bad_epochs = 1
        assert_eq!(scheduler.learning_rate(), 0.1);

        scheduler.step(0.5); // bad_epochs = 2
        assert_eq!(scheduler.learning_rate(), 0.1);

        scheduler.step(0.5); // bad_epochs = 3, triggers reduction
                             // After patience epochs (3), LR should be reduced
        assert!((scheduler.learning_rate() - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_reduce_lr_on_plateau_max_mode() {
        let mut scheduler = ReduceLROnPlateau::new(0.1, ReduceLRMode::Max, 0.1, 3, 1e-4, 0, 1e-8);
        assert_eq!(scheduler.learning_rate(), 0.1);

        // Improving accuracy
        scheduler.step(0.5);
        assert_eq!(scheduler.learning_rate(), 0.1);

        scheduler.step(0.8);
        assert_eq!(scheduler.learning_rate(), 0.1);

        // No improvement - patience counter starts
        scheduler.step(0.8); // bad_epochs = 1
        scheduler.step(0.8); // bad_epochs = 2
        scheduler.step(0.8); // bad_epochs = 3, triggers reduction

        // After patience epochs, LR should be reduced
        assert!((scheduler.learning_rate() - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_reduce_lr_on_plateau_cooldown() {
        let mut scheduler = ReduceLROnPlateau::new(0.1, ReduceLRMode::Min, 0.1, 2, 1e-4, 2, 1e-8);

        // Trigger LR reduction
        scheduler.step(1.0);
        scheduler.step(1.0);
        scheduler.step(1.0);

        // LR should be reduced
        assert!((scheduler.learning_rate() - 0.01).abs() < 1e-9);

        // During cooldown, no further reductions even with bad metrics
        let lr_before = scheduler.learning_rate();
        scheduler.step(10.0); // Very bad metric
        assert_eq!(scheduler.learning_rate(), lr_before); // No change during cooldown

        scheduler.step(10.0);
        assert_eq!(scheduler.learning_rate(), lr_before); // Still in cooldown
    }

    #[test]
    fn test_reduce_lr_on_plateau_min_lr() {
        let mut scheduler = ReduceLROnPlateau::new(0.1, ReduceLRMode::Min, 0.1, 2, 1e-4, 0, 0.001);

        // Trigger multiple LR reductions
        for _ in 0..10 {
            scheduler.step(1.0);
            scheduler.step(1.0);
            scheduler.step(1.0);
        }

        // LR should not go below min_lr
        assert!(scheduler.learning_rate() >= 0.001);
        assert!((scheduler.learning_rate() - 0.001).abs() < 1e-9);
    }

    #[test]
    fn test_reduce_lr_on_plateau_best_metric() {
        let mut scheduler = ReduceLROnPlateau::new(0.1, ReduceLRMode::Min, 0.1, 3, 1e-4, 0, 1e-8);

        scheduler.step(1.0);
        assert!((scheduler.best_metric() - 1.0).abs() < 1e-9);

        scheduler.step(0.5);
        assert!((scheduler.best_metric() - 0.5).abs() < 1e-9);

        scheduler.step(0.6); // Worse than best
        assert!((scheduler.best_metric() - 0.5).abs() < 1e-9); // Best unchanged
    }

    #[test]
    #[should_panic(expected = "initial_lr must be > 0")]
    fn test_reduce_lr_on_plateau_invalid_initial_lr() {
        let _scheduler = ReduceLROnPlateau::new(0.0, ReduceLRMode::Min, 0.1, 10, 1e-4, 0, 1e-8);
    }

    #[test]
    #[should_panic(expected = "factor must be in (0, 1)")]
    fn test_reduce_lr_on_plateau_invalid_factor() {
        let _scheduler = ReduceLROnPlateau::new(0.1, ReduceLRMode::Min, 1.5, 10, 1e-4, 0, 1e-8);
    }

    #[test]
    #[should_panic(expected = "min_lr must be < initial_lr")]
    fn test_reduce_lr_on_plateau_invalid_min_lr() {
        let _scheduler = ReduceLROnPlateau::new(0.1, ReduceLRMode::Min, 0.1, 10, 1e-4, 0, 0.2);
    }

    #[test]
    fn test_onecycle_lr_creation() {
        let scheduler = OneCycleLR::new(0.1, 1000, 0.3, 0.004);
        assert_eq!(scheduler.learning_rate(), 0.004); // Starts at initial_lr
        assert_eq!(scheduler.last_epoch(), 0);
    }

    #[test]
    fn test_onecycle_lr_default() {
        let scheduler = OneCycleLR::default(0.1, 1000);
        assert_eq!(scheduler.learning_rate(), 0.004); // initial_lr = max_lr / 25 = 0.1 / 25 = 0.004
        assert_eq!(scheduler.last_epoch(), 0);
    }

    #[test]
    fn test_onecycle_lr_warmup_phase() {
        let mut scheduler = OneCycleLR::new(0.1, 1000, 0.3, 0.004);

        // At step 0: should be at initial_lr
        assert_eq!(scheduler.learning_rate(), 0.004);

        // Step to middle of warmup (step 150 out of 300 warmup steps)
        for _ in 0..150 {
            scheduler.step();
        }
        let lr_mid_warmup = scheduler.learning_rate();
        assert!(lr_mid_warmup > 0.004 && lr_mid_warmup < 0.1);

        // Step to end of warmup (step 300)
        for _ in 150..300 {
            scheduler.step();
        }
        let lr_end_warmup = scheduler.learning_rate();
        assert_relative_eq!(lr_end_warmup, 0.1, epsilon = 1e-6); // Should be at max_lr
    }

    #[test]
    fn test_onecycle_lr_annealing_phase() {
        let mut scheduler = OneCycleLR::new(0.1, 1000, 0.3, 0.004);

        // Step to end of warmup (step 300)
        for _ in 0..300 {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.learning_rate(), 0.1, epsilon = 1e-6);

        // Step to middle of annealing (step 650 out of 1000 total)
        for _ in 300..650 {
            scheduler.step();
        }
        let lr_mid_annealing = scheduler.learning_rate();
        assert!(lr_mid_annealing > 0.0004 && lr_mid_annealing < 0.1);

        // Step to end of annealing (step 1000)
        for _ in 650..1000 {
            scheduler.step();
        }
        let lr_end_annealing = scheduler.learning_rate();
        assert_relative_eq!(lr_end_annealing, 0.0004, epsilon = 1e-6); // Should be at min_lr
    }

    #[test]
    fn test_onecycle_lr_final_phase() {
        let mut scheduler = OneCycleLR::new(0.1, 1000, 0.3, 0.004);

        // Step beyond total_steps
        for _ in 0..1100 {
            scheduler.step();
        }

        // Should stay at min_lr
        assert_relative_eq!(scheduler.learning_rate(), 0.0004, epsilon = 1e-6);
    }

    #[test]
    fn test_onecycle_lr_set_last_epoch() {
        let mut scheduler = OneCycleLR::new(0.1, 1000, 0.3, 0.004);

        // Set to middle of warmup
        scheduler.set_last_epoch(150);
        assert_eq!(scheduler.last_epoch(), 150);
        let lr_at_150 = scheduler.learning_rate();
        assert!(lr_at_150 > 0.004 && lr_at_150 < 0.1);

        // Set to end of warmup
        scheduler.set_last_epoch(300);
        assert_relative_eq!(scheduler.learning_rate(), 0.1, epsilon = 1e-6);

        // Set to middle of annealing
        scheduler.set_last_epoch(650);
        let lr_at_650 = scheduler.learning_rate();
        assert!(lr_at_650 > 0.0004 && lr_at_650 < 0.1);
    }

    #[test]
    #[should_panic(expected = "max_lr must be > 0")]
    fn test_onecycle_lr_invalid_max_lr() {
        let _scheduler = OneCycleLR::new(0.0, 1000, 0.3, 0.004);
    }

    #[test]
    #[should_panic(expected = "total_steps must be > 0")]
    fn test_onecycle_lr_invalid_total_steps() {
        let _scheduler = OneCycleLR::new(0.1, 0, 0.3, 0.004);
    }

    #[test]
    #[should_panic(expected = "pct_start must be in (0, 1)")]
    fn test_onecycle_lr_invalid_pct_start() {
        let _scheduler = OneCycleLR::new(0.1, 1000, 1.5, 0.004);
    }
}
