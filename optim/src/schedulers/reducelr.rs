//! ReduceLROnPlateau learning rate scheduler
//!
//! Implements ReduceLROnPlateau learning rate schedule,
//! compatible with PyTorch's `torch.optim.lr_scheduler.ReduceLROnPlateau`.

use crate::{Optimizer, Result};
use std::collections::VecDeque;

#[cfg(test)]
mod reducelr_tests {
    use super::*;
    use crate::{ParamGroup, Sgd};
    use approx::assert_relative_eq;
    use coeus_tensor::{CpuBackend, Tensor};

    /// Test ReduceLROnPlateau scheduler creation
    #[test]
    fn test_reducelr_creation() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0_f64, 2.0_f64], vec![2]).unwrap()];
        let mut optimizer = Sgd::new(params, 0.1_f64);
        let scheduler = ReduceLROnPlateau::new(&mut optimizer, Mode::Min, 0.1_f64, 10);

        assert_eq!(scheduler.mode(), Mode::Min);
        assert_eq!(scheduler.factor(), 0.1_f64);
        assert_eq!(scheduler.patience(), 10);
        assert_eq!(scheduler.cooldown_counter(), 0);
        assert_eq!(scheduler.last_epoch(), 0);
        assert!(scheduler.best_score().is_none());
    }

    /// Test ReduceLROnPlateau with custom options
    #[test]
    fn test_reducelr_custom_options() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0_f64, 2.0_f64], vec![2]).unwrap()];
        let mut optimizer = Sgd::new(params, 0.1_f64);
        let scheduler = ReduceLROnPlateau::with_options(
            &mut optimizer,
            Mode::Max,
            0.5_f64,
            5,
            true,      // verbose
            0.001_f64, // threshold
            ThresholdMode::Abs,
            2,               // cooldown
            vec![0.001_f64], // min_lr
            1e-9_f64,        // eps
        );

        assert_eq!(scheduler.mode(), Mode::Max);
        assert_eq!(scheduler.factor(), 0.5_f64);
        assert_eq!(scheduler.patience(), 5);
        assert!(scheduler.min_lr()[0] > 0.0_f64);
    }

    /// Test ReduceLROnPlateau step with improving metrics (min mode)
    #[test]
    fn test_reducelr_step_improving_min() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0_f64, 2.0_f64], vec![2]).unwrap()];
        let mut optimizer = Sgd::new(params, 0.1_f64);
        let mut scheduler = ReduceLROnPlateau::new(&mut optimizer, Mode::Min, 0.1_f64, 10);

        // First step - improving metric
        let reduced = scheduler.step(0.5_f64, None).unwrap();
        assert!(!reduced); // Should not reduce LR
        assert_eq!(scheduler.num_bad_epochs(), 0);
        assert_eq!(scheduler.best_score(), Some(0.5_f64));

        // Second step - further improvement
        let reduced = scheduler.step(0.3_f64, None).unwrap();
        assert!(!reduced); // Should not reduce LR
        assert_eq!(scheduler.num_bad_epochs(), 0);
        assert_eq!(scheduler.best_score(), Some(0.3_f64));
    }

    /// Test ReduceLROnPlateau step with improving metrics (max mode)
    #[test]
    fn test_reducelr_step_improving_max() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0_f64, 2.0_f64], vec![2]).unwrap()];
        let mut optimizer = Sgd::new(params, 0.1_f64);
        let mut scheduler = ReduceLROnPlateau::new(&mut optimizer, Mode::Max, 0.1_f64, 10);

        // First step - improving metric
        let reduced = scheduler.step(0.5_f64, None).unwrap();
        assert!(!reduced); // Should not reduce LR
        assert_eq!(scheduler.num_bad_epochs(), 0);
        assert_eq!(scheduler.best_score(), Some(0.5_f64));

        // Second step - further improvement
        let reduced = scheduler.step(0.7_f64, None).unwrap();
        assert!(!reduced); // Should not reduce LR
        assert_eq!(scheduler.num_bad_epochs(), 0);
        assert_eq!(scheduler.best_score(), Some(0.7_f64));
    }

    /// Test ReduceLROnPlateau step with degrading metrics
    #[test]
    fn test_reducelr_step_degrading() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0_f64, 2.0_f64], vec![2]).unwrap()];
        let mut optimizer = Sgd::new(params, 0.1_f64);
        let mut scheduler = ReduceLROnPlateau::new(&mut optimizer, Mode::Min, 0.1_f64, 3);

        // Set initial best score
        let _ = scheduler.step(0.5_f64, None).unwrap();

        // Degrading metrics for patience + 1 steps
        let _ = scheduler.step(0.6_f64, None).unwrap();
        let _ = scheduler.step(0.7_f64, None).unwrap();
        let _ = scheduler.step(0.8_f64, None).unwrap();
        let _ = scheduler.step(0.9_f64, None).unwrap(); // This should trigger LR reduction

        assert_eq!(scheduler.num_bad_epochs(), 0); // Reset after reduction
        assert_eq!(scheduler.cooldown_counter(), 0); // No cooldown set
                                                     // Check that LR was reduced (should be 0.1 * 0.1 = 0.01)
        assert_relative_eq!(scheduler.get_lr(0).unwrap(), 0.01_f64, epsilon = 1e-10);
    }

    /// Test ReduceLROnPlateau cooldown functionality
    #[test]
    fn test_reducelr_cooldown() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0_f64, 2.0_f64], vec![2]).unwrap()];
        let mut optimizer = Sgd::new(params, 0.1_f64);
        let mut scheduler = ReduceLROnPlateau::with_options(
            &mut optimizer,
            Mode::Min,
            0.1_f64,
            3,
            false,
            0.001_f64,
            ThresholdMode::Rel,
            2, // cooldown
            vec![0.0_f64],
            1e-8_f64,
        );

        // Set initial best score and trigger LR reduction
        let _ = scheduler.step(0.5_f64, None).unwrap();
        let _ = scheduler.step(0.6_f64, None).unwrap();
        let _ = scheduler.step(0.7_f64, None).unwrap();
        let _ = scheduler.step(0.8_f64, None).unwrap();
        let _ = scheduler.step(0.9_f64, None).unwrap(); // Triggers reduction

        assert_eq!(scheduler.cooldown_counter(), 2); // Should be in cooldown

        // During cooldown, should not reduce further
        let _ = scheduler.step(1.0_f64, None).unwrap();
        assert_eq!(scheduler.cooldown_counter(), 1); // Decremented
        assert_eq!(scheduler.num_bad_epochs(), 0); // Reset during cooldown
    }

    /// Test ReduceLROnPlateau threshold functionality
    #[test]
    fn test_reducelr_threshold() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0_f64, 2.0_f64], vec![2]).unwrap()];
        let mut optimizer = Sgd::new(params, 0.1_f64);
        let mut scheduler = ReduceLROnPlateau::with_options(
            &mut optimizer,
            Mode::Min,
            0.1_f64,
            10,
            false,
            0.1_f64, // High threshold
            ThresholdMode::Abs,
            0,
            vec![0.0_f64],
            1e-8_f64,
        );

        // Set initial best score
        let _ = scheduler.step(1.0_f64, None).unwrap();

        // Metric improvement within threshold should not update best score
        let reduced = scheduler.step(0.95_f64, None).unwrap(); // 1.0 - 0.05 = 0.95, threshold = 0.1
        assert!(!reduced);
        assert_eq!(scheduler.best_score(), Some(1.0_f64)); // Best score should not change
    }

    /// Test ReduceLROnPlateau minimum learning rate
    #[test]
    fn test_reducelr_min_lr() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0_f64, 2.0_f64], vec![2]).unwrap()];
        let mut optimizer = Sgd::new(params, 0.1_f64);
        let mut scheduler = ReduceLROnPlateau::with_options(
            &mut optimizer,
            Mode::Min,
            0.01_f64, // Very small factor
            1,        // Reduce after 1 bad epoch
            false,
            0.001_f64,
            ThresholdMode::Rel,
            0,
            vec![0.005_f64], // Set minimum LR
            1e-8_f64,
        );

        // Set initial best score
        let _ = scheduler.step(0.5_f64, None).unwrap();

        // Trigger multiple LR reductions
        for _ in 0..10 {
            let _ = scheduler.step(0.6_f64, None).unwrap();
        }

        // LR should not go below minimum
        assert!(scheduler.get_lr(0).unwrap() >= 0.005_f64);
    }

    /// Test ReduceLROnPlateau with multiple parameter groups
    #[test]
    fn test_reducelr_multiple_groups() {
        let params1 = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0_f64], vec![1]).unwrap()];
        let params2 = vec![Tensor::from_vec(CpuBackend::default(), vec![2.0_f64, 3.0_f64], vec![2]).unwrap()];

        let mut optimizer = Sgd::new(vec![], 0.1_f64); // Start empty
        optimizer.add_param_group(ParamGroup::new(params1, 0.1_f64, 0.0_f64));
        optimizer.add_param_group(ParamGroup::new(params2, 0.2_f64, 0.0_f64));

        let mut scheduler = ReduceLROnPlateau::with_options(
            &mut optimizer,
            Mode::Min,
            0.1_f64,
            5,
            false,
            0.001_f64,
            ThresholdMode::Rel,
            0,
            vec![0.0_f64, 0.0_f64], // min_lr for both groups
            1e-8_f64,
        );

        // Initial LRs
        assert_eq!(scheduler.get_lr(0), Some(0.1_f64)); // Empty group
        assert_eq!(scheduler.get_lr(1), Some(0.1_f64)); // First added group
        assert_eq!(scheduler.get_lr(2), Some(0.2_f64)); // Second added group

        // Trigger LR reduction
        let _ = scheduler.step(0.5_f64, None).unwrap();
        for _ in 0..6 {
            let _ = scheduler.step(0.6_f64, None).unwrap();
        }

        // Both groups should have reduced LR
        assert!(scheduler.get_lr(0).unwrap() < 0.1_f64); // Should be reduced
        assert!(scheduler.get_lr(1).unwrap() < 0.2_f64); // Should be reduced
    }

    /// Test ReduceLROnPlateau state tracking
    #[test]
    fn test_reducelr_state_tracking() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0_f64, 2.0_f64], vec![2]).unwrap()];
        let mut optimizer = Sgd::new(params, 0.1_f64);
        let mut scheduler = ReduceLROnPlateau::new(&mut optimizer, Mode::Min, 0.1_f64, 5);

        // Check initial state
        assert_eq!(scheduler.last_epoch(), 0);
        assert!(scheduler.best_score().is_none());
        assert_eq!(scheduler.num_bad_epochs(), 0);

        // Step with metric
        let _ = scheduler.step(0.5_f64, Some(10)).unwrap();

        assert_eq!(scheduler.last_epoch(), 10);
        assert_eq!(scheduler.best_score(), Some(0.5_f64));
        assert_eq!(scheduler.num_bad_epochs(), 0);
    }

    /// Test ReduceLROnPlateau getter methods
    #[test]
    fn test_reducelr_getters() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0_f64, 2.0_f64], vec![2]).unwrap()];
        let mut optimizer = Sgd::new(params, 0.1_f64);
        let scheduler = ReduceLROnPlateau::new(&mut optimizer, Mode::Min, 0.1_f64, 5);

        assert_eq!(scheduler.mode(), Mode::Min);
        assert_eq!(scheduler.factor(), 0.1_f64);
        assert_eq!(scheduler.patience(), 5);
        assert_eq!(scheduler.cooldown_counter(), 0);
        assert_eq!(scheduler.last_epoch(), 0);
        assert_eq!(scheduler.min_lr().len(), 1);
        assert_eq!(scheduler.min_lr()[0], 0.0_f64);
    }

    /// Test ReduceLROnPlateau is_better method
    #[test]
    fn test_reducelr_is_better() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0_f64, 2.0_f64], vec![2]).unwrap()];
        let mut optimizer_min = Sgd::new(params.clone(), 0.1_f64);
        let scheduler_min = ReduceLROnPlateau::new(&mut optimizer_min, Mode::Min, 0.1_f64, 5);

        let params2 = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0_f64, 2.0_f64], vec![2]).unwrap()];
        let mut optimizer_max = Sgd::new(params2, 0.1_f64);
        let scheduler_max = ReduceLROnPlateau::new(&mut optimizer_max, Mode::Max, 0.1_f64, 5);

        // Test Min mode
        assert!(scheduler_min.is_better(0.4_f64, Some(0.5_f64))); // 0.4 < 0.5, better
        assert!(!scheduler_min.is_better(0.6_f64, Some(0.5_f64))); // 0.6 > 0.5, not better

        // Test Max mode
        assert!(scheduler_max.is_better(0.6_f64, Some(0.5_f64))); // 0.6 > 0.5, better
        assert!(!scheduler_max.is_better(0.4_f64, Some(0.5_f64))); // 0.4 < 0.5, not better

        // Test with no best score (should always be better)
        assert!(scheduler_min.is_better(0.5_f64, None));
        assert!(scheduler_max.is_better(0.5_f64, None));
    }

    /// Test ReduceLROnPlateau with extreme values
    #[test]
    fn test_reducelr_extreme_values() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1e-10_f64, 1e10_f64], vec![2]).unwrap()];
        let mut optimizer = Sgd::new(params, 1e-5_f64);
        let mut scheduler = ReduceLROnPlateau::new(&mut optimizer, Mode::Min, 0.1_f64, 1);

        // Test with very small metric values
        let _ = scheduler.step(1e-15_f64, None).unwrap();
        assert!(scheduler.best_score().unwrap().is_finite());

        // Test with very large metric values
        let _ = scheduler.step(1e15_f64, None).unwrap();
        assert!(scheduler.get_lr(0).unwrap().is_finite());
    }
}

/// ReduceLROnPlateau scheduler
///
/// Reduces learning rate when a metric stops improving. The scheduler
/// monitors a quantity and if no improvement is seen for a 'patience' number
/// of epochs, the learning rate is reduced by a factor.
///
/// Compatible with PyTorch's `torch.optim.lr_scheduler.ReduceLROnPlateau`.
pub struct ReduceLROnPlateau<'a, O, T>
where
    O: Optimizer<T>,
    T: coeus_dtype::FloatDtype,
{
    optimizer: &'a mut O,
    mode: Mode,
    factor: T,
    patience: usize,
    verbose: bool,
    threshold: T,
    threshold_mode: ThresholdMode,
    cooldown: usize,
    min_lr: Vec<T>,
    #[allow(dead_code)]
    eps: T,
    scores: VecDeque<T>,
    best_score: Option<T>,
    num_bad_epochs: usize,
    cooldown_counter: usize,
    last_epoch: usize,
    base_lrs: Vec<T>,
}

/// Mode for determining whether to reduce LR
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    /// Reduce LR when metric stops decreasing (e.g., validation loss)
    Min,
    /// Reduce LR when metric stops increasing (e.g., validation accuracy)
    Max,
}

/// Threshold mode for determining improvement
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ThresholdMode {
    /// Absolute threshold
    Abs,
    /// Relative threshold
    Rel,
}

impl<'a, O, T> ReduceLROnPlateau<'a, O, T>
where
    O: Optimizer<T>,
    T: coeus_dtype::FloatDtype,
{
    /// Create a new ReduceLROnPlateau scheduler
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer to schedule
    /// * `mode` - Whether to reduce LR when metric stops decreasing ('min') or increasing ('max')
    /// * `factor` - Factor by which the learning rate will be reduced (default: 0.1)
    /// * `patience` - Number of epochs with no improvement after which LR will be reduced
    /// * `verbose` - If true, prints a message to stdout for each update (default: false)
    /// * `threshold` - Threshold for measuring the new optimum (default: 1e-4)
    /// * `threshold_mode` - Whether threshold comparison is done on absolute or relative change (default: 'rel')
    /// * `cooldown` - Number of epochs to wait before resuming normal operation after LR has been reduced
    /// * `min_lr` - Minimum learning rate(s) (default: 0.0)
    /// * `eps` - Minimal decay applied to LR (default: 1e-8)
    ///
    /// # Example
    /// ```rust
    /// use coeus_optim::{ReduceLROnPlateau, Sgd};
    /// use coeus_optim::reducelr::Mode;
    /// use coeus_tensor::{Tensor, CpuBackend};
    ///
    /// let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0], vec![2]).unwrap()];
    /// let mut optimizer = Sgd::new(params, 0.1).unwrap();
    /// let scheduler = ReduceLROnPlateau::new(
    ///     &mut optimizer,
    ///     Mode::Min,
    ///     0.1,
    ///     10
    /// );
    /// ```
    pub fn new(optimizer: &'a mut O, mode: Mode, factor: T, patience: usize) -> Self {
        let base_lrs: Vec<T> = (0..optimizer.param_groups().len())
            .filter_map(|i| optimizer.get_lr(i))
            .collect();

        let num_groups = base_lrs.len();
        let min_lr = vec![T::zero(); num_groups];

        Self {
            optimizer,
            mode,
            factor,
            patience,
            verbose: false,
            threshold: T::from(1e-4).unwrap(),
            threshold_mode: ThresholdMode::Rel,
            cooldown: 0,
            min_lr,
            eps: T::from(1e-8).unwrap(),
            scores: VecDeque::new(),
            best_score: None,
            num_bad_epochs: 0,
            cooldown_counter: 0,
            last_epoch: 0,
            base_lrs,
        }
    }

    /// Create with advanced configuration options
    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        optimizer: &'a mut O,
        mode: Mode,
        factor: T,
        patience: usize,
        verbose: bool,
        threshold: T,
        threshold_mode: ThresholdMode,
        cooldown: usize,
        min_lr: Vec<T>,
        eps: T,
    ) -> Self {
        let base_lrs = (0..optimizer.param_groups().len())
            .filter_map(|i| optimizer.get_lr(i))
            .collect();

        Self {
            optimizer,
            mode,
            factor,
            patience,
            verbose,
            threshold,
            threshold_mode,
            cooldown,
            min_lr,
            eps,
            scores: VecDeque::new(),
            best_score: None,
            num_bad_epochs: 0,
            cooldown_counter: 0,
            last_epoch: 0,
            base_lrs,
        }
    }

    /// Step the scheduler with a metric value
    ///
    /// # Arguments
    /// * `metrics` - Current metric value to monitor (e.g., validation loss)
    /// * `epoch` - Optional current epoch number
    ///
    /// # Returns
    /// true if learning rate was reduced, false otherwise
    pub fn step(&mut self, metrics: T, epoch: Option<usize>) -> Result<bool> {
        if let Some(epoch) = epoch {
            self.last_epoch = epoch;
        } else {
            self.last_epoch += 1;
        }

        // In cooldown period, don't reduce LR
        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            self.num_bad_epochs = 0;
            return Ok(false);
        }

        // Add current score to history
        self.scores.push_back(metrics);

        // Keep only the last patience + 1 scores
        while self.scores.len() > self.patience + 1 {
            self.scores.pop_front();
        }

        let reduced = if self.is_better(metrics, self.best_score) {
            // Metric improved, reset bad epochs counter
            self.best_score = Some(metrics);
            self.num_bad_epochs = 0;
            false
        } else {
            // Metric didn't improve
            self.num_bad_epochs += 1;

            if self.num_bad_epochs > self.patience {
                // Reduce learning rate
                self.reduce_lr()?;
                self.cooldown_counter = self.cooldown;
                self.num_bad_epochs = 0;
                true
            } else {
                false
            }
        };

        Ok(reduced)
    }

    /// Check if current score is better than best score
    fn is_better(&self, current: T, best: Option<T>) -> bool {
        match best {
            None => true,
            Some(best_val) => match self.mode {
                Mode::Min => {
                    // For minimization, lower is better
                    match self.threshold_mode {
                        ThresholdMode::Rel => current < best_val * (T::one() - self.threshold),
                        ThresholdMode::Abs => current < best_val - self.threshold,
                    }
                }
                Mode::Max => {
                    // For maximization, higher is better
                    match self.threshold_mode {
                        ThresholdMode::Rel => current > best_val * (T::one() + self.threshold),
                        ThresholdMode::Abs => current > best_val + self.threshold,
                    }
                }
            },
        }
    }

    /// Reduce learning rates by the factor
    fn reduce_lr(&mut self) -> Result<()> {
        for (i, base_lr) in self.base_lrs.iter().enumerate() {
            let new_lr = *base_lr * self.factor;
            let min_lr = self.min_lr.get(i).copied().unwrap_or(T::zero());

            // Ensure LR doesn't go below minimum
            let final_lr = if new_lr < min_lr { min_lr } else { new_lr };

            self.optimizer.set_lr(i, final_lr)?;

            if self.verbose {
                println!(
                    "Reducing learning rate for group {}: {:.6} -> {:.6}",
                    i,
                    num_traits::cast::ToPrimitive::to_f64(base_lr).unwrap(),
                    num_traits::cast::ToPrimitive::to_f64(&final_lr).unwrap()
                );
            }
        }
        Ok(())
    }

    /// Get the current learning rate for a parameter group
    pub fn get_lr(&self, group_index: usize) -> Option<T> {
        self.optimizer.get_lr(group_index)
    }

    /// Get the best score seen so far
    pub fn best_score(&self) -> Option<T> {
        self.best_score
    }

    /// Get the number of bad epochs
    pub fn num_bad_epochs(&self) -> usize {
        self.num_bad_epochs
    }

    /// Get the patience value
    pub fn patience(&self) -> usize {
        self.patience
    }

    /// Get the reduction factor
    pub fn factor(&self) -> T {
        self.factor
    }

    /// Get the mode
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Get the cooldown counter
    pub fn cooldown_counter(&self) -> usize {
        self.cooldown_counter
    }

    /// Get the minimum learning rates
    pub fn min_lr(&self) -> &[T] {
        &self.min_lr
    }

    /// Get the last epoch
    pub fn last_epoch(&self) -> usize {
        self.last_epoch
    }
}
