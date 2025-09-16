//! ReduceLROnPlateau learning rate scheduler
//!
//! Implements ReduceLROnPlateau learning rate schedule,
//! compatible with PyTorch's `torch.optim.lr_scheduler.ReduceLROnPlateau`.

use crate::{Optimizer, Result};
use std::collections::VecDeque;

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
    /// use coeus_tensor::Tensor;
    ///
    /// let params = vec![Tensor::from_vec(vec![1.0, 2.0], vec![2])];
    /// let mut optimizer = Sgd::new(params, 0.1);
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
