//! ExponentialLR learning rate scheduler
//!
//! Implements exponential decay of the learning rate,
//! compatible with PyTorch's `torch.optim.lr_scheduler.ExponentialLR`.

use crate::{Optimizer, Result};

/// ExponentialLR scheduler
///
/// Decays the learning rate exponentially at each step.
///
/// ## Mathematical Formula
///
/// ```text
/// lr_t = lr_0 * γ^t
/// ```
///
/// Where γ is the decay factor (typically < 1.0), and t is the epoch number.
///
/// Compatible with PyTorch's `torch.optim.lr_scheduler.ExponentialLR`.
pub struct ExponentialLR<'a, O, T>
where
    O: Optimizer<T>,
    T: coeus_dtype::FloatDtype,
{
    optimizer: &'a mut O,
    gamma: T,
    last_epoch: usize,
    base_lrs: Vec<T>,
}

impl<'a, O, T> ExponentialLR<'a, O, T>
where
    O: Optimizer<T>,
    T: coeus_dtype::FloatDtype,
{
    /// Create a new ExponentialLR scheduler
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer to schedule
    /// * `gamma` - Multiplicative factor of learning rate decay (default: 0.9)
    ///
    /// # Example
    /// ```rust
    /// use coeus_optim::{ExponentialLR, Sgd};
    /// use coeus_tensor::Tensor;
    ///
    /// let params = vec![Tensor::from_vec(vec![1.0, 2.0], vec![2])];
    /// let mut optimizer = Sgd::new(params, 0.1);
    /// let scheduler = ExponentialLR::new(&mut optimizer, 0.9);
    /// ```
    pub fn new(optimizer: &'a mut O, gamma: T) -> Self {
        let base_lrs = (0..optimizer.param_groups().len())
            .filter_map(|i| optimizer.get_lr(i))
            .collect();

        Self {
            optimizer,
            gamma,
            last_epoch: 0,
            base_lrs,
        }
    }

    /// Get the current learning rate for a parameter group
    pub fn get_lr(&self, group_index: usize) -> Option<T> {
        self.base_lrs.get(group_index).map(|base_lr| {
            // lr_t = lr_0 * γ^t
            let mut lr = *base_lr;
            for _ in 0..self.last_epoch {
                lr = lr * self.gamma;
            }
            lr
        })
    }

    /// Step the scheduler
    ///
    /// Updates the learning rate according to the schedule.
    /// Should be called once per epoch.
    pub fn step(&mut self) -> Result<()> {
        self.last_epoch += 1;
        self.update_lr()
    }

    /// Step the scheduler with custom epoch
    ///
    /// # Arguments
    /// * `epoch` - Current epoch number
    pub fn step_epoch(&mut self, epoch: usize) -> Result<()> {
        self.last_epoch = epoch;
        self.update_lr()
    }

    /// Update learning rates in the optimizer
    fn update_lr(&mut self) -> Result<()> {
        for (i, base_lr) in self.base_lrs.iter().enumerate() {
            let new_lr = self.get_lr(i).unwrap_or(*base_lr);
            self.optimizer.set_lr(i, new_lr)?;
        }
        Ok(())
    }

    /// Get the decay factor gamma
    pub fn gamma(&self) -> T {
        self.gamma
    }

    /// Get the last epoch processed
    pub fn last_epoch(&self) -> usize {
        self.last_epoch
    }

    /// Get the base learning rates
    pub fn base_lrs(&self) -> &[T] {
        &self.base_lrs
    }
}
