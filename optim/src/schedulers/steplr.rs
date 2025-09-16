//! Step learning rate scheduler
//!
//! Decays the learning rate by a factor every `step_size` epochs,
//! compatible with PyTorch's `torch.optim.lr_scheduler.StepLR`.

use crate::{Optimizer, Result};

/// Step learning rate scheduler
///
/// Decays the learning rate by `gamma` every `step_size` epochs.
/// Compatible with PyTorch's `torch.optim.lr_scheduler.StepLR`.
///
/// ## Mathematical Formula
///
/// ```text
/// lr = base_lr * gamma^(floor(epoch / step_size))
/// ```
///
/// # Example
/// ```rust
/// use coeus_optim::{Sgd, StepLR};
/// use coeus_tensor::Tensor;
///
/// let params = vec![Tensor::from_vec(vec![1.0, 2.0], vec![2])];
/// let mut optimizer = Sgd::new(params, 0.1);
///
/// // Decay learning rate by 0.5 every 10 epochs
/// let mut scheduler = StepLR::new(&mut optimizer, 10, 0.5);
///
/// for epoch in 0..50 {
///     // Training loop...
///
///     // Step the scheduler every epoch
///     scheduler.step();
/// }
/// ```
pub struct StepLR<'a, O, T>
where
    O: Optimizer<T>,
    T: coeus_dtype::FloatDtype,
{
    optimizer: &'a mut O,
    step_size: usize,
    gamma: T,
    last_epoch: usize,
    base_lrs: Vec<T>,
}

impl<'a, O, T> StepLR<'a, O, T>
where
    O: Optimizer<T>,
    T: coeus_dtype::FloatDtype,
{
    /// Create a new StepLR scheduler
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer to schedule
    /// * `step_size` - Number of epochs between learning rate decays
    /// * `gamma` - Multiplicative factor of learning rate decay
    pub fn new(optimizer: &'a mut O, step_size: usize, gamma: T) -> Self {
        let base_lrs = (0..optimizer.param_groups().len())
            .filter_map(|i| optimizer.get_lr(i))
            .collect();

        Self {
            optimizer,
            step_size,
            gamma,
            last_epoch: 0,
            base_lrs,
        }
    }

    /// Get the current learning rate for a parameter group
    pub fn get_lr(&self, group_index: usize) -> Option<T> {
        self.base_lrs.get(group_index).map(|base_lr| {
            // lr = base_lr * gamma^(floor(last_epoch / step_size))
            let factor = self.last_epoch / self.step_size;
            let mut lr = *base_lr;
            for _ in 0..factor {
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

    /// Get the step size
    pub fn step_size(&self) -> usize {
        self.step_size
    }

    /// Get the gamma factor
    pub fn gamma(&self) -> T {
        self.gamma
    }

    /// Get the last epoch
    pub fn last_epoch(&self) -> usize {
        self.last_epoch
    }

    /// Get the base learning rates
    pub fn base_lrs(&self) -> &[T] {
        &self.base_lrs
    }
}
