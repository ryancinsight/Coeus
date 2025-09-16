//! CosineAnnealingLR learning rate scheduler
//!
//! Implements cosine annealing learning rate schedule,
//! compatible with PyTorch's `torch.optim.lr_scheduler.CosineAnnealingLR`.

use crate::{Optimizer, Result};

/// CosineAnnealingLR scheduler
///
/// Sets the learning rate using a cosine annealing schedule. The learning rate
/// follows a cosine curve from the initial value to a minimum value over T_max epochs.
///
/// ## Mathematical Formula
///
/// ```text
/// η_t = η_min + (η_max - η_min) * (1 + cos(π * t / T_max)) / 2
/// ```
///
/// Where η_max is the initial learning rate, η_min is the minimum learning rate,
/// t is the current epoch, and T_max is the maximum number of epochs.
///
/// Compatible with PyTorch's `torch.optim.lr_scheduler.CosineAnnealingLR`.
pub struct CosineAnnealingLR<'a, O, T>
where
    O: Optimizer<T>,
    T: coeus_dtype::FloatDtype,
{
    optimizer: &'a mut O,
    t_max: usize,
    eta_min: T,
    last_epoch: usize,
    base_lrs: Vec<T>,
}

impl<'a, O, T> CosineAnnealingLR<'a, O, T>
where
    O: Optimizer<T>,
    T: coeus_dtype::FloatDtype,
{
    /// Create a new CosineAnnealingLR scheduler
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer to schedule
    /// * `t_max` - Maximum number of epochs for annealing
    /// * `eta_min` - Minimum learning rate (default: 0.0)
    ///
    /// # Example
    /// ```rust
    /// use coeus_optim::{CosineAnnealingLR, Sgd};
    /// use coeus_tensor::Tensor;
    ///
    /// let params = vec![Tensor::from_vec(vec![1.0, 2.0], vec![2])];
    /// let mut optimizer = Sgd::new(params, 0.1);
    /// let scheduler = CosineAnnealingLR::new(&mut optimizer, 10, 0.0);
    /// ```
    pub fn new(optimizer: &'a mut O, t_max: usize, eta_min: T) -> Self {
        let base_lrs = (0..optimizer.param_groups().len())
            .filter_map(|i| optimizer.get_lr(i))
            .collect();

        Self {
            optimizer,
            t_max,
            eta_min,
            last_epoch: 0,
            base_lrs,
        }
    }

    /// Get the current learning rate for a parameter group
    pub fn get_lr(&self, group_index: usize) -> Option<T> {
        self.base_lrs.get(group_index).map(|base_lr| {
            // η_t = η_min + (η_max - η_min) * (1 + cos(π * t / T_max)) / 2
            let eta_max = *base_lr;
            let eta_min = self.eta_min;

            let t = self.last_epoch as f64;
            let t_max = self.t_max as f64;

            let cos_arg = std::f64::consts::PI * t / t_max;
            let cos_val = cos_arg.cos();

            let factor =
                (T::from(1.0).unwrap() + T::from(cos_val).unwrap()) / T::from(2.0).unwrap();

            eta_min + (eta_max - eta_min) * factor
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

    /// Get the maximum number of epochs
    pub fn t_max(&self) -> usize {
        self.t_max
    }

    /// Get the minimum learning rate
    pub fn eta_min(&self) -> T {
        self.eta_min
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
