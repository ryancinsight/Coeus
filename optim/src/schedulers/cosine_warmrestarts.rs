//! CosineAnnealingWarmRestarts scheduler implementation
//!
//! Implements the Cosine Annealing with Warm Restarts learning rate scheduler.
//! This scheduler anneals the learning rate to zero following a cosine curve,
//! then restarts from the initial learning rate with a period that increases
//! by a factor of T_mult each restart.
//!
//! ## Mathematical Foundation
//!
//! The learning rate at step t is given by:
//!
//! ```text
//! η_t = η_min + (η_max - η_min) * (1 + cos(π * T_cur / T_i)) / 2
//! ```
//!
//! where:
//! - T_cur is the number of steps since the last restart
//! - T_i is the current period length
//! - η_min is the minimum learning rate
//! - η_max is the maximum learning rate
//!
//! ## References
//!
//! - [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)

use crate::{Optimizer, Result};

/// Cosine Annealing with Warm Restarts scheduler
///
/// Implements cosine annealing with periodic restarts where the period length
/// increases by a multiplicative factor after each restart.
pub struct CosineAnnealingWarmRestarts<'a, O: Optimizer<T>, T: coeus_dtype::FloatDtype> {
    optimizer: &'a mut O,
    /// Initial learning rate
    eta_min: T,
    /// Maximum learning rate
    eta_max: T,
    /// Period multiplier
    t_mult: T,
    /// Current period length
    t_0: usize,
    /// Current step within period
    t_cur: usize,
    /// Last restart step
    t_last: usize,
}

impl<'a, O: Optimizer<T>, T: coeus_dtype::FloatDtype> CosineAnnealingWarmRestarts<'a, O, T> {
    /// Create a new CosineAnnealingWarmRestarts scheduler
    ///
    /// # Arguments
    /// * `optimizer` - Optimizer to schedule
    /// * `eta_min` - Minimum learning rate
    /// * `eta_max` - Maximum learning rate
    /// * `t_0` - Initial period length
    /// * `t_mult` - Period multiplier (default: 1.0)
    ///
    /// # Example
    /// ```rust
    /// use coeus_optim::{Adam, CosineAnnealingWarmRestarts};
    /// use coeus_tensor::Tensor;
    ///
    /// let params = vec![Tensor::from_vec(vec![1.0], vec![1])];
    /// let mut optimizer = Adam::new(params, 0.001);
    /// let mut scheduler = CosineAnnealingWarmRestarts::new(&mut optimizer, 0.0, 0.01, 10);
    /// ```
    pub fn new(optimizer: &'a mut O, eta_min: T, eta_max: T, t_0: usize) -> Self {
        Self::with_t_mult(optimizer, eta_min, eta_max, t_0, T::one())
    }

    /// Create CosineAnnealingWarmRestarts with period multiplier
    ///
    /// # Arguments
    /// * `optimizer` - Optimizer to schedule
    /// * `eta_min` - Minimum learning rate
    /// * `eta_max` - Maximum learning rate
    /// * `t_0` - Initial period length
    /// * `t_mult` - Period multiplier
    pub fn with_t_mult(
        optimizer: &'a mut O,
        eta_min: T,
        eta_max: T,
        t_0: usize,
        t_mult: T,
    ) -> Self {
        // Set initial learning rate
        for group in optimizer.param_groups_mut() {
            group.lr = eta_max;
        }

        Self {
            optimizer,
            eta_min,
            eta_max,
            t_mult,
            t_0,
            t_cur: 0,
            t_last: 0,
        }
    }

    /// Get the current period length
    pub fn t_0(&self) -> usize {
        self.t_0
    }

    /// Get the current step within the period
    pub fn t_cur(&self) -> usize {
        self.t_cur
    }

    /// Get the minimum learning rate
    pub fn eta_min(&self) -> T {
        self.eta_min
    }

    /// Get the maximum learning rate
    pub fn eta_max(&self) -> T {
        self.eta_max
    }

    /// Get the period multiplier
    pub fn t_mult(&self) -> T {
        self.t_mult
    }

    /// Calculate the learning rate for the current step
    fn calculate_lr(&self) -> T {
        // Check if we need to restart
        if self.t_cur == self.t_0 {
            return self.eta_min;
        }

        // Cosine annealing formula
        let progress = T::from(self.t_cur as f64 / self.t_0 as f64).unwrap();
        let cosine = (progress * T::from(std::f64::consts::PI).unwrap()).cos();

        self.eta_min + (self.eta_max - self.eta_min) * (T::one() + cosine) / (T::one() + T::one())
    }

    /// Get the percentage of the current period completed
    pub fn progress(&self) -> f64 {
        if self.t_0 == 0 {
            1.0
        } else {
            self.t_cur as f64 / self.t_0 as f64
        }
    }
}

impl<'a, O: Optimizer<T>, T: coeus_dtype::FloatDtype> CosineAnnealingWarmRestarts<'a, O, T> {
    /// Take a step in the learning rate schedule
    ///
    /// Updates the learning rate based on the current step and restarts
    /// the schedule when the period is complete.
    pub fn step(&mut self) -> Result<()> {
        let new_lr = self.calculate_lr();

        for group in self.optimizer.param_groups_mut() {
            group.lr = new_lr;
        }

        self.t_cur += 1;

        // Check if we need to restart after incrementing
        if self.t_cur == self.t_0 {
            self.t_cur = 0;
            self.t_last = self.t_0;
            let t_mult_f64 = num_traits::ToPrimitive::to_f64(&self.t_mult).unwrap_or(1.0);
            self.t_0 = ((self.t_0 as f64) * t_mult_f64) as usize;

            // Recalculate LR after restart
            let new_lr = self.calculate_lr();
            for group in self.optimizer.param_groups_mut() {
                group.lr = new_lr;
            }
        }

        Ok(())
    }

    /// Get the current learning rates for all parameter groups
    pub fn get_lr(&self) -> Vec<T> {
        (0..self.optimizer.param_groups().len())
            .filter_map(|i| self.optimizer.get_lr(i))
            .collect()
    }

    /// Get the last set learning rates (same as current for CosineAnnealingWarmRestarts)
    pub fn get_last_lr(&self) -> Vec<T> {
        self.get_lr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Adam, ParamGroup};
    use coeus_tensor::Tensor;

    #[test]
    fn test_cosine_warm_restarts_creation() {
        let params = vec![Tensor::from_vec(vec![1.0], vec![1])];
        let mut optimizer = Adam::new(params, 0.001);
        let scheduler = CosineAnnealingWarmRestarts::new(&mut optimizer, 0.0, 0.01, 10);

        assert_eq!(scheduler.t_0(), 10);
        assert_eq!(scheduler.t_cur(), 0);
        assert_eq!(scheduler.eta_min(), 0.0_f64);
        assert_eq!(scheduler.eta_max(), 0.01_f64);
        assert_eq!(scheduler.t_mult(), 1.0_f64);
        assert_eq!(scheduler.optimizer.param_groups()[0].lr, 0.01_f64);
    }

    #[test]
    fn test_cosine_warm_restarts_with_t_mult() {
        let params = vec![Tensor::from_vec(vec![1.0], vec![1])];
        let mut optimizer = Adam::new(params, 0.001);
        let scheduler =
            CosineAnnealingWarmRestarts::with_t_mult(&mut optimizer, 0.0, 0.01, 10, 2.0);

        assert_eq!(scheduler.t_mult(), 2.0_f64);
    }

    #[test]
    fn test_cosine_warm_restarts_step() {
        let params = vec![Tensor::from_vec(vec![1.0], vec![1])];
        let mut optimizer = Adam::new(params, 0.001);
        let mut scheduler = CosineAnnealingWarmRestarts::new(&mut optimizer, 0.0, 0.01, 5);

        // Initial LR should be eta_max
        assert_eq!(scheduler.optimizer.param_groups()[0].lr, 0.01_f64);
        println!("Initial t_0: {}", scheduler.t_0);

        // Take steps until restart (need to take 5 steps to reach t_cur = 5)
        for _ in 0..5 {
            scheduler.step().unwrap();
        }

        // Should have restarted (t_cur should be reset)
        assert_eq!(scheduler.t_cur(), 0);
        assert_eq!(scheduler.t_0(), 5); // t_0 unchanged since t_mult = 1.0

        // LR should be back to eta_max after restart
        assert_eq!(scheduler.optimizer.param_groups()[0].lr, 0.01_f64);
    }

    #[test]
    fn test_cosine_warm_restarts_with_t_mult_step() {
        let params = vec![Tensor::from_vec(vec![1.0], vec![1])];
        let mut optimizer = Adam::new(params, 0.001);
        let mut scheduler =
            CosineAnnealingWarmRestarts::with_t_mult(&mut optimizer, 0.0, 0.01, 3, 2.0);

        // Take steps until first restart
        for _ in 0..3 {
            scheduler.step().unwrap();
        }

        // Should have restarted and doubled the period
        assert_eq!(scheduler.t_cur(), 0);
        assert_eq!(scheduler.t_0(), 6); // 3 * 2 = 6

        // Take more steps
        for _ in 0..6 {
            scheduler.step().unwrap();
        }

        // Should have restarted again and doubled the period again
        assert_eq!(scheduler.t_cur(), 0);
        assert_eq!(scheduler.t_0(), 12); // 6 * 2 = 12
    }

    #[test]
    fn test_cosine_warm_restarts_progress() {
        let params = vec![Tensor::from_vec(vec![1.0], vec![1])];
        let mut optimizer = Adam::new(params, 0.001);
        let mut scheduler = CosineAnnealingWarmRestarts::new(&mut optimizer, 0.0, 0.01, 10);

        assert_eq!(scheduler.progress(), 0.0);

        // Take 5 steps
        for _ in 0..5 {
            scheduler.step().unwrap();
        }

        assert_eq!(scheduler.progress(), 0.5);

        // Take remaining 5 steps
        for _ in 5..10 {
            scheduler.step().unwrap();
        }

        // The last step should have triggered a restart
        // Since we took exactly 10 steps, t_cur should be 10, t_0 should be 10
        // After restart, t_cur should be 0, t_0 should be 10
        assert_eq!(scheduler.t_cur(), 0);
        assert_eq!(scheduler.progress(), 0.0);
    }

    #[test]
    fn test_cosine_warm_restarts_multiple_param_groups() {
        let params1 = vec![Tensor::from_vec(vec![1.0], vec![1])];
        let params2 = vec![Tensor::from_vec(vec![2.0], vec![1])];

        let mut optimizer = Adam::new(params1, 0.001);
        optimizer.add_param_group(ParamGroup::new(params2, 0.001, 0.0));

        let mut scheduler = CosineAnnealingWarmRestarts::new(&mut optimizer, 0.0, 0.01, 5);

        // All parameter groups should have the same learning rate
        let lr1 = scheduler.optimizer.param_groups()[0].lr;
        let lr2 = scheduler.optimizer.param_groups()[1].lr;
        assert_eq!(lr1, lr2);

        scheduler.step().unwrap();

        // All parameter groups should still have the same learning rate
        let lr1_updated = scheduler.optimizer.param_groups()[0].lr;
        let lr2_updated = scheduler.optimizer.param_groups()[1].lr;
        assert_eq!(lr1_updated, lr2_updated);
        assert_eq!(lr1_updated, lr1); // Should still be eta_max after first step
    }
}
