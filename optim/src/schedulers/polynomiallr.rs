//! PolynomialLR scheduler implementation
//!
//! Implements the Polynomial learning rate scheduler which decays the learning
//! rate following a polynomial function. The learning rate decays from the
//! initial value to the minimum value following a power law.
//!
//! ## Mathematical Foundation
//!
//! The learning rate at step t is given by:
//!
//! ```text
//! η_t = η_min + (η_max - η_min) * (1 - t / total_steps)^power
//! ```
//!
//! where:
//! - η_max is the initial learning rate
//! - η_min is the minimum learning rate
//! - t is the current step
//! - total_steps is the total number of steps
//! - power is the polynomial power (default: 1.0 for linear decay)

use crate::{Optimizer, Result};

/// Polynomial learning rate scheduler
///
/// Implements polynomial learning rate decay where the learning rate decreases
/// following a power law from the initial value to the minimum value.
pub struct PolynomialLR<'a, O: Optimizer<T>, T: coeus_dtype::FloatDtype> {
    optimizer: &'a mut O,
    /// Initial learning rate
    eta_max: T,
    /// Minimum learning rate
    eta_min: T,
    /// Total number of training steps
    total_steps: usize,
    /// Current step
    current_step: usize,
    /// Polynomial power (default: 1.0 for linear decay)
    power: T,
}

impl<'a, O: Optimizer<T>, T: coeus_dtype::FloatDtype> PolynomialLR<'a, O, T> {
    /// Create a new PolynomialLR scheduler
    ///
    /// # Arguments
    /// * `optimizer` - Optimizer to schedule
    /// * `eta_max` - Initial learning rate
    /// * `eta_min` - Minimum learning rate
    /// * `total_steps` - Total number of training steps
    /// * `power` - Polynomial power (default: 1.0 for linear decay)
    ///
    /// # Example
    /// ```rust
    /// use coeus_optim::{Adam, PolynomialLR};
    /// use coeus_tensor::{Tensor, CpuBackend};
    ///
    /// let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
    /// let mut optimizer = Adam::new(params, 0.001);
    /// let mut scheduler = PolynomialLR::new(&mut optimizer, 0.001, 1e-6, 1000, 1.0);
    /// ```
    pub fn new(optimizer: &'a mut O, eta_max: T, eta_min: T, total_steps: usize, power: T) -> Self {
        // Set initial learning rate
        let initial_lr = if total_steps == 0 { eta_min } else { eta_max };

        for group in optimizer.param_groups_mut() {
            group.lr = initial_lr;
        }

        Self {
            optimizer,
            eta_max,
            eta_min,
            total_steps,
            current_step: 0,
            power,
        }
    }

    /// Get the current step
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Get the total number of steps
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Get the initial learning rate
    pub fn eta_max(&self) -> T {
        self.eta_max
    }

    /// Get the minimum learning rate
    pub fn eta_min(&self) -> T {
        self.eta_min
    }

    /// Get the polynomial power
    pub fn power(&self) -> T {
        self.power
    }

    /// Calculate the learning rate for the current step
    fn calculate_lr(&self, step: usize) -> T {
        if self.total_steps == 0 || step >= self.total_steps {
            return self.eta_min;
        }

        let progress = T::from(step as f64 / self.total_steps as f64).unwrap();
        let decay = (T::one() - progress).powf(self.power);

        self.eta_min + (self.eta_max - self.eta_min) * decay
    }

    /// Get the percentage of training completed
    pub fn progress(&self) -> f64 {
        if self.total_steps == 0 {
            1.0
        } else {
            self.current_step as f64 / self.total_steps as f64
        }
    }

    /// Check if training is complete
    pub fn is_done(&self) -> bool {
        self.current_step >= self.total_steps
    }
}

impl<'a, O: Optimizer<T>, T: coeus_dtype::FloatDtype> PolynomialLR<'a, O, T> {
    /// Take a step in the learning rate schedule
    ///
    /// Updates the learning rate based on the current step.
    pub fn step(&mut self) -> Result<()> {
        if self.current_step >= self.total_steps {
            return Ok(()); // Already completed
        }

        // Increment step first to calculate learning rate for the next step
        self.current_step += 1;

        let new_lr = self.calculate_lr(self.current_step);

        for group in self.optimizer.param_groups_mut() {
            group.lr = new_lr;
        }

        Ok(())
    }

    /// Get the current learning rates for all parameter groups
    pub fn get_lr(&self) -> Vec<T> {
        (0..self.optimizer.param_groups().len())
            .filter_map(|i| self.optimizer.get_lr(i))
            .collect()
    }

    /// Get the last set learning rates (same as current for PolynomialLR)
    pub fn get_last_lr(&self) -> Vec<T> {
        self.get_lr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Adam, ParamGroup};
    use coeus_tensor::{Tensor, CpuBackend};

    #[test]
    fn test_polynomial_lr_creation() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.001);
        let scheduler: PolynomialLR<'_, Adam<f64>, f64> = PolynomialLR::new(&mut optimizer, 0.001, 1e-6, 1000, 1.0);

        assert_eq!(scheduler.total_steps(), 1000);
        assert_eq!(scheduler.current_step(), 0);
        assert_eq!(scheduler.eta_max(), 0.001_f64);
        assert_eq!(scheduler.eta_min(), 1e-6_f64);
        assert_eq!(scheduler.power(), 1.0_f64);
        assert_eq!(scheduler.optimizer.param_groups()[0].lr, 0.001_f64);
    }

    #[test]
    fn test_polynomial_lr_step() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.001);
        let mut scheduler: PolynomialLR<'_, Adam<f64>, f64> = PolynomialLR::new(&mut optimizer, 0.001, 1e-6, 5, 1.0);

        // Initial LR should be eta_max
        assert_eq!(scheduler.optimizer.param_groups()[0].lr, 0.001_f64);

        // Take first step
        scheduler.step().unwrap();
        assert_eq!(scheduler.current_step(), 1);

        let lr1 = scheduler.optimizer.param_groups()[0].lr;
        // Should be decreasing from eta_max towards eta_min
        assert!(lr1 < 0.001_f64);
        assert!(lr1 > 1e-6_f64);

        // Take steps until completion
        for _ in 1..5 {
            scheduler.step().unwrap();
        }

        // Should be complete
        assert!(scheduler.is_done());
        assert_eq!(scheduler.current_step(), 5);

        // Final LR should be eta_min
        let final_lr = scheduler.optimizer.param_groups()[0].lr;
        assert_eq!(final_lr, 1e-6_f64);
    }

    #[test]
    fn test_polynomial_lr_power() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.001);
        let mut scheduler: PolynomialLR<'_, Adam<f64>, f64> = PolynomialLR::new(&mut optimizer, 0.001, 0.0, 4, 2.0);

        // Take 2 steps (halfway through)
        for _ in 0..2 {
            scheduler.step().unwrap();
        }

        // With power=2, decay should be (1 - 0.5)^2 = 0.25
        // So LR should be 0.0 + (0.001 - 0.0) * 0.25 = 0.00025
        let lr_halfway = scheduler.optimizer.param_groups()[0].lr;
        assert!((lr_halfway - 0.00025_f64).abs() < 1e-5_f64);

        // Take remaining steps
        for _ in 2..4 {
            scheduler.step().unwrap();
        }

        // Final LR should be eta_min
        assert_eq!(scheduler.optimizer.param_groups()[0].lr, 0.0_f64);
    }

    #[test]
    fn test_polynomial_lr_progress() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.001);
        let mut scheduler: PolynomialLR<'_, Adam<f64>, f64> = PolynomialLR::new(&mut optimizer, 0.001, 1e-6, 100, 1.0);

        assert_eq!(scheduler.progress(), 0.0);
        assert!(!scheduler.is_done());

        // Take 50 steps
        for _ in 0..50 {
            scheduler.step().unwrap();
        }

        assert_eq!(scheduler.progress(), 0.5);
        assert!(!scheduler.is_done());

        // Take remaining 50 steps
        for _ in 50..100 {
            scheduler.step().unwrap();
        }

        assert_eq!(scheduler.progress(), 1.0);
        assert!(scheduler.is_done());

        // Additional steps should not change anything
        scheduler.step().unwrap();
        assert_eq!(scheduler.progress(), 1.0);
        assert!(scheduler.is_done());
    }

    #[test]
    fn test_polynomial_lr_multiple_param_groups() {
        let params1 = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
        let params2 = vec![Tensor::from_vec(CpuBackend::default(), vec![2.0], vec![1]).unwrap()];

        let mut optimizer = Adam::new(params1, 0.001);
        optimizer.add_param_group(ParamGroup::new(params2, 0.001, 0.0));

        let mut scheduler: PolynomialLR<'_, Adam<f64>, f64> = PolynomialLR::new(&mut optimizer, 0.001, 1e-6, 10, 1.0);

        // All parameter groups should have the same learning rate
        let lr1 = scheduler.optimizer.param_groups()[0].lr;
        let lr2 = scheduler.optimizer.param_groups()[1].lr;
        assert_eq!(lr1, lr2);

        scheduler.step().unwrap();

        // All parameter groups should still have the same learning rate
        let lr1_updated = scheduler.optimizer.param_groups()[0].lr;
        let lr2_updated = scheduler.optimizer.param_groups()[1].lr;
        assert_eq!(lr1_updated, lr2_updated);
        assert!(lr1_updated < lr1); // Should have decreased
    }

    #[test]
    fn test_polynomial_lr_zero_total_steps() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.001);
        let scheduler: PolynomialLR<'_, Adam<f64>, f64> = PolynomialLR::new(&mut optimizer, 0.001, 1e-6, 0, 1.0);

        // With zero total steps, should immediately be at eta_min
        assert!(scheduler.is_done());
        assert!((scheduler.optimizer.param_groups()[0].lr - 1e-6_f64).abs() < 1e-10_f64);
    }
}
