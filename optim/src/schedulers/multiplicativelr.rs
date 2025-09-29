//! MultiplicativeLR scheduler implementation
//!
//! Implements the MultiplicativeLR learning rate scheduler which multiplies the
//! learning rate by a given factor at each step. This is similar to exponential
//! decay but allows for more flexible multiplicative factors.
//!
//! ## Mathematical Foundation
//!
//! The learning rate at step t is given by:
//!
//! ```text
//! η_t = η_{t-1} * lr_lambda
//! ```
//!
//! where:
//! - η_t is the learning rate at step t
//! - lr_lambda is the multiplicative factor

use crate::{Optimizer, Result};

/// Multiplicative learning rate scheduler
///
/// Multiplies the learning rate by a given factor at each step.
/// Each parameter group can have its own multiplicative factor.
pub struct MultiplicativeLR<'a, O: Optimizer<T>, T: coeus_dtype::FloatDtype> {
    optimizer: &'a mut O,
    /// Multiplicative factors for each parameter group
    lr_lambdas: Vec<T>,
    /// Current step
    current_step: usize,
}

impl<'a, O: Optimizer<T>, T: coeus_dtype::FloatDtype> MultiplicativeLR<'a, O, T> {
    /// Create a new MultiplicativeLR scheduler
    ///
    /// # Arguments
    /// * `optimizer` - Optimizer to schedule
    /// * `lr_lambda` - Multiplicative factor
    ///
    /// # Example
    /// ```rust
    /// use coeus_optim::{Adam, MultiplicativeLR};
    /// use coeus_tensor::{Tensor, CpuBackend};
    ///
    /// let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
    /// let mut optimizer = Adam::new(params, 0.001).unwrap();
    /// let mut scheduler = MultiplicativeLR::new(&mut optimizer, 0.9);
    /// ```
    pub fn new(optimizer: &'a mut O, lr_lambda: T) -> Self {
        Self::with_lr_lambdas(optimizer, vec![lr_lambda])
    }

    /// Create MultiplicativeLR with different factors for each parameter group
    ///
    /// # Arguments
    /// * `optimizer` - Optimizer to schedule
    /// * `lr_lambdas` - Multiplicative factors for each parameter group
    pub fn with_lr_lambdas(optimizer: &'a mut O, lr_lambdas: Vec<T>) -> Self {
        // Extend lr_lambdas if there are more parameter groups than factors
        let mut extended_lambdas = lr_lambdas;
        while extended_lambdas.len() < optimizer.param_groups().len() {
            extended_lambdas.push(T::one()); // Default: no change
        }

        Self {
            optimizer,
            lr_lambdas: extended_lambdas,
            current_step: 0,
        }
    }

    /// Get the current step
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Get the multiplicative factors
    pub fn lr_lambdas(&self) -> &[T] {
        &self.lr_lambdas
    }
}

impl<'a, O: Optimizer<T>, T: coeus_dtype::FloatDtype> MultiplicativeLR<'a, O, T> {
    /// Take a step in the learning rate schedule
    ///
    /// Multiplies the learning rate by the given factor.
    pub fn step(&mut self) -> Result<()> {
        for (group_index, group) in self.optimizer.param_groups_mut().iter_mut().enumerate() {
            if group_index < self.lr_lambdas.len() {
                let lambda = self.lr_lambdas[group_index];
                group.lr = group.lr * lambda;
            }
        }

        self.current_step += 1;
        Ok(())
    }

    /// Get the current learning rates for all parameter groups
    pub fn get_lr(&self) -> Vec<T> {
        (0..self.optimizer.param_groups().len())
            .filter_map(|i| self.optimizer.get_lr(i))
            .collect()
    }

    /// Get the last set learning rates (same as current for MultiplicativeLR)
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
    fn test_multiplicative_lr_creation() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.001);
        let scheduler: MultiplicativeLR<'_, Adam<f64>, f64> = MultiplicativeLR::new(&mut optimizer, 0.9);

        assert_eq!(scheduler.current_step(), 0);
        assert_eq!(scheduler.lr_lambdas(), &[0.9_f64]);
        assert_eq!(scheduler.optimizer.param_groups()[0].lr, 0.001_f64);
    }

    #[test]
    fn test_multiplicative_lr_step() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.001);
        let mut scheduler: MultiplicativeLR<'_, Adam<f64>, f64> = MultiplicativeLR::new(&mut optimizer, 0.9);

        // Initial LR should be base_lr
        assert_eq!(scheduler.optimizer.param_groups()[0].lr, 0.001_f64);

        // Take first step
        scheduler.step().unwrap();
        assert_eq!(scheduler.current_step(), 1);

        let lr1 = scheduler.optimizer.param_groups()[0].lr;
        // Should be base_lr * 0.9 = 0.001 * 0.9
        assert!((lr1 - 0.0009_f64).abs() < 1e-6_f64);

        // Take second step
        scheduler.step().unwrap();
        assert_eq!(scheduler.current_step(), 2);

        let lr2 = scheduler.optimizer.param_groups()[0].lr;
        // Should be base_lr * 0.9^2 = 0.001 * 0.81
        assert!((lr2 - 0.00081_f64).abs() < 1e-6_f64);
    }

    #[test]
    fn test_multiplicative_lr_no_change() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.001);
        let mut scheduler: MultiplicativeLR<'_, Adam<f64>, f64> = MultiplicativeLR::new(&mut optimizer, 1.0);

        // Take several steps with lambda = 1.0 (no change)
        for i in 0..5 {
            scheduler.step().unwrap();
            assert_eq!(scheduler.current_step(), i + 1);
            assert_eq!(scheduler.optimizer.param_groups()[0].lr, 0.001_f64);
        }
    }

    #[test]
    fn test_multiplicative_lr_increase() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.001);
        let mut scheduler: MultiplicativeLR<'_, Adam<f64>, f64> = MultiplicativeLR::new(&mut optimizer, 1.1);

        // Take steps with lambda > 1.0 (increase)
        for i in 0..3 {
            scheduler.step().unwrap();
            let expected_lr = 0.001_f64 * 1.1_f64.powi(i + 1);
            let actual_lr = scheduler.optimizer.param_groups()[0].lr;
            assert!((actual_lr - expected_lr).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn test_multiplicative_lr_multiple_param_groups() {
        let params1 = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
        let params2 = vec![Tensor::from_vec(CpuBackend::default(), vec![2.0], vec![1]).unwrap()];

        let mut optimizer = Adam::new(params1, 0.001);
        optimizer.add_param_group(ParamGroup::new(params2, 0.001, 0.0));

        let mut scheduler: MultiplicativeLR<'_, Adam<f64>, f64> = MultiplicativeLR::with_lr_lambdas(&mut optimizer, vec![0.9, 0.95]);

        // Each parameter group should have its own multiplicative factor
        let lr1_initial = scheduler.optimizer.param_groups()[0].lr;
        let lr2_initial = scheduler.optimizer.param_groups()[1].lr;
        assert_eq!(lr1_initial, lr2_initial);

        scheduler.step().unwrap();

        // Each parameter group should have different learning rates
        let lr1_after = scheduler.optimizer.param_groups()[0].lr;
        let lr2_after = scheduler.optimizer.param_groups()[1].lr;
        assert_ne!(lr1_after, lr2_after);

        // First group should decay faster (0.9 vs 0.95)
        assert!(lr1_after < lr2_after);
    }

    #[test]
    fn test_multiplicative_lr_default_lambda() {
        let params1 = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
        let params2 = vec![Tensor::from_vec(CpuBackend::default(), vec![2.0], vec![1]).unwrap()];

        let mut optimizer = Adam::new(params1, 0.001);
        optimizer.add_param_group(ParamGroup::new(params2, 0.001, 0.0));

        // Only provide one lambda - second group should get default (1.0)
        let mut scheduler: MultiplicativeLR<'_, Adam<f64>, f64> = MultiplicativeLR::with_lr_lambdas(&mut optimizer, vec![0.9]);

        scheduler.step().unwrap();

        // First group should have decayed
        let lr1_after = scheduler.optimizer.param_groups()[0].lr;
        assert!((lr1_after - 0.0009_f64).abs() < 1e-6_f64);

        // Second group should be unchanged (default lambda = 1.0)
        let lr2_after = scheduler.optimizer.param_groups()[1].lr;
        assert_eq!(lr2_after, 0.001_f64);
    }

    #[test]
    fn test_multiplicative_lr_zero_lambda() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.001);
        let mut scheduler: MultiplicativeLR<'_, Adam<f64>, f64> = MultiplicativeLR::new(&mut optimizer, 0.0);

        scheduler.step().unwrap();

        // Learning rate should be zero
        assert_eq!(scheduler.optimizer.param_groups()[0].lr, 0.0_f64);
    }

    #[test]
    fn test_multiplicative_lr_negative_lambda() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.001);
        let mut scheduler: MultiplicativeLR<'_, Adam<f64>, f64> = MultiplicativeLR::new(&mut optimizer, -1.0);

        scheduler.step().unwrap();

        // Learning rate should be negative
        assert_eq!(scheduler.optimizer.param_groups()[0].lr, -0.001_f64);
    }
}
