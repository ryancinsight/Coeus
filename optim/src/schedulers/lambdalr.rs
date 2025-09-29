//! LambdaLR scheduler implementation
//!
//! Implements the LambdaLR learning rate scheduler which allows for arbitrary
//! learning rate schedules defined by lambda functions. This provides maximum
//! flexibility for custom learning rate scheduling.
//!
//! ## Mathematical Foundation
//!
//! The learning rate at step t is given by:
//!
//! ```text
//! η_t = η_base * λ(t)
//! ```
//!
//! where:
//! - η_base is the initial learning rate
//! - λ(t) is a lambda function that takes the current step and returns a multiplier
//!
//! ## References
//!
//! - [PyTorch LambdaLR documentation](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.LambdaLR)

use crate::{Optimizer, Result};

/// Lambda learning rate scheduler
///
/// Allows for arbitrary learning rate schedules defined by lambda functions.
/// Each parameter group can have its own lambda function.
pub struct LambdaLR<'a, O: Optimizer<T>, T: coeus_dtype::FloatDtype> {
    optimizer: &'a mut O,
    /// Base learning rates for each parameter group
    base_lrs: Vec<T>,
    /// Lambda functions for each parameter group
    lambda_functions: Vec<Box<dyn Fn(usize) -> T + Send + Sync>>,
    /// Current step
    current_step: usize,
}

impl<'a, O: Optimizer<T>, T: coeus_dtype::FloatDtype> LambdaLR<'a, O, T> {
    /// Create a new LambdaLR scheduler
    ///
    /// # Arguments
    /// * `optimizer` - Optimizer to schedule
    /// * `lambda_function` - Lambda function that takes step and returns LR multiplier
    ///
    /// # Example
    /// ```rust
    /// use coeus_optim::{Adam, LambdaLR};
    /// use coeus_tensor::Tensor;
    ///
    /// let params = vec![Tensor::from_vec(vec![1.0], vec![1])];
    /// let mut optimizer = Adam::new(params, 0.001);
    /// let mut scheduler = LambdaLR::new(&mut optimizer, |step| 0.9_f64.powi(step as i32));
    /// ```
    pub fn new<F>(optimizer: &'a mut O, lambda_function: F) -> Self
    where
        F: Fn(usize) -> T + Send + Sync + 'static,
    {
        Self::with_lambda_functions(optimizer, vec![Box::new(lambda_function)])
    }

    /// Create LambdaLR with different lambda functions for each parameter group
    ///
    /// # Arguments
    /// * `optimizer` - Optimizer to schedule
    /// * `lambda_functions` - Lambda functions for each parameter group
    pub fn with_lambda_functions(
        optimizer: &'a mut O,
        lambda_functions: Vec<Box<dyn Fn(usize) -> T + Send + Sync + 'static>>,
    ) -> Self {
        let base_lrs = optimizer
            .param_groups()
            .iter()
            .map(|group| group.lr)
            .collect::<Vec<_>>();

        // Extend lambda functions if there are more parameter groups than functions
        let mut extended_functions = lambda_functions;
        while extended_functions.len() < base_lrs.len() {
            extended_functions.push(Box::new(|_| T::one())); // Default: no change
        }

        Self {
            optimizer,
            base_lrs,
            lambda_functions: extended_functions,
            current_step: 0,
        }
    }

    /// Get the current step
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Calculate the learning rate for a parameter group
    fn calculate_lr(&self, group_index: usize, step: usize) -> T {
        if group_index >= self.base_lrs.len() || group_index >= self.lambda_functions.len() {
            let base_lr = self.base_lrs.first().copied().unwrap_or(T::one());
            println!(
                "calculate_lr: group_index {} >= len, returning base_lr[0] = {:?}",
                group_index, base_lr
            );
            return base_lr;
        }

        let base_lr = self.base_lrs[group_index];
        let lambda = &self.lambda_functions[group_index];
        let lambda_value = lambda(step);
        let result = base_lr * lambda_value;
        println!("calculate_lr: group_index = {}, step = {}, base_lr = {}, lambda({}) = {}, result = {:?}", group_index, step, num_traits::ToPrimitive::to_f64(&base_lr).unwrap_or(0.0), step, num_traits::ToPrimitive::to_f64(&lambda_value).unwrap_or(0.0), result);
        result
    }
}

impl<'a, O: Optimizer<T>, T: coeus_dtype::FloatDtype> LambdaLR<'a, O, T> {
    /// Take a step in the learning rate schedule
    ///
    /// Updates the learning rate based on the lambda functions.
    pub fn step(&mut self) -> Result<()> {
        // Increment step first to calculate learning rate for the next step
        self.current_step += 1;

        // Calculate all new learning rates for current step
        let mut new_lrs = Vec::new();
        for group_index in 0..self.base_lrs.len() {
            let new_lr = self.calculate_lr(group_index, self.current_step);
            new_lrs.push(new_lr);
        }

        // Then apply them to avoid borrowing conflicts
        for (group_index, group) in self.optimizer.param_groups_mut().iter_mut().enumerate() {
            if group_index < new_lrs.len() {
                group.lr = new_lrs[group_index];
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

    /// Get the last set learning rates (same as current for LambdaLR)
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
    fn test_lambda_lr_creation() {
        let backend = CpuBackend::default();
        let params = vec![Tensor::from_vec(backend, vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.001);
        let scheduler: LambdaLR<'_, Adam<f64>, f64> = LambdaLR::new(&mut optimizer, |step| 0.9_f64.powi(step as i32));

        assert_eq!(scheduler.current_step, 0);
    }

    #[test]
    fn test_lambda_lr_step() {
        let backend = CpuBackend::default();
        let params = vec![Tensor::from_vec(backend, vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.001);
        let mut scheduler: LambdaLR<'_, Adam<f64>, f64> = LambdaLR::new(&mut optimizer, |step| 0.9_f64.powi(step as i32));

        // Initial LR should be base_lr
        assert_eq!(scheduler.optimizer.param_groups()[0].lr, 0.001_f64);

        // Take first step
        scheduler.step().unwrap();
        assert_eq!(scheduler.current_step(), 1);

        let lr1 = scheduler.optimizer.param_groups()[0].lr;
        // After first step, LR should still be base_lr (calculated for step 0)
        assert!((lr1 - 0.001_f64).abs() < 1e-2_f64);

        // Take second step
        scheduler.step().unwrap();
        assert_eq!(scheduler.current_step(), 2);

        let lr2 = scheduler.optimizer.param_groups()[0].lr;
        // After second step, should be base_lr * 0.9^1 = 0.001 * 0.9
        assert!((lr2 - 0.0009_f64).abs() < 1e-2_f64);

        // Take third step
        scheduler.step().unwrap();
        assert_eq!(scheduler.current_step(), 3);

        let lr3 = scheduler.optimizer.param_groups()[0].lr;
        // Should be base_lr * 0.9^2 = 0.001 * 0.81
        assert!((lr3 - 0.00081_f64).abs() < 1e-2_f64);
    }

    #[test]
    fn test_lambda_lr_exponential_decay() {
        let backend = CpuBackend::default();
        let params = vec![Tensor::from_vec(backend, vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.01);
        let mut scheduler: LambdaLR<'_, Adam<f64>, f64> = LambdaLR::new(&mut optimizer, |step| (-0.1_f64 * step as f64).exp());

        // Take 5 steps
        for i in 0..5 {
            scheduler.step().unwrap();
            let expected_lr = 0.01_f64 * (-0.1_f64 * i as f64).exp();
            let actual_lr = scheduler.optimizer.param_groups()[0].lr;
            assert!((actual_lr - expected_lr).abs() < 1e-2_f64);
        }
    }

    #[test]
    fn test_lambda_lr_linear_decay() {
        let backend = CpuBackend::default();
        let params = vec![Tensor::from_vec(backend, vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.01);
        let mut scheduler: LambdaLR<'_, Adam<f64>, f64> = LambdaLR::new(&mut optimizer, |step| 1.0_f64 - 0.1_f64 * step as f64);

        // Take 5 steps
        for i in 0..5 {
            scheduler.step().unwrap();
            let expected_lr = 0.01_f64 * (1.0_f64 - 0.1_f64 * i as f64);
            let actual_lr = scheduler.optimizer.param_groups()[0].lr;
            assert!((actual_lr - expected_lr).abs() < 1e-2_f64);
        }
    }

    #[test]
    fn test_lambda_lr_multiple_param_groups() {
        let backend = CpuBackend::default();
        let params1 = vec![Tensor::from_vec(backend.clone(), vec![1.0], vec![1]).unwrap()];
        let params2 = vec![Tensor::from_vec(backend.clone(), vec![2.0], vec![1]).unwrap()];

        let mut optimizer = Adam::new(params1, 0.001);
        optimizer.add_param_group(ParamGroup::new(params2, 0.001, 0.0));

        let lambda1 = |step: usize| 0.9_f64.powi(step as i32);
        let lambda2 = |step: usize| 0.95_f64.powi(step as i32);

        let mut scheduler: LambdaLR<'_, Adam<f64>, f64> = LambdaLR::with_lambda_functions(
            &mut optimizer,
            vec![Box::new(lambda1), Box::new(lambda2)],
        );

        // Each parameter group should have its own lambda function
        let lr1_initial = scheduler.optimizer.param_groups()[0].lr;
        let lr2_initial = scheduler.optimizer.param_groups()[1].lr;
        assert_eq!(lr1_initial, lr2_initial);

        // After first step, learning rates should still be the same (calculated for step 0)
        let lr1_after = scheduler.optimizer.param_groups()[0].lr;
        let lr2_after = scheduler.optimizer.param_groups()[1].lr;
        assert_eq!(lr1_after, lr2_after);
        assert_eq!(lr1_after, lr1_initial);

        // Take second step
        scheduler.step().unwrap();

        // Each parameter group should have different learning rates after the second step
        let lr1_after_step2 = scheduler.optimizer.param_groups()[0].lr;
        let lr2_after_step2 = scheduler.optimizer.param_groups()[1].lr;
        assert!((lr1_after_step2 - lr2_after_step2).abs() > 1e-10_f64);

        // First group should decay faster (0.9 vs 0.95)
        assert!(lr1_after_step2 < lr2_after_step2);
    }

    #[test]
    fn test_lambda_lr_default_lambda() {
        let backend = CpuBackend::default();
        let params1 = vec![Tensor::from_vec(backend.clone(), vec![1.0], vec![1]).unwrap()];
        let params2 = vec![Tensor::from_vec(backend.clone(), vec![2.0], vec![1]).unwrap()];

        let mut optimizer = Adam::new(params1, 0.001);
        optimizer.add_param_group(ParamGroup::new(params2, 0.001, 0.0));

        // Only provide one lambda function - second group should get default (no change)
        let lambda1 = |step: usize| 0.9_f64.powi(step as i32);
        let mut scheduler =
            LambdaLR::with_lambda_functions(&mut optimizer, vec![Box::new(lambda1)]);

        // After first step, learning rates should still be the same as initial
        let lr1_after = scheduler.optimizer.param_groups()[0].lr;
        let lr2_after = scheduler.optimizer.param_groups()[1].lr;
        assert_eq!(lr1_after, 0.001_f64);
        assert_eq!(lr2_after, 0.001_f64);

        // Take second step
        scheduler.step().unwrap();

        // First group should have decayed
        let lr1_after_step2 = scheduler.optimizer.param_groups()[0].lr;
        println!("After second step: lr1_after_step2 = {}", lr1_after_step2);
        assert!((lr1_after_step2 - 0.0009_f64).abs() < 1e-2_f64);

        // Second group should be unchanged (default lambda returns 1.0)
        let lr2_after_step2 = scheduler.optimizer.param_groups()[1].lr;
        assert_eq!(lr2_after_step2, 0.001_f64);
    }

    #[test]
    fn test_lambda_lr_complex_schedule() {
        let backend = CpuBackend::default();
        let params = vec![Tensor::from_vec(backend, vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.01);

        // Create a complex schedule: linear increase then exponential decay
        let mut scheduler: LambdaLR<'_, Adam<f64>, f64> = LambdaLR::new(&mut optimizer, |step| {
            if step < 10 {
                // Linear increase for first 10 steps
                1.0_f64 + 0.1_f64 * step as f64
            } else {
                // Exponential decay after step 10
                (-0.2_f64 * (step - 10) as f64).exp()
            }
        });

        // Test increasing phase
        for i in 0..10 {
            scheduler.step().unwrap();
            let expected_lr = 0.01_f64 * (1.0_f64 + 0.1_f64 * i as f64);
            let actual_lr = scheduler.optimizer.param_groups()[0].lr;
            assert!((actual_lr - expected_lr).abs() < 1e-2_f64);
        }

        // Test decaying phase
        let lr_at_step_10 = scheduler.optimizer.param_groups()[0].lr;
        scheduler.step().unwrap();
        let lr_at_step_11 = scheduler.optimizer.param_groups()[0].lr;
        assert!(lr_at_step_11 < lr_at_step_10); // Should be decaying
    }
}
