//! OneCycleLR scheduler implementation
//!
//! Implements the OneCycle learning rate scheduler which varies the learning rate
//! following a one-cycle policy. This scheduler changes the learning rate after
//! every batch and provides a learning rate that increases then decreases in a
//! cosine-like fashion.
//!
//! ## Mathematical Foundation
//!
//! The OneCycleLR scheduler varies the learning rate according to:
//!
//! ```text
//! lr(t) = lr_max * (1 + cos(π * t / total_steps)) / 2
//! ```
//!
//! where `t` is the current step and `total_steps` is the total number of steps.
//!
//! ## References
//!
//! - [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820)

use crate::{Optimizer, Result};

/// OneCycle learning rate scheduler
///
/// Implements the OneCycle learning rate policy which varies the learning rate
/// following a cosine annealing schedule over the course of training.
pub struct OneCycleLR<'a, O: Optimizer<T>, T: coeus_dtype::FloatDtype> {
    optimizer: &'a mut O,
    /// Maximum learning rate
    max_lr: T,
    /// Total number of training steps
    total_steps: usize,
    /// Current step
    current_step: usize,
    /// Number of steps for annealing down (default: total_steps * 0.3)
    anneal_strategy: AnnealStrategy,
    /// Learning rate divisor for final phase (default: 1e4)
    div_factor: T,
    /// Final learning rate divisor (default: 1e4)
    final_div_factor: T,
}

/// Annealing strategy for the OneCycle scheduler
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AnnealStrategy {
    /// Linear annealing
    Linear,
    /// Cosine annealing
    #[default]
    Cosine,
}

impl<'a, O: Optimizer<T>, T: coeus_dtype::FloatDtype> OneCycleLR<'a, O, T> {
    /// Create a new OneCycleLR scheduler
    ///
    /// # Arguments
    /// * `optimizer` - Optimizer to schedule
    /// * `max_lr` - Maximum learning rate
    /// * `total_steps` - Total number of training steps
    /// * `anneal_strategy` - Annealing strategy (default: Cosine)
    /// * `div_factor` - Initial learning rate divisor (default: 25.0)
    /// * `final_div_factor` - Final learning rate divisor (default: 1e4)
    ///
    /// # Example
    /// ```rust
    /// use coeus_optim::{Adam, OneCycleLR};
    /// use coeus_tensor::{Tensor, CpuBackend};
    ///
    /// let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
    /// let mut optimizer = Adam::new(params, 0.001);
    /// let mut scheduler = OneCycleLR::new(&mut optimizer, 0.01, 1000);
    /// ```
    pub fn new(optimizer: &'a mut O, max_lr: T, total_steps: usize) -> Self {
        Self::with_options(
            optimizer,
            max_lr,
            total_steps,
            AnnealStrategy::Cosine,
            T::from(25.0).unwrap(),
            T::from(1e4).unwrap(),
        )
    }

    /// Create OneCycleLR with custom options
    ///
    /// # Arguments
    /// * `optimizer` - Optimizer to schedule
    /// * `max_lr` - Maximum learning rate
    /// * `total_steps` - Total number of training steps
    /// * `anneal_strategy` - Annealing strategy
    /// * `div_factor` - Initial learning rate divisor
    /// * `final_div_factor` - Final learning rate divisor
    pub fn with_options(
        optimizer: &'a mut O,
        max_lr: T,
        total_steps: usize,
        anneal_strategy: AnnealStrategy,
        div_factor: T,
        final_div_factor: T,
    ) -> Self {
        // Set initial learning rate to max_lr / div_factor
        let initial_lr = max_lr / div_factor;
        for group in optimizer.param_groups_mut() {
            group.lr = initial_lr;
        }

        Self {
            optimizer,
            max_lr,
            total_steps,
            current_step: 0,
            anneal_strategy,
            div_factor,
            final_div_factor,
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

    /// Get the maximum learning rate
    pub fn max_lr(&self) -> T {
        self.max_lr
    }

    pub fn div_factor(&self) -> T {
        self.div_factor
    }

    pub fn final_div_factor(&self) -> T {
        self.final_div_factor
    }

    /// Get the annealing strategy
    pub fn anneal_strategy(&self) -> &AnnealStrategy {
        &self.anneal_strategy
    }

    /// Calculate the learning rate for the current step
    fn calculate_lr(&self, step: usize) -> T {
        if step >= self.total_steps - 1 {
            return self.max_lr / self.final_div_factor;
        }

        match self.anneal_strategy {
            AnnealStrategy::Cosine => {
                // Cosine annealing
                let progress = T::from(step as f64 / self.total_steps as f64).unwrap();
                let cosine = (progress * T::from(std::f64::consts::PI).unwrap()).cos();
                self.max_lr * (T::one() + cosine) / (T::one() + T::one())
            }
            AnnealStrategy::Linear => {
                // Linear annealing - adjust for initial LR being max_lr / div_factor
                if step < self.total_steps / 2 {
                    // Increasing phase
                    let phase_steps = self.total_steps / 2;
                    if phase_steps == 0 {
                        return self.max_lr / self.div_factor; // Special case for total_steps < 2
                    }
                    let progress = T::from(step as f64 / phase_steps as f64).unwrap();
                    self.max_lr / self.div_factor
                        + (self.max_lr - self.max_lr / self.div_factor) * progress
                } else {
                    // Decreasing phase
                    let phase_steps = self.total_steps / 2;
                    if phase_steps == 0 {
                        return self.max_lr / self.final_div_factor; // Special case
                    }
                    let progress =
                        T::from((step - self.total_steps / 2) as f64 / phase_steps as f64).unwrap();
                    self.max_lr * (T::one() - progress)
                }
            }
        }
    }

    /// Get the percentage of training completed
    pub fn progress(&self) -> f64 {
        self.current_step as f64 / self.total_steps as f64
    }

    /// Check if training is complete
    pub fn is_done(&self) -> bool {
        self.current_step >= self.total_steps
    }
}

impl<'a, O: Optimizer<T>, T: coeus_dtype::FloatDtype> OneCycleLR<'a, O, T> {
    /// Take a step in the learning rate schedule
    ///
    /// Updates the learning rate based on the current step.
    pub fn step(&mut self) -> Result<()> {
        if self.current_step >= self.total_steps {
            return Ok(()); // Already completed
        }

        // Increment step first to calculate learning rate for the next step
        self.current_step += 1;

        // Calculate learning rate for current step
        let new_lr = self.calculate_lr(self.current_step);

        // Apply the calculated learning rate
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

    /// Get the last set learning rates (same as current for OneCycleLR)
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
    fn test_onecycle_lr_creation() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.001);
        let scheduler: OneCycleLR<'_, Adam<f64>, f64> = OneCycleLR::new(&mut optimizer, 0.01, 100);

        assert_eq!(scheduler.total_steps(), 100);
        assert_eq!(scheduler.current_step(), 0);
        assert_eq!(scheduler.max_lr(), 0.01_f64);
        assert_eq!(scheduler.anneal_strategy(), &AnnealStrategy::Cosine);
    }

    #[test]
    fn test_onecycle_lr_with_options() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.001);
        let scheduler: OneCycleLR<'_, Adam<f64>, f64> = OneCycleLR::with_options(
            &mut optimizer,
            0.01,
            100,
            AnnealStrategy::Linear,
            10.0,
            1000.0,
        );

        assert_eq!(scheduler.total_steps(), 100);
        assert_eq!(scheduler.anneal_strategy(), &AnnealStrategy::Linear);
    }

    #[test]
    fn test_onecycle_lr_step() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.001);
        let mut scheduler: OneCycleLR<'_, Adam<f64>, f64> = OneCycleLR::new(&mut optimizer, 0.01, 10);

        // Initial LR should be max_lr / div_factor = 0.01 / 25 = 0.0004
        assert!((scheduler.optimizer.param_groups()[0].lr - 0.0004_f64).abs() < 1e-6_f64);

        // Take first step
        scheduler.step().unwrap();
        assert_eq!(scheduler.current_step(), 1);

        // LR should increase towards max_lr
        let lr1 = scheduler.optimizer.param_groups()[0].lr;
        assert!(lr1 > 0.0004_f64);

        // Take all remaining steps
        for _ in 1..10 {
            scheduler.step().unwrap();
        }

        // Should be complete
        assert!(scheduler.is_done());
        assert_eq!(scheduler.current_step(), 10);

        // Final LR should be max_lr / final_div_factor = 0.01 / 10000 = 1e-6
        let final_lr = scheduler.optimizer.param_groups()[0].lr;
        assert!((final_lr - 1e-6_f64).abs() < 1e-5_f64);
    }

    #[test]
    fn test_onecycle_lr_linear_annealing() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.001);
        let mut scheduler: OneCycleLR<'_, Adam<f64>, f64> = OneCycleLR::with_options(
            &mut optimizer,
            0.01,
            4, // 4 steps total
            AnnealStrategy::Linear,
            2.0, // div_factor = 2
            1000.0,
        );

        // Initial LR = 0.01 / 2 = 0.005
        assert!((scheduler.optimizer.param_groups()[0].lr - 0.005_f64).abs() < 1e-5_f64);

        // Step 1: halfway through increasing phase (2 steps)
        scheduler.step().unwrap();
        let lr1 = scheduler.optimizer.param_groups()[0].lr;
        assert!(lr1 > 0.005_f64); // Should be increasing

        // Step 2: end of increasing phase, should be max_lr
        scheduler.step().unwrap();
        let lr2 = scheduler.optimizer.param_groups()[0].lr;
        assert!((lr2 - 0.01_f64).abs() < 1e-5_f64); // Should be max_lr

        // Step 3: halfway through decreasing phase
        scheduler.step().unwrap();
        let lr3 = scheduler.optimizer.param_groups()[0].lr;
        println!("Step 3 - lr3 = {}", lr3);
        assert!(lr3 < 0.01_f64); // Should be decreasing

        // Step 4: final step (should be max_lr / final_div_factor)
        scheduler.step().unwrap();
        let lr4 = scheduler.optimizer.param_groups()[0].lr;
        println!("Step 4 - lr4 = {}", lr4);
        assert!((lr4 - 1e-5_f64).abs() < 1e-6_f64); // Should be 0.01 / 1000.0 = 1e-5
    }

    #[test]
    fn test_onecycle_lr_progress() {
        let params = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Adam::new(params, 0.001);
        let mut scheduler: OneCycleLR<'_, Adam<f64>, f64> = OneCycleLR::new(&mut optimizer, 0.01, 100);

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
    fn test_onecycle_lr_multiple_param_groups() {
        let params1 = vec![Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap()];
        let params2 = vec![Tensor::from_vec(CpuBackend::default(), vec![2.0], vec![1]).unwrap()];

        let mut optimizer = Adam::new(params1, 0.001);
        optimizer.add_param_group(ParamGroup::new(params2, 0.001, 0.0));

        let mut scheduler: OneCycleLR<'_, Adam<f64>, f64> = OneCycleLR::new(&mut optimizer, 0.01, 10);

        // All parameter groups should have the same learning rate
        let lr1 = scheduler.optimizer.param_groups()[0].lr;
        let lr2 = scheduler.optimizer.param_groups()[1].lr;
        assert_eq!(lr1, lr2);

        scheduler.step().unwrap();

        // All parameter groups should still have the same learning rate
        let lr1_updated = scheduler.optimizer.param_groups()[0].lr;
        let lr2_updated = scheduler.optimizer.param_groups()[1].lr;
        assert_eq!(lr1_updated, lr2_updated);
        assert!(lr1_updated > lr1); // Should have increased
    }
}
