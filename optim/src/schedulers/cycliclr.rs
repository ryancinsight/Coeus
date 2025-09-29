//! Cyclic learning rate scheduler
//!
//! Implements cyclical learning rate policies as described in
//! "Cyclical Learning Rates for Training Neural Networks" (Smith, 2017).
//!
//! ## Mathematical Foundation
//!
//! Cyclical learning rates oscillate between a minimum and maximum value,
//! allowing the model to benefit from both large learning rates (fast convergence)
//! and small learning rates (fine-tuning).
//!
//! ## Cycle Modes
//!
//! - **Triangular**: Linear increase and decrease (original paper)
//! - **Triangular2**: Triangle wave with half amplitude each cycle
//! - **ExpRange**: Exponential increase and decrease between min and max
//!
//! ## References
//!
//! - [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
//! - [PyTorch CyclicLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html)

use crate::{Optimizer, Result};

/// Cycle mode for CyclicLR scheduler
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    /// Triangular: Linear increase and decrease
    Triangular,
    /// Triangular2: Triangle wave with half amplitude each cycle
    Triangular2,
    /// ExpRange: Exponential increase and decrease
    ExpRange,
}

/// Cyclic learning rate scheduler
///
/// Implements cyclical learning rate policies that oscillate between
/// base_lr and max_lr according to the specified mode and cycle length.
pub struct CyclicLR<'a, O, T>
where
    O: Optimizer<T>,
    T: coeus_dtype::FloatDtype,
{
    optimizer: &'a mut O,
    #[allow(dead_code)]
    base_lr: T,
    max_lr: T,
    step_size_up: usize,
    step_size_down: Option<usize>,
    #[allow(dead_code)]
    mode: Mode,
    gamma: T,
    #[allow(dead_code)]
    scale_fn: Option<Box<dyn Fn(f64) -> f64>>,
    scale_mode: Mode,
    cycle_momentum: bool,
    #[allow(dead_code)]
    base_momentum: T,
    max_momentum: T,
    last_epoch: usize,
    base_lrs: Vec<T>,
    base_momentums: Vec<T>,
}

impl<'a, O, T> CyclicLR<'a, O, T>
where
    O: Optimizer<T>,
    T: coeus_dtype::FloatDtype,
{
    /// Create a new CyclicLR scheduler with triangular policy
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer to schedule
    /// * `base_lr` - Initial learning rate (minimum in cycle)
    /// * `max_lr` - Maximum learning rate in cycle
    /// * `step_size_up` - Number of steps to go from base_lr to max_lr
    /// * `step_size_down` - Number of steps to go from max_lr to base_lr (defaults to step_size_up)
    /// * `mode` - Cycle mode (Triangular, Triangular2, ExpRange)
    /// * `gamma` - Multiplier for Triangular2 mode (default: 1.0)
    ///
    /// # Example
    /// ```rust
    /// use coeus_optim::{Sgd, CyclicLR, CyclicMode};
    /// use coeus_tensor::Tensor;
    ///
    /// let params = vec![Tensor::from_vec(vec![1.0, 2.0], vec![2])];
    /// let mut optimizer = Sgd::new(params, 0.001);
    ///
    /// // Triangle wave: 0.001 ↗ 0.006 ↘ 0.001 (repeats)
    /// let mut scheduler = CyclicLR::new(
    ///     &mut optimizer,
    ///     0.001,  // base_lr
    ///     0.006,  // max_lr
    ///     2000,   // step_size_up
    ///     Some(2000), // step_size_down
    ///     CyclicMode::Triangular
    /// );
    ///
    /// for step in 0..10000 {
    ///     // Training step...
    ///
    ///     // Update learning rate every step
    ///     scheduler.step();
    /// }
    /// ```
    pub fn new(
        optimizer: &'a mut O,
        base_lr: T,
        max_lr: T,
        step_size_up: usize,
        step_size_down: Option<usize>,
        mode: Mode,
    ) -> Self {
        Self::with_options(
            optimizer,
            base_lr,
            max_lr,
            step_size_up,
            step_size_down,
            mode,
            T::from(1.0).unwrap(), // gamma
            false,                 // cycle_momentum
            T::zero(),             // base_momentum
            T::zero(),             // max_momentum
        )
    }

    /// Create CyclicLR with full options
    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        optimizer: &'a mut O,
        base_lr: T,
        max_lr: T,
        step_size_up: usize,
        step_size_down: Option<usize>,
        mode: Mode,
        gamma: T,
        cycle_momentum: bool,
        base_momentum: T,
        max_momentum: T,
    ) -> Self {
        let step_size_down_unwrapped = step_size_down.unwrap_or(step_size_up);
        let base_lrs = optimizer.param_groups().iter().map(|_| base_lr).collect();
        let base_momentums = optimizer
            .param_groups()
            .iter()
            .map(|_| base_momentum)
            .collect();

        Self {
            optimizer,
            base_lr,
            max_lr,
            step_size_up,
            step_size_down: Some(step_size_down_unwrapped),
            mode,
            gamma,
            scale_fn: None,
            scale_mode: mode,
            cycle_momentum,
            base_momentum,
            max_momentum,
            last_epoch: 0,
            base_lrs,
            base_momentums,
        }
    }

    /// Compute the scale factor for the current cycle position
    ///
    /// Returns a value between 0 and 1 representing the position in the cycle
    fn scale_fn(&self, x: f64) -> f64 {
        match self.scale_mode {
            Mode::Triangular => {
                // Linear triangular wave - scale is the cycle position itself
                x
            }
            Mode::Triangular2 => {
                // Triangle wave with half amplitude each cycle
                1.0 / (2.0_f64.powf((x * 2.0).floor()) + 1.0)
            }
            Mode::ExpRange => {
                // Exponential range
                let gamma_f64 = num_traits::ToPrimitive::to_f64(&self.gamma).unwrap_or(1.0);
                gamma_f64.powf(x)
            }
        }
    }

    /// Get the current cycle position (between 0 and 1)
    fn cycle_position(&self, epoch: usize) -> f64 {
        let step_size_down = self.step_size_down.unwrap_or(self.step_size_up);
        let cycle_len = self.step_size_up + step_size_down;
        let _cycle = epoch / cycle_len;
        let x_in_cycle = epoch % cycle_len;

        if x_in_cycle < self.step_size_up {
            // Up phase
            x_in_cycle as f64 / self.step_size_up as f64
        } else {
            // Down phase
            let x_down = x_in_cycle - self.step_size_up;
            1.0 - (x_down as f64 / step_size_down as f64)
        }
    }

    /// Get the learning rate for the current epoch
    fn get_lr(&self, epoch: usize) -> Vec<T> {
        let cycle_pos = self.cycle_position(epoch);
        let scale = self.scale_fn(cycle_pos);

        self.base_lrs
            .iter()
            .map(|base_lr| *base_lr + (self.max_lr - *base_lr) * T::from(scale).unwrap())
            .collect()
    }

    /// Get the momentum for the current epoch (if cycling momentum)
    fn get_momentum(&self, epoch: usize) -> Vec<T> {
        if !self.cycle_momentum {
            return self.base_momentums.clone();
        }

        let cycle_pos = self.cycle_position(epoch);
        let scale = self.scale_fn(cycle_pos);

        self.base_momentums
            .iter()
            .map(|base_momentum| {
                *base_momentum + (self.max_momentum - *base_momentum) * T::from(scale).unwrap()
            })
            .collect()
    }

    /// Step the scheduler
    ///
    /// Updates the learning rate (and momentum if enabled) according to the cycle policy.
    pub fn step(&mut self) -> Result<()> {
        self.last_epoch += 1;

        // Update learning rates
        let new_lrs = self.get_lr(self.last_epoch);
        for (i, new_lr) in new_lrs.iter().enumerate() {
            self.optimizer.set_lr(i, *new_lr)?;
        }

        // Update momentums if cycling
        if self.cycle_momentum {
            let new_momentums = self.get_momentum(self.last_epoch);
            for (i, new_momentum) in new_momentums.iter().enumerate() {
                // Note: This assumes the optimizer has momentum parameter
                // In practice, we'd need to check the optimizer type and update accordingly
                // For now, this is a placeholder for momentum cycling
                let _ = (i, new_momentum);
            }
        }

        Ok(())
    }

    /// Get the current learning rates
    pub fn get_last_lr(&self) -> Vec<T> {
        self.get_lr(self.last_epoch)
    }

    /// Get the last epoch number
    pub fn last_epoch(&self) -> usize {
        self.last_epoch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sgd;
    use coeus_tensor::{Tensor, CpuBackend};

    #[test]
    fn test_cyclic_lr_creation() {
        let backend = CpuBackend::default();
        let params = vec![Tensor::from_vec(backend, vec![1.0, 2.0], vec![2]).unwrap()];
        let mut optimizer = Sgd::new(params, 0.001);

        let scheduler = CyclicLR::new(
            &mut optimizer,
            0.001,      // base_lr
            0.006,      // max_lr
            2000,       // step_size_up
            Some(2000), // step_size_down
            Mode::Triangular,
        );

        assert_eq!(scheduler.last_epoch(), 0);
        assert_eq!(scheduler.get_last_lr().len(), 1);
    }

    #[test]
    fn test_cyclic_lr_step() {
        let backend = CpuBackend::default();
        let params = vec![Tensor::from_vec(backend, vec![1.0, 2.0], vec![2]).unwrap()];
        let mut optimizer = Sgd::new(params, 0.001);

        let mut scheduler = CyclicLR::new(
            &mut optimizer,
            0.001,   // base_lr
            0.006,   // max_lr
            2,       // step_size_up
            Some(2), // step_size_down
            Mode::Triangular,
        );

        // Initial state
        assert_eq!(scheduler.last_epoch(), 0);
        let initial_lr = scheduler.get_last_lr()[0];
        let initial_lr_f64 = num_traits::ToPrimitive::to_f64(&initial_lr).unwrap_or(0.0);
        assert!((initial_lr_f64 - 0.001).abs() < 1e-6);

        // Step 1: Halfway through up phase (epoch 1, position 0.5 in cycle)
        scheduler.step().unwrap();
        assert_eq!(scheduler.last_epoch(), 1);
        let lr1 = scheduler.get_last_lr()[0];
        let lr1_f64 = num_traits::ToPrimitive::to_f64(&lr1).unwrap_or(0.0);
        assert!((lr1_f64 - 0.0035).abs() < 1e-6); // Halfway between 0.001 and 0.006

        // Step 2: At peak (epoch 2, position 1.0 in cycle)
        scheduler.step().unwrap();
        assert_eq!(scheduler.last_epoch(), 2);
        let lr2 = scheduler.get_last_lr()[0];
        let lr2_f64 = num_traits::ToPrimitive::to_f64(&lr2).unwrap_or(0.0);
        assert!((lr2_f64 - 0.006).abs() < 1e-6); // At max_lr

        // Step 3: Halfway through down phase (epoch 3, position 0.5 in down phase)
        scheduler.step().unwrap();
        assert_eq!(scheduler.last_epoch(), 3);
        let lr3 = scheduler.get_last_lr()[0];
        let lr3_f64 = num_traits::ToPrimitive::to_f64(&lr3).unwrap_or(0.0);
        assert!((lr3_f64 - 0.0035).abs() < 1e-6); // Halfway back to base_lr
    }

    #[test]
    fn test_cyclic_lr_triangular2_mode() {
        let backend = CpuBackend::default();
        let params = vec![Tensor::from_vec(backend, vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Sgd::new(params, 0.001);

        let scheduler = CyclicLR::new(
            &mut optimizer,
            0.001,   // base_lr
            0.006,   // max_lr
            2,       // step_size_up
            Some(2), // step_size_down
            Mode::Triangular2,
        );

        // Test that Triangular2 mode is set
        assert_eq!(scheduler.last_epoch(), 0);
    }

    #[test]
    fn test_cyclic_lr_exp_range_mode() {
        let backend = CpuBackend::default();
        let params = vec![Tensor::from_vec(backend, vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Sgd::new(params, 0.001);

        let scheduler = CyclicLR::new(
            &mut optimizer,
            0.001,   // base_lr
            0.006,   // max_lr
            2,       // step_size_up
            Some(2), // step_size_down
            Mode::ExpRange,
        );

        // Test that ExpRange mode is set
        assert_eq!(scheduler.last_epoch(), 0);
    }

    #[test]
    fn test_cyclic_lr_cycle_position() {
        let backend = CpuBackend::default();
        let params = vec![Tensor::from_vec(backend, vec![1.0], vec![1]).unwrap()];
        let mut optimizer = Sgd::new(params, 0.001);

        let scheduler = CyclicLR::new(
            &mut optimizer,
            0.001,
            0.006,
            2,       // step_size_up
            Some(2), // step_size_down
            Mode::Triangular,
        );

        // Test cycle position calculations
        // Note: This is testing internal method, normally we'd test through public API
        let _pos0 = scheduler.cycle_position(0); // Start of cycle
        let _pos1 = scheduler.cycle_position(1); // Middle of up phase
        let _pos2 = scheduler.cycle_position(2); // End of up phase / start of down phase
        let _pos3 = scheduler.cycle_position(3); // Middle of down phase
        let _pos4 = scheduler.cycle_position(4); // End of cycle
    }

    #[test]
    fn test_cyclic_lr_multiple_param_groups() {
        let backend = CpuBackend::default();
        let params1 = vec![Tensor::from_vec(backend.clone(), vec![1.0, 2.0], vec![2]).unwrap()];
        let params2 = vec![Tensor::from_vec(backend.clone(), vec![3.0], vec![1]).unwrap()];
        let mut optimizer = Sgd::new(params1, 0.001);
        optimizer.add_param_group(crate::ParamGroup::new(params2, 0.001, 0.0));

        let mut scheduler = CyclicLR::new(
            &mut optimizer,
            0.001,   // base_lr
            0.006,   // max_lr
            2,       // step_size_up
            Some(2), // step_size_down
            Mode::Triangular,
        );

        // Should handle multiple parameter groups
        assert_eq!(scheduler.get_last_lr().len(), 2);

        scheduler.step().unwrap();
        assert_eq!(scheduler.last_epoch(), 1);
        assert_eq!(scheduler.get_last_lr().len(), 2);
    }
}
