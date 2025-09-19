//! Resilient Backpropagation (Rprop) optimizer
//!
//! Implements the Rprop algorithm which adapts the step size for each parameter
//! individually based on the sign of the gradient, making it resilient to
//! the size of the gradient.

use crate::{BaseOptimizer, Optimizer, ParamGroup, Result};
use coeus_tensor::{Tensor, Mul, Sub};
use std::collections::HashMap;

/// Resilient Backpropagation optimizer
///
/// Implements the Rprop algorithm which adapts the step size for each parameter
/// individually based on the sign of consecutive gradients. This makes the
/// algorithm resilient to the absolute size of gradients and can handle
/// varying gradient scales effectively.
///
/// ## Mathematical Formula
///
/// ```text
/// if sign(g_t) * sign(g_{t-1}) > 0:
///     Δ_t = min(Δ_{t-1} * η_plus, Δ_max)
/// elif sign(g_t) * sign(g_{t-1}) < 0:
///     Δ_t = max(Δ_{t-1} * η_minus, Δ_min)
///     g_t = 0  # reset gradient
/// else:
///     Δ_t = Δ_{t-1}
///
/// p_t = p_{t-1} - sign(g_t) * Δ_t
/// ```
///
/// where:
/// - `Δ_t` is the step size for parameter at time t
/// - `η_plus` is the increase factor (typically 1.2)
/// - `η_minus` is the decrease factor (typically 0.5)
/// - `Δ_min`, `Δ_max` are minimum and maximum step sizes
///
/// ## References
///
/// - Riedmiller, M., & Braun, H. (1993). A direct adaptive method for faster
///   backpropagation learning: the RPROP algorithm. IEEE International Conference
///   on Neural Networks, 586-591.
pub struct Rprop<T: coeus_dtype::FloatDtype> {
    base: BaseOptimizer<T>,
    eta_plus: T,   // increase factor (default: 1.2)
    eta_minus: T,  // decrease factor (default: 0.5)
    delta_min: T,  // minimum step size (default: 1e-6)
    delta_max: T,  // maximum step size (default: 50.0)
}

impl<T: coeus_dtype::FloatDtype> Rprop<T> {
    /// Create a new Rprop optimizer
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Initial learning rate (used as initial step size)
    ///
    /// # Example
    /// ```rust
    /// use coeus_optim::Rprop;
    /// use coeus_tensor::Tensor;
    ///
    /// let params = vec![Tensor::from_vec(vec![1.0, 2.0], vec![2])];
    /// let optimizer = Rprop::new(params, 0.01);
    /// ```
    pub fn new(params: Vec<Tensor<T>>, lr: T) -> Self {
        Self::with_options(
            params,
            lr,
            T::from(1.2).unwrap(),  // eta_plus
            T::from(0.5).unwrap(),  // eta_minus
            T::from(1e-6).unwrap(), // delta_min
            T::from(50.0).unwrap(), // delta_max
        )
    }

    /// Create Rprop with custom options
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Initial learning rate (used as initial step size)
    /// * `eta_plus` - Increase factor for step size
    /// * `eta_minus` - Decrease factor for step size
    /// * `delta_min` - Minimum step size
    /// * `delta_max` - Maximum step size
    pub fn with_options(
        params: Vec<Tensor<T>>,
        lr: T,
        eta_plus: T,
        eta_minus: T,
        delta_min: T,
        delta_max: T,
    ) -> Self {
        let param_group = ParamGroup::new(params, lr, T::zero()); // No weight decay for Rprop
        let base = BaseOptimizer::new(vec![param_group]);
        Self {
            base,
            eta_plus,
            eta_minus,
            delta_min,
            delta_max,
        }
    }
}

impl<T: coeus_dtype::FloatDtype> Optimizer<T> for Rprop<T> {
    fn step(&mut self) -> Result<()> {
        for group in self.base.param_groups_mut() {
            let lr = group.lr;

            for param in &mut group.params {
                if param.grad().is_none() {
                    continue;
                }

                let grad = param.grad().unwrap().clone();

                // Get or create step size buffer (one per parameter element)
                let mut step_size = param.get_buffer("step_size")
                    .unwrap_or_else(|| Tensor::from_vec(vec![lr; param.numel()], param.shape().to_vec()));

                // Get or create previous gradient sign buffer
                let prev_grad_sign = param.get_buffer("prev_grad_sign")
                    .unwrap_or_else(|| Tensor::zeros(param.shape().to_vec()));

                // Compute current gradient sign
                let current_grad_sign = grad.sign();

                // Compute sign product to determine if consecutive gradients have same sign
                let sign_product = current_grad_sign.mul(&prev_grad_sign)?;

                // Update step sizes based on gradient sign consistency
                for i in 0..step_size.numel() {
                    let sign_prod_val = sign_product.data()[i];

                    if sign_prod_val > T::zero() {
                        // Same sign: increase step size
                        step_size.data_mut()[i] = (step_size.data()[i] * self.eta_plus).min(self.delta_max);
                    } else if sign_prod_val < T::zero() {
                        // Different sign: decrease step size and reset gradient
                        step_size.data_mut()[i] = (step_size.data()[i] * self.eta_minus).max(self.delta_min);
                        // Note: Gradient reset would require setting grad back to parameter
                    }
                    // If sign_prod_val == 0, keep step size unchanged
                }

                // Update parameter: p = p - sign(g) * step_size
                let update = current_grad_sign.mul(&step_size)?;
                *param = param.sub(&update)?;

                // Store previous gradient sign for next iteration
                param.set_buffer("prev_grad_sign", current_grad_sign);
                param.set_buffer("step_size", step_size);
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        self.base.zero_grad();
    }

    fn param_groups(&self) -> &[ParamGroup<T>] {
        self.base.param_groups()
    }

    fn param_groups_mut(&mut self) -> &mut [ParamGroup<T>] {
        self.base.param_groups_mut()
    }

    fn add_param_group(&mut self, param_group: ParamGroup<T>) {
        self.base.add_param_group(param_group);
    }

    fn name(&self) -> &str {
        "Rprop"
    }

    fn get_lr(&self, group_index: usize) -> Option<T> {
        self.base.get_lr(group_index)
    }

    fn set_lr(&mut self, group_index: usize, lr: T) -> Result<()> {
        self.base.set_lr(group_index, lr)
    }

    fn state(&self) -> &HashMap<String, Tensor<T>> {
        self.base.state()
    }

    fn state_mut(&mut self) -> &mut HashMap<String, Tensor<T>> {
        self.base.state_mut()
    }
}
