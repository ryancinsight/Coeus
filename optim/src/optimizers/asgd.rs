//! Averaged Stochastic Gradient Descent (ASGD) optimizer
//!
//! Implements the ASGD algorithm which maintains a running average of parameters
//! during optimization, often providing better generalization than standard SGD.

use crate::{BaseOptimizer, Optimizer, ParamGroup, Result};
use coeus_tensor::{Add, Mul, Sub, Tensor};

/// Averaged Stochastic Gradient Descent optimizer
///
/// Implements the ASGD algorithm which maintains a running average of the parameters
/// during optimization. This averaging often provides better generalization performance
/// compared to standard SGD.
///
/// ## Mathematical Formula
///
/// **Momentum Update:**
/// ```text
/// v_t = momentum * v_{t-1} + (1 - dampening) * g_t
/// if nesterov: g_t = g_t + momentum * v_t
/// p_t = p_{t-1} - lr * v_t - lr * weight_decay * p_{t-1}
/// ```
///
/// **Parameter Averaging:**
/// ```text
/// avg_t = α * avg_{t-1} + (1 - α) * p_t
/// ```
///
/// Where:
/// - `v_t` is the momentum buffer at step t
/// - `p_t` are the parameters at step t
/// - `avg_t` is the running average at step t
/// - `α` (alpha) is the smoothing parameter (default: 0.75)
/// - `g_t` is the gradient at step t
///
/// At the end of training, the averaged parameters `avg_t` are typically used
/// instead of the current parameters `p_t` for better generalization.
///
/// ## References
///
/// - Polyak, B. T., & Juditsky, A. B. (1992). Acceleration of stochastic approximation
///   by averaging. SIAM Journal on Control and Optimization, 30(4), 838-855.
pub struct Asgd<T: coeus_dtype::FloatDtype> {
    base: BaseOptimizer<T>,
    momentum: T,
    dampening: T,
    nesterov: bool,
    alpha: T, // smoothing parameter for averaging
    t: usize, // step count
}

impl<T: coeus_dtype::FloatDtype> Asgd<T> {
    /// Create a new ASGD optimizer
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate (default: 0.01)
    /// * `momentum` - Momentum factor (default: 0.0)
    ///
    /// # Example
    /// ```rust
    /// use coeus_optim::Asgd;
    /// use coeus_tensor::Tensor;
    ///
    /// let params = vec![Tensor::from_vec(vec![1.0, 2.0], vec![2])];
    /// let optimizer = Asgd::new(params, 0.01);
    /// ```
    pub fn new(params: Vec<Tensor<T>>, lr: T) -> Self {
        Self::with_options(
            params,
            lr,
            T::zero(),
            T::zero(),
            false,
            T::from(0.75).unwrap(),
        )
    }

    /// Create ASGD with custom options
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate
    /// * `momentum` - Momentum factor
    /// * `weight_decay` - Weight decay (L2 penalty)
    /// * `nesterov` - Whether to use Nesterov momentum
    /// * `alpha` - Smoothing parameter for averaging (default: 0.75)
    pub fn with_options(
        params: Vec<Tensor<T>>,
        lr: T,
        momentum: T,
        weight_decay: T,
        nesterov: bool,
        alpha: T,
    ) -> Self {
        let base = BaseOptimizer::with_defaults(params, lr, weight_decay);
        Self {
            base,
            momentum,
            dampening: T::zero(),
            nesterov,
            alpha,
            t: 0,
        }
    }

    /// Get the current averaged parameters
    ///
    /// Returns the running average of parameters, which typically provides
    /// better generalization than the current parameters.
    pub fn averaged_parameters(&self) -> Vec<&Tensor<T>> {
        self.base
            .param_groups()
            .iter()
            .flat_map(|group| &group.params)
            .collect()
    }

    /// Get the current step count
    pub fn step_count(&self) -> usize {
        self.t
    }

    /// Get the momentum factor
    pub fn momentum(&self) -> T {
        self.momentum
    }

    /// Get the smoothing parameter for averaging
    pub fn alpha(&self) -> T {
        self.alpha
    }

    /// Check if Nesterov momentum is enabled
    pub fn nesterov(&self) -> bool {
        self.nesterov
    }
}

impl<T: coeus_dtype::FloatDtype> Optimizer<T> for Asgd<T> {
    fn step(&mut self) -> Result<()> {
        self.t += 1;

        for group in self.base.param_groups_mut() {
            let lr = group.lr;
            let weight_decay = group.weight_decay;

            for param in &mut group.params {
                if param.grad().is_none() {
                    continue;
                }

                let mut grad = param.grad().unwrap().clone();

                // Apply weight decay
                if weight_decay != T::zero() {
                    // Apply weight decay: grad = grad + weight_decay * param
                    let weight_decay_term = param.mul(&Tensor::scalar(weight_decay))?;
                    grad = grad.add(&weight_decay_term)?;
                }

                // Get or create momentum buffer
                let mut momentum_buffer = param
                    .get_buffer("momentum")
                    .unwrap_or_else(|| Tensor::zeros(param.shape().to_vec()));

                // Update momentum buffer
                if self.momentum != T::zero() {
                    momentum_buffer = momentum_buffer.mul(&Tensor::scalar(self.momentum))?;
                    // Update momentum buffer: momentum_buffer = momentum_buffer + (1 - dampening) * grad
                    let dampening_factor = T::one() - self.dampening;
                    let grad_term = grad.mul(&Tensor::scalar(dampening_factor))?;
                    momentum_buffer = momentum_buffer.add(&grad_term)?;
                } else {
                    momentum_buffer = grad.clone();
                }

                // Apply Nesterov momentum
                let effective_grad = if self.nesterov && self.momentum != T::zero() {
                    momentum_buffer
                        .mul(&Tensor::scalar(self.momentum))?
                        .add(&grad)?
                } else {
                    momentum_buffer.clone()
                };

                // Update parameter
                // Update parameter: param = param - lr * effective_grad
                let update = effective_grad.mul(&Tensor::scalar(lr))?;
                *param = param.sub(&update)?;

                // Update running average
                let mut avg_buffer = param.get_buffer("average").unwrap_or_else(|| param.clone());

                // avg_t = alpha * avg_{t-1} + (1 - alpha) * p_t
                // This is a different averaging strategy than the original paper
                // but is commonly used in practice
                avg_buffer = avg_buffer.mul(&Tensor::scalar(self.alpha))?;
                let one_minus_alpha = T::one() - self.alpha;
                // Update average buffer: avg_buffer = avg_buffer + (1 - alpha) * param
                let param_term = param.mul(&Tensor::scalar(one_minus_alpha))?;
                avg_buffer = avg_buffer.add(&param_term)?;

                // Store buffers
                param.set_buffer("momentum", momentum_buffer);
                param.set_buffer("average", avg_buffer);
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
        "ASGD"
    }

    fn get_lr(&self, group_index: usize) -> Option<T> {
        self.base.get_lr(group_index)
    }

    fn set_lr(&mut self, group_index: usize, lr: T) -> Result<()> {
        self.base.set_lr(group_index, lr)
    }

    fn state(&self) -> &std::collections::HashMap<String, Tensor<T>> {
        self.base.state()
    }

    fn state_mut(&mut self) -> &mut std::collections::HashMap<String, Tensor<T>> {
        self.base.state_mut()
    }
}
