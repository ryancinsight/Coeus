//! RMSprop optimizer
//!
//! Implements the RMSprop (Root Mean Square Propagation) algorithm,
//! compatible with PyTorch's `torch.optim.RMSprop`.

use crate::{BaseOptimizer, Optimizer, ParamGroup, Result};
use coeus_tensor::Tensor;

/// RMSprop optimizer
///
/// Implements the RMSprop algorithm, which divides the learning rate
/// by a running average of the magnitudes of recent gradients.
///
/// ## Mathematical Formula
///
/// ```text
/// v_t = α * v_{t-1} + (1 - α) * g_t²
/// p_t = p_{t-1} - lr * g_t / (√v_t + ε)
/// ```
///
/// Where α is the smoothing constant (typically 0.99), and ε is a small
/// constant for numerical stability.
///
/// Compatible with PyTorch's `torch.optim.RMSprop`.
pub struct Rmsprop<T: coeus_dtype::FloatDtype> {
    base: BaseOptimizer<T>,
    alpha: T,
    eps: T,
    weight_decay: T,
    momentum: Option<T>,
    centered: bool,
}

impl<T: coeus_dtype::FloatDtype> Rmsprop<T> {
    /// Create a new RMSprop optimizer with default parameters
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate (default: 0.01)
    ///
    /// # Example
    /// ```rust
    /// use coeus_optim::Rmsprop;
    /// use coeus_tensor::Tensor;
    ///
    /// let params = vec![Tensor::from_vec(vec![1.0, 2.0], vec![2])];
    /// let optimizer = Rmsprop::new(params, 0.01);
    /// ```
    pub fn new(params: Vec<Tensor<T>>, lr: T) -> Self {
        Self::with_options(
            params,
            lr,
            T::from(0.99).unwrap(),
            T::from(1e-8).unwrap(),
            T::zero(),
            None,
            false,
        )
    }

    /// Create RMSprop with custom parameters
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate
    /// * `alpha` - Smoothing constant (typically 0.99)
    /// * `eps` - Small constant for numerical stability
    /// * `weight_decay` - Weight decay (L2 penalty) coefficient
    /// * `momentum` - Momentum factor (optional)
    /// * `centered` - Whether to use centered RMSprop variant
    pub fn with_options(
        params: Vec<Tensor<T>>,
        lr: T,
        alpha: T,
        eps: T,
        weight_decay: T,
        momentum: Option<T>,
        centered: bool,
    ) -> Self {
        let param_group = ParamGroup::new(params, lr, weight_decay);
        let base = BaseOptimizer::new(vec![param_group]);

        Self {
            base,
            alpha,
            eps,
            weight_decay,
            momentum,
            centered,
        }
    }

    /// Get alpha parameter
    pub fn alpha(&self) -> T {
        self.alpha
    }

    /// Get epsilon parameter
    pub fn eps(&self) -> T {
        self.eps
    }

    /// Get weight decay parameter
    pub fn weight_decay(&self) -> T {
        self.weight_decay
    }

    /// Get momentum parameter
    pub fn momentum(&self) -> Option<T> {
        self.momentum
    }

    /// Check if centered variant is enabled
    pub fn centered(&self) -> bool {
        self.centered
    }
}

impl<T: coeus_dtype::FloatDtype> Optimizer<T> for Rmsprop<T> {
    fn name(&self) -> &str {
        "RMSprop"
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

    fn step(&mut self) -> Result<()> {
        // Collect all updates first to avoid borrowing conflicts
        let mut state_updates = Vec::new();
        let mut param_updates = Vec::new();

        // Process each parameter group
        for group_idx in 0..self.base.param_groups().len() {
            let group = &self.base.param_groups()[group_idx];
            let lr = group.lr;
            let weight_decay = group.weight_decay;

            // Process each parameter in the group
            for param_idx in 0..group.params.len() {
                let param_key = format!(
                    "rmsprop_{}_{}_{:p}",
                    group_idx, param_idx, &group.params[param_idx] as *const _
                );

                // Get gradient for this parameter
                let Some(grad) = group.params[param_idx].grad() else {
                    continue; // Skip parameters without gradients
                };

                // Get or initialize state variables
                let square_avg_key = format!("{}_square_avg", param_key);
                let grad_avg_key = format!("{}_grad_avg", param_key);
                let momentum_buffer_key = format!("{}_momentum_buffer", param_key);

                // Get current state or initialize to zeros
                let square_avg_prev = self
                    .base
                    .state()
                    .get(&square_avg_key)
                    .cloned()
                    .unwrap_or_else(|| Tensor::zeros_like(&grad));
                let grad_avg_prev = if self.centered {
                    self.base
                        .state()
                        .get(&grad_avg_key)
                        .cloned()
                        .unwrap_or_else(|| Tensor::zeros_like(&grad))
                } else {
                    Tensor::zeros_like(&grad)
                };
                let momentum_buffer_prev = if self.momentum.is_some() {
                    self.base
                        .state()
                        .get(&momentum_buffer_key)
                        .cloned()
                        .unwrap_or_else(|| Tensor::zeros_like(&grad))
                } else {
                    Tensor::zeros_like(&grad)
                };

                // Apply weight decay if specified
                let effective_grad = if weight_decay != T::zero() {
                    let param_ref = &group.params[param_idx];
                    let wd_tensor = Tensor::scalar(weight_decay);
                    (&grad + &(param_ref * &wd_tensor)?)?
                } else {
                    grad.clone()
                };

                // Update moving average of squared gradients: v_t = α * v_{t-1} + (1 - α) * g_t²
                let alpha_tensor = Tensor::scalar(self.alpha);
                let one_minus_alpha_tensor = Tensor::scalar(T::one() - self.alpha);
                let square_avg_t = (&alpha_tensor * &square_avg_prev)?;
                let grad_squared = (&effective_grad * &effective_grad)?;
                let grad_squared_term = (&one_minus_alpha_tensor * &grad_squared)?;
                let square_avg_t = (&square_avg_t + &grad_squared_term)?;

                // Update moving average of gradients (for centered RMSprop)
                let grad_avg_t = if self.centered {
                    let grad_avg_update = (&alpha_tensor * &grad_avg_prev)?;
                    let grad_term = (&one_minus_alpha_tensor * &effective_grad)?;
                    (&grad_avg_update + &grad_term)?
                } else {
                    grad_avg_prev.clone()
                };

                // Compute the adaptive learning rate denominator
                // For centered RMSprop: v̂_t = v_t - (moving_avg_g)² + ε
                // For regular RMSprop: v̂_t = v_t + ε
                let eps_tensor = Tensor::scalar(self.eps);
                let adaptive_lr_denom = if self.centered {
                    let grad_avg_squared = (&grad_avg_t * &grad_avg_t)?;
                    let centered_square_avg = (&square_avg_t - &grad_avg_squared)?;
                    let centered_square_avg_sqrt = centered_square_avg.sqrt();
                    (&centered_square_avg_sqrt + &eps_tensor)?
                } else {
                    let square_avg_sqrt = square_avg_t.sqrt();
                    (&square_avg_sqrt + &eps_tensor)?
                };

                // Compute the update
                let lr_tensor = Tensor::scalar(lr);
                let lr_grad = (&lr_tensor * &effective_grad)?;
                let update = (&lr_grad / &adaptive_lr_denom)?;

                // Apply momentum if specified
                let final_update = if let Some(momentum_val) = self.momentum {
                    let momentum_tensor = Tensor::scalar(momentum_val);
                    let momentum_update = (&momentum_tensor * &momentum_buffer_prev)?;
                    let new_momentum_buffer = (&momentum_update + &update)?;
                    state_updates.push((momentum_buffer_key, new_momentum_buffer.clone()));
                    new_momentum_buffer
                } else {
                    update
                };

                // Compute new parameter value
                let param_data = group.params[param_idx].data();
                let update_data = final_update.data();
                let new_param_data: Vec<T> = param_data
                    .iter()
                    .zip(update_data.iter())
                    .map(|(p, u)| *p - *u)
                    .collect();

                let new_param_shape = group.params[param_idx].shape().to_vec();
                let mut new_param = Tensor::from_vec(new_param_data, new_param_shape);

                // Preserve gradient tracking
                if group.params[param_idx].requires_grad() {
                    new_param.set_requires_grad(true);
                }

                // Store state updates
                state_updates.push((square_avg_key, square_avg_t));
                if self.centered {
                    state_updates.push((grad_avg_key, grad_avg_t));
                }
                param_updates.push((group_idx, param_idx, new_param));
            }
        }

        // Apply all state updates
        for (key, tensor) in state_updates {
            self.base.state_mut().insert(key, tensor);
        }

        // Apply all parameter updates
        for (group_idx, param_idx, new_param) in param_updates {
            if let Some(group) = self.base.param_groups_mut().get_mut(group_idx) {
                if let Some(param) = group.params.get_mut(param_idx) {
                    *param = new_param;
                }
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        self.base.zero_grad();
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

/// Builder pattern for RMSprop optimizer
pub struct RMSpropBuilder<T: coeus_dtype::FloatDtype> {
    params: Vec<Tensor<T>>,
    lr: T,
    alpha: T,
    eps: T,
    weight_decay: T,
    momentum: Option<T>,
    centered: bool,
}

impl<T: coeus_dtype::FloatDtype> RMSpropBuilder<T> {
    /// Create a new RMSprop builder
    pub fn new(params: Vec<Tensor<T>>, lr: T) -> Self {
        Self {
            params,
            lr,
            alpha: T::from(0.99).unwrap(),
            eps: T::from(1e-8).unwrap(),
            weight_decay: T::zero(),
            momentum: None,
            centered: false,
        }
    }

    /// Set alpha (smoothing constant)
    pub fn alpha(mut self, alpha: T) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set epsilon
    pub fn eps(mut self, eps: T) -> Self {
        self.eps = eps;
        self
    }

    /// Set weight decay
    pub fn weight_decay(mut self, weight_decay: T) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set momentum
    pub fn momentum(mut self, momentum: T) -> Self {
        self.momentum = Some(momentum);
        self
    }

    /// Enable centered RMSprop
    pub fn centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }

    /// Build the RMSprop optimizer
    pub fn build(self) -> Rmsprop<T> {
        Rmsprop::with_options(
            self.params,
            self.lr,
            self.alpha,
            self.eps,
            self.weight_decay,
            self.momentum,
            self.centered,
        )
    }
}
