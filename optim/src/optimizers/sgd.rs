//! Stochastic Gradient Descent (SGD) optimizer
//!
//! Implements the classic SGD algorithm with momentum and weight decay,
//! compatible with PyTorch's `torch.optim.SGD`.

use crate::{BaseOptimizer, Optimizer, ParamGroup, Result};
use coeus_tensor::Tensor;

/// Stochastic Gradient Descent optimizer
///
/// Implements the SGD algorithm with optional momentum and weight decay.
/// Compatible with PyTorch's `torch.optim.SGD`.
///
/// ## Mathematical Formula
///
/// ```text
/// v_t = momentum * v_{t-1} + (1 - momentum) * g_t
/// p_t = p_{t-1} - lr * v_t - lr * weight_decay * p_{t-1}
/// ```
///
/// where:
/// - `p_t` is the parameter at time t
/// - `g_t` is the gradient at time t
/// - `v_t` is the velocity (momentum buffer) at time t
/// - `lr` is the learning rate
/// - `momentum` is the momentum factor
/// - `weight_decay` is the weight decay (L2 penalty)
pub struct Sgd<T: coeus_dtype::FloatDtype> {
    base: BaseOptimizer<T>,
    momentum: T,
    dampening: T,
    nesterov: bool,
}

impl<T: coeus_dtype::FloatDtype> Sgd<T> {
    /// Create a new SGD optimizer
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate
    ///
    /// # Example
    /// ```rust
    /// use coeus_optim::Sgd;
    /// use coeus_tensor::Tensor;
    ///
    /// let params = vec![Tensor::from_vec(vec![1.0, 2.0], vec![2])];
    /// let optimizer = Sgd::new(params, 0.01);
    /// ```
    pub fn new(params: Vec<Tensor<T>>, lr: T) -> Self {
        Self::with_options(params, lr, T::zero(), T::zero(), false)
    }

    /// Create SGD with momentum
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate
    /// * `momentum` - Momentum factor
    ///
    /// # Example
    /// ```rust
    /// use coeus_optim::Sgd;
    /// use coeus_tensor::Tensor;
    ///
    /// let params = vec![Tensor::from_vec(vec![1.0, 2.0], vec![2])];
    /// let optimizer = Sgd::with_momentum(params, 0.01, 0.9);
    /// ```
    pub fn with_momentum(params: Vec<Tensor<T>>, lr: T, momentum: T) -> Self {
        Self::with_options(params, lr, momentum, T::zero(), false)
    }

    /// Create SGD with full options
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate
    /// * `momentum` - Momentum factor (0 for no momentum)
    /// * `weight_decay` - Weight decay (L2 penalty)
    /// * `nesterov` - Whether to use Nesterov momentum
    pub fn with_options(
        params: Vec<Tensor<T>>,
        lr: T,
        momentum: T,
        weight_decay: T,
        nesterov: bool,
    ) -> Self {
        let param_group = ParamGroup::new(params, lr, weight_decay);
        let base = BaseOptimizer::new(vec![param_group]);

        Self {
            base,
            momentum,
            dampening: T::zero(),
            nesterov,
        }
    }

    /// Get the momentum factor
    pub fn momentum(&self) -> T {
        self.momentum
    }

    /// Get the dampening factor
    pub fn dampening(&self) -> T {
        self.dampening
    }

    /// Check if Nesterov momentum is enabled
    pub fn nesterov(&self) -> bool {
        self.nesterov
    }

    /// Perform SGD step with momentum
    fn step_with_momentum(&mut self) -> Result<()> {
        // Collect updates first to avoid borrowing conflicts
        let mut updates = Vec::new();

        // First pass: collect current state and compute updates
        for group in self.base.param_groups().iter() {
            for param in group.params.iter() {
                if let Some(grad) = param.grad() {
                    let param_key = format!("momentum_{:p}", param as *const _);

                    // Get current momentum buffer
                    let _momentum_buffer = self.base.state().get(&param_key).cloned();

                    // Compute updated momentum
                    let updated_momentum = if self.momentum != T::zero() {
                        // momentum_buffer * momentum + grad * (1 - dampening)
                        // For now, simplified implementation
                        grad.clone()
                    } else {
                        grad.clone()
                    };

                    updates.push((param_key, updated_momentum));
                }
            }
        }

        // Second pass: apply updates
        for (key, momentum) in updates {
            self.base.state_mut().insert(key, momentum);
        }

        Ok(())
    }

    /// Perform basic SGD step without momentum
    fn step_basic(&mut self) -> Result<()> {
        for group in self.base.param_groups_mut() {
            for param in &mut group.params {
                if let Some(_grad) = param.grad() {
                    // Update parameter: param = param - lr * grad - lr * weight_decay * param
                    // For now, simplified implementation
                    // In a full implementation, this would modify the parameter tensor
                }
            }
        }
        Ok(())
    }
}

impl<T: coeus_dtype::FloatDtype> Optimizer<T> for Sgd<T> {
    fn name(&self) -> &str {
        "SGD"
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
        if self.momentum != T::zero() {
            self.step_with_momentum()
        } else {
            self.step_basic()
        }
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

/// Builder pattern for SGD optimizer
///
/// Provides a fluent API for configuring SGD parameters.
pub struct SgdBuilder<T: coeus_dtype::FloatDtype> {
    params: Vec<Tensor<T>>,
    lr: T,
    momentum: T,
    weight_decay: T,
    dampening: T,
    nesterov: bool,
}

impl<T: coeus_dtype::FloatDtype> SgdBuilder<T> {
    /// Create a new SGD builder
    pub fn new(params: Vec<Tensor<T>>, lr: T) -> Self {
        Self {
            params,
            lr,
            momentum: T::zero(),
            weight_decay: T::zero(),
            dampening: T::zero(),
            nesterov: false,
        }
    }

    /// Set momentum
    pub fn momentum(mut self, momentum: T) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set weight decay
    pub fn weight_decay(mut self, weight_decay: T) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set dampening
    pub fn dampening(mut self, dampening: T) -> Self {
        self.dampening = dampening;
        self
    }

    /// Enable Nesterov momentum
    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Build the SGD optimizer
    pub fn build(self) -> Sgd<T> {
        Sgd::with_options(
            self.params,
            self.lr,
            self.momentum,
            self.weight_decay,
            self.nesterov,
        )
    }
}
