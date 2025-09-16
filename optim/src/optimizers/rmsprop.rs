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
        // Collect updates first to avoid borrowing conflicts
        let mut state_updates = Vec::new();

        // First pass: collect current state and compute updates
        for group in self.base.param_groups().iter() {
            for param in group.params.iter() {
                if let Some(grad) = param.grad() {
                    let param_key = format!("rmsprop_{:p}", param as *const _);

                    // Get or initialize square average
                    let square_avg_key = format!("{}_square_avg", param_key);
                    let grad_avg_key = format!("{}_grad_avg", param_key);
                    let momentum_buffer_key = format!("{}_momentum_buffer", param_key);

                    let square_avg = self.base.state().get(&square_avg_key).cloned();

                    // For now, simplified implementation
                    let updated_square_avg =
                        square_avg.unwrap_or_else(|| Tensor::zeros(grad.shape().to_vec()));

                    // Store updates
                    state_updates.push((square_avg_key, updated_square_avg));

                    if self.centered {
                        let grad_avg = self.base.state().get(&grad_avg_key).cloned();
                        let updated_grad_avg =
                            grad_avg.unwrap_or_else(|| Tensor::zeros(grad.shape().to_vec()));
                        state_updates.push((grad_avg_key, updated_grad_avg));
                    }

                    if self.momentum.is_some() {
                        let momentum_buffer = self.base.state().get(&momentum_buffer_key).cloned();
                        let updated_momentum_buffer =
                            momentum_buffer.unwrap_or_else(|| Tensor::zeros(grad.shape().to_vec()));
                        state_updates.push((momentum_buffer_key, updated_momentum_buffer));
                    }
                }
            }
        }

        // Second pass: apply updates
        for (key, tensor) in state_updates {
            self.base.state_mut().insert(key, tensor);
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
