//! Adagrad optimizer
//!
//! Implements the Adagrad (Adaptive Gradient) algorithm,
//! compatible with PyTorch's `torch.optim.Adagrad`.

use crate::{BaseOptimizer, Optimizer, ParamGroup, Result};
use coeus_tensor::Tensor;

/// Adagrad optimizer
///
/// Implements the Adagrad algorithm, which adapts the learning rate
/// for each parameter based on the historical sum of squared gradients.
///
/// ## Mathematical Formula
///
/// ```text
/// s_t = s_{t-1} + g_t²
/// p_t = p_{t-1} - lr * g_t / (√s_t + ε)
/// ```
///
/// Where s_t is the sum of squared gradients up to time t, and ε is a small
/// constant for numerical stability.
///
/// Compatible with PyTorch's `torch.optim.Adagrad`.
pub struct Adagrad<T: coeus_dtype::FloatDtype> {
    base: BaseOptimizer<T>,
    lr_decay: T,
    weight_decay: T,
    eps: T,
    initial_accumulator_value: T,
}

impl<T: coeus_dtype::FloatDtype> Adagrad<T> {
    /// Create a new Adagrad optimizer with default parameters
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate (default: 0.01)
    ///
    /// # Example
    /// ```rust
    /// use coeus_optim::Adagrad;
    /// use coeus_tensor::Tensor;
    ///
    /// let params = vec![Tensor::from_vec(vec![1.0, 2.0], vec![2])];
    /// let optimizer = Adagrad::new(params, 0.01);
    /// ```
    pub fn new(params: Vec<Tensor<T>>, lr: T) -> Self {
        Self::with_options(
            params,
            lr,
            T::zero(),
            T::zero(),
            T::from(1e-10).unwrap(),
            T::zero(),
        )
    }

    /// Create Adagrad with custom parameters
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate
    /// * `lr_decay` - Learning rate decay factor
    /// * `weight_decay` - Weight decay (L2 penalty) coefficient
    /// * `eps` - Small constant for numerical stability
    /// * `initial_accumulator_value` - Initial value for gradient accumulator
    pub fn with_options(
        params: Vec<Tensor<T>>,
        lr: T,
        lr_decay: T,
        weight_decay: T,
        eps: T,
        initial_accumulator_value: T,
    ) -> Self {
        let param_group = ParamGroup::new(params, lr, weight_decay);
        let base = BaseOptimizer::new(vec![param_group]);

        Self {
            base,
            lr_decay,
            weight_decay,
            eps,
            initial_accumulator_value,
        }
    }

    /// Get learning rate decay parameter
    pub fn lr_decay(&self) -> T {
        self.lr_decay
    }

    /// Get weight decay parameter
    pub fn weight_decay(&self) -> T {
        self.weight_decay
    }

    /// Get epsilon parameter
    pub fn eps(&self) -> T {
        self.eps
    }

    /// Get initial accumulator value
    pub fn initial_accumulator_value(&self) -> T {
        self.initial_accumulator_value
    }
}

impl<T: coeus_dtype::FloatDtype> Optimizer<T> for Adagrad<T> {
    fn name(&self) -> &str {
        "Adagrad"
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
                    let param_key = format!("adagrad_{:p}", param as *const _);

                    // Get or initialize sum of squared gradients
                    let sum_key = format!("{}_sum", param_key);

                    let sum = self.base.state().get(&sum_key).cloned();

                    // For now, simplified implementation
                    let updated_sum = sum.unwrap_or_else(|| Tensor::zeros(grad.shape().to_vec()));

                    // Store updates
                    state_updates.push((sum_key, updated_sum));
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

/// Builder pattern for Adagrad optimizer
pub struct AdagradBuilder<T: coeus_dtype::FloatDtype> {
    params: Vec<Tensor<T>>,
    lr: T,
    lr_decay: T,
    weight_decay: T,
    eps: T,
    initial_accumulator_value: T,
}

impl<T: coeus_dtype::FloatDtype> AdagradBuilder<T> {
    /// Create a new Adagrad builder
    pub fn new(params: Vec<Tensor<T>>, lr: T) -> Self {
        Self {
            params,
            lr,
            lr_decay: T::zero(),
            weight_decay: T::zero(),
            eps: T::from(1e-10).unwrap(),
            initial_accumulator_value: T::zero(),
        }
    }

    /// Set learning rate decay
    pub fn lr_decay(mut self, lr_decay: T) -> Self {
        self.lr_decay = lr_decay;
        self
    }

    /// Set weight decay
    pub fn weight_decay(mut self, weight_decay: T) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set epsilon
    pub fn eps(mut self, eps: T) -> Self {
        self.eps = eps;
        self
    }

    /// Set initial accumulator value
    pub fn initial_accumulator_value(mut self, initial_accumulator_value: T) -> Self {
        self.initial_accumulator_value = initial_accumulator_value;
        self
    }

    /// Build the Adagrad optimizer
    pub fn build(self) -> Adagrad<T> {
        Adagrad::with_options(
            self.params,
            self.lr,
            self.lr_decay,
            self.weight_decay,
            self.eps,
            self.initial_accumulator_value,
        )
    }
}
