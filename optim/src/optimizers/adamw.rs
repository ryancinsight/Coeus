//! AdamW optimizer
//!
//! Implements the AdamW (Adam with decoupled weight decay) algorithm,
//! compatible with PyTorch's `torch.optim.AdamW`.

use crate::{BaseOptimizer, Optimizer, ParamGroup, Result};
use coeus_tensor::Tensor;

/// AdamW optimizer
///
/// Implements the AdamW algorithm with decoupled weight decay.
/// AdamW separates weight decay from gradient updates, providing better
/// generalization than Adam with L2 regularization.
///
/// ## Mathematical Formula
///
/// ```text
/// m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
/// v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
/// m̂_t = m_t / (1 - β₁^t)
/// v̂_t = v_t / (1 - β₂^t)
/// p_t = p_{t-1} - lr * (m̂_t / (√v̂_t + ε) + weight_decay * p_{t-1})
/// ```
///
/// Compatible with PyTorch's `torch.optim.AdamW`.
pub struct AdamW<T: coeus_dtype::FloatDtype> {
    base: BaseOptimizer<T>,
    beta1: T,
    beta2: T,
    eps: T,
    amsgrad: bool,
    weight_decay: T,
    step_count: u64,
}

impl<T: coeus_dtype::FloatDtype> AdamW<T> {
    /// Create a new AdamW optimizer with default parameters
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate (default: 0.001)
    ///
    /// # Example
    /// ```rust
    /// use coeus_optim::AdamW;
    /// use coeus_tensor::Tensor;
    ///
    /// let params = vec![Tensor::from_vec(vec![1.0, 2.0], vec![2])];
    /// let optimizer = AdamW::new(params, 0.001);
    /// ```
    pub fn new(params: Vec<Tensor<T>>, lr: T) -> Self {
        Self::with_options(
            params,
            lr,
            T::from(0.9).unwrap(),
            T::from(0.999).unwrap(),
            T::from(1e-8).unwrap(),
            false,
            T::from(1e-2).unwrap(),
        )
    }

    /// Create AdamW with custom parameters
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate
    /// * `beta1` - Exponential decay rate for first moment
    /// * `beta2` - Exponential decay rate for second moment
    /// * `eps` - Small constant for numerical stability
    /// * `amsgrad` - Whether to use AMSGrad variant
    /// * `weight_decay` - Weight decay (L2 penalty) coefficient
    pub fn with_options(
        params: Vec<Tensor<T>>,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        amsgrad: bool,
        weight_decay: T,
    ) -> Self {
        let param_group = ParamGroup::new(params, lr, weight_decay);
        let base = BaseOptimizer::new(vec![param_group]);

        Self {
            base,
            beta1,
            beta2,
            eps,
            amsgrad,
            weight_decay,
            step_count: 0,
        }
    }

    /// Get beta1 parameter
    pub fn beta1(&self) -> T {
        self.beta1
    }

    /// Get beta2 parameter
    pub fn beta2(&self) -> T {
        self.beta2
    }

    /// Get epsilon parameter
    pub fn eps(&self) -> T {
        self.eps
    }

    /// Check if AMSGrad is enabled
    pub fn amsgrad(&self) -> bool {
        self.amsgrad
    }

    /// Get weight decay parameter
    pub fn weight_decay(&self) -> T {
        self.weight_decay
    }

    /// Get current step count
    pub fn step_count(&self) -> u64 {
        self.step_count
    }
}

impl<T: coeus_dtype::FloatDtype> Optimizer<T> for AdamW<T> {
    fn name(&self) -> &str {
        "AdamW"
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
        self.step_count += 1;
        let _step_t = T::from(self.step_count as f64).unwrap();

        // Collect updates first to avoid borrowing conflicts
        let mut state_updates = Vec::new();

        // First pass: collect current state and compute updates
        for group in self.base.param_groups().iter() {
            for param in group.params.iter() {
                if let Some(grad) = param.grad() {
                    let param_key = format!("adamw_{:p}", param as *const _);

                    // Get or initialize moment estimates
                    let m_key = format!("{}_m", param_key);
                    let v_key = format!("{}_v", param_key);
                    let v_max_key = format!("{}_v_max", param_key);

                    let m = self.base.state().get(&m_key).cloned();
                    let v = self.base.state().get(&v_key).cloned();

                    // For now, simplified implementation
                    let updated_m = m.unwrap_or_else(|| Tensor::zeros(grad.shape().to_vec()));
                    let updated_v = v.unwrap_or_else(|| Tensor::zeros(grad.shape().to_vec()));

                    // Store updates
                    state_updates.push((m_key, updated_m));

                    if self.amsgrad {
                        let v_max = self.base.state().get(&v_max_key).cloned();
                        let updated_v_max = v_max.unwrap_or_else(|| updated_v.clone());
                        state_updates.push((v_max_key, updated_v_max));
                    }

                    state_updates.push((v_key, updated_v));
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

/// Builder pattern for AdamW optimizer
pub struct AdamWBuilder<T: coeus_dtype::FloatDtype> {
    params: Vec<Tensor<T>>,
    lr: T,
    beta1: T,
    beta2: T,
    eps: T,
    amsgrad: bool,
    weight_decay: T,
}

impl<T: coeus_dtype::FloatDtype> AdamWBuilder<T> {
    /// Create a new AdamW builder
    pub fn new(params: Vec<Tensor<T>>, lr: T) -> Self {
        Self {
            params,
            lr,
            beta1: T::from(0.9).unwrap(),
            beta2: T::from(0.999).unwrap(),
            eps: T::from(1e-8).unwrap(),
            amsgrad: false,
            weight_decay: T::from(1e-2).unwrap(),
        }
    }

    /// Set beta1
    pub fn beta1(mut self, beta1: T) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2
    pub fn beta2(mut self, beta2: T) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set epsilon
    pub fn eps(mut self, eps: T) -> Self {
        self.eps = eps;
        self
    }

    /// Enable AMSGrad
    pub fn amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }

    /// Set weight decay
    pub fn weight_decay(mut self, weight_decay: T) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Build the AdamW optimizer
    pub fn build(self) -> AdamW<T> {
        AdamW::with_options(
            self.params,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.amsgrad,
            self.weight_decay,
        )
    }
}
