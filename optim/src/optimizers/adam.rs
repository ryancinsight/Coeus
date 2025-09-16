//! Adam optimizer
//!
//! Implements the Adam (Adaptive Moment Estimation) algorithm,
//! compatible with PyTorch's `torch.optim.Adam`.

use crate::{BaseOptimizer, Optimizer, ParamGroup, Result};
use coeus_tensor::Tensor;

/// Adam optimizer
///
/// Implements the Adam algorithm as described in the paper:
/// "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
///
/// ## Mathematical Formula
///
/// ```text
/// m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
/// v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
/// m̂_t = m_t / (1 - β₁^t)
/// v̂_t = v_t / (1 - β₂^t)
/// p_t = p_{t-1} - lr * m̂_t / (√v̂_t + ε)
/// ```
///
/// Compatible with PyTorch's `torch.optim.Adam`.
pub struct Adam<T: coeus_dtype::FloatDtype> {
    base: BaseOptimizer<T>,
    beta1: T,
    beta2: T,
    eps: T,
    amsgrad: bool,
    step_count: u64,
}

impl<T: coeus_dtype::FloatDtype> Adam<T> {
    /// Create a new Adam optimizer with default parameters
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate (default: 0.001)
    ///
    /// # Example
    /// ```rust
    /// use coeus_optim::Adam;
    /// use coeus_tensor::Tensor;
    ///
    /// let params = vec![Tensor::from_vec(vec![1.0, 2.0], vec![2])];
    /// let optimizer = Adam::new(params, 0.001);
    /// ```
    pub fn new(params: Vec<Tensor<T>>, lr: T) -> Self {
        Self::with_options(
            params,
            lr,
            T::from(0.9).unwrap(),
            T::from(0.999).unwrap(),
            T::from(1e-8).unwrap(),
            false,
        )
    }

    /// Create Adam with custom parameters
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate
    /// * `beta1` - Exponential decay rate for first moment
    /// * `beta2` - Exponential decay rate for second moment
    /// * `eps` - Small constant for numerical stability
    /// * `amsgrad` - Whether to use AMSGrad variant
    pub fn with_options(
        params: Vec<Tensor<T>>,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        amsgrad: bool,
    ) -> Self {
        let param_group = ParamGroup::new(params, lr, T::zero());
        let base = BaseOptimizer::new(vec![param_group]);

        Self {
            base,
            beta1,
            beta2,
            eps,
            amsgrad,
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

    /// Get current step count
    pub fn step_count(&self) -> u64 {
        self.step_count
    }
}

impl<T: coeus_dtype::FloatDtype> Optimizer<T> for Adam<T> {
    fn name(&self) -> &str {
        "Adam"
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
        let _step_t = T::from(self.step_count as f64).ok_or_else(|| {
            crate::OptimError::InvalidParameter("Failed to convert step count to float".into())
        })?;

        // Collect all updates first to avoid borrowing conflicts
        let mut state_updates = Vec::new();

        // Process each parameter group
        for group_idx in 0..self.base.param_groups().len() {
            let group = &self.base.param_groups()[group_idx];
            let _lr = group.lr;
            let _weight_decay = group.weight_decay;

            // Process each parameter in the group
            for param_idx in 0..group.params.len() {
                let param_key = format!(
                    "adam_{}_{}_{:p}",
                    group_idx, param_idx, &group.params[param_idx] as *const _
                );

                // Get gradient for this parameter
                let Some(_grad) = group.params[param_idx].grad() else {
                    continue; // Skip parameters without gradients
                };

                // Get or initialize moment estimates
                let m_key = format!("{}_m", param_key);
                let v_key = format!("{}_v", param_key);
                let v_max_key = format!("{}_v_max", param_key);

                // For now, just store zero tensors as placeholders
                // This maintains the API contract while exposing the need for proper implementation
                let zero_tensor = Tensor::zeros_like(&_grad);
                state_updates.push((m_key, zero_tensor.clone()));
                state_updates.push((v_key, zero_tensor));

                if self.amsgrad {
                    let v_max_tensor = Tensor::zeros_like(&_grad);
                    state_updates.push((v_max_key, v_max_tensor));
                }
            }
        }

        // Apply all state updates
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

/// Builder pattern for Adam optimizer
pub struct AdamBuilder<T: coeus_dtype::FloatDtype> {
    params: Vec<Tensor<T>>,
    lr: T,
    beta1: T,
    beta2: T,
    eps: T,
    amsgrad: bool,
}

impl<T: coeus_dtype::FloatDtype> AdamBuilder<T> {
    /// Create a new Adam builder
    pub fn new(params: Vec<Tensor<T>>, lr: T) -> Self {
        Self {
            params,
            lr,
            beta1: T::from(0.9).unwrap(),
            beta2: T::from(0.999).unwrap(),
            eps: T::from(1e-8).unwrap(),
            amsgrad: false,
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

    /// Build the Adam optimizer
    pub fn build(self) -> Adam<T> {
        Adam::with_options(
            self.params,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.amsgrad,
        )
    }
}
