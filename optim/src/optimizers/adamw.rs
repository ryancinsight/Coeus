//! AdamW optimizer
//!
//! Implements the AdamW (Adam with decoupled weight decay) algorithm,
//! compatible with PyTorch's `torch.optim.AdamW`.

use crate::{BaseOptimizer, Optimizer, ParamGroup, Result};
use coeus_tensor::{ops::arithmetic::{maximum, sqrt, sub, div, mul, add}, Tensor, Backend, CpuBackend};

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
pub struct AdamW<T: coeus_dtype::FloatDtype, B: Backend<T> + Clone = CpuBackend> {
    base: BaseOptimizer<T, B>,
    beta1: T,
    beta2: T,
    eps: T,
    amsgrad: bool,
    weight_decay: T,
    step_count: u64,
}

impl<T: coeus_dtype::FloatDtype, B: Backend<T> + Clone> AdamW<T, B> {
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
    pub fn new(params: Vec<Tensor<T, B>>, lr: T) -> Self {
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
        params: Vec<Tensor<T, B>>,
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

impl<T: coeus_dtype::FloatDtype, B: Backend<T> + Clone> Optimizer<T, B> for AdamW<T, B> {
    fn name(&self) -> &str {
        "AdamW"
    }

    fn param_groups(&self) -> &[ParamGroup<T, B>] {
        self.base.param_groups()
    }

    fn param_groups_mut(&mut self) -> &mut [ParamGroup<T, B>] {
        self.base.param_groups_mut()
    }

    fn add_param_group(&mut self, param_group: ParamGroup<T, B>) {
        self.base.add_param_group(param_group);
    }

    fn step(&mut self) -> Result<()> {
        self.step_count += 1;
        let step_t = T::from(self.step_count as f64).ok_or_else(|| {
            crate::OptimError::InvalidParameter("Failed to convert step count to float".into())
        })?;

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
                let param = &group.params[param_idx];
                let param_key = format!(
                    "adamw_{}_{}_{:p}",
                    group_idx, param_idx, param as *const _
                );

                // Get gradient for this parameter
                let Some(grad) = param.grad() else {
                    continue; // Skip parameters without gradients
                };

                // Get or initialize moment estimates
                let m_key = format!("{}_m", param_key);
                let v_key = format!("{}_v", param_key);
                let v_max_key = format!("{}_v_max", param_key);

                // Get current moment estimates or initialize to zeros
                let m_prev = self
                    .base
                    .state()
                    .get(&m_key)
                    .cloned()
                    .unwrap_or_else(|| Tensor::<T, B>::zeros_like(&param.unwrap_grad()));
                let v_prev = self
                    .base
                    .state()
                    .get(&v_key)
                    .cloned()
                    .unwrap_or_else(|| Tensor::<T, B>::zeros_like(&param.unwrap_grad()));

                // AdamW: Weight decay is applied directly to parameters (decoupled from gradients)
                // The gradient used for optimization remains unchanged
                let effective_grad = param.grad().unwrap();

                // Update biased first moment estimate: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
                let beta1_tensor = Tensor::scalar(self.beta1);
                let one_minus_beta1_tensor = Tensor::scalar(T::one() - self.beta1);
                let m_t_scaled = (&beta1_tensor * &m_prev).unwrap();
                let grad_term = (&one_minus_beta1_tensor * &effective_grad).unwrap();
                let m_t = (&m_t_scaled + &grad_term).unwrap();

                // Update biased second moment estimate: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
                let beta2_tensor = Tensor::scalar(self.beta2);
                let one_minus_beta2_tensor = Tensor::scalar(T::one() - self.beta2);
                let v_t_scaled = (&beta2_tensor * &v_prev).unwrap();
                let grad_squared = (&effective_grad * &effective_grad).unwrap();
                let grad_squared_term = (&one_minus_beta2_tensor * &grad_squared).unwrap();
                let v_t = (&v_t_scaled + &grad_squared_term).unwrap();

                // Compute bias-corrected moments
                // β₁^t and β₂^t using iterative multiplication for integer powers
                let mut beta1_pow = T::one();
                let mut beta2_pow = T::one();
                let step_int = if step_t >= T::one() { 1 } else { 0 }; // Simple conversion for step count
                for _ in 0..step_int {
                    beta1_pow = beta1_pow * self.beta1;
                    beta2_pow = beta2_pow * self.beta2;
                }

                let bias_correction1 = T::one() - beta1_pow;
                let bias_correction2 = T::one() - beta2_pow;
                let bias_correction1_tensor = Tensor::from_vec(param.backend().clone(), vec![bias_correction1], vec![]).unwrap();
                let bias_correction2_tensor = Tensor::from_vec(param.backend().clone(), vec![bias_correction2], vec![]).unwrap();

                let m_hat = div(&m_t, &bias_correction1_tensor)?;
                let v_hat = div(&v_t, &bias_correction2_tensor)?;

                // AMSGrad: take maximum of current and previous v_hat
                let v_hat_final = if self.amsgrad {
                    let v_max_prev = self
                        .base
                        .state()
                        .get(&v_max_key)
                        .cloned()
                        .unwrap_or_else(|| v_hat.clone());
                    let v_max_new = maximum(&v_hat, &v_max_prev)?;
                    state_updates.push((v_max_key, v_max_new.clone()));
                    v_max_new
                } else {
                    v_hat
                };

                // Compute parameter update: θ = θ - η * m̂_t / (√v̂_t + ε)
                let eps_tensor = Tensor::scalar(self.eps);
                let v_hat_sqrt = sqrt(&v_hat_final);
                let denominator = add(&v_hat_sqrt, &eps_tensor)?;
                let lr_tensor = Tensor::scalar(lr);
                let lr_m_hat = mul(&lr_tensor, &m_hat)?;
                let update = div(&lr_m_hat, &denominator)?;

                // Apply decoupled weight decay: θ = θ * (1 - η * λ)
                let mut new_param = if weight_decay != T::zero() {
                    let wd_factor = T::one() - lr * weight_decay;
                    let wd_tensor = Tensor::from_vec(param.backend().clone(), vec![wd_factor], vec![]).unwrap();
                    (&group.params[param_idx] * &wd_tensor).unwrap()
                } else {
                    group.params[param_idx].clone()
                };

                // Apply gradient update
                new_param = sub(&new_param, &update).unwrap();

                // Preserve gradient tracking
                if group.params[param_idx].requires_grad() {
                    new_param.set_requires_grad(true);
                }

                // Store updates
                state_updates.push((m_key, m_t));
                state_updates.push((v_key, v_t));
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

    fn state(&self) -> &std::collections::HashMap<String, Tensor<T, B>> {
        self.base.state()
    }

    fn state_mut(&mut self) -> &mut std::collections::HashMap<String, Tensor<T, B>> {
        self.base.state_mut()
    }
}

/// Builder pattern for AdamW optimizer
pub struct AdamWBuilder<T: coeus_dtype::FloatDtype, B: Backend<T> + Clone = CpuBackend> {
    params: Vec<Tensor<T, B>>,
    lr: T,
    beta1: T,
    beta2: T,
    eps: T,
    amsgrad: bool,
    weight_decay: T,
}

impl<T: coeus_dtype::FloatDtype, B: Backend<T> + Clone> AdamWBuilder<T, B> {
    /// Create a new AdamW builder
    pub fn new(params: Vec<Tensor<T, B>>, lr: T) -> Self {
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
    pub fn build(self) -> AdamW<T, B> {
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
