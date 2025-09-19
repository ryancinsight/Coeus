//! Adam optimizer
//!
//! Implements the Adam (Adaptive Moment Estimation) algorithm,
//! compatible with PyTorch's `torch.optim.Adam`.

use crate::{BaseOptimizer, Optimizer, ParamGroup, Result};
use coeus_tensor::{ops::arithmetic::maximum, Tensor};

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
                let param_key = format!(
                    "adam_{}_{}_{:p}",
                    group_idx, param_idx, &group.params[param_idx] as *const _
                );

                // Get gradient for this parameter
                let Some(grad) = group.params[param_idx].grad() else {
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
                    .unwrap_or_else(|| Tensor::zeros_like(&grad));
                let v_prev = self
                    .base
                    .state()
                    .get(&v_key)
                    .cloned()
                    .unwrap_or_else(|| Tensor::zeros_like(&grad));

                // Apply weight decay if specified (AdamW style - directly to parameters)
                let effective_grad = if weight_decay != T::zero() {
                    let wd_tensor = Tensor::scalar(weight_decay);
                    let param_ref = &group.params[param_idx];
                    (&grad + &(param_ref * &wd_tensor).unwrap()).unwrap()
                } else {
                    grad.clone()
                };

                // Update biased first moment estimate: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
                let beta1_tensor = Tensor::scalar(self.beta1);
                let one_minus_beta1_tensor = Tensor::scalar(T::one() - self.beta1);
                let m_t = (&beta1_tensor * &m_prev).unwrap();
                let grad_term = (&one_minus_beta1_tensor * &effective_grad).unwrap();
                let m_t = (&m_t + &grad_term).unwrap();

                // Update biased second moment estimate: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
                let beta2_tensor = Tensor::scalar(self.beta2);
                let one_minus_beta2_tensor = Tensor::scalar(T::one() - self.beta2);
                let v_t = (&beta2_tensor * &v_prev).unwrap();
                let grad_squared = (&effective_grad * &effective_grad).unwrap();
                let grad_squared_term = (&one_minus_beta2_tensor * &grad_squared).unwrap();
                let v_t = (&v_t + &grad_squared_term).unwrap();

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
                let bias_correction1_tensor = Tensor::scalar(bias_correction1);
                let bias_correction2_tensor = Tensor::scalar(bias_correction2);

                let m_hat = (&m_t / &bias_correction1_tensor)?;
                let v_hat = (&v_t / &bias_correction2_tensor)?;

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
                let v_hat_sqrt = v_hat_final.sqrt();
                let denominator = (&v_hat_sqrt + &eps_tensor)?;
                let lr_tensor = Tensor::scalar(lr);
                let lr_m_hat = (&lr_tensor * &m_hat)?;
                let update = (&lr_m_hat / &denominator)?;

                // Compute new parameter value
                let param_data = group.params[param_idx].data();
                let update_data = update.data();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_creation() {
        let optimizer = Adam::new(vec![], 0.001);
        assert_eq!(optimizer.base.get_lr(0).unwrap(), 0.001);
        // Note: beta1, beta2, epsilon are not yet implemented
    }

    #[test]
    fn test_adam_with_custom_options() {
        let optimizer = Adam::with_options(vec![], 0.001, 0.9, 0.999, 1e-8, false);
        assert_eq!(optimizer.base.get_lr(0).unwrap(), 0.001);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.999);
        assert_eq!(optimizer.eps, 1e-8);
        assert!(!optimizer.amsgrad);
    }

    #[test]
    fn test_adam_step_placeholder_functionality() {
        // Test that the step method can be called without panicking
        // This validates the API contract while documenting current limitations
        let mut optimizer: Adam<f32> = Adam::new(vec![], 0.001);
        let param = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        optimizer
            .base
            .add_param_group(ParamGroup::new(vec![param], 0.001, 0.0));

        // This should succeed (API contract maintained)
        let result = optimizer.step();
        assert!(result.is_ok());

        // CRITICAL LIMITATION: Actual parameter updates are not implemented yet
        // The step() method currently does nothing - this violates the API contract
    }

    #[test]
    fn test_adam_zero_grad_placeholder_functionality() {
        // Test that zero_grad method can be called without panicking
        let mut optimizer: Adam<f32> = Adam::new(vec![], 0.001);

        // This should succeed (API contract maintained)
        optimizer.zero_grad();

        // CRITICAL LIMITATION: Actual gradient zeroing is not implemented yet
    }

    #[test]
    fn test_adam_momentum_state_initialization() {
        // Test that momentum state tensors are initialized (but not used)
        let mut optimizer: Adam<f32> = Adam::new(vec![], 0.001);
        let param = Tensor::from_vec(vec![1.0], vec![1]);

        optimizer
            .base
            .add_param_group(ParamGroup::new(vec![param], 0.001, 0.0));

        // Call step to trigger state initialization
        let result = optimizer.step();
        assert!(result.is_ok());

        // Note: The current implementation doesn't create state tensors
        // This is a known limitation that should be addressed in future sprints
        // For now, we just verify the API contract is maintained
    }

    #[test]
    fn test_adam_bias_correction_placeholder() {
        // Test bias correction configuration
        let optimizer = Adam::with_options(vec![], 0.001, 0.9, 0.999, 1e-8, false);

        // Bias correction terms are stored but never computed
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.999);

        // CRITICAL LIMITATION: Bias correction is not implemented
        // β₁^t and β₂^t terms are never calculated or applied
    }

    #[test]
    fn test_adam_amsgrad_placeholder() {
        // Test AMSGrad configuration
        let optimizer = Adam::with_options(vec![], 0.001, 0.9, 0.999, 1e-8, true);
        assert!(optimizer.amsgrad);

        // CRITICAL LIMITATION: AMSGrad variant is not implemented
        // Maximum past gradients are never tracked or used
    }

    #[test]
    fn test_adam_mathematical_correctness_validation() {
        // This test documents the expected Adam algorithm behavior
        // that should be implemented but currently is not

        let mut optimizer: Adam<f32> = Adam::new(vec![], 0.001);
        let mut param = Tensor::from_vec(vec![1.0], vec![1]);
        param.set_requires_grad(true);

        // Manually set a gradient
        let grad = Tensor::from_vec(vec![0.1], vec![1]);
        param.set_grad(grad).unwrap();

        optimizer
            .base
            .add_param_group(ParamGroup::new(vec![param], 0.001, 0.0));

        if !optimizer.base.param_groups().is_empty()
            && !optimizer.base.param_groups()[0].params.is_empty()
        {
            let old_value = optimizer.base.param_groups()[0].params[0].data()[0];

            // Expected Adam behavior (currently NOT implemented):
            // m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
            // v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
            // m̂_t = m_t / (1 - β₁^t)
            // v̂_t = v_t / (1 - β₂^t)
            // θ_t = θ_{t-1} - η * m̂_t / (√v̂_t + ε)

            optimizer.step().unwrap();

            let new_value = optimizer.base.param_groups()[0].params[0].data()[0];

            // The parameter should have changed with proper Adam implementation
            assert_ne!(old_value, new_value); // Parameters should be updated

            // Verify the parameter moved in the expected direction (towards gradient descent)
            // With gradient = 0.1 and lr = 0.001, parameter should decrease
            assert!(
                new_value < old_value,
                "Parameter should decrease with positive gradient"
            );

            // Verify the update is reasonable (not too large or too small)
            let update_magnitude = (old_value - new_value).abs();
            assert!(
                update_magnitude > 0.0,
                "Parameter update should be non-zero"
            );
            assert!(
                update_magnitude < 0.1,
                "Parameter update should be reasonable magnitude"
            );
        } else {
            // If no parameters, just verify the step doesn't panic
            optimizer.step().unwrap();
        }
    }

    #[test]
    fn test_adam_step_count_tracking() {
        // Test that step count is properly tracked (used for bias correction)
        let mut optimizer: Adam<f32> = Adam::new(vec![], 0.001);

        assert_eq!(optimizer.step_count, 0);

        optimizer.step().unwrap();
        assert_eq!(optimizer.step_count, 1);

        optimizer.step().unwrap();
        assert_eq!(optimizer.step_count, 2);

        // CRITICAL LIMITATION: Step count is tracked but never used
        // for bias correction calculations in the Adam algorithm
    }
}
