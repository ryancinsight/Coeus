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
        // Collect momentum buffers first to avoid borrowing conflicts
        let mut momentum_buffers = std::collections::HashMap::new();
        let mut param_keys = Vec::new();

        // First pass: collect momentum buffers
        for (group_idx, group) in self.base.param_groups().iter().enumerate() {
            for (param_idx, param) in group.params.iter().enumerate() {
                if param.grad().is_some() {
                    let param_key = format!("momentum_{}_{:p}", group_idx, param as *const _);
                    let momentum_buffer = self
                        .base
                        .state()
                        .get(&param_key)
                        .cloned()
                        .unwrap_or_else(|| Tensor::zeros_like(&param.grad().unwrap()));
                    momentum_buffers.insert(param_key.clone(), momentum_buffer);
                    param_keys.push((group_idx, param_idx, param_key));
                }
            }
        }

        // Second pass: compute updates and collect new state
        let mut new_params = Vec::new();
        let mut new_state = std::collections::HashMap::new();

        for (group_idx, group) in self.base.param_groups().iter().enumerate() {
            let lr = group.lr;
            let weight_decay = group.weight_decay;

            for (param_idx, param) in group.params.iter().enumerate() {
                if let Some(grad) = param.grad() {
                    let param_key = format!("momentum_{}_{:p}", group_idx, param as *const _);
                    let momentum_buffer = momentum_buffers
                        .get(&param_key)
                        .cloned()
                        .unwrap_or_else(|| Tensor::zeros_like(&grad));

                    // Apply weight decay if specified
                    let weight_decay_term = if weight_decay != T::zero() {
                        let wd_tensor = Tensor::scalar(weight_decay);
                        let param_ref = param;
                        (param_ref * &wd_tensor).unwrap()
                    } else {
                        Tensor::zeros_like(&grad)
                    };
                    let effective_grad = (&grad + &weight_decay_term).unwrap();

                    // Update momentum: momentum_buffer = momentum * momentum_buffer + (1-dampening) * effective_grad
                    let dampening_factor = Tensor::scalar(T::one() - self.dampening);
                    let grad_term = (&effective_grad * &dampening_factor).unwrap();

                    let momentum_factor = Tensor::scalar(self.momentum);
                    let momentum_term = (&momentum_buffer * &momentum_factor).unwrap();

                    let updated_momentum = (&momentum_term + &grad_term).unwrap();

                    // Apply Nesterov momentum if enabled
                    let final_momentum = if self.nesterov {
                        let nesterov_term = (&updated_momentum * &momentum_factor).unwrap();
                        let nesterov_grad = (&effective_grad * &dampening_factor).unwrap();
                        (&nesterov_term + &nesterov_grad).unwrap()
                    } else {
                        updated_momentum.clone()
                    };

                    // Compute parameter update: lr * final_momentum
                    let lr_tensor = Tensor::scalar(lr);
                    let update = (&lr_tensor * &final_momentum).unwrap();

                    // Apply update: param = param - update
                    let new_param_data: Vec<T> = param
                        .data()
                        .iter()
                        .zip(update.data().iter())
                        .map(|(p, u)| *p - *u)
                        .collect();

                    // Create new tensor with updated data
                    let mut new_param = Tensor::from_vec(new_param_data, param.shape().to_vec());

                    // Preserve gradient tracking
                    if param.requires_grad() {
                        new_param.set_requires_grad(true);
                    }

                    new_params.push((group_idx, param_idx, new_param));
                    new_state.insert(param_key, updated_momentum);
                }
            }
        }

        // Third pass: apply updates
        for group in self.base.param_groups_mut() {
            let group_idx = 0; // Simplified for now
            for (param_idx, param) in group.params.iter_mut().enumerate() {
                if param.grad().is_some() {
                    if let Some((_, _, new_param)) = new_params
                        .iter()
                        .find(|(g_idx, p_idx, _)| *g_idx == group_idx && *p_idx == param_idx)
                    {
                        *param = new_param.clone();
                    }
                }
            }
        }

        // Update state
        for (key, value) in new_state {
            self.base.state_mut().insert(key, value);
        }
        Ok(())
    }

    /// Perform basic SGD step without momentum
    fn step_basic(&mut self) -> Result<()> {
        for group in self.base.param_groups_mut() {
            let lr = group.lr;
            let weight_decay = group.weight_decay;

            for param in &mut group.params {
                if let Some(grad) = param.grad() {
                    // Compute parameter update: param = param - lr * grad - lr * weight_decay * param

                    // Apply weight decay if specified
                    let weight_decay_term = if weight_decay != T::zero() {
                        let wd_tensor = Tensor::scalar(weight_decay);
                        let param_ref = &*param; // Create immutable reference
                        (param_ref * &wd_tensor).unwrap()
                    } else {
                        Tensor::zeros_like(&grad)
                    };
                    let effective_grad = (&grad + &weight_decay_term).unwrap();

                    // Compute update: lr * effective_grad
                    let lr_tensor = Tensor::scalar(lr);
                    let update = (&lr_tensor * &effective_grad).unwrap();

                    // Apply update: param = param - update
                    let new_param_data: Vec<T> = param
                        .data()
                        .iter()
                        .zip(update.data().iter())
                        .map(|(p, u)| *p - *u)
                        .collect();

                    // Create new tensor with updated data
                    let mut new_param = Tensor::from_vec(new_param_data, param.shape().to_vec());

                    // Preserve gradient tracking
                    if param.requires_grad() {
                        new_param.set_requires_grad(true);
                    }

                    // Replace the parameter
                    *param = new_param;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_creation() {
        let optimizer = Sgd::new(vec![], 0.01);
        assert_eq!(optimizer.base.get_lr(0).unwrap(), 0.01);
        // Note: momentum and weight_decay are not yet implemented
    }

    #[test]
    fn test_optimizer_register_parameter() {
        let mut optimizer: Sgd<f32> = Sgd::new(vec![], 0.01);
        let param = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        optimizer
            .base
            .add_param_group(ParamGroup::new(vec![param], 0.01, 0.0));

        // The optimizer starts with an empty param group, so adding one makes it 2
        // But we verify that the API works correctly
        assert!(!optimizer.base.param_groups().is_empty());
    }

    #[test]
    fn test_sgd_step_placeholder_functionality() {
        // Test that the step method can be called without panicking
        // This validates the API contract while documenting current limitations
        let mut optimizer = Sgd::new(vec![], 0.01);
        let param = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        optimizer
            .base
            .add_param_group(ParamGroup::new(vec![param], 0.01, 0.0));

        // This should succeed (API contract maintained)
        let result = optimizer.step();
        assert!(result.is_ok());

        // CRITICAL LIMITATION: Actual parameter updates are not implemented yet
        // The step() method currently does nothing - this violates the API contract
        // and represents a critical production readiness gap that must be addressed
    }

    #[test]
    fn test_sgd_zero_grad_placeholder_functionality() {
        // Test that zero_grad method can be called without panicking
        let mut optimizer = Sgd::new(vec![], 0.01);

        // This should succeed (API contract maintained)
        optimizer.zero_grad();

        // CRITICAL LIMITATION: Actual gradient zeroing is not implemented yet
        // The zero_grad() method currently does nothing - this violates the API contract
    }

    #[test]
    fn test_sgd_with_momentum_placeholder() {
        // Test momentum configuration (API level only)
        let optimizer = Sgd::with_momentum(vec![], 0.01, 0.9);
        assert_eq!(optimizer.momentum, 0.9);

        // CRITICAL LIMITATION: Momentum computation is not implemented yet
        // The momentum factor is stored but never used in parameter updates
    }

    #[test]
    fn test_sgd_nesterov_placeholder() {
        // Test Nesterov momentum configuration (API level only)
        let mut optimizer = Sgd::new(vec![], 0.01);
        optimizer.nesterov = true;

        // CRITICAL LIMITATION: Nesterov momentum computation is not implemented yet
        // The nesterov flag is stored but never used in parameter updates
    }

    #[test]
    fn test_sgd_weight_decay_placeholder() {
        // Test weight decay configuration (API level only)
        let mut optimizer = Sgd::new(vec![], 0.01);
        let param_group = ParamGroup::new(vec![], 0.01, 0.0001);
        optimizer.base.add_param_group(param_group);

        // CRITICAL LIMITATION: Weight decay is not implemented yet
        // Weight decay values are stored but never used in parameter updates
    }

    #[test]
    fn test_sgd_mathematical_correctness_validation() {
        // This test validates the actual mathematical behavior of SGD

        let mut param = Tensor::from_vec(vec![2.0], vec![1]);
        param.set_requires_grad(true);

        // Manually set a gradient
        let grad = Tensor::from_vec(vec![1.0], vec![1]);
        param.set_grad(grad).unwrap();

        let mut optimizer: Sgd<f32> = Sgd::new(vec![param], 0.01);

        if !optimizer.base.param_groups().is_empty()
            && !optimizer.base.param_groups()[0].params.is_empty()
        {
            let _old_value = optimizer.base.param_groups()[0].params[0].data()[0];

            // This call should update the parameter: param = param - lr * grad
            optimizer.step().unwrap();

            // The parameter should have changed: 2.0 - 0.01 * 1.0 = 1.99
            let new_value = optimizer.base.param_groups()[0].params[0].data()[0];
            assert!(
                (new_value - 1.99).abs() < 1e-6,
                "Expected parameter to be updated to ~1.99, got {}",
                new_value
            );

            // Verify the gradient is preserved
            assert!(optimizer.base.param_groups()[0].params[0].requires_grad());
        } else {
            panic!("Parameter group should contain parameters");
        }
    }

    #[test]
    fn test_sgd_with_weight_decay() {
        // Test SGD with weight decay (L2 regularization)

        let mut param = Tensor::from_vec(vec![2.0], vec![1]);
        param.set_requires_grad(true);

        // Manually set a gradient
        let grad = Tensor::from_vec(vec![1.0], vec![1]);
        param.set_grad(grad).unwrap();

        let mut optimizer: Sgd<f32> = Sgd::new(vec![param], 0.01);

        // Set weight decay on the parameter group
        if let Some(group) = optimizer.base.param_groups_mut().first_mut() {
            group.weight_decay = 0.01;
        }

        if !optimizer.base.param_groups().is_empty()
            && !optimizer.base.param_groups()[0].params.is_empty()
        {
            let _old_value = optimizer.base.param_groups()[0].params[0].data()[0];

            optimizer.step().unwrap();

            // With weight decay: param = param - lr * (grad + weight_decay * param)
            // = 2.0 - 0.01 * (1.0 + 0.01 * 2.0) = 2.0 - 0.01 * 1.02 = 1.9898
            let new_value = optimizer.base.param_groups()[0].params[0].data()[0];
            assert!(
                (new_value - 1.9898).abs() < 1e-4,
                "Expected parameter with weight decay to be ~1.9898, got {}",
                new_value
            );
        } else {
            panic!("Parameter group should contain parameters");
        }
    }

    #[test]
    fn test_sgd_with_momentum() {
        // Test SGD with momentum

        let mut param = Tensor::from_vec(vec![2.0], vec![1]);
        param.set_requires_grad(true);

        // Manually set a gradient
        let grad = Tensor::from_vec(vec![1.0], vec![1]);
        param.set_grad(grad.clone()).unwrap();

        let mut optimizer: Sgd<f32> = Sgd::with_momentum(vec![param], 0.01, 0.9);

        if !optimizer.base.param_groups().is_empty()
            && !optimizer.base.param_groups()[0].params.is_empty()
        {
            let _old_value = optimizer.base.param_groups()[0].params[0].data()[0];

            // First step: momentum buffer is zero, so update = lr * grad
            optimizer.step().unwrap();

            let first_update_value = optimizer.base.param_groups()[0].params[0].data()[0];
            assert!(
                (first_update_value - 1.99).abs() < 1e-6,
                "First step should update to ~1.99, got {}",
                first_update_value
            );

            // Set the same gradient for second step
            optimizer.base.param_groups_mut()[0].params[0]
                .set_grad(grad)
                .unwrap();

            // Second step: momentum = 0.9 * 1.0 + (1-0) * 1.0 = 1.9, update = lr * 1.9 = 0.019
            optimizer.step().unwrap();

            let second_update_value = optimizer.base.param_groups()[0].params[0].data()[0];
            // Correct expectation: 1.99 - 0.019 = 1.971
            assert!(
                (second_update_value - 1.971).abs() < 1e-6,
                "Second step should update to ~1.971, got {}",
                second_update_value
            );
        } else {
            panic!("Parameter group should contain parameters");
        }
    }
}
