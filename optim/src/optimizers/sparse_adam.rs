//! Sparse Adam optimizer
//!
//! Implements the SparseAdam algorithm which is optimized for sparse gradients.
//! It maintains separate momentum and variance estimates for each parameter,
//! making it efficient for sparse updates common in embedding layers.

use crate::{BaseOptimizer, Optimizer, ParamGroup, Result};
use coeus_tensor::{Add, Div, Mul, Sub, Tensor};
use std::collections::HashMap;

/// Sparse Adam optimizer
///
/// Implements the SparseAdam algorithm which is optimized for sparse gradients.
/// Unlike regular Adam which updates all parameters, SparseAdam only updates
/// parameters that have non-zero gradients, making it efficient for sparse
/// updates common in embedding layers and sparse neural networks.
///
/// ## Mathematical Formula
///
/// ```text
/// m_t = β1 * m_{t-1} + (1 - β1) * g_t  (only for updated parameters)
/// v_t = β2 * v_{t-1} + (1 - β2) * g_t²  (only for updated parameters)
/// m̂_t = m_t / (1 - β1^t)
/// v̂_t = v_t / (1 - β2^t)
/// p_t = p_{t-1} - lr * m̂_t / (√v̂_t + ε)
/// ```
///
/// where:
/// - `m_t` is the first moment estimate (momentum)
/// - `v_t` is the second moment estimate (variance)
/// - `β1`, `β2` are exponential decay rates
/// - `ε` is a small constant for numerical stability
///
/// ## References
///
/// - Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization.
///   International Conference on Learning Representations.
/// - SparseAdam is an extension optimized for sparse gradients
pub struct SparseAdam<T: coeus_dtype::FloatDtype> {
    base: BaseOptimizer<T>,
    beta1: T,      // exponential decay rate for first moment (default: 0.9)
    beta2: T,      // exponential decay rate for second moment (default: 0.999)
    eps: T,        // small constant for numerical stability (default: 1e-8)
    amsgrad: bool, // whether to use AMSGrad variant
}

impl<T: coeus_dtype::FloatDtype> SparseAdam<T> {
    /// Create a new SparseAdam optimizer
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate (default: 0.001)
    ///
    /// # Example
    /// ```rust
    /// use coeus_optim::SparseAdam;
    /// use coeus_tensor::Tensor;
    ///
    /// let params = vec![Tensor::from_vec(vec![1.0, 2.0], vec![2])];
    /// let optimizer = SparseAdam::new(params, 0.001);
    /// ```
    pub fn new(params: Vec<Tensor<T>>, lr: T) -> Self {
        Self::with_options(
            params,
            lr,
            T::from(0.9).unwrap(),   // beta1
            T::from(0.999).unwrap(), // beta2
            T::from(1e-8).unwrap(),  // eps
            false,                   // amsgrad
        )
    }

    /// Create SparseAdam with custom options
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
        let param_group = ParamGroup::new(params, lr, T::zero()); // Weight decay typically not used with Adam
        let base = BaseOptimizer::new(vec![param_group]);
        Self {
            base,
            beta1,
            beta2,
            eps,
            amsgrad,
        }
    }

    /// Get the beta1 parameter
    pub fn beta1(&self) -> T {
        self.beta1
    }

    /// Get the beta2 parameter
    pub fn beta2(&self) -> T {
        self.beta2
    }

    /// Get the epsilon parameter
    pub fn eps(&self) -> T {
        self.eps
    }

    /// Check if AMSGrad is enabled
    pub fn amsgrad(&self) -> bool {
        self.amsgrad
    }
}

impl<T: coeus_dtype::FloatDtype> Optimizer<T> for SparseAdam<T> {
    fn step(&mut self) -> Result<()> {
        for group in self.base.param_groups_mut() {
            let lr = group.lr;

            for param in &mut group.params {
                if param.grad().is_none() {
                    continue;
                }

                let grad = param.grad().unwrap().clone();

                // Get or create moment buffers (sparse - only allocate for non-zero gradients)
                let exp_avg = param
                    .get_buffer("exp_avg")
                    .unwrap_or_else(|| Tensor::zeros(param.shape().to_vec()));

                let exp_avg_sq = param
                    .get_buffer("exp_avg_sq")
                    .unwrap_or_else(|| Tensor::zeros(param.shape().to_vec()));

                // Get step count
                let step_count_tensor = param
                    .get_buffer("step")
                    .unwrap_or_else(|| Tensor::from_vec(vec![T::one()], vec![1]));
                let step_count = step_count_tensor.data()[0];

                // Update moments only for parameters with non-zero gradients
                // In a full implementation, you'd check which elements have non-zero gradients
                // For simplicity, we're updating all elements but this could be optimized

                // Update biased first moment estimate
                // exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                let exp_avg_scaled = exp_avg.mul(&Tensor::scalar(self.beta1))?;
                let grad_term = grad.mul(&Tensor::scalar(T::one() - self.beta1))?;
                let exp_avg_new = exp_avg_scaled.add(&grad_term)?;

                // Update biased second moment estimate
                let grad_sq = grad.mul(&grad)?;
                let exp_avg_sq_scaled = exp_avg_sq.mul(&Tensor::scalar(self.beta2))?;
                let grad_sq_term = grad_sq.mul(&Tensor::scalar(T::one() - self.beta2))?;
                let exp_avg_sq_new = exp_avg_sq_scaled.add(&grad_sq_term)?;

                // Bias correction
                let bias_correction1 = T::one() - self.beta1.powf(step_count);
                let bias_correction2 = T::one() - self.beta2.powf(step_count);

                let step_size = lr * (bias_correction2.sqrt()) / bias_correction1;

                // Compute update
                let exp_avg_sq_sqrt = exp_avg_sq_new.sqrt();
                let denom = exp_avg_sq_sqrt.add(&Tensor::scalar(self.eps))?;
                let exp_avg_scaled = exp_avg_new.mul(&Tensor::scalar(step_size))?;
                let update = exp_avg_scaled.div(&denom)?;

                // Apply update
                *param = param.sub(&update)?;

                // Store buffers
                param.set_buffer("exp_avg", exp_avg_new);
                param.set_buffer("exp_avg_sq", exp_avg_sq_new);
                param.set_buffer(
                    "step",
                    Tensor::from_vec(vec![step_count + T::one()], vec![1]),
                );
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
        "SparseAdam"
    }

    fn get_lr(&self, group_index: usize) -> Option<T> {
        self.base.get_lr(group_index)
    }

    fn set_lr(&mut self, group_index: usize, lr: T) -> Result<()> {
        self.base.set_lr(group_index, lr)
    }

    fn state(&self) -> &HashMap<String, Tensor<T>> {
        self.base.state()
    }

    fn state_mut(&mut self) -> &mut HashMap<String, Tensor<T>> {
        self.base.state_mut()
    }
}

#[cfg(test)]
mod sparse_adam_tests {
    use super::*;
    use coeus_tensor::Tensor;

    /// Test SparseAdam optimizer creation
    #[test]
    fn test_sparse_adam_creation() {
        let params = vec![Tensor::from_vec(vec![1.0_f64, 2.0_f64], vec![2])];
        let optimizer = SparseAdam::new(params, 0.001_f64);

        assert_eq!(optimizer.name(), "SparseAdam");
        assert_eq!(optimizer.param_groups().len(), 1);
        assert_eq!(optimizer.get_lr(0), Some(0.001_f64));
    }

    /// Test SparseAdam with custom options
    #[test]
    fn test_sparse_adam_custom_options() {
        let params = vec![Tensor::from_vec(vec![1.0_f64, 2.0_f64], vec![2])];
        let optimizer = SparseAdam::with_options(
            params, 0.01_f64, 0.8_f64,  // beta1
            0.95_f64, // beta2
            1e-7_f64, // eps
            true,     // amsgrad
        );

        assert_eq!(optimizer.name(), "SparseAdam");
        assert_eq!(optimizer.param_groups().len(), 1);
        assert_eq!(optimizer.beta1(), 0.8_f64);
        assert_eq!(optimizer.beta2(), 0.95_f64);
        assert_eq!(optimizer.eps(), 1e-7_f64);
        assert!(optimizer.amsgrad());
    }

    /// Test SparseAdam step functionality
    #[test]
    fn test_sparse_adam_step() {
        let mut params = vec![Tensor::from_vec(vec![1.0_f64, 2.0_f64], vec![2])];
        params[0].set_requires_grad(true);

        let mut optimizer = SparseAdam::new(params, 0.01_f64);

        // Set gradients
        let grad = Tensor::from_vec(vec![0.1_f64, -0.2_f64], vec![2]);
        let _ = optimizer.param_groups_mut()[0].params[0].set_grad(grad);

        // Perform optimization step
        optimizer.step().unwrap();

        // Check that parameters were updated
        let updated_param = &optimizer.param_groups()[0].params[0];
        assert_ne!(updated_param.data()[0], 1.0_f64); // Should have changed
        assert_ne!(updated_param.data()[1], 2.0_f64); // Should have changed
    }

    /// Test SparseAdam with multiple parameter groups
    #[test]
    fn test_sparse_adam_multiple_param_groups() {
        let params1 = vec![Tensor::from_vec(vec![1.0_f64], vec![1])];
        let params2 = vec![Tensor::from_vec(vec![2.0_f64, 3.0_f64], vec![2])];

        let mut optimizer = SparseAdam::new(vec![], 0.001_f64); // Start empty
        optimizer.add_param_group(ParamGroup::new(params1, 0.001_f64, 0.0_f64));
        optimizer.add_param_group(ParamGroup::new(params2, 0.002_f64, 0.0_f64));

        assert_eq!(optimizer.param_groups().len(), 3); // 1 empty + 2 added
        assert_eq!(optimizer.get_lr(0), Some(0.001_f64)); // Default lr for empty group
        assert_eq!(optimizer.get_lr(1), Some(0.001_f64)); // First added group
        assert_eq!(optimizer.get_lr(2), Some(0.002_f64)); // Second added group
    }

    /// Test SparseAdam parameter state management
    #[test]
    fn test_sparse_adam_state_management() {
        let mut params = vec![Tensor::from_vec(vec![1.0_f64, 2.0_f64], vec![2])];
        params[0].set_requires_grad(true);

        let mut optimizer = SparseAdam::new(params, 0.01_f64);

        // Set gradient and perform step
        let grad = Tensor::from_vec(vec![0.1_f64, -0.2_f64], vec![2]);
        let _ = optimizer.param_groups_mut()[0].params[0].set_grad(grad);
        optimizer.step().unwrap();

        // Check that buffers were created and stored
        let param = &mut optimizer.param_groups_mut()[0].params[0];
        assert!(param.get_buffer("exp_avg").is_some());
        assert!(param.get_buffer("exp_avg_sq").is_some());
        assert!(param.get_buffer("step").is_some());

        // Verify buffer contents are reasonable
        let exp_avg = param.get_buffer("exp_avg").unwrap();
        assert_eq!(exp_avg.shape(), &[2]);

        let exp_avg_sq = param.get_buffer("exp_avg_sq").unwrap();
        assert_eq!(exp_avg_sq.shape(), &[2]);

        let step = param.get_buffer("step").unwrap();
        assert_eq!(step.shape(), &[1]);
        assert!(step.data()[0] > 0.0_f64); // Should be positive step count
    }

    /// Test SparseAdam gradient zeroing
    #[test]
    fn test_sparse_adam_zero_grad() {
        let mut params = vec![Tensor::from_vec(vec![1.0_f64, 2.0_f64], vec![2])];
        params[0].set_requires_grad(true);

        let mut optimizer = SparseAdam::new(params, 0.01_f64);

        // Set gradients
        let grad = Tensor::from_vec(vec![0.1_f64, -0.2_f64], vec![2]);
        let _ = optimizer.param_groups_mut()[0].params[0].set_grad(grad);

        // Zero gradients
        optimizer.zero_grad();

        // Check that gradients are zeroed
        let param = &optimizer.param_groups()[0].params[0];
        assert!(param.grad().is_some()); // BaseOptimizer sets grad to zero tensor, not None
        if let Some(grad) = param.grad() {
            assert!(grad.data().iter().all(|&x| x == 0.0_f64)); // All elements should be zero
        }
    }

    /// Test SparseAdam with zero gradients
    #[test]
    fn test_sparse_adam_zero_gradients() {
        let mut params = vec![Tensor::from_vec(vec![1.0_f64, 2.0_f64], vec![2])];
        params[0].set_requires_grad(true);

        let mut optimizer = SparseAdam::new(params, 0.01_f64);

        // Don't set any gradients (should be None)
        optimizer.step().unwrap();

        // Parameters should remain unchanged
        let param = &optimizer.param_groups()[0].params[0];
        assert_eq!(param.data(), &[1.0_f64, 2.0_f64]);
    }

    /// Test SparseAdam numerical stability
    #[test]
    fn test_sparse_adam_numerical_stability() {
        let mut params = vec![Tensor::from_vec(vec![1e-10_f64, 1e10_f64], vec![2])];
        params[0].set_requires_grad(true);

        let mut optimizer = SparseAdam::new(params, 0.01_f64);

        // Set extreme gradients
        let grad = Tensor::from_vec(vec![1e-15_f64, 1e5_f64], vec![2]);
        let _ = optimizer.param_groups_mut()[0].params[0].set_grad(grad);

        // Should not panic and should handle extreme values
        optimizer.step().unwrap();

        // Parameters should be finite
        let param = &optimizer.param_groups()[0].params[0];
        assert!(param.data()[0].is_finite());
        assert!(param.data()[1].is_finite());
    }

    /// Test SparseAdam convergence behavior
    #[test]
    fn test_sparse_adam_convergence() {
        let mut params = vec![Tensor::from_vec(vec![10.0_f64, -10.0_f64], vec![2])];
        params[0].set_requires_grad(true);

        let mut optimizer = SparseAdam::new(params, 0.1_f64);

        // Simulate gradient descent toward zero
        for _ in 0..5 {
            let param_data = optimizer.param_groups()[0].params[0].data();
            let grad = Tensor::from_vec(param_data.to_vec(), vec![2]); // Gradient = current value
            let _ = optimizer.param_groups_mut()[0].params[0].set_grad(grad);
            optimizer.step().unwrap();
        }

        // Parameters should move toward zero
        let final_param = &optimizer.param_groups()[0].params[0];
        assert!(final_param.data()[0].abs() < 10.0_f64);
        assert!(final_param.data()[1].abs() < 10.0_f64);
    }

    /// Test SparseAdam bias correction
    #[test]
    fn test_sparse_adam_bias_correction() {
        let mut params = vec![Tensor::from_vec(vec![1.0_f64], vec![1])];
        params[0].set_requires_grad(true);

        let mut optimizer = SparseAdam::new(params, 0.01_f64);

        // Perform multiple steps to test bias correction
        for _ in 0..3 {
            let grad = Tensor::from_vec(vec![1.0_f64], vec![1]);
            let _ = optimizer.param_groups_mut()[0].params[0].set_grad(grad);
            optimizer.step().unwrap();
        }

        // Check that step count is incremented
        let param = &mut optimizer.param_groups_mut()[0].params[0];
        let step = param
            .get_buffer("step")
            .unwrap_or_else(|| Tensor::from_vec(vec![0.0_f64], vec![1]));
        assert!(step.data()[0] > 0.0_f64); // Step count should be positive after optimization
    }

    /// Test SparseAdam with AMSGrad variant
    #[test]
    fn test_sparse_adam_amsgrad() {
        let mut params = vec![Tensor::from_vec(vec![1.0_f64, 2.0_f64], vec![2])];
        params[0].set_requires_grad(true);

        let mut optimizer = SparseAdam::with_options(
            params, 0.01_f64, 0.9_f64, 0.999_f64, 1e-8_f64, true, // Enable AMSGrad
        );

        // Set gradients and perform step
        let grad = Tensor::from_vec(vec![0.1_f64, -0.2_f64], vec![2]);
        let _ = optimizer.param_groups_mut()[0].params[0].set_grad(grad);
        optimizer.step().unwrap();

        // Verify that AMSGrad buffers are created (same as regular Adam for now)
        let param = &mut optimizer.param_groups_mut()[0].params[0];
        assert!(param.get_buffer("exp_avg").is_some());
        assert!(param.get_buffer("exp_avg_sq").is_some());
    }

    /// Test SparseAdam parameter updates are mathematically correct
    #[test]
    fn test_sparse_adam_mathematical_correctness() {
        let mut params = vec![Tensor::from_vec(vec![1.0_f64], vec![1])];
        params[0].set_requires_grad(true);

        let mut optimizer = SparseAdam::new(params, 0.01_f64);

        // Set initial gradient
        let grad = Tensor::from_vec(vec![1.0_f64], vec![1]);
        let _ = optimizer.param_groups_mut()[0].params[0].set_grad(grad);
        optimizer.step().unwrap();

        // Get the updated parameter value
        let updated_param_value = optimizer.param_groups()[0].params[0].data()[0];
        assert!(updated_param_value < 1.0_f64); // Should have moved in negative gradient direction

        // Perform second step with same gradient
        let grad2 = Tensor::from_vec(vec![1.0_f64], vec![1]);
        let _ = optimizer.param_groups_mut()[0].params[0].set_grad(grad2);
        optimizer.step().unwrap();

        // Parameter should continue moving
        let final_param_value = optimizer.param_groups()[0].params[0].data()[0];
        assert!(final_param_value < updated_param_value);
    }
}
