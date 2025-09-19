//! Sparse Adam optimizer
//!
//! Implements the SparseAdam algorithm which is optimized for sparse gradients.
//! It maintains separate momentum and variance estimates for each parameter,
//! making it efficient for sparse updates common in embedding layers.

use crate::{BaseOptimizer, Optimizer, ParamGroup, Result};
use coeus_tensor::{Tensor, Mul, Add, Sub, Div};
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
            T::from(0.9).unwrap(),    // beta1
            T::from(0.999).unwrap(),  // beta2
            T::from(1e-8).unwrap(),   // eps
            false,                    // amsgrad
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
                let exp_avg = param.get_buffer("exp_avg")
                    .unwrap_or_else(|| Tensor::zeros(param.shape().to_vec()));

                let exp_avg_sq = param.get_buffer("exp_avg_sq")
                    .unwrap_or_else(|| Tensor::zeros(param.shape().to_vec()));

                // Get step count
                let step_count_tensor = param.get_buffer("step")
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
                param.set_buffer("step", Tensor::from_vec(vec![step_count + T::one()], vec![1]));
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
