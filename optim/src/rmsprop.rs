//! RMSprop (Root Mean Square Propagation) optimizer.
//!
//! This module implements the RMSprop optimizer with optional momentum and centering.

use std::collections::HashMap;
use std::marker::PhantomData;

use coeus_backend::Backend;
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::{DenseStorage, Storage, StorageFromVec};
use coeus_tensor::Tensor;

use crate::optimizer_core::{Optimizer, ParamState};
use crate::Parameter;

/// RMSprop (Root Mean Square Propagation) optimizer.
///
/// RMSprop divides the learning rate by an exponentially decaying average of squared gradients.
/// This helps with the diminishing learning rates problem in AdaGrad while maintaining
/// adaptive learning rates.
///
/// # Algorithm
///
/// ```text
/// square_avg = alpha * square_avg + (1 - alpha) * grad^2
/// param = param - lr * grad / sqrt(square_avg + eps)
/// ```
///
/// With momentum:
/// ```text
/// grad = grad + weight_decay * param  # L2 regularization
/// square_avg = alpha * square_avg + (1 - alpha) * grad^2
/// momentum_buffer = momentum * momentum_buffer + grad
/// param = param - lr * momentum_buffer / sqrt(square_avg + eps)
/// ```
///
/// With centering:
/// ```text
/// grad_avg = alpha * grad_avg + (1 - alpha) * grad
/// square_avg = alpha * square_avg + (1 - alpha) * grad^2
/// param = param - lr * grad / sqrt(square_avg - grad_avg^2 + eps)
/// ```
///
/// # Hyperparameters
///
/// - `lr`: Learning rate (default: 0.01)
/// - `alpha`: Smoothing constant (default: 0.99)
/// - `eps`: Numerical stability constant (default: 1e-8)
/// - `weight_decay`: L2 regularization factor (default: 0.0)
/// - `momentum`: Momentum factor (default: 0.0, no momentum)
/// - `centered`: Whether to center the second moment (default: false)
///
/// # Examples
///
/// ```rust
/// use coeus_optim::rmsprop::RMSprop;
/// use coeus_dtype::float::Float32;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
///
/// // Create RMSprop with default hyperparameters
/// let mut optimizer = RMSprop::<CpuBackend, DenseStorage<Float32>, Float32>::default(0.01);
/// ```
#[derive(Debug)]
pub struct RMSprop<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    /// Parameter states
    param_states: Vec<ParamState<B, S, T>>,
    /// Learning rate
    lr: f64,
    /// Smoothing constant (α)
    alpha: f64,
    /// Numerical stability constant
    eps: f64,
    /// Weight decay (L2 regularization)
    weight_decay: f64,
    /// Momentum factor
    momentum: f64,
    /// Whether to use centered RMSprop
    centered: bool,
    /// Phantom data
    _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> RMSprop<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    /// Create a new RMSprop optimizer.
    ///
    /// # Arguments
    /// * `lr` - Learning rate
    /// * `alpha` - Smoothing constant (0 < alpha < 1)
    /// * `eps` - Numerical stability constant
    /// * `weight_decay` - L2 regularization factor
    /// * `momentum` - Momentum factor
    /// * `centered` - Whether to use centered RMSprop
    pub fn new(lr: f64, alpha: f64, eps: f64, weight_decay: f64, momentum: f64, centered: bool) -> Self {
        Self {
            param_states: Vec::new(),
            lr,
            alpha,
            eps,
            weight_decay,
            momentum,
            centered,
            _phantom: PhantomData,
        }
    }

    /// Create RMSprop with default hyperparameters.
    pub fn default(lr: f64) -> Self {
        Self::new(lr, 0.99, 1e-8, 0.0, 0.0, false)
    }

    /// Create RMSprop with momentum.
    pub fn with_momentum(lr: f64, momentum: f64) -> Self {
        Self::new(lr, 0.99, 1e-8, 0.0, momentum, false)
    }

    /// Create centered RMSprop.
    pub fn centered_rmsprop(lr: f64) -> Self {
        Self::new(lr, 0.99, 1e-8, 0.0, 0.0, true)
    }

    /// Get alpha (smoothing constant) value
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get momentum value
    pub fn momentum(&self) -> f64 {
        self.momentum
    }

    /// Check if centered RMSprop is enabled
    pub fn centered(&self) -> bool {
        self.centered
    }
}

impl<B, S, T> Optimizer<B, S, T> for RMSprop<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    fn name(&self) -> &str {
        "RMSprop"
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        self.param_states.iter().map(|ps| ps.param.clone()).collect()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter<B, S, T>> {
        self.param_states
            .iter()
            .map(|ps| (ps.name.clone(), ps.param.clone()))
            .collect()
    }

    fn add_param(&mut self, param: Parameter<B, S, T>, name: String) {
        let mut param_state = ParamState::new(param, name);

        // Initialize RMSprop state
        let shape = param_state.param.shape().dims();
        let square_avg = Tensor::zeros(shape).unwrap();
        param_state.init_state("square_avg".to_string(), square_avg);

        if self.centered {
            let grad_avg = Tensor::zeros(shape).unwrap();
            param_state.init_state("grad_avg".to_string(), grad_avg);
        }

        if self.momentum > 0.0 {
            let momentum_buffer = Tensor::zeros(shape).unwrap();
            param_state.init_state("momentum_buffer".to_string(), momentum_buffer);
        }

        self.param_states.push(param_state);
    }

    fn remove_param(&mut self, name: &str) {
        self.param_states.retain(|ps| ps.name != name);
    }

    fn has_param(&self, name: &str) -> bool {
        self.param_states.iter().any(|ps| ps.name == name)
    }

    fn lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn weight_decay(&self) -> f64 {
        self.weight_decay
    }

    fn set_weight_decay(&mut self, weight_decay: f64) {
        self.weight_decay = weight_decay;
    }

    fn zero_grad(&mut self) {
        for param_state in &mut self.param_states {
            if let Some(ref mut param) = param_state.param.grad_mut() {
                param.zero_();
            }
        }
    }

    fn step(&mut self) -> Result<(), crate::error::OptimError> {
        let lr = T::from(self.lr).unwrap();
        let alpha = T::from(self.alpha).unwrap();
        let eps = T::from(self.eps).unwrap();
        let weight_decay = T::from(self.weight_decay).unwrap();
        let momentum = T::from(self.momentum).unwrap();
        let one = T::from(1.0).unwrap();
        let one_minus_alpha = one - alpha;

        for param_state in &mut self.param_states {
            let grad = param_state.grad()?;

            // Apply weight decay if specified (L2 regularization)
            let effective_grad = if self.weight_decay > 0.0 {
                grad + &(&param_state.param * weight_decay)
            } else {
                grad.clone()
            };

            // Update square average: square_avg = alpha * square_avg + (1 - alpha) * grad^2
            let square_avg = param_state.get_state_mut("square_avg")
                .ok_or_else(|| crate::error::OptimError::InvalidState {
                    param_name: param_state.name.clone(),
                    state_key: "square_avg".to_string(),
                })?;

            let grad_squared = &effective_grad * &effective_grad;
            *square_avg = &(&square_avg * alpha) + &(&grad_squared * one_minus_alpha);

            let mut denom = square_avg.sqrt() + eps;

            if self.centered {
                // For centered RMSprop: denom = sqrt(square_avg - grad_avg^2 + eps)
                let grad_avg = param_state.get_state_mut("grad_avg")
                    .ok_or_else(|| crate::error::OptimError::InvalidState {
                        param_name: param_state.name.clone(),
                        state_key: "grad_avg".to_string(),
                    })?;

                *grad_avg = &(&grad_avg * alpha) + &(&effective_grad * one_minus_alpha);

                let grad_avg_squared = grad_avg * grad_avg;
                denom = (&square_avg - &grad_avg_squared).sqrt() + eps;
            }

            if self.momentum > 0.0 {
                // Update momentum buffer: momentum_buffer = momentum * momentum_buffer + grad / denom
                let momentum_buffer = param_state.get_state_mut("momentum_buffer")
                    .ok_or_else(|| crate::error::OptimError::InvalidState {
                        param_name: param_state.name.clone(),
                        state_key: "momentum_buffer".to_string(),
                    })?;

                let grad_norm = &effective_grad / &denom;
                *momentum_buffer = &(&momentum_buffer * momentum) + &grad_norm;
                param_state.param -= &(&momentum_buffer * lr);
            } else {
                // Standard RMSprop: param = param - lr * grad / denom
                let update = &(&effective_grad * lr) / &denom;
                param_state.param -= &update;
            }
        }

        Ok(())
    }

    fn state_dict(&self) -> HashMap<String, Tensor<B, S, T>> {
        let mut state = HashMap::new();
        for param_state in &self.param_states {
            state.insert(param_state.name.clone(), param_state.param.clone());
            for (key, tensor) in &param_state.state {
                state.insert(format!("{}.{}", param_state.name, key), tensor.clone());
            }
        }
        state
    }

    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor<B, S, T>>) -> Result<(), crate::error::OptimError> {
        for param_state in &mut self.param_states {
            if let Some(param) = state_dict.get(&param_state.name) {
                if param.shape().dims() != param_state.param.shape().dims() {
                    return Err(crate::error::OptimError::ShapeMismatch {
                        param_name: param_state.name.clone(),
                        expected: param_state.param.shape().dims().to_vec(),
                        actual: param.shape().dims().to_vec(),
                    });
                }
                param_state.param = param.clone();
            }

            // Load RMSprop state
            let square_avg_key = format!("{}.square_avg", param_state.name);
            if let Some(square_avg) = state_dict.get(&square_avg_key) {
                param_state.init_state("square_avg".to_string(), square_avg.clone());
            }

            if self.centered {
                let grad_avg_key = format!("{}.grad_avg", param_state.name);
                if let Some(grad_avg) = state_dict.get(&grad_avg_key) {
                    param_state.init_state("grad_avg".to_string(), grad_avg.clone());
                }
            }

            if self.momentum > 0.0 {
                let momentum_key = format!("{}.momentum_buffer", param_state.name);
                if let Some(momentum_buffer) = state_dict.get(&momentum_key) {
                    param_state.init_state("momentum_buffer".to_string(), momentum_buffer.clone());
                }
            }
        }

        Ok(())
    }
}

impl<B, S, T> Default for RMSprop<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    fn default() -> Self {
        Self::new(0.01, 0.99, 1e-8, 0.0, 0.0, false)
    }
}
