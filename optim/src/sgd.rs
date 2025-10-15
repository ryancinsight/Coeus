//! Stochastic Gradient Descent (SGD) optimizer.
//!
//! This module implements the SGD optimizer with optional momentum, weight decay,
//! and Nesterov acceleration.

use std::collections::HashMap;
use std::marker::PhantomData;

use coeus_backend::Backend;
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::{DenseStorage, Storage, StorageFromVec};
use coeus_tensor::Tensor;

use crate::optimizer_core::{Optimizer, OptimizerConfig, ParamState};
use crate::Parameter;

/// Stochastic Gradient Descent (SGD) optimizer with momentum.
///
/// SGD updates parameters using the gradient of the loss function:
/// ```text
/// param = param - lr * grad
/// ```
///
/// With momentum:
/// ```text
/// velocity = momentum * velocity + (1 - dampening) * grad
/// param = param - lr * velocity
/// ```
///
/// With Nesterov momentum:
/// ```text
/// velocity = momentum * velocity + grad
/// param = param - lr * (momentum * velocity + (1 + momentum) * grad)
/// ```
///
/// # Hyperparameters
///
/// - `lr`: Learning rate (default: 0.01)
/// - `momentum`: Momentum factor (default: 0.0, no momentum)
/// - `weight_decay`: L2 regularization factor (default: 0.0)
/// - `dampening`: Dampening factor for momentum (default: 0.0)
/// - `nesterov`: Enable Nesterov momentum (default: false)
///
/// # Examples
///
/// ```rust
/// use coeus_optim::sgd::SGD;
/// use coeus_dtype::float::Float32;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
///
/// // Create SGD optimizer
/// let mut optimizer = SGD::<CpuBackend, DenseStorage<Float32>, Float32>::new(0.01, 0.9, 0.0, 0.0, false);
/// ```
#[derive(Debug)]
pub struct SGD<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    /// Parameter states
    param_states: Vec<ParamState<B, S, T>>,
    /// Learning rate
    lr: f64,
    /// Momentum factor
    momentum: f64,
    /// Dampening factor
    dampening: f64,
    /// Weight decay (L2 regularization)
    weight_decay: f64,
    /// Whether to use Nesterov momentum
    nesterov: bool,
    /// Phantom data
    _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> SGD<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    /// Create a new SGD optimizer.
    ///
    /// # Arguments
    /// * `lr` - Learning rate
    /// * `momentum` - Momentum factor (0.0 = no momentum)
    /// * `weight_decay` - L2 regularization factor
    /// * `dampening` - Dampening factor for momentum
    /// * `nesterov` - Whether to use Nesterov momentum
    pub fn new(lr: f64, momentum: f64, weight_decay: f64, dampening: f64, nesterov: bool) -> Self {
        Self {
            param_states: Vec::new(),
            lr,
            momentum,
            dampening,
            weight_decay,
            nesterov,
            _phantom: PhantomData,
        }
    }

    /// Create SGD with momentum.
    pub fn with_momentum(lr: f64, momentum: f64) -> Self {
        Self::new(lr, momentum, 0.0, 0.0, false)
    }

    /// Create SGD with momentum and weight decay.
    pub fn with_momentum_weight_decay(lr: f64, momentum: f64, weight_decay: f64) -> Self {
        Self::new(lr, momentum, weight_decay, 0.0, false)
    }

    /// Create SGD with Nesterov momentum.
    pub fn nesterov_momentum(lr: f64, momentum: f64) -> Self {
        Self::new(lr, momentum, 0.0, 0.0, true)
    }

    /// Get momentum value
    pub fn momentum(&self) -> f64 {
        self.momentum
    }

    /// Get dampening value
    pub fn dampening(&self) -> f64 {
        self.dampening
    }

    /// Check if Nesterov momentum is enabled
    pub fn nesterov(&self) -> bool {
        self.nesterov
    }
}

impl<B, S, T> Optimizer<B, S, T> for SGD<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt + core::ops::Add<Output = T> + core::ops::Sub<Output = T> + core::ops::Mul<Output = T> + core::ops::Div<Output = T>,
{
    fn name(&self) -> &str {
        "SGD"
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
        if self.momentum > 0.0 {
            // Initialize momentum buffer
            let velocity = Tensor::zeros(param.shape().dims()).unwrap();
            param_state.init_state("momentum_buffer".to_string(), velocity);
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
            if let Some(ref mut grad) = param_state.grad {
                grad.zero_();
            }
        }
    }

    fn step(&mut self) -> Result<(), crate::error::OptimError> {
        let lr = T::from(self.lr).unwrap();
        let weight_decay = T::from(self.weight_decay).unwrap();
        let momentum = T::from(self.momentum).unwrap();
        let dampening = T::from(self.dampening).unwrap();
        let one = T::from(1.0).unwrap();

        for param_state in &mut self.param_states {
            // Get parameter and gradient
            let param_data = param_state.param.as_slice();
            let grad = param_state.grad()?;

            // Apply weight decay if specified
            let effective_grad = if self.weight_decay > 0.0 {
                use crate::ops::arithmetic::{scalar_add, scalar_mul};
                let weight_decay_term = scalar_mul(&param_state.param, weight_decay)?;
                scalar_add(&grad, &weight_decay_term)?
            } else {
                grad.clone()
            };

            if self.momentum > 0.0 {
                // Momentum-based update
                let velocity_key = "momentum_buffer";
                let velocity = param_state.get_state_mut(velocity_key)
                    .ok_or_else(|| crate::error::OptimError::InvalidState {
                        message: format!("Missing momentum buffer for parameter '{}'", param_state.name),
                    })?;

                if self.nesterov {
                    // Nesterov momentum: v = momentum * v + g, p = p - lr * (momentum * v + g)
                    use crate::ops::arithmetic::{scalar_mul, scalar_add};
                    let momentum_velocity = scalar_mul(&velocity, momentum)?;
                    let new_velocity = scalar_add(&momentum_velocity, &scalar_mul(&effective_grad, one - dampening)?)?;
                    let nesterov_grad = scalar_add(&momentum_velocity, &scalar_mul(&effective_grad, one + momentum)?)?;
                    *velocity = new_velocity;
                    param_state.param -= &scalar_mul(&nesterov_grad, lr)?;
                } else {
                    // Standard momentum: v = momentum * v + (1-dampening) * g, p = p - lr * v
                    use crate::ops::arithmetic::{scalar_mul, scalar_add};
                    *velocity = scalar_add(&scalar_mul(&velocity, momentum)?, &scalar_mul(&effective_grad, one - dampening)?)?;
                    param_state.param -= &scalar_mul(&velocity, lr)?;
                }
            } else {
                // Standard SGD: p = p - lr * g
                use crate::ops::arithmetic::scalar_mul;
                param_state.param -= &scalar_mul(&effective_grad, lr)?;
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
                        expected: param_state.param.shape().dims().to_vec(),
                        actual: param.shape().dims().to_vec(),
                    });
                }
                param_state.param = param.clone();
            }

            // Load momentum buffer if it exists
            if self.momentum > 0.0 {
                let velocity_key = format!("{}.momentum_buffer", param_state.name);
                if let Some(velocity) = state_dict.get(&velocity_key) {
                    param_state.init_state("momentum_buffer".to_string(), velocity.clone());
                }
            }
        }
        Ok(())
    }
}

impl<B, S, T> Default for SGD<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    fn default() -> Self {
        Self::new(0.01, 0.0, 0.0, 0.0, false)
    }
}
