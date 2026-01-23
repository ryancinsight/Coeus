//! Stochastic Gradient Descent (SGD) optimizer.
//!
//! This module implements the SGD optimizer with optional momentum, weight decay,
//! and Nesterov acceleration.

use std::collections::HashMap;
use std::marker::PhantomData;

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
use tensor::{
    ops::arithmetic::{add, mul},
    Tensor,
};

use crate::optimizer::{BaseOptimizer, ParamGroup};
use crate::optimizer_core::{Optimizer, ParamState};

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
/// ```rust,no_run
/// use backend::CpuBackend;
/// use dtype::float::Float32;
/// use optim::sgd::SGD;
///
/// let _optimizer: SGD<CpuBackend<Float32>, Float32> = SGD::new(0.01, 0.9, 0.0, 0.0, false);
/// ```
#[derive(Debug)]
pub struct SGD<B, T>
where
    B: Backend<Data = T> + Clone,
    T: DataType + FloatExt + num_traits::FromPrimitive,
{
    /// Parameter states
    param_states: Vec<ParamState<B, DenseStorage<T>, T>>,
    param_groups: Vec<ParamGroup<B, DenseStorage<T>, T>>,
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
    _phantom: PhantomData<(B, T)>,
}

impl<B, T> SGD<B, T>
where
    B: Backend<Data = T> + Clone,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive,
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
            param_groups: Vec::new(),
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

impl<B, T> BaseOptimizer<B, DenseStorage<T>, T> for SGD<B, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync,
    T: DataType
        + FloatExt
        + num_traits::FromPrimitive
        + core::ops::Add<Output = T>
        + core::ops::Sub<Output = T>
        + core::ops::Mul<Output = T>
        + core::ops::Div<Output = T>,
{
    fn step(&mut self) -> Result<usize, crate::OptimError> {
        <Self as Optimizer<B, DenseStorage<T>, T>>::step(self)
    }

    fn step_cpu(&mut self) -> Result<usize, crate::OptimError> {
        <Self as Optimizer<B, DenseStorage<T>, T>>::step(self)
    }

    fn zero_grad(&mut self) {
        <Self as Optimizer<B, DenseStorage<T>, T>>::zero_grad(self);
    }

    fn add_param_group(&mut self, params: Vec<Tensor<B, DenseStorage<T>, T>>) {
        for tensor in params.clone().into_iter() {
            let mut param_state =
                ParamState::new(tensor.clone(), format!("param_{}", self.param_states.len()));

            if self.momentum > 0.0 {
                let velocity = Tensor::zeros(tensor.shape().dims()).unwrap();
                param_state.init_state("momentum_buffer".to_string(), velocity);
            }

            self.param_states.push(param_state);
        }

        self.param_groups.push(ParamGroup::new(
            params,
            self.lr as f32,
            self.weight_decay as f32,
        ));
    }

    fn get_lr(&self) -> f32 {
        self.lr as f32
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr as f64;
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    fn state_dict(&self) -> HashMap<String, Tensor<B, DenseStorage<T>, T>> {
        <Self as Optimizer<B, DenseStorage<T>, T>>::state_dict(self)
    }

    fn load_state_dict(
        &mut self,
        state_dict: HashMap<String, Tensor<B, DenseStorage<T>, T>>,
    ) -> Result<(), crate::OptimError> {
        <Self as Optimizer<B, DenseStorage<T>, T>>::load_state_dict(self, state_dict)
    }

    fn param_groups(&self) -> &[ParamGroup<B, DenseStorage<T>, T>] {
        &self.param_groups
    }

    fn param_groups_mut(&mut self) -> &mut [ParamGroup<B, DenseStorage<T>, T>] {
        &mut self.param_groups
    }
}

impl<B, T> Optimizer<B, DenseStorage<T>, T> for SGD<B, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync,
    T: DataType
        + FloatExt
        + num_traits::FromPrimitive
        + core::ops::Add<Output = T>
        + core::ops::Sub<Output = T>
        + core::ops::Mul<Output = T>
        + core::ops::Div<Output = T>,
{
    fn name(&self) -> &str {
        "SGD"
    }

    fn parameters(&self) -> Vec<Tensor<B, DenseStorage<T>, T>> {
        self.param_states
            .iter()
            .map(|ps| ps.param.clone())
            .collect()
    }

    fn named_parameters(&self) -> HashMap<String, Tensor<B, DenseStorage<T>, T>> {
        self.param_states
            .iter()
            .map(|ps| (ps.name.clone(), ps.param.clone()))
            .collect()
    }

    fn add_param(
        &mut self,
        param: &mut Tensor<B, DenseStorage<T>, T>,
        name: String,
    ) -> Result<(), crate::error::OptimError> {
        if !param.requires_grad() {
            return Err(crate::error::OptimError::InvalidParameter {
                param: name.clone(),
                value: "requires_grad=false".to_string(),
                reason: "parameter must require gradients for optimization".to_string(),
            });
        }

        if self.has_param(&name) {
            return Err(crate::error::OptimError::InvalidParameter {
                param: name,
                value: "already exists".to_string(),
                reason: "parameter name must be unique".to_string(),
            });
        }

        let param_clone = param.clone();
        let mut param_state = ParamState::new(param_clone, name);
        if self.momentum > 0.0 {
            // Initialize momentum buffer
            let velocity = Tensor::zeros(param.shape().dims())
                .map_err(|e| crate::error::OptimError::TensorError { source: e })?;
            param_state.init_state("momentum_buffer".to_string(), velocity);
        }
        self.param_states.push(param_state);
        Ok(())
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

    fn set_lr(&mut self, lr: f64) -> Result<(), crate::error::OptimError> {
        if lr <= 0.0 {
            return Err(crate::error::OptimError::InvalidParameter {
                param: "lr".to_string(),
                value: lr.to_string(),
                reason: "Learning rate must be positive".to_string(),
            });
        }
        self.lr = lr;
        Ok(())
    }

    fn weight_decay(&self) -> f64 {
        self.weight_decay
    }

    fn set_weight_decay(&mut self, weight_decay: f64) -> Result<(), crate::error::OptimError> {
        if weight_decay < 0.0 {
            return Err(crate::error::OptimError::InvalidParameter {
                param: "weight_decay".to_string(),
                value: weight_decay.to_string(),
                reason: "Weight decay must be non-negative".to_string(),
            });
        }
        self.weight_decay = weight_decay;
        Ok(())
    }

    fn zero_grad(&mut self) {
        for param_state in &mut self.param_states {
            // Zero gradients in the tensor
            if let Ok(()) = param_state.param.zero_grad() {
                // Successfully zeroed gradients
            }
        }
    }

    fn step(&mut self) -> Result<usize, crate::error::OptimError> {
        let lr = T::from(self.lr).unwrap();
        let weight_decay = T::from(self.weight_decay).unwrap();
        let momentum = T::from(self.momentum).unwrap();
        let dampening = T::from(self.dampening).unwrap();
        let one = T::from(1.0).unwrap();
        let mut updated = 0usize;

        for param_state in &mut self.param_states {
            // Get gradient from tensor (PyTorch-style: gradients are stored on tensors)
            let grad = match param_state.param.grad() {
                Ok(grad) => grad,
                Err(_) => continue,
            };

            // Apply weight decay if specified
            let effective_grad = if self.weight_decay > 0.0 {
                let weight_decay_t = Tensor::from_vec_with_backend(vec![weight_decay], &[], param_state.param.backend().clone())
                     .map_err(|e| crate::OptimError::TensorError { source: e })?;
                let weight_decay_term = mul(&param_state.param, &weight_decay_t)?;
                add(&grad, &weight_decay_term)?
            } else {
                grad
            };

            if self.momentum > 0.0 {
                // Momentum-based update
                let velocity_key = "momentum_buffer";
                let param_name = param_state.name.clone();
                let velocity = param_state.get_state_mut(velocity_key).ok_or_else(|| {
                    crate::error::OptimError::InvalidState {
                        param_name,
                        state_key: velocity_key.to_string(),
                    }
                })?;

                let momentum_t = Tensor::from_vec_with_backend(vec![momentum], &[], effective_grad.backend().clone())
                     .map_err(|e| crate::OptimError::TensorError { source: e })?;
                     
                let one_minus_dampening_t = Tensor::from_vec_with_backend(vec![one - dampening], &[], effective_grad.backend().clone())
                     .map_err(|e| crate::OptimError::TensorError { source: e })?;

                let new_velocity = add(
                    &mul(&*velocity, &momentum_t)?,
                    &mul(&effective_grad, &one_minus_dampening_t)?,
                )?;
                velocity
                    .as_mut_slice()
                    .copy_from_slice(new_velocity.as_slice());

                let update_dir = if self.nesterov {
                    add(&effective_grad, &mul(&*velocity, &momentum_t)?)?
                } else {
                    velocity.clone()
                };

                let lr_t = Tensor::from_vec_with_backend(vec![lr], &[], effective_grad.backend().clone())
                     .map_err(|e| crate::OptimError::TensorError { source: e })?;
                let param_update = mul(&update_dir, &lr_t)?;
                if param_state.param.as_slice().len() != param_update.as_slice().len() {
                    return Err(crate::error::OptimError::ShapeMismatch {
                        param_name: param_state.name.clone(),
                        expected: param_state.param.shape().dims().to_vec(),
                        actual: param_update.shape().dims().to_vec(),
                    });
                }
                for (p, u) in param_state
                    .param
                    .as_mut_slice()
                    .iter_mut()
                    .zip(param_update.as_slice().iter())
                {
                    *p = *p - *u;
                }
            } else {
                // Standard SGD: p = p - lr * g
                let lr_t = Tensor::from_vec_with_backend(vec![lr], &[], effective_grad.backend().clone())
                     .map_err(|e| crate::OptimError::TensorError { source: e })?;
                let param_update = mul(&effective_grad, &lr_t)?;
                if param_state.param.as_slice().len() != param_update.as_slice().len() {
                    return Err(crate::error::OptimError::ShapeMismatch {
                        param_name: param_state.name.clone(),
                        expected: param_state.param.shape().dims().to_vec(),
                        actual: param_update.shape().dims().to_vec(),
                    });
                }
                for (p, u) in param_state
                    .param
                    .as_mut_slice()
                    .iter_mut()
                    .zip(param_update.as_slice().iter())
                {
                    *p = *p - *u;
                }
            }

            updated += 1;
        }

        Ok(updated)
    }

    fn state_dict(&self) -> HashMap<String, Tensor<B, DenseStorage<T>, T>> {
        let mut state = HashMap::new();
        for param_state in &self.param_states {
            state.insert(param_state.name.clone(), param_state.param.clone());
            for (key, tensor) in &param_state.state {
                state.insert(format!("{}.{}", param_state.name, key), tensor.clone());
            }
        }
        state
    }

    fn load_state_dict(
        &mut self,
        state_dict: HashMap<String, Tensor<B, DenseStorage<T>, T>>,
    ) -> Result<(), crate::error::OptimError> {
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

impl<B, T> Default for SGD<B, T>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + FloatExt + num_traits::FromPrimitive,
{
    fn default() -> Self {
        Self::new(0.01, 0.0, 0.0, 0.0, false)
    }
}
