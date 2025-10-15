//! Adam (Adaptive Moment Estimation) optimizer.
//!
//! This module implements the Adam optimizer with bias correction and adaptive
//! learning rates using first and second moment estimates.

use std::collections::HashMap;
use std::marker::PhantomData;

use coeus_backend::Backend;
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::{DenseStorage, Storage, StorageFromVec};
use coeus_tensor::Tensor;

use crate::optimizer_core::{Optimizer, ParamState};
use crate::Parameter;

/// Adam (Adaptive Moment Estimation) optimizer.
///
/// Implements the Adam algorithm from "Adam: A Method for Stochastic Optimization"
/// (Kingma & Ba, 2014): <https://arxiv.org/abs/1412.6980>
///
/// Adam combines the benefits of AdaGrad and RMSprop by maintaining per-parameter
/// adaptive learning rates using first and second moment estimates with bias correction.
///
/// # Algorithm
///
/// ```text
/// m_t = beta1 * m_{t-1} + (1 - beta1) * grad
/// v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
/// m_hat = m_t / (1 - beta1^t)
/// v_hat = v_t / (1 - beta2^t)
/// param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
/// ```
///
/// # Hyperparameters
///
/// - `lr`: Learning rate (default: 0.001)
/// - `beta1`: First moment decay rate (default: 0.9)
/// - `beta2`: Second moment decay rate (default: 0.999)
/// - `epsilon`: Numerical stability constant (default: 1e-8)
/// - `weight_decay`: L2 regularization factor (default: 0.0)
///
/// # Examples
///
/// ```rust
/// use coeus_optim::adam::Adam;
/// use coeus_dtype::float::Float32;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
///
/// // Create Adam with default hyperparameters
/// let mut optimizer = Adam::<CpuBackend, DenseStorage<Float32>, Float32>::new(0.001, 0.9, 0.999, 1e-8, 0.0);
/// ```
#[derive(Debug)]
pub struct Adam<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    /// Parameter states
    param_states: Vec<ParamState<B, S, T>>,
    /// Learning rate
    lr: f64,
    /// First moment decay rate
    beta1: f64,
    /// Second moment decay rate
    beta2: f64,
    /// Numerical stability constant
    epsilon: f64,
    /// Weight decay (L2 regularization)
    weight_decay: f64,
    /// Timestep counter
    t: usize,
    /// Phantom data
    _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> Adam<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    /// Create a new Adam optimizer.
    ///
    /// # Arguments
    /// * `lr` - Learning rate
    /// * `beta1` - First moment decay rate
    /// * `beta2` - Second moment decay rate
    /// * `epsilon` - Numerical stability constant
    /// * `weight_decay` - L2 regularization factor
    pub fn new(lr: f64, beta1: f64, beta2: f64, epsilon: f64, weight_decay: f64) -> Self {
        Self {
            param_states: Vec::new(),
            lr,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            t: 0,
            _phantom: PhantomData,
        }
    }

    /// Create Adam with default hyperparameters.
    pub fn default(lr: f64) -> Self {
        Self::new(lr, 0.9, 0.999, 1e-8, 0.0)
    }

    /// Create Adam with custom beta values.
    pub fn with_betas(lr: f64, beta1: f64, beta2: f64) -> Self {
        Self::new(lr, beta1, beta2, 1e-8, 0.0)
    }

    /// Get beta1 value
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Get beta2 value
    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    /// Get epsilon value
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Get current timestep
    pub fn timestep(&self) -> usize {
        self.t
    }
}

impl<B, S, T> Optimizer<B, S, T> for Adam<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    fn name(&self) -> &str {
        "Adam"
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

        // Initialize Adam state: m (first moment) and v (second moment)
        let shape = param_state.param.shape().dims();
        let m = Tensor::zeros(shape).unwrap();
        let v = Tensor::zeros(shape).unwrap();

        param_state.init_state("m".to_string(), m);
        param_state.init_state("v".to_string(), v);

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
        self.t += 1;

        let lr = T::from(self.lr).unwrap();
        let beta1 = T::from(self.beta1).unwrap();
        let beta2 = T::from(self.beta2).unwrap();
        let epsilon = T::from(self.epsilon).unwrap();
        let weight_decay = T::from(self.weight_decay).unwrap();
        let one = T::from(1.0).unwrap();
        let t = T::from(self.t as f64).unwrap();

        for param_state in &mut self.param_states {
            let grad = param_state.grad()?;

            // Apply weight decay if specified
            let effective_grad = if self.weight_decay > 0.0 {
                grad + &(&param_state.param * weight_decay)
            } else {
                grad.clone()
            };

            // Get or create moment estimates
            let m = param_state.get_state_mut("m")
                .ok_or_else(|| crate::error::OptimError::InvalidState {
                    param_name: param_state.name.clone(),
                    state_key: "m".to_string(),
                })?;

            let v = param_state.get_state_mut("v")
                .ok_or_else(|| crate::error::OptimError::InvalidState {
                    param_name: param_state.name.clone(),
                    state_key: "v".to_string(),
                })?;

            // Update biased first moment estimate
            // m_t = beta1 * m_{t-1} + (1 - beta1) * grad
            *m = &(&m * beta1) + &(&effective_grad * (one - beta1));

            // Update biased second raw moment estimate
            // v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
            let grad_squared = &effective_grad * &effective_grad;
            *v = &(&v * beta2) + &(&grad_squared * (one - beta2));

            // Compute bias-corrected first moment
            // m_hat = m_t / (1 - beta1^t)
            let beta1_t = beta1.powf(t);
            let m_hat = m / (one - beta1_t);

            // Compute bias-corrected second moment
            // v_hat = v_t / (1 - beta2^t)
            let beta2_t = beta2.powf(t);
            let v_hat = v / (one - beta2_t);

            // Compute parameter update
            // param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
            let v_hat_sqrt = v_hat.sqrt();
            let denominator = &v_hat_sqrt + &epsilon;
            let update = &(&m_hat * lr) / &denominator;

            param_state.param -= &update;
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
        state.insert("timestep".to_string(), Tensor::from_vec(vec![T::from(self.t as f64).unwrap()], &[1]).unwrap());
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

            // Load moment estimates
            let m_key = format!("{}.m", param_state.name);
            let v_key = format!("{}.v", param_state.name);

            if let Some(m) = state_dict.get(&m_key) {
                param_state.init_state("m".to_string(), m.clone());
            }
            if let Some(v) = state_dict.get(&v_key) {
                param_state.init_state("v".to_string(), v.clone());
            }
        }

        if let Some(t_tensor) = state_dict.get("timestep") {
            if let Some(&t_val) = t_tensor.as_slice().first() {
                self.t = t_val.to_f64().unwrap() as usize;
            }
        }

        Ok(())
    }
}

impl<B, S, T> Default for Adam<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    fn default() -> Self {
        Self::new(0.001, 0.9, 0.999, 1e-8, 0.0)
    }
}
