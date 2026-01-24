//! AdamW (Adam with decoupled weight decay) optimizer.
//!
//! This module implements the AdamW optimizer, which fixes weight decay regularization
//! in Adam by decoupling it from the gradient update.
//!
//! AdamW algorithm:
//! ```text
//! m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
//! v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
//! m̂_t = m_t / (1 - β₁^t)
//! v̂_t = v_t / (1 - β₂^t)
//! θ_t = θ_{t-1} - η * (λ * θ_{t-1} + m̂_t / (√v̂_t + ε))
//! ```
//! where η is the learning rate and λ is the weight decay coefficient.

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::gpu_backend::GpuAcceleratedOptimizer;
use crate::optimizer::BaseOptimizer;
use crate::optimizer_core::{Optimizer, ParamState};
use crate::Parameter;

/// AdamW optimizer with optional GPU acceleration
#[derive(Debug)]
pub struct AdamW<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive,
{
    param_states: Vec<ParamState<B, S, T>>,
    param_groups: Vec<crate::optimizer::ParamGroup<B, S, T>>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    t: u64, // timestep for bias correction
}

impl<B, S, T> AdamW<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive,
{
    /// Create AdamW optimizer with default hyperparameters
    ///
    /// # Arguments
    /// * `params` - Parameter tensors to optimize
    /// * `lr` - Learning rate (must be > 0)
    pub fn new(params: Vec<tensor::Tensor<B, S, T>>, lr: f64) -> Self {
        assert!(lr > 0.0, "Learning rate must be positive, got {}", lr);
        Self::with_hyperparams(params, lr, 0.9, 0.999, 1e-8, 0.01)
    }

    /// Create AdamW optimizer with custom hyperparameters
    ///
    /// # Arguments
    /// * `params` - Parameter tensors to optimize
    /// * `lr` - Learning rate (must be > 0)
    /// * `beta1` - Exponential decay rate for first moment (must be in [0, 1))
    /// * `beta2` - Exponential decay rate for second moment (must be in [0, 1))
    /// * `eps` - Small constant for numerical stability (must be >= 0)
    /// * `weight_decay` - Weight decay factor (must be >= 0)
    pub fn with_hyperparams(
        params: Vec<tensor::Tensor<B, S, T>>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    ) -> Self {
        // Validate hyperparameters
        assert!(lr > 0.0, "Learning rate must be positive, got {}", lr);
        assert!(
            (0.0..1.0).contains(&beta1),
            "beta1 must be in range [0, 1), got {}",
            beta1
        );
        assert!(
            (0.0..1.0).contains(&beta2),
            "beta2 must be in range [0, 1), got {}",
            beta2
        );
        assert!(eps >= 0.0, "eps must be non-negative, got {}", eps);
        assert!(
            weight_decay >= 0.0,
            "weight_decay must be non-negative, got {}",
            weight_decay
        );

        let mut optimizer = Self {
            param_states: Vec::new(),
            param_groups: Vec::new(),
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t: 0,
        };
        optimizer.add_param_group(params);
        optimizer
    }

    /// Get beta1 parameter
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Get beta2 parameter
    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    /// Get epsilon parameter
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Get weight decay parameter
    pub fn weight_decay(&self) -> f64 {
        self.weight_decay
    }
}

impl<B, S, T> BaseOptimizer<B, S, T> for AdamW<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive,
{
    fn step(&mut self) -> Result<usize, crate::OptimError> {
        self.step_cpu()
    }

    fn step_cpu(&mut self) -> Result<usize, crate::OptimError> {
        use tensor::ops::arithmetic::{add, div, mul, sub};
        use tensor::ops::math::sqrt;

        self.t += 1;

        let lr = T::from(self.lr).unwrap();
        let beta1 = T::from(self.beta1).unwrap();
        let beta2 = T::from(self.beta2).unwrap();
        let eps = T::from(self.eps).unwrap();
        let weight_decay = T::from(self.weight_decay).unwrap();
        let one = T::from(1.0).unwrap();

        let t_val = self.t as f64;
        let bias_correction1 = T::from(1.0 / (1.0 - self.beta1.powf(t_val))).unwrap();
        let bias_correction2 = T::from(1.0 / (1.0 - self.beta2.powf(t_val))).unwrap();

        for param_state in &mut self.param_states {
            let grad = match param_state.param.grad() {
                Ok(tensor_grad) => {
                    let dense_grad: DenseStorage<T> = tensor_grad
                        .storage()
                        .to_dense()
                        .map_err(|_| crate::OptimError::GradientNotAvailable)?;
                    match S::from_vec(dense_grad.as_slice().to_vec(), tensor_grad.shape().dims()) {
                        Ok(converted_storage) => {
                            let backend = tensor_grad.backend().clone();
                            Tensor::from_storage(converted_storage, backend)
                        }
                        Err(_) => return Err(crate::OptimError::GradientNotAvailable),
                    }
                }
                Err(_) => return Err(crate::OptimError::GradientNotAvailable),
            };

            // In AdamW, weight decay is applied directly to weights, not to gradients
            // θ = θ - lr * wd * θ
            if self.weight_decay > 0.0 {
                let wd_lr = weight_decay * lr;
                let wd_lr_t = Tensor::from_vec_with_backend(vec![wd_lr], &[], param_state.param.backend().clone())
                     .map_err(|e| crate::OptimError::TensorError { source: e })?;
                let wd_term = mul(&param_state.param, &wd_lr_t)?;
                param_state.param = sub(&param_state.param, &wd_term)?;
            }

            let param_name = param_state.name.clone();
            {
                let m = param_state.get_state_mut("m").ok_or_else(|| {
                    crate::error::OptimError::InvalidState {
                        param_name: param_name.clone(),
                        state_key: "m".to_string(),
                    }
                })?;
                
                let beta1_t = Tensor::from_vec_with_backend(vec![beta1], &[], grad.backend().clone())
                     .map_err(|e| crate::OptimError::TensorError { source: e })?;
                let beta1_m = mul(m, &beta1_t)?;
                
                let one_minus_beta1 = one - beta1;
                let one_minus_beta1_t = Tensor::from_vec_with_backend(vec![one_minus_beta1], &[], grad.backend().clone())
                     .map_err(|e| crate::OptimError::TensorError { source: e })?;
                let one_minus_beta1_grad = mul(&grad, &one_minus_beta1_t)?;
                *m = add(&beta1_m, &one_minus_beta1_grad)?;
            }

            {
                let v = param_state.get_state_mut("v").ok_or_else(|| {
                    crate::error::OptimError::InvalidState {
                        param_name,
                        state_key: "v".to_string(),
                    }
                })?;
                let grad_squared = mul(&grad, &grad)?;
                
                let beta2_t = Tensor::from_vec_with_backend(vec![beta2], &[], grad.backend().clone())
                     .map_err(|e| crate::OptimError::TensorError { source: e })?;
                let beta2_v = mul(v, &beta2_t)?;
                
                let one_minus_beta2 = one - beta2;
                let one_minus_beta2_t = Tensor::from_vec_with_backend(vec![one_minus_beta2], &[], grad.backend().clone())
                     .map_err(|e| crate::OptimError::TensorError { source: e })?;
                let one_minus_beta2_grad_sq = mul(&grad_squared, &one_minus_beta2_t)?;
                *v = add(&beta2_v, &one_minus_beta2_grad_sq)?;
            }

            let m_ref = param_state.get_state("m").unwrap();
            let v_ref = param_state.get_state("v").unwrap();
            
            let bc1_t = Tensor::from_vec_with_backend(vec![bias_correction1], &[], grad.backend().clone())
                 .map_err(|e| crate::OptimError::TensorError { source: e })?;
            let m_hat = mul(m_ref, &bc1_t)?;
            
            let bc2_t = Tensor::from_vec_with_backend(vec![bias_correction2], &[], grad.backend().clone())
                 .map_err(|e| crate::OptimError::TensorError { source: e })?;
            let v_hat = mul(v_ref, &bc2_t)?;

            let eps_t = Tensor::from_vec_with_backend(vec![eps], &[], grad.backend().clone())
                 .map_err(|e| crate::OptimError::TensorError { source: e })?;
            let v_hat_plus_eps = add(&v_hat, &eps_t)?;
            let v_hat_sqrt = sqrt(&v_hat_plus_eps)?;
            let update_ratio = div(&m_hat, &v_hat_sqrt)?;
            
            let lr_t = Tensor::from_vec_with_backend(vec![lr], &[], grad.backend().clone())
                 .map_err(|e| crate::OptimError::TensorError { source: e })?;
            let scaled_update = mul(&update_ratio, &lr_t)?;
            param_state.param = sub(&param_state.param, &scaled_update)?;
        }

        Ok(self.param_states.len())
    }

    fn zero_grad(&mut self) {
        for param_state in &mut self.param_states {
            let _ = param_state.param.zero_grad();
        }
    }

    fn add_param_group(&mut self, params: Vec<tensor::Tensor<B, S, T>>) {
        for tensor in params.clone().into_iter() {
            let mut param_state =
                ParamState::new(tensor.clone(), format!("param_{}", self.param_states.len()));

            let shape = tensor.shape().dims().to_vec();
            let m = Tensor::zeros(&shape).unwrap();
            let v = Tensor::zeros(&shape).unwrap();

            param_state.init_state("m".to_string(), m);
            param_state.init_state("v".to_string(), v);

            self.param_states.push(param_state);
        }

        self.param_groups.push(crate::optimizer::ParamGroup::new(
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
        if !self.param_groups.is_empty() {
            self.param_groups[0].lr = lr;
        }
    }

    fn state_dict(&self) -> std::collections::HashMap<String, tensor::Tensor<B, S, T>> {
        let mut state = std::collections::HashMap::new();
        for param_state in &self.param_states {
            state.insert(param_state.name.clone(), param_state.param.clone());
            for (key, tensor) in &param_state.state {
                state.insert(format!("{}.{}", param_state.name, key), tensor.clone());
            }
        }
        let step_tensor = Tensor::from_vec(vec![T::from(self.t as f64).unwrap()], &[1]).unwrap();
        state.insert("step".to_string(), step_tensor);
        state
    }

    fn load_state_dict(
        &mut self,
        state_dict: std::collections::HashMap<String, tensor::Tensor<B, S, T>>,
    ) -> Result<(), crate::OptimError> {
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
            let m_key = format!("{}.m", param_state.name);
            if let Some(m) = state_dict.get(&m_key) {
                param_state.init_state("m".to_string(), m.clone());
            }
            let v_key = format!("{}.v", param_state.name);
            if let Some(v) = state_dict.get(&v_key) {
                param_state.init_state("v".to_string(), v.clone());
            }
        }
        if let Some(step_tensor) = state_dict.get("step") {
            if let Some(&step_val) = step_tensor.as_slice().first() {
                self.t = step_val.to_f64().unwrap() as u64;
            }
        }
        Ok(())
    }

    fn param_groups(&self) -> &[crate::optimizer::ParamGroup<B, S, T>] {
        &self.param_groups
    }

    fn param_groups_mut(&mut self) -> &mut [crate::optimizer::ParamGroup<B, S, T>] {
        &mut self.param_groups
    }
}

impl<B, S, T> Optimizer<B, S, T> for AdamW<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive,
{
    fn name(&self) -> &str {
        "AdamW"
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        self.param_states
            .iter()
            .map(|ps| ps.param.clone())
            .collect()
    }

    fn named_parameters(&self) -> std::collections::HashMap<String, Parameter<B, S, T>> {
        self.param_states
            .iter()
            .map(|ps| (ps.name.clone(), ps.param.clone()))
            .collect()
    }

    fn add_param(
        &mut self,
        param: &mut Parameter<B, S, T>,
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
        let shape = param.shape().dims();
        let m = Tensor::zeros(shape)
            .map_err(|e| crate::error::OptimError::TensorError { source: e })?;
        let v = Tensor::zeros(shape)
            .map_err(|e| crate::error::OptimError::TensorError { source: e })?;
        param_state.init_state("m".to_string(), m);
        param_state.init_state("v".to_string(), v);
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
                reason: "learning rate must be positive".to_string(),
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
                reason: "weight decay must be non-negative".to_string(),
            });
        }
        self.weight_decay = weight_decay;
        for group in &mut self.param_groups {
            group.weight_decay = weight_decay as f32;
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.lr()
    }

    fn set_learning_rate(&mut self, lr: f64) -> Result<(), crate::error::OptimError> {
        <Self as Optimizer<B, S, T>>::set_lr(self, lr)
    }

    fn zero_grad(&mut self) {
        <Self as BaseOptimizer<B, S, T>>::zero_grad(self);
    }

    fn step(&mut self) -> Result<usize, crate::error::OptimError> {
        <Self as BaseOptimizer<B, S, T>>::step(self)
    }

    fn state_dict(&self) -> std::collections::HashMap<String, Tensor<B, S, T>> {
        <Self as BaseOptimizer<B, S, T>>::state_dict(self)
    }

    fn load_state_dict(
        &mut self,
        state_dict: std::collections::HashMap<String, Tensor<B, S, T>>,
    ) -> Result<(), crate::error::OptimError> {
        <Self as BaseOptimizer<B, S, T>>::load_state_dict(self, state_dict)
    }
}

impl<B, S, T> GpuAcceleratedOptimizer for AdamW<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive,
{
}
