//! Adagrad (Adaptive Gradient) optimizer.
//!
//! This module implements the Adagrad optimizer, which adapts the learning rate
//! per parameter based on the accumulation of squared gradients.
//!
//! Adagrad algorithm:
//! ```text
//! G_t = G_{t-1} + g_t²
//! θ_t = θ_{t-1} - (α / √(G_t + ε)) * g_t
//! ```

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{Storage, StorageFromVec};
use tensor::Tensor;

use crate::optimizer::BaseOptimizer;
use crate::optimizer_core::{Optimizer, ParamState};
use crate::Parameter;

/// Adagrad (Adaptive Gradient) optimizer
///
/// Adapts learning rate per parameter based on historical squared gradients.
/// Effective for sparse data and features with different frequencies.
#[derive(Debug)]
pub struct Adagrad<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + storage::StorageToDense<T> + tensor::ops::arithmetic::traits::TensorStorageArithmetic<T>,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive,
{
    param_states: Vec<ParamState<B, S, T>>,
    param_groups: Vec<crate::optimizer::ParamGroup<B, S, T>>,
    lr: f64,
    lr_decay: f64, // Learning rate decay factor
    weight_decay: f64,
    initial_accumulator_value: f64, // Initial value for gradient accumulator
    eps: f64,                       // Small constant for numerical stability
}

impl<B, S, T> Adagrad<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + storage::StorageToDense<T> + tensor::ops::arithmetic::traits::TensorStorageArithmetic<T>,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive,
{
    /// Create Adagrad optimizer with default hyperparameters
    ///
    /// # Arguments
    /// * `params` - Parameter tensors to optimize
    /// * `lr` - Learning rate (must be > 0)
    pub fn new(params: Vec<tensor::Tensor<B, S, T>>, lr: f64) -> Self {
        assert!(lr > 0.0, "Learning rate must be positive, got {}", lr);
        Self::with_hyperparams(params, lr, 0.0, 1e-10, 0.0, 0.0)
    }

    /// Create Adagrad optimizer with custom hyperparameters
    ///
    /// # Arguments
    /// * `params` - Parameter tensors to optimize
    /// * `lr` - Learning rate (must be > 0)
    /// * `lr_decay` - Learning rate decay factor (must be >= 0)
    /// * `weight_decay` - L2 regularization factor (must be >= 0)
    /// * `initial_accumulator_value` - Initial value for gradient accumulator (must be >= 0)
    /// * `eps` - Small constant for numerical stability (must be >= 0)
    pub fn with_hyperparams(
        params: Vec<tensor::Tensor<B, S, T>>,
        lr: f64,
        lr_decay: f64,
        weight_decay: f64,
        initial_accumulator_value: f64,
        eps: f64,
    ) -> Self {
        // Validate hyperparameters
        assert!(lr > 0.0, "Learning rate must be positive, got {}", lr);
        assert!(
            lr_decay >= 0.0,
            "lr_decay must be non-negative, got {}",
            lr_decay
        );
        assert!(
            weight_decay >= 0.0,
            "weight_decay must be non-negative, got {}",
            weight_decay
        );
        assert!(
            initial_accumulator_value >= 0.0,
            "initial_accumulator_value must be non-negative, got {}",
            initial_accumulator_value
        );
        assert!(eps >= 0.0, "eps must be non-negative, got {}", eps);

        let mut optimizer = Self {
            param_states: Vec::new(),
            param_groups: Vec::new(),
            lr,
            lr_decay,
            weight_decay,
            initial_accumulator_value,
            eps,
        };
        optimizer.add_param_group(params);
        optimizer
    }

    /// Create Adagrad optimizer with default learning rate
    pub fn default(lr: f64) -> Self {
        Self::new(Vec::new(), lr)
    }
}

impl<B, S, T> BaseOptimizer<B, S, T> for Adagrad<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + storage::StorageToDense<T> + tensor::ops::arithmetic::traits::TensorStorageArithmetic<T>,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive,
{
    fn step(&mut self) -> Result<usize, crate::OptimError> {
        self.step_cpu()
    }

    fn step_cpu(&mut self) -> Result<usize, crate::OptimError> {
        use tensor::ops::arithmetic::{add, div, mul, sub};
        use tensor::ops::math::sqrt;

        let lr = T::from(self.lr).unwrap();
        let eps = T::from(self.eps).unwrap();
        let weight_decay = T::from(self.weight_decay).unwrap();
        let initial_accum_val = T::from(self.initial_accumulator_value).unwrap();

        for param_state in &mut self.param_states {
            // Get gradient from tensor
            let grad = match param_state.param.grad() {
                Ok(tensor_grad) => {
                    // Convert gradient storage type to match optimizer's expected type
                    match S::from_vec(tensor_grad.as_slice().to_vec(), tensor_grad.shape().dims()) {
                        Ok(converted_storage) => {
                            let backend = tensor_grad.backend().clone();
                            Tensor::from_storage(converted_storage, backend)
                        }
                        Err(_) => return Err(crate::OptimError::GradientNotAvailable),
                    }
                }
                Err(_) => return Err(crate::OptimError::GradientNotAvailable),
            };

            // Apply weight decay if specified (L2 regularization)
            // Apply weight decay if specified (L2 regularization)
            let effective_grad = if self.weight_decay > 0.0 {
                let weight_decay_t = Tensor::from_vec_with_backend(vec![weight_decay], &[], param_state.param.backend().clone())
                     .map_err(|e| crate::OptimError::TensorError { source: e })?;
                let weight_decay_term = mul(&param_state.param, &weight_decay_t)?;
                add(&grad, &weight_decay_term)?
            } else {
                grad.clone()
            };

            // Update gradient accumulator: state_sum = state_sum + grad²
            let param_name = param_state.name.clone();
            let accumulator = param_state.get_state_mut("state_sum").ok_or_else(|| {
                crate::error::OptimError::InvalidState {
                    param_name: param_name.clone(),
                    state_key: "state_sum".to_string(),
                }
            })?;

            let grad_squared = mul(&effective_grad, &effective_grad)?;
            let accumulator_val = add(accumulator, &grad_squared)?;

            // Update the stored accumulator
            let initial_accum_t = Tensor::from_vec_with_backend(vec![initial_accum_val], &[], effective_grad.backend().clone())
                 .map_err(|e| crate::OptimError::TensorError { source: e })?;
            *accumulator = add(&accumulator_val, &initial_accum_t)?;

            // Apply learning rate decay (if specified)
            let effective_lr = if self.lr_decay > 0.0 {
                // In place of step decay, use a smooth decay: lr / (1 + decay * time)
                // For simplicity, we'll assume decay affects each step uniformly
                T::from(self.lr / (1.0 + self.lr_decay)).unwrap()
            } else {
                lr
            };

            // Compute adaptive learning rate: lr / sqrt(state_sum + eps)
            let eps_t = Tensor::from_vec_with_backend(vec![eps], &[], effective_grad.backend().clone())
                 .map_err(|e| crate::OptimError::TensorError { source: e })?;
            let accumulator_with_eps = add(&accumulator_val, &eps_t)?;
            let sqrt_accum = sqrt(&accumulator_with_eps)?;
            // Create tensor filled with effective_lr for element-wise division
            let lr_tensor = Tensor::from_vec_with_backend(
                vec![effective_lr; sqrt_accum.as_slice().len()],
                sqrt_accum.shape().dims(),
                sqrt_accum.backend().clone(),
            )?;
            let adaptive_lr = div(&lr_tensor, &sqrt_accum)?;

            // Parameter update: θ = θ - (adaptive_lr * grad)
            let scaled_grad = mul(&effective_grad, &adaptive_lr)?;
            param_state.param = sub(&param_state.param, &scaled_grad)?;
        }

        Ok(self.param_states.len())
    }

    fn zero_grad(&mut self) {
        for param_state in &mut self.param_states {
            let _ = param_state.param.zero_grad();
        }
    }

    fn add_param_group(&mut self, params: Vec<tensor::Tensor<B, S, T>>) {
        // Create parameter states for each tensor with Adagrad-specific state
        for tensor in params.clone().into_iter() {
            let mut param_state =
                ParamState::new(tensor.clone(), format!("param_{}", self.param_states.len()));

            // Initialize Adagrad state: sum of squared gradients accumulator
            let shape = tensor.shape().dims().to_vec();
            let state_sum = Tensor::from_vec_with_backend(
                vec![T::from(self.initial_accumulator_value).unwrap(); shape.iter().product()],
                &shape,
                tensor.backend().clone(),
            )
            .unwrap(); // Accumulates sum of squared gradients

            param_state.init_state("state_sum".to_string(), state_sum);

            self.param_states.push(param_state);
        }

        // Create parameter group
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

        // Save parameters and their Adagrad state (gradient accumulator)
        for param_state in &self.param_states {
            state.insert(param_state.name.clone(), param_state.param.clone());

            // Save Adagrad state (sum of squared gradients)
            for (key, tensor) in &param_state.state {
                state.insert(format!("{}.{}", param_state.name, key), tensor.clone());
            }
        }

        // Save hyperparameters for state reconstruction
        let lr_tensor = Tensor::from_vec(vec![T::from(self.lr).unwrap()], &[1]).unwrap();
        state.insert("lr".to_string(), lr_tensor);

        let lr_decay_tensor =
            Tensor::from_vec(vec![T::from(self.lr_decay).unwrap()], &[1]).unwrap();
        state.insert("lr_decay".to_string(), lr_decay_tensor);

        state
    }

    fn load_state_dict(
        &mut self,
        state_dict: std::collections::HashMap<String, tensor::Tensor<B, S, T>>,
    ) -> Result<(), crate::OptimError> {
        for param_state in &mut self.param_states {
            // Load parameter
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

            // Load Adagrad state (gradient accumulator)
            let state_sum_key = format!("{}.state_sum", param_state.name);
            if let Some(state_sum) = state_dict.get(&state_sum_key) {
                param_state.init_state("state_sum".to_string(), state_sum.clone());
            }
        }

        // Load hyperparameters
        if let Some(lr_tensor) = state_dict.get("lr") {
            if let Some(&lr_val) = lr_tensor.as_slice().first() {
                self.lr = lr_val.to_f64().unwrap();
            }
        }

        if let Some(lr_decay_tensor) = state_dict.get("lr_decay") {
            if let Some(&lr_decay_val) = lr_decay_tensor.as_slice().first() {
                self.lr_decay = lr_decay_val.to_f64().unwrap();
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

impl<B, S, T> Optimizer<B, S, T> for Adagrad<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + storage::StorageToDense<T> + tensor::ops::arithmetic::traits::TensorStorageArithmetic<T>,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive,
{
    fn name(&self) -> &str {
        "Adagrad"
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
        // Check if parameter requires gradients
        if !param.requires_grad() {
            return Err(crate::error::OptimError::InvalidParameter {
                param: name.clone(),
                value: "requires_grad=false".to_string(),
                reason: "parameter must require gradients for optimization".to_string(),
            });
        }

        // Check if parameter already exists
        if self.has_param(&name) {
            return Err(crate::error::OptimError::InvalidParameter {
                param: name,
                value: "already exists".to_string(),
                reason: "parameter name must be unique".to_string(),
            });
        }

        let param_clone = param.clone();
        let mut param_state = ParamState::new(param_clone, name);

        // Initialize Adagrad state (gradient accumulator)
        let shape = param.shape().dims();
        let initial_accumulator = Tensor::from_vec_with_backend(
            vec![T::from(self.initial_accumulator_value).unwrap(); shape.iter().product()],
            shape,
            param.backend().clone(),
        )
        .map_err(|e| crate::error::OptimError::TensorError { source: e })?;
        param_state.init_state("state_sum".to_string(), initial_accumulator);

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
        // Update parameter groups if they exist
        for group in &mut self.param_groups {
            group.weight_decay = weight_decay as f32;
        }
        Ok(())
    }

    /// Get the learning rate (alias for lr)
    fn learning_rate(&self) -> f64 {
        self.lr()
    }

    /// Set the learning rate (alias for set_lr)
    fn set_learning_rate(&mut self, lr: f64) -> Result<(), crate::error::OptimError> {
        <Self as Optimizer<B, S, T>>::set_lr(self, lr)
    }

    fn zero_grad(&mut self) {
        <Self as BaseOptimizer<B, S, T>>::zero_grad(self);
    }

    fn step(&mut self) -> Result<usize, crate::OptimError> {
        <Self as BaseOptimizer<B, S, T>>::step(self)
    }

    fn state_dict(&self) -> std::collections::HashMap<String, Tensor<B, S, T>> {
        <Self as BaseOptimizer<B, S, T>>::state_dict(self)
    }

    fn load_state_dict(
        &mut self,
        state_dict: std::collections::HashMap<String, Tensor<B, S, T>>,
    ) -> Result<(), crate::OptimError> {
        <Self as BaseOptimizer<B, S, T>>::load_state_dict(self, state_dict)
    }
}
