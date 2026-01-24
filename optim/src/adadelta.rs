use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::optimizer::BaseOptimizer;
use crate::optimizer_core::{Optimizer, ParamState};
use crate::Parameter;

#[derive(Debug)]
pub struct Adadelta<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive,
{
    param_states: Vec<ParamState<B, S, T>>,
    param_groups: Vec<crate::optimizer::ParamGroup<B, S, T>>,
    lr: f64,
    rho: f64,
    eps: f64,
    weight_decay: f64,
}

impl<B, S, T> Adadelta<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive,
{
    pub fn new(params: Vec<tensor::Tensor<B, S, T>>, lr: f64) -> Self {
        assert!(lr > 0.0, "Learning rate must be positive, got {}", lr);
        Self::with_hyperparams(params, lr, 0.9, 1e-6, 0.0)
    }

    pub fn with_hyperparams(
        params: Vec<tensor::Tensor<B, S, T>>,
        lr: f64,
        rho: f64,
        eps: f64,
        weight_decay: f64,
    ) -> Self {
        assert!(lr > 0.0, "Learning rate must be positive, got {}", lr);
        assert!(
            (0.0..=1.0).contains(&rho),
            "rho must be in range [0, 1], got {}",
            rho
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
            rho,
            eps,
            weight_decay,
        };
        optimizer.add_param_group(params);
        optimizer
    }
}

impl<B, S, T> BaseOptimizer<B, S, T> for Adadelta<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive,
{
    fn step(&mut self) -> Result<usize, crate::OptimError> {
        self.step_cpu()
    }

    fn step_cpu(&mut self) -> Result<usize, crate::OptimError> {
        use tensor::ops::arithmetic::{add, div, mul};
        use tensor::ops::math::sqrt;

        let lr = T::from(self.lr).unwrap();
        let rho = T::from(self.rho).unwrap();
        let one = T::from(1.0).unwrap();
        let eps = T::from(self.eps).unwrap();
        let weight_decay = T::from(self.weight_decay).unwrap();

        let mut updated = 0usize;
        for param_state in &mut self.param_states {
            let grad = match param_state.param.grad() {
                Ok(tensor_grad) => {
                    let dense_grad: DenseStorage<T> = match tensor_grad.storage().to_dense() {
                        Ok(dense) => dense,
                        Err(_) => continue,
                    };
                    match S::from_vec(dense_grad.as_slice().to_vec(), tensor_grad.shape().dims()) {
                        Ok(converted_storage) => {
                            let backend = tensor_grad.backend().clone();
                            Tensor::from_storage(converted_storage, backend)
                        }
                        Err(_) => continue,
                    }
                }
                Err(_) => continue,
            };

            let effective_grad = if self.weight_decay > 0.0 {
                let weight_decay_t = Tensor::from_vec_with_backend(vec![weight_decay], &[], param_state.param.backend().clone())
                     .map_err(|e| crate::OptimError::TensorError { source: e })?;
                let weight_decay_term = mul(&param_state.param, &weight_decay_t)?;
                add(&grad, &weight_decay_term)?
            } else {
                grad
            };

            let param_name = param_state.name.clone();

            let acc_delta_old = param_state
                .get_state("acc_delta")
                .ok_or_else(|| crate::error::OptimError::InvalidState {
                    param_name: param_name.clone(),
                    state_key: "acc_delta".to_string(),
                })?
                .clone();

            let square_avg_prev = param_state
                .get_state("square_avg")
                .ok_or_else(|| crate::error::OptimError::InvalidState {
                    param_name: param_name.clone(),
                    state_key: "square_avg".to_string(),
                })?
                .clone();

            let grad_sq = mul(&effective_grad, &effective_grad)?;
            
            let rho_t = Tensor::from_vec_with_backend(vec![rho], &[], effective_grad.backend().clone())
                 .map_err(|e| crate::OptimError::TensorError { source: e })?;
            let one_minus_rho = one - rho;
            let one_minus_rho_t = Tensor::from_vec_with_backend(vec![one_minus_rho], &[], effective_grad.backend().clone())
                 .map_err(|e| crate::OptimError::TensorError { source: e })?;
            
            let rho_square_avg = mul(&square_avg_prev, &rho_t)?;
            let one_minus_rho_grad_sq = mul(&grad_sq, &one_minus_rho_t)?;
            let square_avg_new = add(&rho_square_avg, &one_minus_rho_grad_sq)?;

            *param_state.get_state_mut("square_avg").ok_or_else(|| {
                crate::error::OptimError::InvalidState {
                    param_name: param_name.clone(),
                    state_key: "square_avg".to_string(),
                }
            })? = square_avg_new.clone();

            let eps_t = Tensor::from_vec_with_backend(vec![eps], &[], effective_grad.backend().clone())
                 .map_err(|e| crate::OptimError::TensorError { source: e })?;

            let numerator = sqrt(&add(&acc_delta_old, &eps_t)?)?;
            let denominator = sqrt(&add(&square_avg_new, &eps_t)?)?;
            let rms = div(&numerator, &denominator)?;
            let delta = mul(&rms, &effective_grad)?;

            {
                let acc_delta = param_state.get_state_mut("acc_delta").ok_or_else(|| {
                    crate::error::OptimError::InvalidState {
                        param_name: param_name.clone(),
                        state_key: "acc_delta".to_string(),
                    }
                })?;
                let delta_sq = mul(&delta, &delta)?;
                let rho_acc_delta = mul(acc_delta, &rho_t)?;
                let _one_minus_rho = one - rho;
                let one_minus_rho_delta_sq = mul(&delta_sq, &one_minus_rho_t)?;
                *acc_delta = add(&rho_acc_delta, &one_minus_rho_delta_sq)?;
            }

            let lr_t = Tensor::from_vec_with_backend(vec![lr], &[], effective_grad.backend().clone())
                 .map_err(|e| crate::OptimError::TensorError { source: e })?;
            let scaled_delta = mul(&delta, &lr_t)?;
            for (p, d) in param_state
                .param
                .as_mut_slice()
                .iter_mut()
                .zip(scaled_delta.as_slice().iter().copied())
            {
                *p = *p - d;
            }
            updated += 1;
        }

        Ok(updated)
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
            let square_avg = Tensor::zeros(&shape).unwrap();
            let acc_delta = Tensor::zeros(&shape).unwrap();

            param_state.init_state("square_avg".to_string(), square_avg);
            param_state.init_state("acc_delta".to_string(), acc_delta);
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
        for group in &mut self.param_groups {
            group.lr = lr;
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

            for key in ["square_avg", "acc_delta"] {
                let state_key = format!("{}.{}", param_state.name, key);
                if let Some(t) = state_dict.get(&state_key) {
                    param_state.init_state(key.to_string(), t.clone());
                }
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

impl<B, S, T> Optimizer<B, S, T> for Adadelta<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive,
{
    fn name(&self) -> &str {
        "Adadelta"
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

        let mut param_state = ParamState::new(param.clone(), name);
        let shape = param.shape().dims().to_vec();
        let square_avg = Tensor::zeros(&shape)
            .map_err(|e| crate::error::OptimError::TensorError { source: e })?;
        let acc_delta = Tensor::zeros(&shape)
            .map_err(|e| crate::error::OptimError::TensorError { source: e })?;
        param_state.init_state("square_avg".to_string(), square_avg);
        param_state.init_state("acc_delta".to_string(), acc_delta);
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
        for group in &mut self.param_groups {
            group.lr = lr as f32;
        }
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
