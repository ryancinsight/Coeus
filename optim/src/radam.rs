use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::optimizer::BaseOptimizer;
use crate::optimizer_core::{Optimizer, ParamState};
use crate::Parameter;

#[derive(Debug)]
pub struct RAdam<B, S, T>
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
    t: u64,
}

impl<B, S, T> RAdam<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive,
{
    pub fn new(params: Vec<tensor::Tensor<B, S, T>>, lr: f64) -> Self {
        assert!(lr > 0.0, "Learning rate must be positive, got {}", lr);
        Self::with_hyperparams(params, lr, 0.9, 0.999, 1e-8, 0.0)
    }

    pub fn with_hyperparams(
        params: Vec<tensor::Tensor<B, S, T>>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    ) -> Self {
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
}

impl<B, S, T> BaseOptimizer<B, S, T> for RAdam<B, S, T>
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

        self.t += 1;

        let lr_t = T::from(self.lr).unwrap();
        let beta1 = T::from(self.beta1).unwrap();
        let beta2 = T::from(self.beta2).unwrap();
        let eps = T::from(self.eps).unwrap();
        let weight_decay = T::from(self.weight_decay).unwrap();
        let one = T::from(1.0).unwrap();

        let t_val = self.t as f64;
        let beta1_t = self.beta1.powf(t_val);
        let beta2_t = self.beta2.powf(t_val);

        let bias_correction1 = 1.0 - beta1_t;
        let bias_correction2 = 1.0 - beta2_t;

        let rho_inf = 2.0 / (1.0 - self.beta2) - 1.0;
        let rho_t = rho_inf - (2.0 * t_val * beta2_t) / bias_correction2;

        let r_t = if rho_t > 4.0 {
            (((rho_t - 4.0) * (rho_t - 2.0) * rho_inf)
                / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                .sqrt()
        } else {
            0.0
        };

        let step_size = if rho_t > 4.0 {
            self.lr * r_t * bias_correction2.sqrt() / bias_correction1
        } else {
            self.lr / bias_correction1
        };


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
                let weight_decay_t: Tensor<B, S, T> = Tensor::from_vec_with_backend(vec![weight_decay], &[], param_state.param.backend().clone())
                     .map_err(|e| crate::OptimError::TensorError { source: e })?;
                let wd = mul(&param_state.param, &weight_decay_t)?;
                add(&grad, &wd)?
            } else {
                grad.clone()
            };

            let param_name = param_state.name.clone();

            {
                let m = param_state.get_state_mut("m").ok_or_else(|| {
                    crate::error::OptimError::InvalidState {
                        param_name: param_name.clone(),
                        state_key: "m".to_string(),
                    }
                })?;
                
                let beta1_t: Tensor<B, S, T> = Tensor::from_vec_with_backend(vec![beta1], &[], effective_grad.backend().clone())
                     .map_err(|e| crate::OptimError::TensorError { source: e })?;
                let beta1_m = mul(m, &beta1_t)?;
                
                let one_minus_beta1_t: Tensor<B, S, T> = Tensor::from_vec_with_backend(vec![one - beta1], &[], effective_grad.backend().clone())
                     .map_err(|e| crate::OptimError::TensorError { source: e })?;
                let one_minus_beta1_grad = mul(&effective_grad, &one_minus_beta1_t)?;
                *m = add(&beta1_m, &one_minus_beta1_grad)?;
            }

            {
                let v = param_state.get_state_mut("v").ok_or_else(|| {
                    crate::error::OptimError::InvalidState {
                        param_name,
                        state_key: "v".to_string(),
                    }
                })?;
                let grad_squared = mul(&effective_grad, &effective_grad)?;
                
                let beta2_t: Tensor<B, S, T> = Tensor::from_vec_with_backend(vec![beta2], &[], effective_grad.backend().clone())
                     .map_err(|e| crate::OptimError::TensorError { source: e })?;
                let beta2_v = mul(v, &beta2_t)?;
                
                let one_minus_beta2_t: Tensor<B, S, T> = Tensor::from_vec_with_backend(vec![one - beta2], &[], effective_grad.backend().clone())
                     .map_err(|e| crate::OptimError::TensorError { source: e })?;
                let one_minus_beta2_grad_sq = mul(&grad_squared, &one_minus_beta2_t)?;
                *v = add(&beta2_v, &one_minus_beta2_grad_sq)?;
            }

            let m_ref = param_state.get_state("m").unwrap();
            let v_ref = param_state.get_state("v").unwrap();

            let update = if rho_t > 4.0 {
                let eps_t: Tensor<B, S, T> = Tensor::from_vec_with_backend(vec![eps], &[], effective_grad.backend().clone())
                     .map_err(|e| crate::OptimError::TensorError { source: e })?;
                let v_plus_eps = add(v_ref, &eps_t)?;
                let v_sqrt = sqrt(&v_plus_eps)?;
                div(m_ref, &v_sqrt)?
            } else {
                m_ref.clone()
            };

            let step_size_tensor: Tensor<B, S, T> = Tensor::from_vec_with_backend(vec![T::from(step_size).unwrap()], &[], effective_grad.backend().clone())
                 .map_err(|e| crate::OptimError::TensorError { source: e })?;
            let scaled_update = mul(&update, &step_size_tensor)?;
            for (p, u) in param_state
                .param
                .as_mut_slice()
                .iter_mut()
                .zip(scaled_update.as_slice().iter().copied())
            {
                *p = *p - u;
            }
            updated += 1;
        }

        let _ = lr_t;
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
            for key in ["m", "v"] {
                let state_key = format!("{}.{}", param_state.name, key);
                if let Some(t) = state_dict.get(&state_key) {
                    param_state.init_state(key.to_string(), t.clone());
                }
            }
        }
        if let Some(step_tensor) = state_dict.get("step") {
            if let Some(&step_val) = step_tensor.as_slice().first() {
                self.t = step_val.to_f64().unwrap_or(0.0) as u64;
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

impl<B, S, T> Optimizer<B, S, T> for RAdam<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::Float + num_traits::FromPrimitive,
{
    fn name(&self) -> &str {
        "RAdam"
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
        let m = Tensor::zeros(&shape)
            .map_err(|e| crate::error::OptimError::TensorError { source: e })?;
        let v = Tensor::zeros(&shape)
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
