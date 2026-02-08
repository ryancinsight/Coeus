//! Averaged Stochastic Gradient Descent (ASGD) optimizer.

use std::collections::HashMap;
use std::marker::PhantomData;

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};

use tensor::Tensor;

use crate::optimizer::{BaseOptimizer, ParamGroup};
use crate::optimizer_core::{Optimizer, ParamState};

/// Averaged Stochastic Gradient Descent (ASGD) optimizer.
#[derive(Debug)]
pub struct ASGD<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::FromPrimitive,
{
    param_states: Vec<ParamState<B, S, T>>,
    param_groups: Vec<ParamGroup<B, S, T>>,
    lr: f64,
    lambd: f64,
    alpha: f64,
    t0: f64,
    weight_decay: f64,
    _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> ASGD<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::FromPrimitive,
{
    pub fn new(lr: f64, lambd: f64, alpha: f64, t0: f64, weight_decay: f64) -> Self {
        Self {
            param_states: Vec::new(),
            param_groups: Vec::new(),
            lr,
            lambd,
            alpha,
            t0,
            weight_decay,
            _phantom: PhantomData,
        }
    }
}

impl<B, S, T> BaseOptimizer<B, S, T> for ASGD<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync,
    T: DataType + FloatExt + num_traits::FromPrimitive + core::ops::Add<Output = T> + core::ops::Sub<Output = T> + core::ops::Mul<Output = T> + core::ops::Div<Output = T>,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
{
    fn step(&mut self) -> crate::Result<usize> {
        <Self as Optimizer<B, S, T>>::step(self)
    }

    fn step_cpu(&mut self) -> crate::Result<usize> {
        <Self as Optimizer<B, S, T>>::step(self)
    }

    fn zero_grad(&mut self) {
        <Self as Optimizer<B, S, T>>::zero_grad(self);
    }

    fn add_param_group(&mut self, params: Vec<Tensor<B, S, T>>) {
        for tensor in params.clone().into_iter() {
            let mut param_state = ParamState::new(tensor.clone(), format!("param_{}", self.param_states.len()));
            let shape = tensor.shape().dims().to_vec();
            param_state.init_state("eta".to_string(), Tensor::from_vec_with_backend(vec![T::from_f64(self.lr).unwrap()], &[], tensor.backend().clone()).unwrap());
            param_state.init_state("mu".to_string(), Tensor::from_vec_with_backend(vec![T::from_f64(1.0).unwrap()], &[], tensor.backend().clone()).unwrap());
            param_state.init_state("ax".to_string(), Tensor::zeros(&shape).unwrap());
            param_state.init_state("step".to_string(), Tensor::from_vec_with_backend(vec![T::zero()], &[], tensor.backend().clone()).unwrap());
            self.param_states.push(param_state);
        }
        self.param_groups.push(ParamGroup::new(params, self.lr as f32, self.weight_decay as f32));
    }

    fn get_lr(&self) -> f32 { self.lr as f32 }
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr as f64;
        for group in &mut self.param_groups { group.lr = lr; }
    }

    fn state_dict(&self) -> HashMap<String, Tensor<B, S, T>> { <Self as Optimizer<B, S, T>>::state_dict(self) }
    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor<B, S, T>>) -> crate::Result<()> {
        <Self as Optimizer<B, S, T>>::load_state_dict(self, state_dict)
    }

    fn param_groups(&self) -> &[ParamGroup<B, S, T>] { &self.param_groups }
    fn param_groups_mut(&mut self) -> &mut [ParamGroup<B, S, T>] { &mut self.param_groups }
}

impl<B, S, T> Optimizer<B, S, T> for ASGD<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync,
    T: DataType + FloatExt + num_traits::FromPrimitive + core::ops::Add<Output = T> + core::ops::Sub<Output = T> + core::ops::Mul<Output = T> + core::ops::Div<Output = T>,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
{
    fn name(&self) -> &str { "ASGD" }
    fn parameters(&self) -> Vec<Tensor<B, S, T>> { self.param_states.iter().map(|ps| ps.param.clone()).collect() }
    fn named_parameters(&self) -> HashMap<String, Tensor<B, S, T>> { self.param_states.iter().map(|ps| (ps.name.clone(), ps.param.clone())).collect() }

    fn add_param(&mut self, param: &mut Tensor<B, S, T>, name: String) -> crate::Result<()> {
        let mut ps = ParamState::new(param.clone(), name);
        let shape = param.shape().dims().to_vec();
        ps.init_state("eta".to_string(), Tensor::from_vec_with_backend(vec![T::from_f64(self.lr).unwrap()], &[], param.backend().clone()).unwrap());
        ps.init_state("mu".to_string(), Tensor::from_vec_with_backend(vec![T::from_f64(1.0).unwrap()], &[], param.backend().clone()).unwrap());
        ps.init_state("ax".to_string(), Tensor::zeros(&shape).unwrap());
        ps.init_state("step".to_string(), Tensor::from_vec_with_backend(vec![T::zero()], &[], param.backend().clone()).unwrap());
        self.param_states.push(ps);
        Ok(())
    }

    fn remove_param(&mut self, name: &str) { self.param_states.retain(|ps| ps.name != name); }
    fn has_param(&self, name: &str) -> bool { self.param_states.iter().any(|ps| ps.name == name) }
    fn lr(&self) -> f64 { self.lr }
    fn set_lr(&mut self, lr: f64) -> crate::Result<()> { self.lr = lr; Ok(()) }
    fn weight_decay(&self) -> f64 { self.weight_decay }
    fn set_weight_decay(&mut self, wd: f64) -> crate::Result<()> { self.weight_decay = wd; Ok(()) }

    fn zero_grad(&mut self) {
        for ps in &mut self.param_states { let _ = ps.param.zero_grad(); }
    }

    fn step(&mut self) -> crate::Result<usize> {
        let lambd = T::from_f64(self.lambd).unwrap();
        let alpha = T::from_f64(self.alpha).unwrap();
        let t0 = T::from_f64(self.t0).unwrap();
        let one = T::from_f64(1.0).unwrap();

        for ps in &mut self.param_states {
            let grad = ps.param.grad().map_err(|_| crate::OptimError::GradientNotAvailable)?;
            
            // weight decay
            let effective_grad = if self.weight_decay > 0.0 {
                let wd_t: Tensor<B, DenseStorage<T>, T> = Tensor::from_vec_with_backend(vec![T::from_f64(self.weight_decay).unwrap()], &[], ps.param.backend().clone()).unwrap();
                tensor::ops::add(&grad, &tensor::ops::mul(&ps.param, &wd_t)?)?
            } else { grad };

            // Temporarily remove state tensors from the map to avoid borrow checker issues
            let mut step_t = ps.state.remove("step").ok_or_else(|| crate::OptimError::BackendError { message: "step state missing".to_string() })?;
            let mut eta_t = ps.state.remove("eta").ok_or_else(|| crate::OptimError::BackendError { message: "eta state missing".to_string() })?;
            let mut mu_t = ps.state.remove("mu").ok_or_else(|| crate::OptimError::BackendError { message: "mu state missing".to_string() })?;
            let mut ax = ps.state.remove("ax").ok_or_else(|| crate::OptimError::BackendError { message: "ax state missing".to_string() })?;

            {
                let step_val = step_t.storage().as_slice()[0] + one;
                step_t.storage_mut().as_mut_slice()[0] = step_val;

                let eta = eta_t.storage().as_slice()[0];
                let mu = mu_t.storage().as_slice()[0];

                // p = p * (1 - lambd * eta) - eta * g
                let decay = one - lambd * eta;
                let p_slice = ps.param.storage_mut().as_mut_slice();
                let g_slice = effective_grad.storage().as_slice();

                for (p, g) in p_slice.iter_mut().zip(g_slice.iter()) {
                    *p = *p * decay - eta * *g;
                }

                // Averaging
                if mu != one {
                    let ax_slice = ax.storage_mut().as_mut_slice();
                    for (a, p) in ax_slice.iter_mut().zip(p_slice.iter()) {
                        *a = *a + mu * (*p - *a);
                    }
                } else if step_val > t0 {
                     mu_t.storage_mut().as_mut_slice()[0] = T::from_f64(1.0).unwrap() / (step_val - t0);
                     ax.storage_mut().as_mut_slice().copy_from_slice(p_slice);
                }

                // update eta
                eta_t.storage_mut().as_mut_slice()[0] = T::from_f64(self.lr).unwrap() / (one + lambd * T::from_f64(self.lr).unwrap() * step_val).powf(alpha);
            }

            // Put them back
            ps.state.insert("step".to_string(), step_t);
            ps.state.insert("eta".to_string(), eta_t);
            ps.state.insert("mu".to_string(), mu_t);
            ps.state.insert("ax".to_string(), ax);
        }
        Ok(self.param_states.len())
    }

    fn state_dict(&self) -> HashMap<String, Tensor<B, S, T>> {
        let mut dict = HashMap::new();
        for ps in &self.param_states {
            for (k, v) in &ps.state {
                dict.insert(format!("{}.{}", ps.name, k), v.clone());
            }
        }
        dict
    }

    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor<B, S, T>>) -> crate::Result<()> {
        for ps in &mut self.param_states {
            for k in ["step", "eta", "mu", "ax"] {
                if let Some(v) = state_dict.get(&format!("{}.{}", ps.name, k)) {
                    ps.state.insert(k.to_string(), v.clone());
                }
            }
        }
        Ok(())
    }
}
