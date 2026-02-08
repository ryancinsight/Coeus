//! Resilient Backpropagation (Rprop) optimizer.

use std::collections::HashMap;
use std::marker::PhantomData;

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::optimizer::{BaseOptimizer, ParamGroup};
use crate::optimizer_core::{Optimizer, ParamState};

/// Resilient Backpropagation (Rprop) optimizer.
#[derive(Debug)]
pub struct Rprop<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::FromPrimitive,
{
    param_states: Vec<ParamState<B, S, T>>,
    param_groups: Vec<ParamGroup<B, S, T>>,
    lr: f64,
    etas: (f64, f64),
    step_sizes: (f64, f64),
    _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> Rprop<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::FromPrimitive,
{
    pub fn new(lr: f64, etas: (f64, f64), step_sizes: (f64, f64)) -> Self {
        Self {
            param_states: Vec::new(),
            param_groups: Vec::new(),
            lr,
            etas,
            step_sizes,
            _phantom: PhantomData,
        }
    }
}

impl<B, S, T> BaseOptimizer<B, S, T> for Rprop<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync,
    T: DataType + FloatExt + num_traits::FromPrimitive + core::ops::Add<Output = T> + core::ops::Sub<Output = T> + core::ops::Mul<Output = T> + core::ops::Div<Output = T>,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
{
    fn step(&mut self) -> crate::Result<usize> { <Self as Optimizer<B, S, T>>::step(self) }
    fn step_cpu(&mut self) -> crate::Result<usize> { <Self as Optimizer<B, S, T>>::step(self) }
    fn zero_grad(&mut self) { <Self as Optimizer<B, S, T>>::zero_grad(self); }

    fn add_param_group(&mut self, params: Vec<Tensor<B, S, T>>) {
        for tensor in params.clone().into_iter() {
            let mut ps = ParamState::new(tensor.clone(), format!("param_{}", self.param_states.len()));
            let shape = tensor.shape().dims().to_vec();
            ps.init_state("step_size".to_string(), Tensor::full(&shape, T::from_f64(self.lr).unwrap()).unwrap());
            ps.init_state("prev_grad".to_string(), Tensor::zeros(&shape).unwrap());
            self.param_states.push(ps);
        }
        self.param_groups.push(ParamGroup::new(params, self.lr as f32, 0.0));
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

impl<B, S, T> Optimizer<B, S, T> for Rprop<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync,
    T: DataType + FloatExt + num_traits::FromPrimitive + core::ops::Add<Output = T> + core::ops::Sub<Output = T> + core::ops::Mul<Output = T> + core::ops::Div<Output = T>,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
{
    fn name(&self) -> &str { "Rprop" }
    fn parameters(&self) -> Vec<Tensor<B, S, T>> { self.param_states.iter().map(|ps| ps.param.clone()).collect() }
    fn named_parameters(&self) -> HashMap<String, Tensor<B, S, T>> { self.param_states.iter().map(|ps| (ps.name.clone(), ps.param.clone())).collect() }

    fn add_param(&mut self, param: &mut Tensor<B, S, T>, name: String) -> crate::Result<()> {
        let mut ps = ParamState::new(param.clone(), name);
        let shape = param.shape().dims().to_vec();
        ps.init_state("step_size".to_string(), Tensor::full(&shape, T::from_f64(self.lr).unwrap()).unwrap());
        ps.init_state("prev_grad".to_string(), Tensor::zeros(&shape).unwrap());
        self.param_states.push(ps);
        Ok(())
    }

    fn remove_param(&mut self, name: &str) { self.param_states.retain(|ps| ps.name != name); }
    fn has_param(&self, name: &str) -> bool { self.param_states.iter().any(|ps| ps.name == name) }
    fn lr(&self) -> f64 { self.lr }
    fn set_lr(&mut self, lr: f64) -> crate::Result<()> { self.lr = lr; Ok(()) }
    fn weight_decay(&self) -> f64 { 0.0 }
    fn set_weight_decay(&mut self, _wd: f64) -> crate::Result<()> { Ok(()) }

    fn zero_grad(&mut self) {
        for ps in &mut self.param_states { let _ = ps.param.zero_grad(); }
    }

    fn step(&mut self) -> crate::Result<usize> {
        let (eta_minus, eta_plus) = (T::from_f64(self.etas.0).unwrap(), T::from_f64(self.etas.1).unwrap());
        let (step_min, step_max) = (T::from_f64(self.step_sizes.0).unwrap(), T::from_f64(self.step_sizes.1).unwrap());
        let zero = T::zero();

        for ps in &mut self.param_states {
            let grad = ps.param.grad().map_err(|_| crate::OptimError::GradientNotAvailable)?;
            
            // Temporarily remove from state map to avoid borrow checker issues with multiple mutable borrows
            let mut step_size = ps.state.remove("step_size").ok_or_else(|| crate::OptimError::BackendError { message: "step_size state missing".to_string() })?;
            let mut prev_grad = ps.state.remove("prev_grad").ok_or_else(|| crate::OptimError::BackendError { message: "prev_grad state missing".to_string() })?;

            {
                let p_slice = ps.param.storage_mut().as_mut_slice();
                let g_slice = grad.storage().as_slice();
                let ss_slice = step_size.as_mut_slice();
                let pg_slice = prev_grad.as_mut_slice();

                for i in 0..p_slice.len() {
                    let sign = g_slice[i] * pg_slice[i];
                    if sign > zero {
                        ss_slice[i] = (ss_slice[i] * eta_plus).min(step_max);
                        p_slice[i] = p_slice[i] - g_slice[i].signum() * ss_slice[i];
                        pg_slice[i] = g_slice[i];
                    } else if sign < zero {
                        ss_slice[i] = (ss_slice[i] * eta_minus).max(step_min);
                        pg_slice[i] = zero;
                    } else {
                        p_slice[i] = p_slice[i] - g_slice[i].signum() * ss_slice[i];
                        pg_slice[i] = g_slice[i];
                    }
                }
            }

            // Put them back
            ps.state.insert("step_size".to_string(), step_size);
            ps.state.insert("prev_grad".to_string(), prev_grad);
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
            if let Some(ss) = state_dict.get(&format!("{}.step_size", ps.name)) {
                ps.state.insert("step_size".to_string(), ss.clone());
            }
            if let Some(pg) = state_dict.get(&format!("{}.prev_grad", ps.name)) {
                ps.state.insert("prev_grad".to_string(), pg.clone());
            }
        }
        Ok(())
    }
}
