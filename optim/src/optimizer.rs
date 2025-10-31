//! Base optimizer traits and types

use crate::OptimError;
use std::collections::HashMap;

use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// Base trait for all optimizers
pub trait BaseOptimizer<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType,
{
    /// Perform one optimization step
    /// Returns the number of parameters updated
    fn step(&mut self) -> Result<usize, OptimError>;

    /// Perform one optimization step using CPU computation
    /// Returns the number of parameters updated
    fn step_cpu(&mut self) -> Result<usize, OptimError>;

    /// Zero out gradients for all parameters
    fn zero_grad(&mut self);

    /// Add a parameter group to the optimizer
    fn add_param_group(&mut self, params: Vec<tensor::Tensor<B, S, T>>);

    /// Get the current learning rate
    fn get_lr(&self) -> f32;

    /// Set the learning rate
    fn set_lr(&mut self, lr: f32);

    /// Get optimizer state as a dictionary
    fn state_dict(&self) -> HashMap<String, tensor::Tensor<B, S, T>>;

    /// Load optimizer state from a dictionary
    fn load_state_dict(
        &mut self,
        state_dict: HashMap<String, tensor::Tensor<B, S, T>>,
    ) -> Result<(), OptimError>;

    /// Get parameter groups
    fn param_groups(&self) -> &[ParamGroup<B, S, T>];

    /// Get mutable parameter groups
    fn param_groups_mut(&mut self) -> &mut [ParamGroup<B, S, T>];
}

/// Parameter group configuration
#[derive(Debug, Clone)]
pub struct ParamGroup<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType,
{
    pub params: Vec<crate::Parameter<B, S, T>>, // Parameter tensors
    pub lr: f32,
    pub weight_decay: f32,
    pub maximize: bool,
}

impl<B, S, T> ParamGroup<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType,
{
    /// Create a new parameter group
    pub fn new(params: Vec<crate::Parameter<B, S, T>>, lr: f32, weight_decay: f32) -> Self {
        Self {
            params,
            lr,
            weight_decay,
            maximize: false,
        }
    }

    /// Get the parameters
    pub fn parameters(&self) -> &[crate::Parameter<B, S, T>] {
        &self.params
    }

    /// Get mutable access to parameters
    pub fn parameters_mut(&mut self) -> &mut [crate::Parameter<B, S, T>] {
        &mut self.params
    }
}

/// Re-export the trait from optimizer_core
pub use crate::optimizer_core::Optimizer;
