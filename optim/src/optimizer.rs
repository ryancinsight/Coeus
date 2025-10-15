//! Base optimizer traits and types

use crate::OptimError;
use std::collections::HashMap;

/// Base trait for all optimizers
pub trait BaseOptimizer<B, S, T>
where
    B: coeus_tensor::Backend,
    S: coeus_tensor::Storage<T>,
    T: coeus_tensor::DataType,
{
    /// Perform one optimization step
    fn step(&mut self) -> Result<(), OptimError>;

    /// Zero out gradients for all parameters
    fn zero_grad(&mut self);

    /// Add a parameter group to the optimizer
    fn add_param_group(&mut self, params: Vec<coeus_tensor::Tensor<B, S, T>>);

    /// Get the current learning rate
    fn get_lr(&self) -> f32;

    /// Set the learning rate
    fn set_lr(&mut self, lr: f32);

    /// Get optimizer state as a dictionary
    fn state_dict(&self) -> HashMap<String, coeus_tensor::Tensor<B, S, T>>;

    /// Load optimizer state from a dictionary
    fn load_state_dict(&mut self, state_dict: HashMap<String, coeus_tensor::Tensor<B, S, T>>) -> Result<(), OptimError>;
}

/// Parameter group configuration
#[derive(Debug, Clone)]
pub struct ParamGroup {
    pub params: Vec<String>, // Parameter names
    pub lr: f32,
    pub weight_decay: f32,
    pub maximize: bool,
}

/// Re-export the trait from optimizer_core
pub use crate::optimizer_core::Optimizer;
