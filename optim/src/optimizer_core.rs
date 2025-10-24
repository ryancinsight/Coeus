//! Core optimizer structures and traits.
//!
//! This module contains the fundamental traits and structures shared
//! across all optimizer implementations.

use std::collections::HashMap;
use std::fmt;

use coeus_backend::Backend;
use coeus_dtype::DataType;
use coeus_storage::{Storage, StorageFromVec};
use coeus_tensor::Tensor;

use crate::Parameter;

/// Core trait for all optimizers.
///
/// This trait defines the interface that all optimizers must implement.
/// It provides methods for parameter management, state updates, and optimization steps.
pub trait Optimizer<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + coeus_dtype::traits::FloatExt,
{
    /// Get the name of this optimizer
    fn name(&self) -> &str;

    /// Get all parameters being optimized
    fn parameters(&self) -> Vec<Parameter<B, S, T>>;

    /// Get all parameters with their names
    fn named_parameters(&self) -> HashMap<String, Parameter<B, S, T>>;

    /// Add a parameter to be optimized
    ///
    /// # Errors
    /// Returns an error if the parameter is invalid or already exists
    fn add_param(
        &mut self,
        param: &mut Parameter<B, S, T>,
        name: String,
    ) -> Result<(), crate::error::OptimError>;

    /// Remove a parameter from optimization
    fn remove_param(&mut self, name: &str);

    /// Check if a parameter exists
    fn has_param(&self, name: &str) -> bool;

    /// Get the learning rate
    fn lr(&self) -> f64;

    /// Set the learning rate
    ///
    /// # Errors
    /// Returns an error if lr <= 0
    fn set_lr(&mut self, lr: f64) -> Result<(), crate::error::OptimError>;

    /// Get the learning rate (alias for lr)
    fn learning_rate(&self) -> f64 {
        self.lr()
    }

    /// Set the learning rate (alias for set_lr)
    fn set_learning_rate(&mut self, lr: f64) -> Result<(), crate::error::OptimError> {
        self.set_lr(lr)
    }

    /// Get the weight decay (L2 regularization)
    fn weight_decay(&self) -> f64;

    /// Set the weight decay
    ///
    /// # Errors
    /// Returns an error if weight_decay < 0
    fn set_weight_decay(&mut self, weight_decay: f64) -> Result<(), crate::error::OptimError>;

    /// Zero all parameter gradients
    fn zero_grad(&mut self);

    /// Perform one optimization step
    ///
    /// This method updates all parameters based on their gradients
    /// using the specific optimization algorithm.
    ///
    /// # Returns
    /// Returns the number of parameters updated on success
    ///
    /// # Errors
    /// Returns an error if any parameter update fails
    fn step(&mut self) -> Result<usize, crate::error::OptimError>;

    /// Get current parameter values (for debugging/state inspection)
    fn state_dict(&self) -> HashMap<String, Tensor<B, S, T>>;

    /// Load parameter values from state dict
    ///
    /// # Errors
    /// Returns an error if parameter shapes don't match or parameters are missing
    fn load_state_dict(
        &mut self,
        state_dict: HashMap<String, Tensor<B, S, T>>,
    ) -> Result<(), crate::error::OptimError>;
}

/// Common optimizer hyperparameters
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Learning rate
    pub lr: f64,
    /// Weight decay (L2 regularization coefficient)
    pub weight_decay: f64,
    /// Whether to use weight decay
    pub use_weight_decay: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            weight_decay: 0.0,
            use_weight_decay: false,
        }
    }
}

impl OptimizerConfig {
    /// Create a new optimizer configuration
    pub fn new(lr: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            weight_decay,
            use_weight_decay: weight_decay > 0.0,
        }
    }

    /// Set learning rate
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    /// Set weight decay
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self.use_weight_decay = weight_decay > 0.0;
        self
    }
}

/// Parameter state for optimizers that maintain per-parameter state
///
/// This is used by optimizers like Adam, RMSprop, etc. that need to
/// track momentum, variance, or other statistics per parameter.
#[derive(Debug, Clone)]
pub struct ParamState<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType,
{
    /// Parameter name
    pub name: String,
    /// Parameter tensor
    pub param: Parameter<B, S, T>,
    /// Gradient tensor (if available)
    pub grad: Option<Tensor<B, S, T>>,
    /// Optimizer-specific state (e.g., momentum, variance)
    pub state: HashMap<String, Tensor<B, S, T>>,
}

impl<B, S, T> ParamState<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType,
{
    /// Create a new parameter state
    pub fn new(param: Parameter<B, S, T>, name: String) -> Self {
        Self {
            name,
            param,
            grad: None,
            state: HashMap::new(),
        }
    }

    /// Update gradient
    pub fn set_grad(&mut self, grad: Option<Tensor<B, S, T>>) {
        self.grad = grad;
    }

    /// Get gradient (returns error if no gradient available)
    pub fn grad(&self) -> Result<&Tensor<B, S, T>, crate::error::OptimError> {
        self.grad
            .as_ref()
            .ok_or(crate::error::OptimError::GradientNotAvailable)
    }

    /// Check if parameter has gradient
    pub fn has_grad(&self) -> bool {
        self.grad.is_some()
    }

    /// Initialize optimizer state for this parameter
    pub fn init_state(&mut self, key: String, tensor: Tensor<B, S, T>) {
        self.state.insert(key, tensor);
    }

    /// Get optimizer state
    pub fn get_state(&self, key: &str) -> Option<&Tensor<B, S, T>> {
        self.state.get(key)
    }

    /// Get mutable optimizer state
    pub fn get_state_mut(&mut self, key: &str) -> Option<&mut Tensor<B, S, T>> {
        self.state.get_mut(key)
    }

    /// Clear all optimizer state
    pub fn clear_state(&mut self) {
        self.state.clear();
    }
}

impl<B, S, T> fmt::Display for ParamState<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ParamState(name={}, param_shape={:?}, has_grad={}, state_keys={})",
            self.name,
            self.param.shape().dims(),
            self.has_grad(),
            self.state.len()
        )
    }
}
