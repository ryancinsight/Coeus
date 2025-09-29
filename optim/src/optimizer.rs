//! Core optimizer trait and parameter group management
//!
//! Defines the core `Optimizer` trait and `ParamGroup` structure
//! that all optimization algorithms implement.

use crate::Result;
use coeus_tensor::{Backend, Tensor, CpuBackend};
use std::collections::HashMap;

/// A group of parameters with their optimization settings
///
/// Compatible with PyTorch's parameter group concept
#[derive(Clone)]
pub struct ParamGroup<T: coeus_dtype::FloatDtype, B: Backend<T> + Clone = coeus_tensor::CpuBackend> {
    /// Parameters in this group
    pub params: Vec<Tensor<T, B>>,
    /// Learning rate for this group
    pub lr: T,
    /// Weight decay (L2 regularization) for this group
    pub weight_decay: T,
    /// Additional optimizer-specific parameters
    pub options: HashMap<String, T>,
}

impl<T: coeus_dtype::FloatDtype, B: Backend<T> + Clone> ParamGroup<T, B> {
    /// Create a new parameter group
    pub fn new(params: Vec<Tensor<T, B>>, lr: T, weight_decay: T) -> Self {
        Self {
            params,
            lr,
            weight_decay,
            options: HashMap::new(),
        }
    }

    /// Create a parameter group from a slice of parameters
    pub fn from_params(params: &[Tensor<T, B>], lr: T, weight_decay: T) -> Self {
        Self::new(params.to_vec(), lr, weight_decay)
    }

    /// Add an optimizer-specific option
    pub fn with_option(mut self, key: impl Into<String>, value: T) -> Self {
        self.options.insert(key.into(), value);
        self
    }

    /// Get an optimizer-specific option
    pub fn get_option(&self, key: &str) -> Option<&T> {
        self.options.get(key)
    }

    /// Get all parameters in this group
    pub fn parameters(&self) -> &[Tensor<T, B>] {
        &self.params
    }

    /// Get mutable access to parameters
    pub fn parameters_mut(&mut self) -> &mut [Tensor<T, B>] {
        &mut self.params
    }
}

/// Core optimizer trait
///
/// Defines the interface that all optimization algorithms must implement.
/// Compatible with PyTorch's optimizer interface.
pub trait Optimizer<T: coeus_dtype::FloatDtype, B: Backend<T> + Clone = coeus_tensor::CpuBackend> {
    /// Get the optimizer's name
    fn name(&self) -> &str;

    /// Get the parameter groups
    fn param_groups(&self) -> &[ParamGroup<T, B>];

    /// Get mutable access to parameter groups
    fn param_groups_mut(&mut self) -> &mut [ParamGroup<T, B>];

    /// Add a new parameter group
    fn add_param_group(&mut self, param_group: ParamGroup<T, B>);

    /// Perform a single optimization step
    ///
    /// This method updates all parameters based on their gradients.
    /// Should be called after computing gradients via backpropagation.
    fn step(&mut self) -> Result<()>;

    /// Zero out all parameter gradients
    ///
    /// This should be called at the beginning of each training iteration
    /// to clear the gradients from the previous iteration.
    fn zero_grad(&mut self);

    /// Get the current learning rate for a parameter group
    fn get_lr(&self, group_index: usize) -> Option<T>;

    /// Set the learning rate for a parameter group
    fn set_lr(&mut self, group_index: usize, lr: T) -> Result<()>;

    /// Get optimizer state for a parameter
    fn state(&self) -> &HashMap<String, Tensor<T, B>>;

    /// Get mutable optimizer state
    fn state_mut(&mut self) -> &mut HashMap<String, Tensor<T, B>>;

    /// Get the default parameter group
    fn default_param_group(&self) -> &ParamGroup<T, B> {
        &self.param_groups()[0]
    }

    /// Get mutable access to the default parameter group
    fn default_param_group_mut(&mut self) -> &mut ParamGroup<T, B> {
        &mut self.param_groups_mut()[0]
    }

    /// Get all parameters across all groups
    fn parameters(&self) -> Vec<&Tensor<T, B>> {
        self.param_groups()
            .iter()
            .flat_map(|group| group.parameters())
            .collect()
    }

    /// Get mutable access to all parameters
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, B>> {
        self.param_groups_mut()
            .iter_mut()
            .flat_map(|group| group.parameters_mut())
            .collect()
    }
}

/// Base optimizer struct that can be extended by specific algorithms
///
/// Provides common functionality that most optimizers need
pub struct BaseOptimizer<T: coeus_dtype::FloatDtype, B: Backend<T> + Clone = coeus_tensor::CpuBackend> {
    /// Parameter groups
    param_groups: Vec<ParamGroup<T, B>>,
    /// Optimizer state (momentum, variance, etc.)
    state: HashMap<String, Tensor<T, B>>,
}

impl<T: coeus_dtype::FloatDtype, B: Backend<T> + Clone> BaseOptimizer<T, B> {
    /// Create a new base optimizer
    pub fn new(param_groups: Vec<ParamGroup<T, B>>) -> Self {
        Self {
            param_groups,
            state: HashMap::new(),
        }
    }

    /// Create a base optimizer with default parameter group
    pub fn with_defaults(params: Vec<Tensor<T, B>>, lr: T, weight_decay: T) -> Self {
        let param_group = ParamGroup::new(params, lr, weight_decay);
        Self::new(vec![param_group])
    }
}

impl<T: coeus_dtype::FloatDtype, B: Backend<T> + Clone> Optimizer<T, B> for BaseOptimizer<T, B> {
    fn name(&self) -> &str {
        "BaseOptimizer"
    }

    fn param_groups(&self) -> &[ParamGroup<T, B>] {
        &self.param_groups
    }

    fn param_groups_mut(&mut self) -> &mut [ParamGroup<T, B>] {
        &mut self.param_groups
    }

    fn add_param_group(&mut self, param_group: ParamGroup<T, B>) {
        self.param_groups.push(param_group);
    }

    fn step(&mut self) -> Result<()> {
        // Base implementation does nothing
        // Subclasses should override this
        Ok(())
    }

    fn zero_grad(&mut self) {
        for group in &mut self.param_groups {
            for param in &mut group.params {
                if param.grad().is_some() {
                    // Zero out the gradient using the built-in method
                    param.zero_grad();
                }
            }
        }
    }

    fn get_lr(&self, group_index: usize) -> Option<T> {
        self.param_groups.get(group_index).map(|group| group.lr)
    }

    fn set_lr(&mut self, group_index: usize, lr: T) -> Result<()> {
        if let Some(group) = self.param_groups.get_mut(group_index) {
            group.lr = lr;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Parameter group index {} out of bounds", group_index).into())
        }
    }

    fn state(&self) -> &HashMap<String, Tensor<T, B>> {
        &self.state
    }

    fn state_mut(&mut self) -> &mut HashMap<String, Tensor<T, B>> {
        &mut self.state
    }
}

/// Helper function to check if a parameter requires gradients
pub fn param_requires_grad<T: coeus_dtype::Dtype>(param: &Tensor<T, CpuBackend>) -> bool {
    param.requires_grad()
}

/// Helper function to get parameter gradient
pub fn param_grad<T: coeus_dtype::Dtype>(param: &Tensor<T, CpuBackend>) -> Option<Tensor<T, CpuBackend>> {
    param.grad()
}

/// Helper function to get parameter gradient (returns owned copy)
pub fn param_grad_mut<T: coeus_dtype::Dtype, B: Backend<T> + Clone>(param: &Tensor<T, B>) -> Option<Tensor<T, B>> {
    param.grad()
}
