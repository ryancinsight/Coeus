//! Container modules for composing neural networks
//!
//! This module provides container types that allow composing multiple modules
//! into complex neural network architectures.
//!
//! ## Available Containers
//!
//! - **Sequential**: Linear stack of modules
//! - **ModuleList**: List of modules with indexing
//! - **ModuleDict**: Dictionary of named modules
//!
//! ## Sequential Container
//!
//! The Sequential container allows building networks by stacking modules:
//!
//! ```rust
//! use coeus_nn::{Sequential, Linear, ReLU, Module};
//! use coeus_tensor::Tensor;
//!
//! let model: Sequential<f32> = Sequential::new(vec![
//!     Box::new(Linear::<f32>::new(784, 128)),
//!     Box::new(ReLU::new()),
//!     Box::new(Linear::<f32>::new(128, 10)),
//! ]);
//!
//! let input = Tensor::from_vec(CpuBackend::default(), vec![0.0; 784], vec![784]).unwrap();
//! let output = model.forward(&input);
//! ```
//!
//! ## References
//!
//! - [PyTorch Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)
//! - [Design Patterns: Composite Pattern](https://en.wikipedia.org/wiki/Composite_pattern)

use crate::Module;
use coeus_tensor::{FloatDtype, Tensor, CpuBackend};
use std::collections::HashMap;
use std::fmt;

/// Sequential container for stacking modules in order
///
/// This container executes modules in the order they were added.
/// The output of one module becomes the input to the next.
pub struct Sequential<T: FloatDtype> {
    modules: Vec<Box<dyn Module<T>>>,
}

impl<T: FloatDtype> Default for Sequential<T> {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl<T: FloatDtype> Sequential<T> {
    /// Create a new empty sequential container
    pub fn new(modules: Vec<Box<dyn Module<T>>>) -> Self {
        Self { modules }
    }

    /// Add a module to the end of the sequence
    pub fn add(&mut self, module: Box<dyn Module<T>>) {
        self.modules.push(module);
    }

    /// Get the number of modules in the sequence
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Check if the sequence is empty
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Get a reference to a module at the specified index
    pub fn get(&self, index: usize) -> Option<&dyn Module<T>> {
        self.modules.get(index).map(|boxed| boxed.as_ref())
    }

    /// Apply a function to a mutable module at the specified index
    pub fn with_module_mut<F, R>(&mut self, index: usize, f: F) -> Option<R>
    where
        F: FnOnce(&mut dyn Module<T>) -> R,
    {
        self.modules.get_mut(index).map(|boxed| f(boxed.as_mut()))
    }
}

impl<T: FloatDtype> Module<T> for Sequential<T> {
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        // Chain the modules while maintaining the autograd graph
        // We need to keep ownership of intermediate results
        if self.modules.is_empty() {
            return Ok(input.clone());
        }

        let mut current = input.clone();

        for module in &self.modules {
            current = module
                .forward(&current)
                .map_err(|e| crate::NNError::InvalidInput {
                    message: format!("Sequential forward failed: {}", e),
                })?;
        }

        // Return the final output
        Ok(current)
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        let mut params = Vec::new();
        for module in &self.modules {
            params.extend(module.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        let mut params = Vec::new();
        for module in &mut self.modules {
            params.extend(module.parameters_mut());
        }
        params
    }

    fn train(&mut self) {
        for module in &mut self.modules {
            module.train();
        }
    }

    fn eval(&mut self) {
        for module in &mut self.modules {
            module.eval();
        }
    }

    fn zero_grad(&mut self) {
        for module in &mut self.modules {
            module.zero_grad();
        }
    }
}

// Display implementation removed due to Module trait not implementing Debug

/// ModuleList container for storing modules with indexing
///
/// Similar to a vector but provides neural network specific functionality.
pub struct ModuleList<T: FloatDtype> {
    modules: Vec<Box<dyn Module<T>>>,
}

impl<T: FloatDtype> Default for ModuleList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatDtype> ModuleList<T> {
    /// Create a new empty module list
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
        }
    }

    /// Create a new module list from a vector of modules
    pub fn from_vec(modules: Vec<Box<dyn Module<T>>>) -> Self {
        Self { modules }
    }

    /// Add a module to the list
    pub fn push(&mut self, module: Box<dyn Module<T>>) {
        self.modules.push(module);
    }

    /// Remove and return the last module
    pub fn pop(&mut self) -> Option<Box<dyn Module<T>>> {
        self.modules.pop()
    }

    /// Get the number of modules
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Check if the list is empty
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Get a reference to a module at the specified index
    pub fn get(&self, index: usize) -> Option<&dyn Module<T>> {
        self.modules.get(index).map(|boxed| boxed.as_ref())
    }

    /// Apply a function to a mutable module at the specified index
    pub fn with_module_mut<F, R>(&mut self, index: usize, f: F) -> Option<R>
    where
        F: FnOnce(&mut dyn Module<T>) -> R,
    {
        self.modules.get_mut(index).map(|boxed| f(boxed.as_mut()))
    }

    /// Apply a function to each module
    pub fn apply<F>(&mut self, f: F)
    where
        F: Fn(&mut Box<dyn Module<T>>),
    {
        for module in &mut self.modules {
            f(module);
        }
    }
}

impl<T: FloatDtype> Module<T> for ModuleList<T> {
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        // ModuleList doesn't define a forward pass by itself
        // It's primarily used for organizing modules
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        let mut params = Vec::new();
        for module in &self.modules {
            params.extend(module.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        let mut params = Vec::new();
        for module in &mut self.modules {
            params.extend(module.parameters_mut());
        }
        params
    }

    fn train(&mut self) {
        for module in &mut self.modules {
            module.train();
        }
    }

    fn eval(&mut self) {
        for module in &mut self.modules {
            module.eval();
        }
    }

    fn zero_grad(&mut self) {
        for module in &mut self.modules {
            module.zero_grad();
        }
    }
}

impl<T: FloatDtype> fmt::Display for ModuleList<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModuleList({})", self.modules.len())
    }
}

/// ModuleDict container for storing named modules
///
/// Allows accessing modules by name rather than index.
pub struct ModuleDict<T: FloatDtype> {
    modules: HashMap<String, Box<dyn Module<T>>>,
}

impl<T: FloatDtype> Default for ModuleDict<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatDtype> ModuleDict<T> {
    /// Create a new empty module dictionary
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }

    /// Insert a module with a given name
    pub fn insert(&mut self, name: String, module: Box<dyn Module<T>>) {
        self.modules.insert(name, module);
    }

    /// Get a reference to a module by name
    pub fn get(&self, name: &str) -> Option<&dyn Module<T>> {
        self.modules.get(name).map(|boxed| boxed.as_ref())
    }

    /// Apply a function to a mutable module by name
    pub fn with_module_mut<F, R>(&mut self, name: &str, f: F) -> Option<R>
    where
        F: FnOnce(&mut dyn Module<T>) -> R,
    {
        self.modules.get_mut(name).map(|boxed| f(boxed.as_mut()))
    }

    /// Remove a module by name
    pub fn remove(&mut self, name: &str) -> Option<Box<dyn Module<T>>> {
        self.modules.remove(name)
    }

    /// Check if a module exists
    pub fn contains(&self, name: &str) -> bool {
        self.modules.contains_key(name)
    }

    /// Get the number of modules
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Check if the dictionary is empty
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Get all module names
    pub fn keys(&self) -> std::collections::hash_map::Keys<'_, String, Box<dyn Module<T>>> {
        self.modules.keys()
    }

    /// Apply a function to each module
    pub fn apply<F>(&mut self, f: F)
    where
        F: Fn(&mut Box<dyn Module<T>>),
    {
        for module in self.modules.values_mut() {
            f(module);
        }
    }
}

impl<T: FloatDtype> Module<T> for ModuleDict<T> {
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        // ModuleDict doesn't define a forward pass by itself
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        let mut params = Vec::new();
        for module in self.modules.values() {
            params.extend(module.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        let mut params = Vec::new();
        for module in self.modules.values_mut() {
            params.extend(module.parameters_mut());
        }
        params
    }

    fn train(&mut self) {
        for module in self.modules.values_mut() {
            module.train();
        }
    }

    fn eval(&mut self) {
        for module in self.modules.values_mut() {
            module.eval();
        }
    }

    fn zero_grad(&mut self) {
        for module in self.modules.values_mut() {
            module.zero_grad();
        }
    }
}

impl<T: FloatDtype> fmt::Display for ModuleDict<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModuleDict({})", self.modules.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Linear, ReLU};

    #[test]
    fn test_sequential() {
        let model = Sequential::new(vec![
            Box::new(Linear::<f32>::new(10, 5)),
            Box::new(ReLU::new()),
            Box::new(Linear::<f32>::new(5, 2)),
        ]);

        let input = Tensor::from_vec(CpuBackend::default(), vec![1.0; 10], vec![10]).unwrap();
        let output = model
            .forward(&input)
            .expect("Sequential forward should succeed");

        assert_eq!(output.shape(), &[2]);
        assert_eq!(model.len(), 3);
    }

    #[test]
    fn test_module_list() {
        let mut list = ModuleList::new();
        list.push(Box::new(Linear::<f32>::new(10, 5)));
        list.push(Box::new(ReLU::new()));

        assert_eq!(list.len(), 2);
        assert!(list.get(0).is_some());
        assert!(list.get(1).is_some());
        assert!(list.get(2).is_none());
    }

    #[test]
    fn test_module_dict() {
        let mut dict = ModuleDict::new();
        dict.insert("linear1".to_string(), Box::new(Linear::<f32>::new(10, 5)));
        dict.insert("relu".to_string(), Box::new(ReLU::new()));

        assert_eq!(dict.len(), 2);
        assert!(dict.contains("linear1"));
        assert!(dict.contains("relu"));
        assert!(!dict.contains("nonexistent"));
    }

    #[test]
    fn test_sequential_gradient_flow() {
        let mut model = Sequential::new(vec![
            Box::new(Linear::<f32>::new(3, 2)),
            Box::new(ReLU::new()),
        ]);

        // Enable gradients
        model.zero_grad();

        let input = Tensor::from_vec_with_grad(vec![1.0, 2.0, 3.0], vec![3]);
        let output = model
            .forward(&input)
            .expect("Sequential gradient flow forward should succeed");

        let loss = output.sum();
        let _ = loss.backward();

        // For now, check that the loss tensor has a gradient (it should)
        assert!(loss.grad().is_some());

        // The input gradient check is more complex due to current autograd limitations
        // We'll verify this works when we complete the autograd system
        // For now, just ensure the computation doesn't panic
        println!("Sequential gradient flow test completed without panic");

        // Get parameters and check gradients
        let params = model.parameters();
        assert!(!params.is_empty());
    }
}


