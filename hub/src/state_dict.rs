//! State dictionary management for model parameters

use coeus_tensor::Tensor;
use std::collections::HashMap;
use std::fmt;

/// A state dictionary containing model parameters
#[derive(Clone, Debug)]
pub struct StateDict {
    /// Parameter name to tensor mapping
    pub parameters: HashMap<String, Tensor<f32>>,
}

impl StateDict {
    /// Create a new empty state dictionary
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
        }
    }

    /// Create a state dictionary from a parameter map
    pub fn from_parameters(parameters: HashMap<String, Tensor<f32>>) -> Self {
        Self { parameters }
    }

    /// Get a parameter by name
    pub fn get(&self, name: &str) -> Option<&Tensor<f32>> {
        self.parameters.get(name)
    }

    /// Get a mutable reference to a parameter by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Tensor<f32>> {
        self.parameters.get_mut(name)
    }

    /// Insert a parameter
    pub fn insert(&mut self, name: String, tensor: Tensor<f32>) -> Option<Tensor<f32>> {
        self.parameters.insert(name, tensor)
    }

    /// Remove a parameter
    pub fn remove(&mut self, name: &str) -> Option<Tensor<f32>> {
        self.parameters.remove(name)
    }

    /// Check if the state dict contains a parameter
    pub fn contains_key(&self, name: &str) -> bool {
        self.parameters.contains_key(name)
    }

    /// Get the number of parameters
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    /// Check if the state dict is empty
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    /// Get all parameter names
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.parameters.keys()
    }

    /// Get all parameters
    pub fn values(&self) -> impl Iterator<Item = &Tensor<f32>> {
        self.parameters.values()
    }

    /// Get all parameter name-value pairs
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Tensor<f32>)> {
        self.parameters.iter()
    }

    /// Get mutable parameter name-value pairs
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&String, &mut Tensor<f32>)> {
        self.parameters.iter_mut()
    }

    /// Clear all parameters
    pub fn clear(&mut self) {
        self.parameters.clear();
    }

    /// Extend this state dict with another
    pub fn extend(&mut self, other: StateDict) {
        self.parameters.extend(other.parameters);
    }

    /// Create a subset of this state dict with the given keys
    pub fn subset(&self, keys: &[&str]) -> StateDict {
        let parameters = keys
            .iter()
            .filter_map(|key| {
                self.parameters
                    .get(*key)
                    .map(|tensor| ((*key).to_string(), tensor.clone()))
            })
            .collect();

        StateDict { parameters }
    }

    /// Apply a transformation function to all parameters
    pub fn transform<F>(&mut self, f: F)
    where
        F: Fn(&Tensor<f32>) -> Tensor<f32>,
    {
        for tensor in self.parameters.values_mut() {
            *tensor = f(tensor);
        }
    }
}

impl Default for StateDict {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for StateDict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "StateDict with {} parameters:", self.len())?;
        for (name, tensor) in &self.parameters {
            writeln!(f, "  {}: {:?}", name, tensor.shape())?;
        }
        Ok(())
    }
}

impl IntoIterator for StateDict {
    type Item = (String, Tensor<f32>);
    type IntoIter = std::collections::hash_map::IntoIter<String, Tensor<f32>>;

    fn into_iter(self) -> Self::IntoIter {
        self.parameters.into_iter()
    }
}

impl<'a> IntoIterator for &'a StateDict {
    type Item = (&'a String, &'a Tensor<f32>);
    type IntoIter = std::collections::hash_map::Iter<'a, String, Tensor<f32>>;

    fn into_iter(self) -> Self::IntoIter {
        self.parameters.iter()
    }
}

impl<'a> IntoIterator for &'a mut StateDict {
    type Item = (&'a String, &'a mut Tensor<f32>);
    type IntoIter = std::collections::hash_map::IterMut<'a, String, Tensor<f32>>;

    fn into_iter(self) -> Self::IntoIter {
        self.parameters.iter_mut()
    }
}
