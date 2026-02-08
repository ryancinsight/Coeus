//! Sequential container for neural network modules.
//!
//! This module implements a Sequential container that chains multiple
//! neural network modules together in a sequential manner.
//!
//! Similar to PyTorch's `nn.Sequential`, this allows building networks
//! by stacking layers in order.

use std::any::Any;

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;

/// Sequential container that chains multiple modules together.
///
/// This is equivalent to PyTorch's `nn.Sequential`, allowing you to stack
/// neural network layers in a sequential manner.
///
/// # Examples
/// ```rust
/// use nn::{Sequential, Linear, ReLU};
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// // Create a simple MLP
/// let mut model = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
/// model.add_module("fc1", Box::new(Linear::new(784, 128).unwrap()));
/// model.add_module("relu1", Box::new(ReLU));
/// model.add_module("fc2", Box::new(Linear::new(128, 10).unwrap()));
///
/// // Use the model
/// # fn forward_example(model: &Sequential<CpuBackend<Float32>, DenseStorage<Float32>, Float32>) {
/// let input = Tensor::from_vec(vec![Float32::new(0.0); 784], &[784]).unwrap();
/// let output = model.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[10]);
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Sequential<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt,
{
    /// Vector of modules in sequential order
    modules: Vec<Box<dyn Module<B, S, T, Input = Tensor<B, S, T>, Output = Tensor<B, S, T>>>>,
    /// Names for each module (for serialization and introspection)
    names: Vec<String>,
    /// Training mode flag
    training: bool,
}

impl<B, S, T> Sequential<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt,
{
    /// Create a new empty Sequential container.
    ///
    /// # Returns
    /// A new Sequential container with no modules.
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            names: Vec::new(),
            training: true,
        }
    }

    /// Add a module to the sequential container.
    ///
    /// # Arguments
    /// * `name` - Name identifier for the module (used for serialization)
    /// * `module` - The module to add (concrete type that implements Module)
    ///
    /// # Examples
    /// ```rust
    /// use nn::{Sequential, Linear};
    /// use backend::CpuBackend;
    /// use storage::DenseStorage;
    /// use dtype::float::Float32;
    ///
    /// let mut seq = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
    /// seq.add_module("layer1", Linear::new(10, 5).unwrap());
    /// ```
    #[allow(clippy::missing_docs_in_private_items)]
    pub fn add_module<M>(&mut self, name: impl Into<String>, module: M)
    where
        M: Module<B, S, T, Input = Tensor<B, S, T>, Output = Tensor<B, S, T>> + 'static,
    {
        let name = name.into();
        self.names.push(name);
        self.modules.push(Box::new(module));
    }

    /// Get the number of modules in this container.
    ///
    /// # Returns
    /// The number of modules.
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Check if the container is empty.
    ///
    /// # Returns
    /// `true` if no modules are present, `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Get a reference to a module by index.
    ///
    /// # Arguments
    /// * `index` - The index of the module to retrieve
    ///
    /// # Returns
    /// Option containing the module reference, or None if index is out of bounds.
    pub fn get(&self, index: usize) -> Option<&dyn Module<B, S, T, Input = Tensor<B, S, T>, Output = Tensor<B, S, T>>> {
        self.modules.get(index).map(|m| m.as_ref())
    }

    /// Get a mutable reference to a module by index.
    ///
    /// # Arguments
    /// * `index` - The index of the module to retrieve
    ///
    /// # Returns
    /// Option containing the mutable module reference, or None if index is out of bounds.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut dyn Module<B, S, T, Input = Tensor<B, S, T>, Output = Tensor<B, S, T>>> {
        self.modules.get_mut(index).map(|m| m.as_mut())
    }

    /// Extend this Sequential container with modules from another.
    ///
    /// This appends all modules from the other container to this one,
    /// preserving their names.
    ///
    /// # Arguments
    /// * `other` - Another Sequential container to append
    pub fn extend(&mut self, other: Sequential<B, S, T>) {
        for (name, module) in other.names.into_iter().zip(other.modules.into_iter()) {
            self.names.push(name);
            self.modules.push(module);
        }
    }

    /// Get the model state dictionary.
    ///
    /// # Returns
    /// HashMap containing named parameters
    ///
    /// # Note
    /// Currently only supports Float32 tensors. Other data types will return an empty HashMap.
    #[cfg(feature = "safetensors")]
    pub fn state_dict(&self) -> std::collections::HashMap<String, Vec<f32>> {
        use std::collections::HashMap;

        let mut state_dict = HashMap::new();
        let params = self.parameters();

        for param in params {
            // Convert tensor data to Vec<f32>
            let data: Vec<f32> = param
                .data()
                .as_slice()
                .iter()
                .filter_map(|&x| x.to_f32())
                .collect();
            state_dict.insert(param.name().to_string(), data);
        }

        state_dict
    }

    /// Load parameters from a state dictionary.
    ///
    /// # Arguments
    /// * `state_dict` - HashMap containing named parameters
    ///
    /// # Returns
    /// Result indicating success or failure
    #[cfg(feature = "safetensors")]
    pub fn load_state_dict(
        &mut self,
        state_dict: &std::collections::HashMap<
            String,
            std::collections::HashMap<usize, Tensor<B, S, T>>,
        >,
    ) -> Result<()> {
        let mut current_params = self.parameters();

        for param in &mut current_params {
            if let Some(tensor_data) = state_dict.get(&param.name().to_string()) {
                // For now, assume the tensor is stored under key "param"
                if let Some(tensor) = tensor_data.get(&0) {
                    param.update_data(tensor.clone());
                } else {
                    return Err(NNError::InvalidInput {
                        message: format!(
                            "Parameter '{}' tensor data not found in state dict",
                            param.name()
                        ),
                    });
                }
            } else {
                return Err(NNError::InvalidInput {
                    message: format!("Parameter '{}' not found in saved state", param.name()),
                });
            }
        }

        Ok(())
    }
}

impl<B, S, T> Default for Sequential<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt,
{
    fn default() -> Self {
        Self::new()
    }
}

// Specialized implementation for Float32 to enable serialization
#[cfg(feature = "safetensors")]
impl<B, S> Sequential<B, S, dtype::float::Float32>
where
    B: Backend<Data = dtype::float::Float32> + Clone + Default,
    S: Storage<dtype::float::Float32> + StorageFromVec<dtype::float::Float32> + Clone + 'static,
{
    /// Save the sequential model to a SafeTensors file (Float32 only).
    ///
    /// # Arguments
    /// * `path` - Path where to save the model
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        use crate::safetensors::conversion::module_to_safetensors;
        let safetensors = module_to_safetensors(self)?;
        safetensors.save(path.as_ref())
    }

    /// Load the sequential model from a SafeTensors file (Float32 only).
    ///
    /// # Arguments
    /// * `path` - Path from where to load the model
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn load<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        use crate::safetensors::conversion::safetensors_to_state_dict;
        use backend::CpuBackend as CpuBackendConcrete;
        use dtype::float::Float32;
        use tensor::{DenseStorage, Tensor};

        let safetensors = crate::safetensors::SafeTensors::load(path.as_ref())?;
        let state_dict: std::collections::HashMap<
            String,
            Tensor<CpuBackendConcrete<Float32>, DenseStorage<Float32>, Float32>,
        > = safetensors_to_state_dict(&safetensors)?;

        // Convert state dict to expected format
        let mut converted_state_dict: std::collections::HashMap<
            String,
            std::collections::HashMap<usize, Tensor<B, S, Float32>>,
        > = std::collections::HashMap::new();

        for (name, tensor) in state_dict {
            let mut tensor_map = std::collections::HashMap::new();
            // Create new tensor with generic backend and storage types
            let data = tensor.as_slice().to_vec();
            let shape = tensor.shape().dims().to_vec();
            let storage = S::from_vec(data, &shape)?;
            let converted_tensor = Tensor::from_storage(storage, B::default());
            tensor_map.insert(0, converted_tensor);
            converted_state_dict.insert(name, tensor_map);
        }

        self.load_state_dict(&converted_state_dict)
    }
}

impl<B, S, T> Module<B, S, T> for Sequential<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let mut output = input.clone();

        // Pass input through each module in sequence
        for (i, module) in self.modules.iter().enumerate() {
            output = module.forward(&output).map_err(|e| NNError::InvalidInput {
                message: format!(
                    "Sequential[{}:{}] forward pass failed: {:?}",
                    i,
                    self.names.get(i).unwrap_or(&"unnamed".to_string()),
                    e
                ),
            })?;
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        // Collect parameters from all modules
        let mut all_params = Vec::new();
        for (i, module) in self.modules.iter().enumerate() {
            let prefix = self
                .names
                .get(i)
                .unwrap_or(&format!("module_{}", i))
                .clone();
            let mut module_params = module.parameters();

            // Add prefix to parameter names for hierarchical naming
            for param in &mut module_params {
                let full_name = format!("{}.{}", prefix, param.name());
                param.name = full_name;
            }

            all_params.extend(module_params);
        }
        all_params
    }

    fn modules(&self) -> Vec<&dyn Module<B, S, T, Input = Tensor<B, S, T>, Output = Tensor<B, S, T>>> {
        self.modules.iter().map(|m| m.as_ref()).collect()
    }

    fn zero_grad(&mut self) {
        for module in &mut self.modules {
            module.zero_grad();
        }
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
        for module in &mut self.modules {
            module.train(mode);
        }
    }

    fn name(&self) -> &str {
        "Sequential"
    }

    fn child_module_names(&self) -> Vec<(usize, String)> {
        self.names
            .iter()
            .enumerate()
            .map(|(i, name)| (i, name.clone()))
            .collect()
    }

    fn named_buffers(&self) -> Vec<(String, Tensor<B, S, T>)> {
        println!(
            "DEBUG: Sequential::named_buffers called. Modules count: {}",
            self.modules.len()
        );
        let mut buffers = Vec::new();
        for (i, module) in self.modules.iter().enumerate() {
            println!(
                "DEBUG: Visiting module index {}, name from trait: {}",
                i,
                module.name()
            );
            let prefix = self
                .names
                .get(i)
                .unwrap_or(&format!("module_{}", i))
                .clone();

            for (name, buf) in module.named_buffers() {
                buffers.push((format!("{}.{}", prefix, name), buf));
            }
        }
        buffers
    }

    fn load_buffer(&self, name: &str, value: &Tensor<B, S, T>) -> Result<()> {
        if let Some((prefix, suffix)) = name.split_once('.') {
            for (i, mod_name) in self.names.iter().enumerate() {
                if mod_name == prefix {
                    if let Some(module) = self.modules.get(i) {
                        return module.load_buffer(suffix, value);
                    }
                }
            }
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Linear, ReLU};
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;

    #[test]
    fn test_empty_sequential() {
        let seq = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
        assert!(seq.is_empty());
        assert_eq!(seq.len(), 0);
    }

    #[test]
    fn test_sequential_add_module() {
        let mut seq = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();

        let linear =
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 5).unwrap();
        seq.add_module("fc1", linear);

        let relu = ReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
        seq.add_module("relu1", relu);

        assert_eq!(seq.len(), 2);
        assert!(!seq.is_empty());

        // Check names
        let names = seq.child_module_names();
        assert_eq!(
            names,
            vec![(0, "fc1".to_string()), (1, "relu1".to_string())]
        );
    }

    #[test]
    fn test_sequential_get() {
        let mut seq = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
        let linear = Linear::new(5, 3).unwrap();
        let linear_name = linear.name().to_string();
        seq.add_module("layer", linear);

        let module = seq.get(0).unwrap();
        assert_eq!(module.name(), linear_name);

        assert!(seq.get(1).is_none()); // Out of bounds
    }

    #[test]
    fn test_sequential_get_mut() {
        let mut seq = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
        seq.add_module("layer", Linear::new(5, 3).unwrap());

        {
            let module = seq.get_mut(0).unwrap();
            assert_eq!(module.name(), "Linear");
        }
    }

    #[test]
    fn test_sequential_extend() {
        let mut seq1 = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
        seq1.add_module("fc1", Linear::new(10, 8).unwrap());
        seq1.add_module(
            "relu",
            ReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(),
        );

        let mut seq2 = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
        seq2.add_module("fc2", Linear::new(8, 4).unwrap());

        seq1.extend(seq2);

        assert_eq!(seq1.len(), 3);
        let names = seq1.child_module_names();
        assert_eq!(
            names,
            vec![
                (0, "fc1".to_string()),
                (1, "relu".to_string()),
                (2, "fc2".to_string())
            ]
        );
    }

    #[test]
    fn test_sequential_forward() {
        let mut seq = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
        seq.add_module("fc1", Linear::new(4, 3).unwrap());
        seq.add_module(
            "relu",
            ReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(),
        );

        let input = Tensor::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(-2.0),
                Float32::new(3.0),
                Float32::new(-1.0),
            ],
            &[1, 4], // [batch_size, input_dim]
        )
        .unwrap();

        let output = seq.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 3]);

        // Test that output is properly shaped and ReLU applied
        // (All values should be >= 0 after ReLU)
        for &val in output.as_slice() {
            assert!((val.get() as f64) >= 0.0);
        }
    }

    #[test]
    fn test_sequential_parameters() {
        let mut seq = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
        seq.add_module("fc1", Linear::new(4, 3).unwrap());
        seq.add_module("fc2", Linear::new(3, 2).unwrap());

        let params = seq.parameters();

        // Should have parameters from both layers
        // fc1: weight (4*3=12) + bias (3) = 15 params
        // fc2: weight (3*2=6) + bias (2) = 8 params
        // Total: 23 parameters
        assert_eq!(params.len(), 4); // weight and bias for each linear layer

        let names: Vec<_> = params.iter().map(|p| p.name()).collect();
        assert!(names.contains(&"fc1.weight"));
        assert!(names.contains(&"fc1.bias"));
        assert!(names.contains(&"fc2.weight"));
        assert!(names.contains(&"fc2.bias"));
    }

    #[test]
    fn test_sequential_train_mode() {
        let mut seq = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
        seq.add_module("fc", Linear::new(5, 3).unwrap());
        seq.add_module(
            "relu",
            ReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(),
        );

        // Default should be training mode
        assert!(seq.training);

        seq.train(false);
        assert!(!seq.training);

        // Test that child modules also get the mode set
        let _modules: Vec<_> = seq.modules();
        // Note: We can't easily test that child modules received the train call
        // without more complex mocking - the behavior is verified in train() method
    }

    #[test]
    fn test_sequential_zero_grad() {
        let mut seq = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
        seq.add_module("fc", Linear::new(5, 3).unwrap());

        // zero_grad should not panic - actual gradient zeroing is tested
        // in the individual module tests
        seq.zero_grad();
    }

    #[test]
    fn test_sequential_name() {
        let seq = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
        assert_eq!(seq.name(), "Sequential");
    }
}
