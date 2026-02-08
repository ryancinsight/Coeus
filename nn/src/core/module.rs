//! Module trait and utilities for neural network components.

use std::collections::HashMap;

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

#[cfg(feature = "safetensors")]
use crate::core::error::NNError;
use crate::core::error::Result;
use crate::core::parameter::Parameter;

/// Core trait for neural network modules.
///
/// All neural network components must implement this trait to provide:
/// - Forward pass computation
/// - Parameter management
/// - Training mode control
/// - Module introspection
///
/// # Type Parameters
/// - `B`: The backend type (e.g., `CpuBackend`)
/// - `T`: The data type (e.g., `Float32`)
///
/// # Examples
/// ```rust
/// use nn::{Module, Parameter, error};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
///
/// struct MyModule {
///     weight: Parameter<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
/// }
///
/// impl Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32> for MyModule {
///     fn forward(&self, input: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>)
///         -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, error::NNError> {
///         // Implementation here
///         Ok(Tensor::zeros(&input.shape().dims())?) // Placeholder
///     }
///
///     fn parameters(&self) -> Vec<Parameter<CpuBackend<Float32>, DenseStorage<Float32>, Float32>> {
///         vec![self.weight.clone()]
///     }
///
///     fn modules(&self) -> Vec<&dyn Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>> {
///         vec![]
///     }
///
///     fn zero_grad(&mut self) {
///         // Implementation here
///     }
///
///     fn train(&mut self, _mode: bool) {
///         // Implementation here
///     }
///
///     fn name(&self) -> &str {
///         "MyModule"
///     }
/// }
/// ```
pub trait Module<B, S, T>: std::fmt::Debug + std::any::Any
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt,
{
    /// Input type for the module's forward pass.
    type Input;
    /// Output type for the module's forward pass.
    type Output;

    /// Perform forward pass through the module.
    ///
    /// # Arguments
    /// * `input` - Input to the module
    ///
    /// # Returns
    /// Result containing the output, or an error if the forward pass fails.
    fn forward(&self, input: &Self::Input) -> Result<Self::Output>;

    /// Perform forward pass with autograd support.
    fn forward_autograd(&self, input: &Self::Input) -> Result<Self::Output>
    where
        T: FloatExt + From<f64>,
        B: Backend<Data = T> + Default + 'static,
        S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + 'static,
    {
        // Default implementation: just call regular forward
        self.forward(input)
    }

    /// Get all learnable parameters in this module.
    ///
    /// This includes parameters from this module and all submodules.
    /// Parameters are returned in a consistent order for gradient updates.
    ///
    /// # Returns
    /// Vector of all parameters in this module hierarchy.
    fn parameters(&self) -> Vec<Parameter<B, S, T>>;

    /// Get mutable references to all parameters for gradient updates.
    ///
    /// This is used by optimizers to update parameter values during training.
    /// The default implementation returns an empty vector; concrete modules
    /// should override to provide mutable parameter access.
    ///
    /// # Returns
    /// Vector of mutable parameter references.
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<B, S, T>> {
        Vec::new() // Default: no mutable parameters
    }

    /// Get all submodules in this module.
    ///
    /// # Returns
    /// Vector of references to child modules.
    /// Note: Submodules must all have Tensor -> Tensor signature for object safety in this vector.
    fn modules(&self) -> Vec<&dyn Module<B, S, T, Input = Tensor<B, S, T>, Output = Tensor<B, S, T>>> {
        Vec::new() // Default implementation for leaf modules
    }

    /// Get mutable references to all submodules.
    ///
    /// This is used for recursive operations like zero_grad() and train().
    ///
    /// # Returns
    /// Vector of mutable submodule references.
    fn modules_mut(&mut self) -> Vec<&mut dyn Module<B, S, T, Input = Tensor<B, S, T>, Output = Tensor<B, S, T>>> {
        Vec::new() // Default: no mutable submodules
    }

    /// Zero all gradients in this module and submodules.
    ///
    /// This should recursively call `zero_grad()` on all parameters.
    fn zero_grad(&mut self) {
        // Zero gradients on all parameters
        for param in self.parameters_mut() {
            param.zero_grad();
        }

        // Recursively zero gradients on submodules
        for module in self.modules_mut() {
            module.zero_grad();
        }
    }

    /// Set training mode for this module and submodules.
    ///
    /// # Arguments
    /// * `mode` - `true` for training mode, `false` for evaluation mode
    ///
    /// Training mode typically enables dropout and batch normalization updates.
    /// Evaluation mode disables these for inference.
    fn train(&mut self, mode: bool) {
        // Recursively set training mode on submodules
        for module in self.modules_mut() {
            module.train(mode);
        }
    }

    /// Get the name/type of this module.
    ///
    /// # Returns
    /// A string identifier for this module type.
    fn name(&self) -> &str;

    /// Get the names of child modules for serialization.
    ///
    /// This is used by Sequential and other container modules to provide
    /// custom names for their children in the state dict.
    ///
    /// # Returns
    /// A vector of (index, name) pairs for child modules, or empty for leaf modules.
    fn child_module_names(&self) -> Vec<(usize, String)> {
        Vec::new() // Default: no custom names
    }

    /// Get this module as an Any trait object for downcasting.
    ///
    /// This is used for runtime type checking and downcasting to concrete module types.
    ///
    /// # Returns
    /// This module as a trait object that can be downcast.
    fn as_any(&self) -> &dyn std::any::Any
    where
        Self: Sized,
    {
        self as &dyn std::any::Any
    }

    /// Get this module as a mutable Any trait object for downcasting.
    ///
    /// This is used for runtime type checking and downcasting to concrete module types.
    ///
    /// # Returns
    /// This module as a mutable trait object that can be downcast.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any
    where
        Self: Sized,
    {
        self as &mut dyn std::any::Any
    }

    /// Get all named buffers (non-parameter state) in this module.
    ///
    /// This is used for serialization of state like BatchNorm running stats.
    ///
    /// # Returns
    /// Vector of (name, tensor) pairs.
    fn named_buffers(&self) -> Vec<(String, Tensor<B, S, T>)> {
        Vec::new() // Default: no buffers
    }

    /// Load a buffer (non-parameter state) into this module.
    ///
    /// # Arguments
    /// * `name` - The name of the buffer
    /// * `value` - The tensor value to load
    fn load_buffer(&self, _name: &str, _value: &Tensor<B, S, T>) -> Result<()> {
        Ok(()) // Default: ignore
    }

    /// Clone the module into a Box.
    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>>;
}

impl<B, S, T> Clone for Box<dyn Module<B, S, T, Input = Tensor<B, S, T>, Output = Tensor<B, S, T>>>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt,
{
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Extension methods for Module trait.
pub trait ModuleExt<
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt,
>: Module<B, S, T>
{
    /// Count total number of parameters in this module.
    ///
    /// # Returns
    /// Total number of elements across all parameters.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.data().len()).sum()
    }

    /// Check if this module has any learnable parameters.
    ///
    /// # Returns
    /// `true` if the module has parameters that require gradients.
    fn has_parameters(&self) -> bool {
        !self.parameters().is_empty()
    }

    /// Get a summary of the module structure.
    ///
    /// # Returns
    /// A string representation of the module hierarchy.
    fn summary(&self) -> String {
        let mut summary = format!("{} ({} parameters)", self.name(), self.parameters().len());
        for module in self.modules() {
            summary.push_str(&format!("\n  {}", module.name()));
        }
        summary
    }
}

// Auto-implement ModuleExt for all Module implementors
impl<
        B: Backend<Data = T> + Clone,
        S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
        T: DataType + FloatExt,
        M: Module<B, S, T>,
    > ModuleExt<B, S, T> for M
{
}

/// State dictionary for serializing and deserializing model parameters.
///
/// Maps parameter names to their tensor data for saving/loading model state.
pub type StateDict<T> = HashMap<String, Vec<T>>;

/// Extension trait for Module serialization and deserialization.
///
/// This trait provides PyTorch-compatible model serialization and checkpointing.
/// All neural network modules automatically implement this trait.
///
/// # Examples
/// ```rust
/// use nn::{Linear, ModuleSerialize};
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
/// use std::collections::HashMap;
/// use std::path::Path;
///
/// // Create and train a model
/// let mut model = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(784, 10).unwrap();
///
/// // Get current parameters
/// let state_dict = model.state_dict();
/// println!("Parameters: {:?}", state_dict.keys());
///
/// // Save to file
/// model.save(Path::new("model.json")).unwrap();
///
/// // Load from file
/// let mut new_model = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(784, 10).unwrap();
/// new_model.load(Path::new("model.json")).unwrap();
/// # std::fs::remove_file("model.json").ok(); // Cleanup
/// ```
#[cfg(feature = "safetensors")]
pub trait ModuleSerialize<
    B: Backend<Data = T> + Clone + std::default::Default,
    S: Storage<T> + Clone + 'static + storage::StorageFromVec<T> + storage::StorageToDense<T>,
    T: DataType + serde::Serialize + serde::de::DeserializeOwned,
>: Module<B, S, T>
{
    /// Recursively collect parameters from this module and submodules.
    ///
    /// # Arguments
    /// * `prefix` - The prefix for parameter names (used for hierarchical naming)
    /// * `state` - The state dictionary to populate
    fn collect_state_dict(&self, prefix: &str, state: &mut StateDict<T>) {
        // Collect parameters from this module
        let params = self.parameters();
        for param in params {
            let full_name = if prefix.is_empty() {
                param.name().to_string()
            } else {
                format!("{}.{}", prefix, param.name())
            };
            state.insert(full_name, param.data().as_slice().to_vec());
        }
    }

    /// Helper method to recursively collect parameters with shape information.
    fn collect_state_with_shapes(
        &self,
        prefix: &str,
        state: &mut std::collections::HashMap<String, (Vec<T>, Vec<usize>)>,
    ) {
        // Collect parameters with shapes from this module
        let params = self.parameters();
        for param in params {
            let full_name = if prefix.is_empty() {
                param.name().to_string()
            } else {
                format!("{}.{}", prefix, param.name())
            };
            state.insert(
                full_name,
                (
                    param.data().as_slice().to_vec(),
                    param.data().shape().dims().to_vec(),
                ),
            );
        }
    }
    /// Get the state dictionary containing all learnable parameters.
    ///
    /// The state dict maps parameter names to their flattened tensor data.
    /// Parameter names include the full hierarchical path (e.g., "layer.weight").
    ///
    /// # Returns
    /// A HashMap mapping parameter names to their serialized data.
    fn state_dict(&self) -> StateDict<T> {
        let mut state = HashMap::new();
        self.collect_state_dict("", &mut state);
        state
    }

    /// Load parameters from a state dictionary.
    ///
    /// This updates the module's parameters with the values from the state dict.
    /// Missing parameters in the state dict are ignored (for partial loading).
    /// Extra parameters in the state dict are ignored.
    ///
    /// # Arguments
    /// * `state_dict` - The state dictionary containing parameter data
    ///
    /// # Returns
    /// Result indicating success or failure to load state.
    ///
    /// # Errors
    /// Returns `NNError::SerializationError` if parameter shapes don't match.
    fn load_state_dict(&mut self, state_dict: &StateDict<T>) -> Result<()> {
        self.load_state_dict_with_prefix("", state_dict)
    }

    /// Load parameters from a state dictionary with a prefix.
    ///
    /// This recursively loads parameters from submodules and handles hierarchical naming.
    ///
    /// # Arguments
    /// * `prefix` - The prefix for parameter names (used for hierarchical naming)
    /// * `state_dict` - The state dictionary containing parameter data
    ///
    /// # Returns
    /// Result indicating success or failure of the load operation.
    fn load_state_dict_with_prefix(
        &mut self,
        prefix: &str,
        state_dict: &StateDict<T>,
    ) -> Result<()> {
        let modules = self.modules();

        // If this module has submodules, only load from them (not from self.parameters())
        // to avoid duplicates, since self.parameters() includes submodule parameters
        if !modules.is_empty() {
            // Trait-level implementation is limited - concrete module types
            // should override this method for proper submodule parameter loading
            Ok(())
        } else {
            // Leaf module: load its own parameters
            let params = self.parameters();
            for mut param in params {
                let full_name = if prefix.is_empty() {
                    param.name().to_string()
                } else {
                    format!("{}.{}", prefix, param.name())
                };

                if let Some(data) = state_dict.get(&full_name) {
                    // This is a simplified implementation
                    // Real implementation would need to reshape and validate data
                    // For now, assume the data is the correct shape
                    let tensor = Tensor::from_vec(data.clone(), param.data().shape().dims())
                        .map_err(|_| NNError::SerializationError {
                            message: format!("Failed to create tensor for parameter {}", full_name),
                        })?;

                    // Update the parameter data
                    *param.data_mut() = tensor;
                }
                // Missing parameters are ignored (for partial loading)
            }
            Ok(())
        }
    }

    /// Save the module's state to a JSON file.
    ///
    /// # Arguments
    /// * `path` - Path to save the state dictionary
    ///
    /// # Returns
    /// Result indicating success or failure of the save operation.
    ///
    /// # Errors
    /// Returns `NNError::SerializationError` if serialization or file I/O fails.
    fn save(&self, path: &std::path::Path) -> Result<()> {
        let state = self.state_dict();
        let json =
            serde_json::to_string_pretty(&state).map_err(|e| NNError::SerializationError {
                message: format!("Failed to serialize state dict: {}", e),
            })?;
        std::fs::write(path, json).map_err(|e| NNError::SerializationError {
            message: format!("Failed to write state dict to file: {}", e),
        })?;
        Ok(())
    }

    /// Load the module's state from a JSON file.
    ///
    /// # Arguments
    /// * `path` - Path to load the state dictionary from
    ///
    /// # Returns
    /// Result indicating success or failure of the load operation.
    ///
    /// # Errors
    /// Returns `NNError::SerializationError` if deserialization or file I/O fails.
    fn load(&mut self, path: &std::path::Path) -> Result<()> {
        let json = std::fs::read_to_string(path).map_err(|e| NNError::SerializationError {
            message: format!("Failed to read state dict from file: {}", e),
        })?;
        let state_dict: StateDict<T> =
            serde_json::from_str(&json).map_err(|e| NNError::SerializationError {
                message: format!("Failed to deserialize state dict: {}", e),
            })?;
        self.load_state_dict(&state_dict)
    }

    /// Save the module's state to a SafeTensors file.
    ///
    /// SafeTensors is a secure serialization format that prevents arbitrary
    /// code execution when loading untrusted files.
    ///
    /// # Arguments
    /// * `path` - Path to save the SafeTensors file
    ///
    /// # Returns
    /// Result indicating success or failure of the save operation.
    ///
    /// # Errors
    /// Returns `NNError::SerializationError` if serialization or file I/O fails.
    #[cfg(feature = "safetensors")]
    fn save_safetensors(&self, path: &std::path::Path) -> Result<()> {
        // Collect parameters with shape information
        let mut state_with_shapes = std::collections::HashMap::new();
        self.collect_state_with_shapes("", &mut state_with_shapes);

        let safetensors = crate::io::safetensors::SafeTensors::from_state_dict(&state_with_shapes)?;
        safetensors.save(path)
    }

    /// Load the module's state from a SafeTensors file.
    ///
    /// # Arguments
    /// * `path` - Path to load the SafeTensors file from
    ///
    /// # Returns
    /// Result indicating success or failure of the load operation.
    ///
    /// # Errors
    /// Returns `NNError::SerializationError` if deserialization or file I/O fails.
    #[cfg(feature = "safetensors")]
    fn load_safetensors(&mut self, path: &std::path::Path) -> Result<()> {
        let safetensors = crate::io::safetensors::SafeTensors::load(path)?;
        let state_with_shapes = safetensors.to_state_dict()?;

        // Convert to regular state dict format for loading
        let mut state_dict = StateDict::new();
        for (name, (data, _shape)) in state_with_shapes {
            state_dict.insert(name, data);
        }

        self.load_state_dict(&state_dict)
    }
}

/// Helper macro for implementing Module trait.
///
/// This macro reduces boilerplate when implementing the Module trait
/// by providing default implementations for common methods.
///
/// # Usage
/// ```rust
/// use crate::core::module::module;
///
/// #[module]
/// impl MyModule {
///     // Your module implementation here
/// }
/// ```
#[macro_export]
macro_rules! module {
    ($(#[$attr:meta])* $vis:vis struct $name:ident $(<$($gen:tt),*>)? {
        $($field_vis:vis $field_name:ident : $field_ty:ty),* $(,)?
    }) => {
        $(#[$attr])* $vis struct $name $(<$($gen),*>)? {
            $($field_vis $field_name : $field_ty),*
        }

        impl $(<$($gen),*>)? $name $(<$($gen),*>)? {
            /// Get the name of this module type.
            pub fn name(&self) -> &str {
                stringify!($name)
            }

            /// Get all parameters in this module.
            pub fn parameters(&self) -> Vec<&dyn $crate::core::parameter::ParameterTrait> {
                vec![] // Default: no parameters
            }

            /// Get mutable references to all parameters.
            pub fn parameters_mut(&mut self) -> Vec<&mut dyn $crate::core::parameter::ParameterTrait> {
                vec![] // Default: no parameters
            }

            /// Get child modules.
            pub fn modules(&self) -> Vec<&dyn Module> {
                vec![] // Default: no submodules
            }

            /// Get child module names with their indices.
            pub fn child_module_names(&self) -> Vec<(usize, String)> {
                vec![] // Default: no named submodules
            }

            /// Zero all gradients.
            pub fn zero_grad(&mut self) {
                // Default: do nothing
            }

            /// Set training mode.
            pub fn train(&mut self, _mode: bool) {
                // Default: do nothing
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::NNError;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;
    use tensor::Tensor;

    // Mock parameter for testing
    #[derive(Debug)]
    struct MockParameter {
        #[allow(dead_code)]
        name: String,
        #[allow(dead_code)]
        data: Vec<f32>,
    }

    impl MockParameter {
        fn new(name: &str, data: Vec<f32>) -> Self {
            Self {
                name: name.to_string(),
                data,
            }
        }

        fn zero_grad(&mut self) {
            // Mock implementation - do nothing
        }
    }

    // Mock module for testing
    #[derive(Debug)]
    struct MockModule {
        param1: MockParameter,
        param2: MockParameter,
    }

    impl MockModule {
        fn new() -> Self {
            Self {
                param1: MockParameter::new("param1", vec![1.0, 2.0, 3.0]),
                param2: MockParameter::new("param2", vec![4.0, 5.0]),
            }
        }
    }

    impl Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32> for MockModule {
        type Input = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;
        type Output = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

        fn forward(
            &self,
            _input: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
        ) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>> {
            // Mock implementation
            Err(NNError::InvalidInput {
                message: "Mock forward not implemented".to_string(),
            })
        }

        fn parameters(
            &self,
        ) -> Vec<crate::Parameter<CpuBackend<Float32>, DenseStorage<Float32>, Float32>> {
            vec![
                Parameter::new(
                    Tensor::from_vec(
                        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
                        &[3],
                    )
                    .unwrap(),
                    "param1".to_string(),
                ),
                Parameter::new(
                    Tensor::from_vec(vec![Float32::new(4.0), Float32::new(5.0)], &[2]).unwrap(),
                    "param2".to_string(),
                ),
            ]
        }

        fn zero_grad(&mut self) {
            self.param1.zero_grad();
            self.param2.zero_grad();
        }

        fn train(&mut self, _mode: bool) {
            // Mock implementation
        }

        fn name(&self) -> &str {
            "MockModule"
        }

        fn clone_box(
            &self,
        ) -> Box<dyn Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32, Input = Self::Input, Output = Self::Output>> {
            // Re-create the mock module (since MockParameter isn't easily cloneable in this snippet,
            // but we can just make a new one or implement Clone for MockModule properly.
            // For testing, new is fine or we can clone fields if we derive Clone).
            Box::new(MockModule::new())
        }
    }

    #[test]
    fn test_module_parameters() {
        let module = MockModule::new();
        let params = module.parameters();

        assert_eq!(params.len(), 2);
        assert_eq!(params[0].name(), "param1");
        assert_eq!(params[1].name(), "param2");
    }

    #[test]
    fn test_module_name() {
        let module = MockModule::new();
        assert_eq!(module.name(), "MockModule");
    }

    #[test]
    fn test_module_summary() {
        let module = MockModule::new();
        let summary = module.summary();

        assert!(summary.contains("MockModule"));
        assert!(summary.contains("2 parameters"));
    }
}
