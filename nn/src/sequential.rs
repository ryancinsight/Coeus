//! Sequential container for chaining neural network modules.

use std::collections::HashMap;

use coeus_backend::Backend;
use coeus_dtype::{DataType, traits::FloatExt};
use coeus_storage::{Storage, StorageFromVec, DenseStorage};
use coeus_tensor::Tensor;

use crate::error::{NNError, Result};
use crate::module::Module;
use crate::parameter::Parameter;

/// A sequential container that chains modules in order.
///
/// Modules are executed in the order they were added, with the output
/// of one module becoming the input to the next.
///
/// # Examples
/// ```rust
/// use coeus_nn::{Sequential, Linear, ReLU, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// // Create a simple MLP
/// let mut model = Sequential::<CpuBackend, DenseStorage<Float32>, Float32>::new();
/// model.add_module("fc1".to_string(), Linear::<CpuBackend, DenseStorage<Float32>, Float32>::new(784, 128).unwrap());
/// model.add_module("relu1".to_string(), ReLU::new());
/// model.add_module("fc2".to_string(), Linear::<CpuBackend, DenseStorage<Float32>, Float32>::new(128, 10).unwrap());
///
/// // Forward pass
/// let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[32, 784]).unwrap();
/// let output = model.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[32, 10]);
/// ```
// #[derive(Debug)] // TODO: Add Debug when Module trait supports it
pub struct Sequential<B: Backend, S, T>
where
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType,
{
    /// Ordered list of modules
    modules: Vec<Box<dyn Module<B, S, T>>>,
    /// Module names for lookup
    names: Vec<String>,
    /// Name-to-index mapping for efficient lookup
    name_to_index: HashMap<String, usize>,
}

impl<B: Backend, S, T> Sequential<B, S, T>
where
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType,
{
    /// Create a new empty sequential container.
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            names: Vec::new(),
            name_to_index: HashMap::new(),
        }
    }

    /// Add a module to the end of the sequence.
    ///
    /// # Arguments
    /// * `name` - Unique name for the module
    /// * `module` - The module to add
    ///
    /// # Panics
    /// Panics if a module with the same name already exists.
    pub fn add_module<M: Module<B, S, T> + 'static>(&mut self, name: String, module: M)
    where
        B: Clone,
    {
        assert!(
            !self.name_to_index.contains_key(&name),
            "Module with name '{}' already exists",
            name
        );

        let index = self.modules.len();
        self.modules.push(Box::new(module));
        self.names.push(name.clone());
        self.name_to_index.insert(name, index);
    }

    /// Get a module by name.
    ///
    /// # Arguments
    /// * `name` - Name of the module to retrieve
    ///
    /// # Returns
    /// Reference to the module, or an error if not found.
    pub fn get_module(&self, name: &str) -> Result<&dyn Module<B, S, T>> {
        let index = self
            .name_to_index
            .get(name)
            .ok_or_else(|| NNError::ModuleNotFound {
                name: name.to_string(),
            })?;
        Ok(&*self.modules[*index])
    }

    /// Get the number of modules in the sequence.
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Check if the sequential container is empty.
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Get the names of all modules in order.
    pub fn module_names(&self) -> &[String] {
        &self.names
    }
}

impl<B, S, T> Module<B, S, T> for Sequential<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType,
{
    fn forward(
        &self,
        input: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>> {
        if self.modules.is_empty() {
            return Err(NNError::InvalidConfiguration {
                message: "Sequential container is empty".to_string(),
            });
        }

        let requires_grad = input.requires_grad();
        let mut current = input.clone();

        for module in &self.modules {
            current = module.forward(&current)?;
        }

        Ok(current.requires_grad_(requires_grad))
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        let mut params = Vec::new();
        for module in &self.modules {
            params.extend(module.parameters());
        }
        params
    }

    fn modules(&self) -> Vec<&dyn Module<B, S, T>> {
        self.modules.iter().map(|m| &**m).collect()
    }

    fn zero_grad(&mut self) {
        for module in &mut self.modules {
            module.zero_grad();
        }
    }

    fn train(&mut self, mode: bool) {
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
}

impl<B: Backend + Default, S, T> Default for Sequential<B, S, T>
where
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S, T> crate::module::ModuleSerialize<B, S, T> for Sequential<B, S, T>
where
    B: Backend + Clone + std::default::Default,
    S: Storage<T> + Clone + 'static + coeus_storage::StorageFromVec<T>,
    T: DataType + serde::Serialize + serde::de::DeserializeOwned,
{
    // Default implementations from trait are sufficient
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Linear, ModuleSerialize};
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;

    // Mock module for testing
    #[derive(Debug)]
    struct MockModule {
        scale: f32,
    }

    impl MockModule {
        fn new(scale: f32) -> Self {
            Self { scale }
        }
    }

    impl<B, S, T> Module<B, S, T> for MockModule
    where
        B: Backend + Clone + Default,
        S: Storage<T> + StorageFromVec<T> + Clone + 'static,
        T: DataType + FloatExt + std::ops::Mul<Output = T> + Copy,
    {
        fn forward(
            &self,
            input: &Tensor<B, S, T>,
        ) -> Result<Tensor<B, S, T>> {
            // Simple scaling operation using generic T
            let scale_t = T::from(self.scale).ok_or_else(|| {
                crate::NNError::InvalidInput {
                    message: "Cannot convert scale to T".to_string(),
                }
            })?;
            let scaled_data: Vec<T> = input
                .as_slice()
                .iter()
                .map(|&x| x * scale_t)
                .collect();

            Tensor::from_vec(scaled_data, input.shape().dims()).map_err(Into::into)
        }

        fn parameters(&self) -> Vec<Parameter<B, S, T>> {
            Vec::new() // No parameters
        }

        fn train(&mut self, _mode: bool) {
            // No-op
        }

        fn zero_grad(&mut self) {
            // No-op: MockModule has no parameters
        }

        fn name(&self) -> &str {
            "MockModule"
        }
    }

    #[test]
    fn test_sequential_empty() {
        let seq = Sequential::<CpuBackend, DenseStorage<Float32>, Float32>::new();

        assert!(seq.is_empty());
        assert_eq!(seq.len(), 0);

        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[2, 3]).unwrap();
        let result = seq.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_sequential_add_module() {
        let mut seq = Sequential::<CpuBackend, DenseStorage<Float32>, Float32>::new();

        seq.add_module("scale1".to_string(), MockModule::new(2.0));
        seq.add_module("scale2".to_string(), MockModule::new(3.0));

        assert_eq!(seq.len(), 2);
        assert_eq!(
            seq.module_names(),
            &["scale1".to_string(), "scale2".to_string()]
        );
    }

    #[test]
    fn test_sequential_forward() {
        let mut seq = Sequential::<CpuBackend, DenseStorage<Float32>, Float32>::new();

        // Add modules: first *2, then *3, so final result should be *6
        seq.add_module("scale1".to_string(), MockModule::new(2.0));
        seq.add_module("scale2".to_string(), MockModule::new(3.0));

        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        )
        .unwrap();

        let output = seq.forward(&input).unwrap();

        // Check that each element was scaled by 2 * 3 = 6
        let expected = [6.0, 12.0];
        let actual: Vec<f32> = output.as_slice().iter().map(|x| x.get()).collect();

        approx::assert_relative_eq!(actual[0], expected[0]);
        approx::assert_relative_eq!(actual[1], expected[1]);
    }

    #[test]
    fn test_sequential_get_module() {
        let mut seq = Sequential::<CpuBackend, DenseStorage<Float32>, Float32>::new();
        seq.add_module("test_module".to_string(), MockModule::new(1.0));

        let module = seq.get_module("test_module").unwrap();
        assert_eq!(module.name(), "MockModule");
    }

    #[test]
    fn test_sequential_module_not_found() {
        let seq = Sequential::<CpuBackend, DenseStorage<Float32>, Float32>::new();

        let result = seq.get_module("nonexistent");
        assert!(result.is_err());

        if let Err(NNError::ModuleNotFound { name }) = result {
            assert_eq!(name, "nonexistent");
        } else {
            panic!("Expected ModuleNotFound error");
        }
    }

    #[test]
    fn test_sequential_parameters() {
        let mut seq = Sequential::<CpuBackend, DenseStorage<Float32>, Float32>::new();

        // Add a module with parameters (Linear layer)
        let linear = Linear::<CpuBackend, DenseStorage<Float32>, Float32>::new(5, 3).unwrap();
        seq.add_module("linear".to_string(), linear);

        let params = seq.parameters();
        assert!(!params.is_empty()); // Should have weight and bias parameters
    }

    #[test]
    fn test_sequential_serialization() {
        let mut seq = Sequential::<CpuBackend, DenseStorage<Float32>, Float32>::new();

        // Add a Linear layer with parameters
        let linear = Linear::<CpuBackend, DenseStorage<Float32>, Float32>::new(5, 3).unwrap();
        seq.add_module("linear".to_string(), linear);

        // Get state dict
        let state_dict = seq.state_dict();

        // Should contain parameters with hierarchical names (uses add_module name "linear")
        assert!(state_dict.contains_key("linear.weight"));
        assert!(state_dict.contains_key("linear.bias"));

        // Check shapes
        assert_eq!(state_dict["linear.weight"].len(), 15); // 5 * 3 = 15
        assert_eq!(state_dict["linear.bias"].len(), 3); // 3 bias terms
    }

    #[test]
    #[should_panic]
    fn test_sequential_duplicate_names() {
        let mut seq = Sequential::<CpuBackend, DenseStorage<Float32>, Float32>::new();
        seq.add_module("test".to_string(), MockModule::new(1.0));
        seq.add_module("test".to_string(), MockModule::new(2.0)); // Should panic
    }
}
