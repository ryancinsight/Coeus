//! Tensor serialization and deserialization
//!
//! Provides functionality to save and load tensors in various formats,
//! compatible with PyTorch's serialization interface.

use crate::{Dtype, Tensor, TensorError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Trait for tensor serialization operations
pub trait TensorSerializable: Dtype {
    /// Save tensor to file
    fn save_tensor(
        tensor: &Tensor<Self>,
        path: impl AsRef<Path>,
        format: SerializationFormat,
    ) -> Result<(), Box<dyn std::error::Error>>;

    /// Load tensor from file
    fn load_tensor(
        path: impl AsRef<Path>,
        format: SerializationFormat,
    ) -> Result<Tensor<Self>, Box<dyn std::error::Error>>;
}

/// Serialization format for tensors
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SerializationFormat {
    /// Binary format (fast, compact)
    Binary,
    /// JSON format (human-readable, cross-platform)
    Json,
}

/// Tensor data structure for serialization (supports common numeric types)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SerializableTensor {
    /// Tensor data as flat vector (stored as f64 for JSON compatibility)
    pub data: Vec<f64>,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Whether gradients are required
    pub requires_grad: bool,
    /// Device information (reserved for future GPU support)
    pub device: String,
    /// Data type information
    pub dtype: String,
}

impl From<&Tensor<f32>> for SerializableTensor {
    fn from(tensor: &Tensor<f32>) -> Self {
        SerializableTensor {
            data: tensor.data().iter().map(|&x| x as f64).collect(),
            shape: tensor.shape().to_vec(),
            requires_grad: tensor.requires_grad(),
            device: "cpu".to_string(),
            dtype: "f32".to_string(),
        }
    }
}

impl TryFrom<SerializableTensor> for Tensor<f32> {
    type Error = TensorError;

    fn try_from(serializable: SerializableTensor) -> Result<Self, Self::Error> {
        let data: Vec<f32> = serializable.data.iter().map(|&x| x as f32).collect();
        let mut tensor = Tensor::from_vec(data, serializable.shape);
        if serializable.requires_grad {
            tensor.set_requires_grad(true);
        }
        Ok(tensor)
    }
}

impl From<&Tensor<f64>> for SerializableTensor {
    fn from(tensor: &Tensor<f64>) -> Self {
        SerializableTensor {
            data: tensor.data().to_vec(),
            shape: tensor.shape().to_vec(),
            requires_grad: tensor.requires_grad(),
            device: "cpu".to_string(),
            dtype: "f64".to_string(),
        }
    }
}

impl TryFrom<SerializableTensor> for Tensor<f64> {
    type Error = TensorError;

    fn try_from(serializable: SerializableTensor) -> Result<Self, Self::Error> {
        let mut tensor = Tensor::from_vec(serializable.data, serializable.shape);
        if serializable.requires_grad {
            tensor.set_requires_grad(true);
        }
        Ok(tensor)
    }
}

/// State dictionary for model parameters
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct StateDict {
    /// Parameter name to tensor mapping
    pub parameters: HashMap<String, SerializableTensor>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl StateDict {
    /// Create a new empty state dictionary
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a parameter to the state dictionary (f32)
    pub fn insert_f32<S: Into<String>>(&mut self, key: S, tensor: &Tensor<f32>) {
        self.parameters.insert(key.into(), tensor.into());
    }

    /// Add a parameter to the state dictionary (f64)
    pub fn insert_f64<S: Into<String>>(&mut self, key: S, tensor: &Tensor<f64>) {
        self.parameters.insert(key.into(), tensor.into());
    }

    /// Get a parameter from the state dictionary
    pub fn get<S: AsRef<str>>(&self, key: S) -> Option<&SerializableTensor> {
        self.parameters.get(key.as_ref())
    }

    /// Remove a parameter from the state dictionary
    pub fn remove<S: AsRef<str>>(&mut self, key: S) -> Option<SerializableTensor> {
        self.parameters.remove(key.as_ref())
    }

    /// Get all parameter names
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.parameters.keys()
    }

    /// Check if the state dictionary is empty
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    /// Get the number of parameters
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    /// Add metadata
    pub fn set_metadata<S: Into<String>>(&mut self, key: S, value: S) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get metadata
    pub fn get_metadata<S: AsRef<str>>(&self, key: S) -> Option<&String> {
        self.metadata.get(key.as_ref())
    }
}

impl Tensor<f32> {
    /// Save f32 tensor to file
    ///
    /// # Arguments
    /// * `path` - File path to save to
    /// * `format` - Serialization format to use
    ///
    /// # Example
    /// ```rust,no_run
    /// use coeus_tensor::{Tensor, serialization::SerializationFormat};
    ///
    /// let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    /// tensor.save("tensor.bin", SerializationFormat::Binary).unwrap();
    /// ```
    pub fn save<P: AsRef<Path>>(
        &self,
        path: P,
        format: SerializationFormat,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let serializable: SerializableTensor = self.into();

        match format {
            SerializationFormat::Binary => {
                let encoded = bincode::serialize(&serializable)?;
                let mut file = File::create(path)?;
                file.write_all(&encoded)?;
            }
            SerializationFormat::Json => {
                let json = serde_json::to_string_pretty(&serializable)?;
                let mut file = File::create(path)?;
                file.write_all(json.as_bytes())?;
            }
        }

        Ok(())
    }

    /// Load f32 tensor from file
    ///
    /// # Arguments
    /// * `path` - File path to load from
    /// * `format` - Serialization format to use
    ///
    /// # Example
    /// ```rust,no_run
    /// use coeus_tensor::{Tensor, serialization::SerializationFormat};
    ///
    /// let tensor = Tensor::<f32>::load("tensor.bin", SerializationFormat::Binary).unwrap();
    /// ```
    pub fn load<P: AsRef<Path>>(
        path: P,
        format: SerializationFormat,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        match format {
            SerializationFormat::Binary => {
                let mut file = File::open(path)?;
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer)?;
                let serializable: SerializableTensor = bincode::deserialize(&buffer)?;
                Ok(serializable.try_into()?)
            }
            SerializationFormat::Json => {
                let mut file = File::open(path)?;
                let mut json_str = String::new();
                file.read_to_string(&mut json_str)?;
                let serializable: SerializableTensor = serde_json::from_str(&json_str)?;
                Ok(serializable.try_into()?)
            }
        }
    }
}

impl Tensor<f64> {
    /// Save f64 tensor to file
    pub fn save<P: AsRef<Path>>(
        &self,
        path: P,
        format: SerializationFormat,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let serializable: SerializableTensor = self.into();

        match format {
            SerializationFormat::Binary => {
                let encoded = bincode::serialize(&serializable)?;
                let mut file = File::create(path)?;
                file.write_all(&encoded)?;
            }
            SerializationFormat::Json => {
                let json = serde_json::to_string_pretty(&serializable)?;
                let mut file = File::create(path)?;
                file.write_all(json.as_bytes())?;
            }
        }

        Ok(())
    }

    /// Load f64 tensor from file
    pub fn load<P: AsRef<Path>>(
        path: P,
        format: SerializationFormat,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        match format {
            SerializationFormat::Binary => {
                let mut file = File::open(path)?;
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer)?;
                let serializable: SerializableTensor = bincode::deserialize(&buffer)?;
                Ok(serializable.try_into()?)
            }
            SerializationFormat::Json => {
                let mut file = File::open(path)?;
                let mut json_str = String::new();
                file.read_to_string(&mut json_str)?;
                let serializable: SerializableTensor = serde_json::from_str(&json_str)?;
                Ok(serializable.try_into()?)
            }
        }
    }
}

impl StateDict {
    /// Save state dictionary to file
    ///
    /// # Arguments
    /// * `path` - File path to save to
    /// * `format` - Serialization format to use
    ///
    /// # Example
    /// ```rust,no_run
    /// use coeus_tensor::{Tensor, serialization::{StateDict, SerializationFormat}};
    ///
    /// let mut state_dict = StateDict::new();
    /// let param = Tensor::<f32>::from_vec(vec![1.0, 2.0], vec![2]);
    /// state_dict.insert_f32("weight", &param);
    ///
    /// state_dict.save("model.bin", SerializationFormat::Binary).unwrap();
    /// ```
    pub fn save<P: AsRef<Path>>(
        &self,
        path: P,
        format: SerializationFormat,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match format {
            SerializationFormat::Binary => {
                let encoded = bincode::serialize(self)?;
                let mut file = File::create(path)?;
                file.write_all(&encoded)?;
            }
            SerializationFormat::Json => {
                let json = serde_json::to_string_pretty(self)?;
                let mut file = File::create(path)?;
                file.write_all(json.as_bytes())?;
            }
        }

        Ok(())
    }

    /// Load state dictionary from file
    ///
    /// # Arguments
    /// * `path` - File path to load from
    /// * `format` - Serialization format to use
    ///
    /// # Example
    /// ```rust,no_run
    /// use coeus_tensor::serialization::{StateDict, SerializationFormat};
    ///
    /// let state_dict = StateDict::load("model.bin", SerializationFormat::Binary).unwrap();
    /// let weight = state_dict.get("weight").unwrap();
    /// ```
    pub fn load<P: AsRef<Path>>(
        path: P,
        format: SerializationFormat,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        match format {
            SerializationFormat::Binary => {
                let mut file = File::open(path)?;
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer)?;
                let state_dict: StateDict = bincode::deserialize(&buffer)?;
                Ok(state_dict)
            }
            SerializationFormat::Json => {
                let mut file = File::open(path)?;
                let mut json_str = String::new();
                file.read_to_string(&mut json_str)?;
                let state_dict: StateDict = serde_json::from_str(&json_str)?;
                Ok(state_dict)
            }
        }
    }

    /// Convert state dictionary to f32 tensor map
    ///
    /// # Example
    /// ```rust,no_run
    /// use coeus_tensor::serialization::{StateDict, SerializationFormat};
    ///
    /// let state_dict = StateDict::load("model.bin", SerializationFormat::Binary).unwrap();
    /// let tensors = state_dict.to_tensors_f32().unwrap();
    /// ```
    pub fn to_tensors_f32(
        &self,
    ) -> Result<HashMap<String, Tensor<f32>>, Box<dyn std::error::Error>> {
        let mut tensors = HashMap::new();

        for (name, serializable) in &self.parameters {
            match serializable.dtype.as_str() {
                "f32" => {
                    let tensor = Tensor::<f32>::try_from(serializable.clone())?;
                    tensors.insert(name.clone(), tensor);
                }
                _ => return Err(format!("Unsupported dtype: {}", serializable.dtype).into()),
            }
        }

        Ok(tensors)
    }

    /// Convert state dictionary to f64 tensor map
    pub fn to_tensors_f64(
        &self,
    ) -> Result<HashMap<String, Tensor<f64>>, Box<dyn std::error::Error>> {
        let mut tensors = HashMap::new();

        for (name, serializable) in &self.parameters {
            match serializable.dtype.as_str() {
                "f64" => {
                    let tensor = Tensor::<f64>::try_from(serializable.clone())?;
                    tensors.insert(name.clone(), tensor);
                }
                _ => return Err(format!("Unsupported dtype: {}", serializable.dtype).into()),
            }
        }

        Ok(tensors)
    }

    /// Create state dictionary from tensor map
    ///
    /// # Arguments
    /// * `tensors` - Map of parameter names to tensors
    ///
    /// # Example
    /// ```rust,no_run
    /// use coeus_tensor::{Tensor, serialization::StateDict};
    /// use std::collections::HashMap;
    ///
    /// let mut tensors = HashMap::new();
    /// tensors.insert("weight".to_string(), Tensor::<f32>::from_vec(vec![1.0, 2.0], vec![2]));
    ///
    /// let state_dict = StateDict::from_tensors_f32(tensors).unwrap();
    /// ```
    /// Create state dictionary from f32 tensor map
    ///
    /// # Arguments
    /// * `tensors` - Map of parameter names to tensors
    ///
    /// # Example
    /// ```rust,no_run
    /// use coeus_tensor::{Tensor, serialization::StateDict};
    /// use std::collections::HashMap;
    ///
    /// let mut tensors = HashMap::new();
    /// tensors.insert("weight".to_string(), Tensor::<f32>::from_vec(vec![1.0, 2.0], vec![2]));
    /// let state_dict = StateDict::from_tensors_f32(tensors).unwrap();
    /// ```
    pub fn from_tensors_f32(
        tensors: HashMap<String, Tensor<f32>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut state_dict = StateDict::new();

        for (name, tensor) in tensors {
            state_dict.insert_f32(name, &tensor);
        }

        Ok(state_dict)
    }

    /// Create state dictionary from f64 tensor map
    ///
    /// # Arguments
    /// * `tensors` - Map of parameter names to tensors
    pub fn from_tensors_f64(
        tensors: HashMap<String, Tensor<f64>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut state_dict = StateDict::new();

        for (name, tensor) in tensors {
            state_dict.insert_f64(name, &tensor);
        }

        Ok(state_dict)
    }
}

// Implement TensorSerializable for f32
impl TensorSerializable for f32 {
    fn save_tensor(
        tensor: &Tensor<Self>,
        path: impl AsRef<Path>,
        format: SerializationFormat,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let serializable: SerializableTensor = tensor.into();

        match format {
            SerializationFormat::Binary => {
                let encoded = bincode::serialize(&serializable)?;
                let mut file = File::create(path)?;
                file.write_all(&encoded)?;
            }
            SerializationFormat::Json => {
                let json = serde_json::to_string_pretty(&serializable)?;
                let mut file = File::create(path)?;
                file.write_all(json.as_bytes())?;
            }
        }

        Ok(())
    }

    fn load_tensor(
        path: impl AsRef<Path>,
        format: SerializationFormat,
    ) -> Result<Tensor<Self>, Box<dyn std::error::Error>> {
        match format {
            SerializationFormat::Binary => {
                let mut file = File::open(path)?;
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer)?;
                let serializable: SerializableTensor = bincode::deserialize(&buffer)?;
                Ok(serializable.try_into()?)
            }
            SerializationFormat::Json => {
                let mut file = File::open(path)?;
                let mut json_str = String::new();
                file.read_to_string(&mut json_str)?;
                let serializable: SerializableTensor = serde_json::from_str(&json_str)?;
                Ok(serializable.try_into()?)
            }
        }
    }
}

// Implement TensorSerializable for f64
impl TensorSerializable for f64 {
    fn save_tensor(
        tensor: &Tensor<Self>,
        path: impl AsRef<Path>,
        format: SerializationFormat,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let serializable: SerializableTensor = tensor.into();

        match format {
            SerializationFormat::Binary => {
                let encoded = bincode::serialize(&serializable)?;
                let mut file = File::create(path)?;
                file.write_all(&encoded)?;
            }
            SerializationFormat::Json => {
                let json = serde_json::to_string_pretty(&serializable)?;
                let mut file = File::create(path)?;
                file.write_all(json.as_bytes())?;
            }
        }

        Ok(())
    }

    fn load_tensor(
        path: impl AsRef<Path>,
        format: SerializationFormat,
    ) -> Result<Tensor<Self>, Box<dyn std::error::Error>> {
        match format {
            SerializationFormat::Binary => {
                let mut file = File::open(path)?;
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer)?;
                let serializable: SerializableTensor = bincode::deserialize(&buffer)?;
                Ok(serializable.try_into()?)
            }
            SerializationFormat::Json => {
                let mut file = File::open(path)?;
                let mut json_str = String::new();
                file.read_to_string(&mut json_str)?;
                let serializable: SerializableTensor = serde_json::from_str(&json_str)?;
                Ok(serializable.try_into()?)
            }
        }
    }
}

/// Helper functions for common serialization tasks
pub mod helpers {
    use super::*;

    /// Save multiple tensors with automatic naming
    ///
    /// # Arguments
    /// * `tensors` - Iterator of tensors to save
    /// * `base_path` - Base path for saving (will append indices)
    /// * `format` - Serialization format
    ///
    /// # Example
    /// ```rust,no_run
    /// use coeus_tensor::{Tensor, serialization::{helpers, SerializationFormat}};
    ///
    /// let tensors = vec![
    ///     Tensor::<f32>::from_vec(vec![1.0, 2.0], vec![2]),
    ///     Tensor::<f32>::from_vec(vec![3.0, 4.0], vec![2]),
    /// ];
    ///
    /// helpers::save_tensors_f32(tensors.iter(), "tensor", SerializationFormat::Binary).unwrap();
    /// // Saves as: tensor_0.bin, tensor_1.bin
    /// ```
    pub fn save_tensors_f32<'a, I, P>(
        tensors: I,
        base_path: P,
        format: SerializationFormat,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        I: Iterator<Item = &'a Tensor<f32>>,
        P: AsRef<Path>,
    {
        let base_path_str = base_path.as_ref().to_string_lossy();

        for (i, tensor) in tensors.enumerate() {
            let filename = format!(
                "{}_{}.{}",
                base_path_str,
                i,
                match format {
                    SerializationFormat::Binary => "bin",
                    SerializationFormat::Json => "json",
                }
            );
            <f32 as TensorSerializable>::save_tensor(tensor, &filename, format)?;
        }

        Ok(())
    }

    pub fn save_tensors_f64<'a, I, P>(
        tensors: I,
        base_path: P,
        format: SerializationFormat,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        I: Iterator<Item = &'a Tensor<f64>>,
        P: AsRef<Path>,
    {
        let base_path_str = base_path.as_ref().to_string_lossy();

        for (i, tensor) in tensors.enumerate() {
            let filename = format!(
                "{}_{}.{}",
                base_path_str,
                i,
                match format {
                    SerializationFormat::Binary => "bin",
                    SerializationFormat::Json => "json",
                }
            );
            <f64 as TensorSerializable>::save_tensor(tensor, &filename, format)?;
        }

        Ok(())
    }

    /// Load multiple f32 tensors with automatic naming
    ///
    /// # Arguments
    /// * `count` - Number of tensors to load
    /// * `base_path` - Base path for loading
    /// * `format` - Serialization format
    ///
    /// # Example
    /// ```rust,no_run
    /// use coeus_tensor::serialization::{helpers, SerializationFormat};
    ///
    /// let tensors = helpers::load_tensors_f32(2, "tensor", SerializationFormat::Binary).unwrap();
    /// ```
    pub fn load_tensors_f32<P>(
        count: usize,
        base_path: P,
        format: SerializationFormat,
    ) -> Result<Vec<Tensor<f32>>, Box<dyn std::error::Error>>
    where
        P: AsRef<Path>,
    {
        let base_path_str = base_path.as_ref().to_string_lossy();
        let mut tensors = Vec::with_capacity(count);

        for i in 0..count {
            let filename = format!(
                "{}_{}.{}",
                base_path_str,
                i,
                match format {
                    SerializationFormat::Binary => "bin",
                    SerializationFormat::Json => "json",
                }
            );
            let tensor = <f32 as TensorSerializable>::load_tensor(&filename, format)?;
            tensors.push(tensor);
        }

        Ok(tensors)
    }

    /// Load multiple f64 tensors with automatic naming
    ///
    /// # Arguments
    /// * `count` - Number of tensors to load
    /// * `base_path` - Base path for loading
    /// * `format` - Serialization format
    pub fn load_tensors_f64<P>(
        count: usize,
        base_path: P,
        format: SerializationFormat,
    ) -> Result<Vec<Tensor<f64>>, Box<dyn std::error::Error>>
    where
        P: AsRef<Path>,
    {
        let base_path_str = base_path.as_ref().to_string_lossy();
        let mut tensors = Vec::with_capacity(count);

        for i in 0..count {
            let filename = format!(
                "{}_{}.{}",
                base_path_str,
                i,
                match format {
                    SerializationFormat::Binary => "bin",
                    SerializationFormat::Json => "json",
                }
            );
            let tensor = <f64 as TensorSerializable>::load_tensor(&filename, format)?;
            tensors.push(tensor);
        }

        Ok(tensors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_tensor_serialization_binary() {
        let original = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let mut original_with_grad = original.clone();
        original_with_grad.set_requires_grad(true);

        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("tensor.bin");

        // Save tensor
        original_with_grad
            .save(&file_path, SerializationFormat::Binary)
            .unwrap();

        // Load tensor
        let loaded = Tensor::<f32>::load(&file_path, SerializationFormat::Binary).unwrap();

        // Verify data
        assert_eq!(original_with_grad.data(), loaded.data());
        assert_eq!(original_with_grad.shape(), loaded.shape());
        assert_eq!(original_with_grad.requires_grad(), loaded.requires_grad());
    }

    #[test]
    fn test_tensor_serialization_json() {
        let original = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("tensor.json");

        // Save tensor
        original
            .save(&file_path, SerializationFormat::Json)
            .unwrap();

        // Load tensor
        let loaded = Tensor::<f32>::load(&file_path, SerializationFormat::Json).unwrap();

        // Verify data
        assert_eq!(original.data(), loaded.data());
        assert_eq!(original.shape(), loaded.shape());
    }

    #[test]
    fn test_state_dict_operations() {
        let mut state_dict = StateDict::new();

        let weight1 = Tensor::<f32>::from_vec(vec![1.0, 2.0], vec![2]);
        let weight2 = Tensor::<f32>::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        state_dict.insert_f32("layer1.weight", &weight1);
        state_dict.insert_f32("layer2.weight", &weight2);
        state_dict.set_metadata("version", "1.0");

        assert_eq!(state_dict.len(), 2);
        assert!(state_dict.get("layer1.weight").is_some());
        assert!(state_dict.get("nonexistent").is_none());
        assert_eq!(state_dict.get_metadata("version"), Some(&"1.0".to_string()));
    }

    #[test]
    fn test_state_dict_serialization() {
        let mut original_state_dict = StateDict::new();

        let weight = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let bias = Tensor::<f32>::from_vec(vec![0.1, 0.2], vec![2]);

        original_state_dict.insert_f32("weight", &weight);
        original_state_dict.insert_f32("bias", &bias);
        original_state_dict.set_metadata("epoch", "42");

        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("state_dict.bin");

        // Save state dict
        original_state_dict
            .save(&file_path, SerializationFormat::Binary)
            .unwrap();

        // Load state dict
        let loaded_state_dict = StateDict::load(&file_path, SerializationFormat::Binary).unwrap();

        // Verify contents
        assert_eq!(loaded_state_dict.len(), 2);
        assert!(loaded_state_dict.get("weight").is_some());
        assert!(loaded_state_dict.get("bias").is_some());
        assert_eq!(
            loaded_state_dict.get_metadata("epoch"),
            Some(&"42".to_string())
        );

        // Convert to tensors and verify
        let tensors = loaded_state_dict.to_tensors_f32().unwrap();
        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors["weight"].shape(), &[2, 2]);
        assert_eq!(tensors["bias"].shape(), &[2]);
    }
}
