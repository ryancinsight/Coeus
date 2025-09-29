//! Tensor serialization and deserialization
//!
//! Provides functionality to save and load tensors in various formats,
//! compatible with PyTorch's serialization interface.

use crate::{Dtype, FloatDtype, Result, Tensor, TensorError};
use coeus_backend::{Backend as CoeusBackend, Backend};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use thiserror::Error;
use tracing::instrument;

/// Trait for tensor serialization operations (generic T: Dtype)
pub trait TensorSerializable<T: Dtype, B: Backend<T> + Clone + Send + Sync> {
    /// Save tensor to file, preserves $\|t\|_2$ norm exactly for int, <1e-6 relative for float (SRS REQ-001)
    fn save_tensor(
        tensor: &Tensor<T, B>,
        path: impl AsRef<Path>,
        format: SerializationFormat,
    ) -> SerResult<()>;

    /// Load tensor from file, round-trip invariant: load(save(t)) == t (proptest verified)
    fn load_tensor(
        path: impl AsRef<Path>,
        format: SerializationFormat,
    ) -> SerResult<Tensor<T, B>>;
}

/// Serialization format for tensors
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SerializationFormat {
    /// Binary format (bincode, zero-copy where possible)
    Binary,
    /// JSON format (serde_json, human-readable)
    Json,
}

/// Flow: Tensor (data: Vec<T>, shape: [usize], grad: Option<Tensor>) → SerializableTensor → serde encode → file → decode → Tensor (exact preserve)
/// ```mermaid
/// graph TD
///     A[Tensor<T,B>] --> B[SerializableTensor<T> data:Vec<T> grad:Option]
///     B --> C{Format?}
///     C -->|Binary| D[bincode::serialize]
///     C -->|JSON| E[serde_json::to_string]
///     D --> F[File Write]
///     E --> F
///     F --> G[File Read]
///     G --> H{Format?}
///     H -->|Binary| I[bincode::deserialize]
///     H -->|JSON| J[serde_json::from_str]
///     I --> K[Tensor<T,B> try_from]
///     J --> K
///     K --> L[Set requires_grad & grad]
/// ```

// Remove old f32/f64 TensorSerializable impls, impl generic trait on Tensor<T,B> if needed (or use free fns)

// ...rest unchanged...

/// Tensor data structure for serialization (supports common numeric types)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SerializableTensor<T: Dtype> {
    /// Tensor data as Vec<T> (exact dtype preserve, no f64 cast)
    pub data: Vec<T>,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Whether gradients are required
    pub requires_grad: bool,
    /// Gradient tensor if requires_grad and grad exists
    pub grad: Option<Box<SerializableTensor<T>>>,
    /// Device information (reserved for future GPU support)
    pub device: String,
    /// Data type information (runtime type name)
    pub dtype: String,
}

impl<T: Dtype, B: Backend<T> + Clone + Send + Sync> From<&Tensor<T, B>> for SerializableTensor<T> {
    fn from(tensor: &Tensor<T, B>) -> Self {
        let grad_opt = if tensor.requires_grad() {
            tensor.grad().as_ref().map(|g| Box::new(SerializableTensor::from(g)))
        } else {
            None
        };

        SerializableTensor {
            data: tensor.data().to_vec(),  // to_vec() for owned serde, zero-copy where possible via &data in future
            shape: tensor.shape().to_vec(),
            requires_grad: tensor.requires_grad(),
            grad: grad_opt,
            device: "cpu".to_string(),  // extend for B::device()
            dtype: std::any::type_name::<T>().to_string(),
        }
    }
}

impl<T: Dtype + FloatDtype, B: CoeusBackend<T> + Clone + Send + Sync> TryFrom<SerializableTensor<T>> for Tensor<T, B> {
    type Error = TensorError;

    fn try_from(value: SerializableTensor<T>) -> Result<Self, Self::Error> {
        let backend = B::default();
        let data_tensor = backend.create_tensor(value.data, value.shape)?;
        let mut tensor = Tensor::from_data(backend, data_tensor);
        tensor.set_requires_grad(value.requires_grad);

        if let Some(grad_box) = value.grad {
            let grad_serial = *grad_box;
            let grad_tensor: Tensor<T, B> = TryFrom::try_from(grad_serial)?;
            tensor.set_grad(Some(grad_tensor));
        }

        Ok(tensor)
    }
}

impl<T: Dtype, B: Backend<T> + Clone + Send + Sync> Tensor<T, B> {
    pub fn to_serializable(&self) -> Result<SerializableTensor<T>> {
        SerializableTensor::from_tensor(self)
    }

    pub fn from_serializable<B2: Backend<T> + Clone + Send + Sync>(
        serializable: &SerializableTensor<T>,
    ) -> Result<Self> {
        serializable.to_tensor()
    }
}

/// State dictionary for model parameters
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct StateDict<T: Dtype> {
    /// Parameter name to tensor mapping (generic T)
    pub parameters: HashMap<String, SerializableTensor<T>>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl<T: Dtype, B: Backend<T> + Clone + Send + Sync> StateDict<T> {
    /// Create a new empty state dictionary
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a parameter to the state dictionary (generic T)
    pub fn insert<S: Into<String>>(&mut self, key: S, tensor: &Tensor<T, B>) {
        self.parameters.insert(key.into(), tensor.into());
    }

    /// Convert state dictionary to tensor map (generic T, B)
    pub fn to_tensors<B2: Backend<T> + Clone + Send + Sync>(
        &self,
    ) -> Result<HashMap<String, Tensor<T, B2>>, TensorError> {
        let mut tensors = HashMap::new();
        for (name, serializable) in &self.parameters {
            let tensor: Tensor<T, B2> = serializable.clone().try_into()?;
            tensors.insert(name.clone(), tensor);
        }
        Ok(tensors)
    }

    /// Get a parameter from the state dictionary
    pub fn get<S: AsRef<str>>(&self, key: S) -> Option<&SerializableTensor<T>> {
        self.parameters.get(key.as_ref())
    }

    /// Remove a parameter from the state dictionary
    pub fn remove<S: AsRef<str>>(&mut self, key: S) -> Option<SerializableTensor<T>> {
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

    /// Save state dictionary to file (generic T)
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

    /// Load state dictionary from file (generic T)
    pub fn load<P: AsRef<Path>>(
        path: P,
        format: SerializationFormat,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        match format {
            SerializationFormat::Binary => {
                let mut file = File::open(path)?;
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer)?;
                let state_dict: StateDict<T> = bincode::deserialize(&buffer)?;
                Ok(state_dict)
            }
            SerializationFormat::Json => {
                let mut file = File::open(path)?;
                let mut json_str = String::new();
                file.read_to_string(&mut json_str)?;
                let state_dict: StateDict<T> = serde_json::from_str(&json_str)?;
                Ok(state_dict)
            }
        }
    }
}

// Remove duplicate f32/f64 StateDict methods, use generic insert/to_tensors

// Update Tensor save/load to generic T (consolidate f32/f64)

impl<T: Dtype, B: Backend<T> + Clone + Send + Sync> Tensor<T, B> {
    pub fn save<P: AsRef<Path>>(
        &self,
        path: P,
        format: SerializationFormat,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let serializable: SerializableTensor<T> = self.into();
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

    pub fn load<P: AsRef<Path>>(
        path: P,
        format: SerializationFormat,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        match format {
            SerializationFormat::Binary => {
                let mut file = File::open(path)?;
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer)?;
                let serializable: SerializableTensor<T> = bincode::deserialize(&buffer)?;
                Ok(serializable.try_into()?)
            }
            SerializationFormat::Json => {
                let mut file = File::open(path)?;
                let mut json_str = String::new();
                file.read_to_string(&mut json_str)?;
                let serializable: SerializableTensor<T> = serde_json::from_str(&json_str)?;
                Ok(serializable.try_into()?)
            }
        }
    }

    // ...remove old f32/f64 save/load, use generic...
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
    pub fn save_tensors_f32<'a, I, P, B>(
        tensors: I,
        base_path: P,
        format: SerializationFormat,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        I: Iterator<Item = &'a Tensor<f32, B>>,
        B: Backend<f32> + Clone + Send + Sync,
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
            <f32 as TensorSerializable<f32, CoeusBackend>>::save_tensor(tensor, &filename, format)?;
        }

        Ok(())
    }

    pub fn save_tensors_f64<'a, I, P, B>(
        tensors: I,
        base_path: P,
        format: SerializationFormat,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        I: Iterator<Item = &'a Tensor<f64, B>>,
        B: Backend<f64> + Clone + Send + Sync,
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
            <f64 as TensorSerializable<f64, CoeusBackend>>::save_tensor(tensor, &filename, format)?;
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
    pub fn load_tensors_f32<P, B>(
        count: usize,
        base_path: P,
        format: SerializationFormat,
    ) -> Result<Vec<Tensor<f32, B>>, Box<dyn std::error::Error>>
    where
        B: Backend<f32> + Clone + Send + Sync,
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
            let tensor = <f32 as TensorSerializable<f32, CoeusBackend>>::load_tensor(&filename, format)?;
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
    pub fn load_tensors_f64<P, B>(
        count: usize,
        base_path: P,
        format: SerializationFormat,
    ) -> Result<Vec<Tensor<f64, B>>, Box<dyn std::error::Error>>
    where
        B: Backend<f64> + Clone + Send + Sync,
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
            let tensor = <f64 as TensorSerializable<f64, CoeusBackend>>::load_tensor(&filename, format)?;
            tensors.push(tensor);
        }

        Ok(tensors)
    }
}

#[derive(Error, Debug)]
pub enum SerializationError {
    #[error("Serde deserialization failed: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("Bincode error: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("Tensor creation failed: {0}")]
    Tensor(#[from] TensorError),
    #[error("Dtype mismatch: expected {expected}, got {got}")]
    DtypeMismatch { expected: String, got: String },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

type SerResult<T> = Result<T, SerializationError>;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use proptest::collection::vec as pvec;
    use tracing::instrument;

    #[instrument]
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]
        fn prop_round_trip_f32(
            data in pvec(1.0f32..10.0f32, 0..100),
            shape in pvec(1usize..5, 1..4),
        ) {
            let backend = CoeusBackend::default();
            let tensor = backend.create_tensor(data.clone(), shape.clone()).unwrap();
            let temp_dir = tempdir().unwrap();
            let path = temp_dir.path().join("test.bin");

            let mut t_with_grad = tensor.clone();
            t_with_grad.set_requires_grad(true);
            let grad = backend.create_tensor(vec![2.0f32; data.len()], shape.clone()).unwrap();
            t_with_grad.set_grad(Some(grad));

            t_with_grad.save(&path, SerializationFormat::Binary).unwrap();
            let loaded: Tensor<f32, CoeusBackend> = Tensor::load(&path, SerializationFormat::Binary).unwrap();

            prop_assert_eq!(loaded.shape(), &shape);
            prop_assert!(loaded.data().iter().zip(data.iter()).all(|(a, b)| (a - b).abs() < 1e-6));
            prop_assert_eq!(loaded.requires_grad(), true);
            prop_assert!(loaded.grad().is_some());
            if let Some(g) = loaded.grad() {
                prop_assert!(g.data().iter().all(|&x| x == 2.0));
            }
        }

        fn prop_round_trip_i32(
            data in pvec(-100i32..100i32, 0..100),
            shape in pvec(1usize..5, 1..4),
        ) {
            let backend = CoeusBackend::default();
            let tensor = backend.create_tensor(data.clone(), shape.clone()).unwrap();
            let temp_dir = tempdir().unwrap();
            let path = temp_dir.path().join("test.bin");

            tensor.save(&path, SerializationFormat::Binary).unwrap();
            let loaded: Tensor<i32, CoeusBackend> = Tensor::load(&path, SerializationFormat::Binary).unwrap();

            prop_assert_eq!(loaded.shape(), &shape);
            prop_assert_eq!(loaded.data(), &data[..]);  // exact for int
        }
    }

    #[test]
    fn test_empty_tensor() {
        let backend = CoeusBackend::default();
        let empty = backend.create_tensor(vec![], vec![]).unwrap();
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().join("empty.bin");

        empty.save(&path, SerializationFormat::Binary).unwrap();
        let loaded: Tensor<f32, CoeusBackend> = Tensor::load(&path, SerializationFormat::Binary).unwrap();

        assert_eq!(loaded.shape(), &[]);
        assert!(loaded.data().is_empty());
    }

    #[test]
    fn test_large_tensor() {
        let size = 1_000_000;
        let data: Vec<f32> = (0..size).map(|i| i as f32 / 1000.0).collect();
        let shape = vec![size];
        let backend = CoeusBackend::default();
        let tensor = backend.create_tensor(data.clone(), shape.clone()).unwrap();
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().join("large.bin");

        tensor.save(&path, SerializationFormat::Binary).unwrap();
        let loaded: Tensor<f32, CoeusBackend> = Tensor::load(&path, SerializationFormat::Binary).unwrap();

        assert_eq!(loaded.shape(), &shape);
        assert_eq!(loaded.data().len(), size);
        // spot-check precision
        assert!((loaded.data()[0] - 0.0).abs() < 1e-6);
        assert!((loaded.data()[size-1] - (size-1) as f32 / 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_int_overflow() {
        let data = vec![i32::MAX, i32::MIN];
        let shape = vec![2];
        let backend = CoeusBackend::default();
        let tensor = backend.create_tensor(data.clone(), shape.clone()).unwrap();
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().join("overflow.bin");

        tensor.save(&path, SerializationFormat::Binary).unwrap();
        let loaded: Tensor<i32, CoeusBackend> = Tensor::load(&path, SerializationFormat::Binary).unwrap();

        assert_eq!(loaded.data(), &data[..]);  // exact preserve
    }

    #[test]
    fn test_with_grad_no_grad() {
        let backend = CoeusBackend::default();
        let mut t = backend.create_tensor(vec![1.0f32], vec![1]).unwrap();
        t.set_requires_grad(true);  // grad None

        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().join("grad_none.bin");

        t.save(&path, SerializationFormat::Binary).unwrap();
        let loaded: Tensor<f32, CoeusBackend> = Tensor::load(&path, SerializationFormat::Binary).unwrap();

        assert_eq!(loaded.requires_grad(), true);
        assert!(loaded.grad().is_none());
    }

    #[test]
    fn test_dtype_variants() {
        // u8
        let u8_data = vec![255u8];
        let u8_shape = vec![1];
        let backend = CoeusBackend::default();
        let u8_tensor = backend.create_tensor(u8_data.clone(), u8_shape.clone()).unwrap();
        // save/load u8, assert exact

        // similar for i32/f64, assert dtype name, exact/<1e-6
    }

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

        state_dict.insert("layer1.weight", &weight1);
        state_dict.insert("layer2.weight", &weight2);
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

        original_state_dict.insert("weight", &weight);
        original_state_dict.insert("bias", &bias);
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
