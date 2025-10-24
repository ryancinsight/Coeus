//! SafeTensors format for secure model serialization.
//!
//! SafeTensors is a secure serialization format for tensors that prevents
//! arbitrary code execution when loading untrusted files. It stores tensors
//! in a binary format with metadata describing tensor shapes and data types.
//!
//! # Format Structure
//! ```text
//! +-------------------+
//! | Header (JSON)     |  <- Metadata about tensors
//! +-------------------+
//! | Tensor 1 Data     |  <- Raw tensor bytes
//! +-------------------+
//! | Tensor 2 Data     |  <- Raw tensor bytes
//! +-------------------+
//! | ...               |
//! +-------------------+
//! ```
//!
//! # Header Format
//! The header is a JSON object mapping tensor names to their metadata:
//! ```json
//! {
//!   "tensor_name": {
//!     "dtype": "F32",        // Data type
//!     "shape": [1, 2, 3],    // Tensor shape
//!     "data_offsets": [0, 24] // Start and end byte offsets in data section
//!   }
//! }
//! ```

/// Type alias for state dictionary entries: (data, shape).
type StateDict<T> = HashMap<String, (Vec<T>, Vec<usize>)>;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{NNError, Result};
use coeus_dtype::DataType;
use coeus_storage::{Storage, StorageFromVec};

/// Supported data types in SafeTensors format.
///
/// Matches PyTorch dtypes for compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafeDtype {
    /// 32-bit float
    F32,
    /// 64-bit float
    F64,
    /// 16-bit float (future support)
    F16,
    /// 8-bit signed integer (future support)
    I8,
    /// 16-bit signed integer (future support)
    I16,
    /// 32-bit signed integer (future support)
    I32,
    /// 64-bit signed integer (future support)
    I64,
    /// 8-bit unsigned integer (future support)
    U8,
    /// 16-bit unsigned integer (future support)
    U16,
    /// 32-bit unsigned integer (future support)
    U32,
    /// 64-bit unsigned integer (future support)
    U64,
}

impl SafeDtype {
    /// Get the size in bytes for this data type.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            SafeDtype::F32 => 4,
            SafeDtype::F64 => 8,
            SafeDtype::F16 => 2,
            SafeDtype::I8 => 1,
            SafeDtype::I16 => 2,
            SafeDtype::I32 => 4,
            SafeDtype::I64 => 8,
            SafeDtype::U8 => 1,
            SafeDtype::U16 => 2,
            SafeDtype::U32 => 4,
            SafeDtype::U64 => 8,
        }
    }
}

/// Metadata for a single tensor in SafeTensors format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    /// Data type of the tensor
    pub dtype: SafeDtype,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Byte offsets [start, end] in the data section
    pub data_offsets: [usize; 2],
}

/// SafeTensors header containing metadata for all tensors.
pub type SafeTensorsHeader = HashMap<String, TensorMetadata>;

/// SafeTensors format representation.
///
/// Contains the header metadata and raw tensor data.
#[derive(Debug, Clone)]
pub struct SafeTensors {
    /// Header with tensor metadata
    pub header: SafeTensorsHeader,
    /// Raw concatenated tensor data
    pub data: Vec<u8>,
}

impl SafeTensors {
    /// Create a SafeTensors from a state dictionary with shape information.
    ///
    /// # Arguments
    /// * `state_dict` - State dictionary mapping parameter names to (tensor_data, shape)
    ///
    /// # Returns
    /// A SafeTensors instance containing the serialized tensors.
    ///
    /// # Errors
    /// Returns `NNError::SerializationError` if tensor data types are unsupported.
    pub fn from_state_dict<T: DataType>(
        state_dict: &HashMap<String, (Vec<T>, Vec<usize>)>,
    ) -> Result<Self> {
        let mut header = SafeTensorsHeader::new();
        let mut data = Vec::new();
        let mut offset = 0;

        for (name, (tensor_data, shape)) in state_dict {
            // Determine dtype - for now we only support F32 and F64
            let dtype = match std::any::TypeId::of::<T>() {
                id if id == std::any::TypeId::of::<coeus_dtype::float::Float32>() => SafeDtype::F32,
                id if id == std::any::TypeId::of::<coeus_dtype::float::Float64>() => SafeDtype::F64,
                _ => {
                    return Err(NNError::SerializationError {
                        message: format!("Unsupported data type for tensor '{}'", name),
                    })
                }
            };

            // Serialize tensor data to bytes
            let tensor_bytes = serialize_tensor_data(tensor_data)?;
            let data_len = tensor_bytes.len();

            // Add to header
            header.insert(
                name.clone(),
                TensorMetadata {
                    dtype,
                    shape: shape.clone(),
                    data_offsets: [offset, offset + data_len],
                },
            );

            // Append to data
            data.extend_from_slice(&tensor_bytes);
            offset += data_len;
        }

        Ok(Self { header, data })
    }

    /// Load a state dictionary from SafeTensors.
    ///
    /// # Returns
    /// A state dictionary mapping parameter names to tensor data.
    ///
    /// # Errors
    /// Returns `NNError::SerializationError` if deserialization fails.
    pub fn to_state_dict<T: DataType>(&self) -> Result<StateDict<T>> {
        let mut state_dict = HashMap::new();

        for (name, metadata) in &self.header {
            // Extract tensor data from the data section
            let start = metadata.data_offsets[0];
            let end = metadata.data_offsets[1];
            let tensor_bytes = &self.data[start..end];

            // Deserialize tensor data
            let tensor_data = deserialize_tensor_data::<T>(tensor_bytes, &metadata.shape)?;

            state_dict.insert(name.clone(), (tensor_data, metadata.shape.clone()));
        }

        Ok(state_dict)
    }

    /// Serialize SafeTensors to bytes.
    ///
    /// # Returns
    /// A byte vector containing the complete SafeTensors file.
    ///
    /// # Errors
    /// Returns `NNError::SerializationError` if JSON serialization fails.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        // Serialize header to JSON
        let header_json =
            serde_json::to_string(&self.header).map_err(|e| NNError::SerializationError {
                message: format!("Failed to serialize header: {}", e),
            })?;

        // Convert to UTF-8 bytes and pad to 8-byte alignment
        let mut header_bytes = header_json.as_bytes().to_vec();
        let padding_len = (8 - (header_bytes.len() % 8)) % 8;
        header_bytes.extend(vec![0u8; padding_len]);

        // Combine header and data
        let mut result = header_bytes;
        result.extend_from_slice(&self.data);

        Ok(result)
    }

    /// Deserialize SafeTensors from bytes.
    ///
    /// # Arguments
    /// * `bytes` - The complete SafeTensors file as bytes
    ///
    /// # Returns
    /// A SafeTensors instance.
    ///
    /// # Errors
    /// Returns `NNError::SerializationError` if deserialization fails.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        // Find the first null byte to locate end of header
        let header_end =
            bytes
                .iter()
                .position(|&b| b == 0)
                .ok_or_else(|| NNError::SerializationError {
                    message: "Invalid SafeTensors format: no null terminator found".to_string(),
                })?;

        // Extract and deserialize header
        let header_json =
            std::str::from_utf8(&bytes[..header_end]).map_err(|e| NNError::SerializationError {
                message: format!("Invalid UTF-8 in header: {}", e),
            })?;
        let header: SafeTensorsHeader =
            serde_json::from_str(header_json).map_err(|e| NNError::SerializationError {
                message: format!("Failed to deserialize header: {}", e),
            })?;

        // Extract data section (skip padding to 8-byte alignment)
        let data_start = ((header_end + 8) / 8) * 8; // Round up to next 8-byte boundary
        let data = bytes[data_start..].to_vec();

        Ok(Self { header, data })
    }

    /// Save SafeTensors to a file.
    ///
    /// # Arguments
    /// * `path` - Path to save the SafeTensors file
    ///
    /// # Returns
    /// Result indicating success or failure.
    ///
    /// # Errors
    /// Returns `NNError::SerializationError` if file I/O fails.
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes).map_err(|e| NNError::SerializationError {
            message: format!("Failed to write SafeTensors file: {}", e),
        })?;
        Ok(())
    }

    /// Load SafeTensors from a file.
    ///
    /// # Arguments
    /// * `path` - Path to the SafeTensors file
    ///
    /// # Returns
    /// A SafeTensors instance.
    ///
    /// # Errors
    /// Returns `NNError::SerializationError` if file I/O or deserialization fails.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| NNError::SerializationError {
            message: format!("Failed to read SafeTensors file: {}", e),
        })?;
        Self::from_bytes(&bytes)
    }
}

/// Serialize tensor data to bytes in little-endian format.
///
/// # Arguments
/// * `tensor_data` - Slice of tensor data to serialize
///
/// # Returns
/// Byte vector containing the serialized tensor data.
fn serialize_tensor_data<T: DataType>(tensor_data: &[T]) -> Result<Vec<u8>> {
    // Use unsafe code to reinterpret the data as bytes
    // This is safe because we're just serializing to bytes for storage
    let bytes = unsafe {
        std::slice::from_raw_parts(
            tensor_data.as_ptr() as *const u8,
            std::mem::size_of_val(tensor_data),
        )
    };

    Ok(bytes.to_vec())
}

/// Deserialize tensor data from bytes in little-endian format.
///
/// # Arguments
/// * `bytes` - Byte slice containing serialized tensor data
/// * `shape` - Expected shape of the tensor
///
/// # Returns
/// Vector containing the deserialized tensor data.
fn deserialize_tensor_data<T: DataType>(bytes: &[u8], shape: &[usize]) -> Result<Vec<T>> {
    let expected_len: usize = shape.iter().product();
    let type_size = std::mem::size_of::<T>();
    let expected_bytes = expected_len * type_size;

    if bytes.len() != expected_bytes {
        return Err(NNError::SerializationError {
            message: format!(
                "Byte length mismatch: expected {} bytes, got {} bytes",
                expected_bytes,
                bytes.len()
            ),
        });
    }

    // Use unsafe code to reinterpret bytes as tensor data
    // This is safe because we validated the byte length matches the expected tensor size
    let tensor_data =
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, expected_len) };

    Ok(tensor_data.to_vec())
}

/// Model conversion utilities for PyTorch ↔ SafeTensors interoperability
pub mod conversion {
    use super::*;
    use crate::error::{NNError, Result};
    use crate::Module;
    use coeus_backend::{Backend, CpuBackend};
    use coeus_dtype::float::Float32;
    use coeus_tensor::{DenseStorage, Tensor};

    /// Convert a Coeus module's parameters to SafeTensors format
    ///
    /// # Arguments
    /// * `module` - The module to extract parameters from (must use Float32)
    ///
    /// # Returns
    /// SafeTensors instance containing the serialized model parameters
    ///
    /// # Errors
    /// Returns `NNError::SerializationError` if conversion fails
    pub fn module_to_safetensors<M, B, S>(module: &M) -> Result<SafeTensors>
    where
        M: Module<B, S, Float32>,
        B: Backend + Clone,
        S: Storage<Float32> + StorageFromVec<Float32> + Clone + 'static,
    {
        let mut safetensors_data = HashMap::new();

        // Get parameters from the module
        let params = module.parameters();

        for (i, param) in params.iter().enumerate() {
            let tensor = param.data();

            // Extract Float32 data
            let data: Vec<Float32> = tensor.as_slice().to_vec();

            let shape = tensor.shape().dims().to_vec();
            let name = format!("param_{}", i);
            safetensors_data.insert(name, (data, shape));
        }

        SafeTensors::from_state_dict(&safetensors_data)
    }

    /// Convert SafeTensors format to Coeus state dictionary
    ///
    /// # Arguments
    /// * `safetensors` - SafeTensors instance to convert
    ///
    /// # Returns
    /// Coeus state dictionary (tensor name → tensor)
    ///
    /// # Errors
    /// Returns `NNError::SerializationError` if conversion fails
    #[allow(clippy::type_complexity)]
    pub fn safetensors_to_state_dict<T>(
        safetensors: &SafeTensors,
    ) -> Result<
        std::collections::HashMap<String, Tensor<CpuBackend<T>, DenseStorage<Float32>, Float32>>,
    >
    where
        T: DataType,
    {
        let mut state_dict = std::collections::HashMap::new();
        let safetensors_dict: StateDict<Float32> = safetensors.to_state_dict()?;

        for (name, (data, shape)) in safetensors_dict.into_iter() {
            let tensor = Tensor::from_vec(data, &shape)?;
            state_dict.insert(name, tensor);
        }

        Ok(state_dict)
    }

    /// Convert PyTorch-style state dict (JSON) to SafeTensors
    ///
    /// # Arguments
    /// * `pytorch_state_dict` - JSON representation of PyTorch state dict
    ///
    /// # Returns
    /// SafeTensors instance
    ///
    /// # Errors
    /// Returns `NNError::SerializationError` if conversion fails
    pub fn pytorch_json_to_safetensors(
        pytorch_state_dict: &serde_json::Value,
    ) -> Result<SafeTensors> {
        if let serde_json::Value::Object(map) = pytorch_state_dict {
            let mut safetensors_data = StateDict::new();

            for (name, value) in map {
                if let serde_json::Value::Object(tensor_info) = value {
                    // Extract shape and data from PyTorch format
                    // This is a simplified implementation - real conversion would
                    // handle various PyTorch tensor formats
                    if let (Some(shape_val), Some(data_val)) =
                        (tensor_info.get("shape"), tensor_info.get("data"))
                    {
                        if let (
                            serde_json::Value::Array(shape_arr),
                            serde_json::Value::Array(data_arr),
                        ) = (shape_val, data_val)
                        {
                            let shape: Vec<usize> = shape_arr
                                .iter()
                                .filter_map(|v| v.as_u64().map(|n| n as usize))
                                .collect();

                            let data: Vec<f32> = data_arr
                                .iter()
                                .filter_map(|v| v.as_f64().map(|n| n as f32))
                                .collect();

                            // Convert to Float32
                            let float32_data: Vec<Float32> =
                                data.into_iter().map(Float32::new).collect();
                            safetensors_data.insert(name.clone(), (float32_data, shape));
                        }
                    }
                }
            }

            SafeTensors::from_state_dict(&safetensors_data)
        } else {
            Err(NNError::SerializationError {
                message: "Invalid PyTorch state dict format".to_string(),
            })
        }
    }

    /// Convert SafeTensors to PyTorch-style JSON format
    ///
    /// # Arguments
    /// * `safetensors` - SafeTensors instance to convert
    ///
    /// # Returns
    /// JSON representation compatible with PyTorch
    ///
    /// # Errors
    /// Returns `NNError::SerializationError` if conversion fails
    pub fn safetensors_to_pytorch_json(safetensors: &SafeTensors) -> Result<serde_json::Value> {
        let state_dict: StateDict<Float32> = safetensors.to_state_dict()?;
        let mut pytorch_dict = serde_json::Map::new();

        for (name, (data, shape)) in state_dict {
            let data_f32: Vec<f32> = data.into_iter().map(|x| x.get()).collect();
            let shape_u64: Vec<u64> = shape.into_iter().map(|s| s as u64).collect();

            let tensor_info = serde_json::json!({
                "shape": shape_u64,
                "data": data_f32,
                "dtype": "float32"
            });

            pytorch_dict.insert(name, tensor_info);
        }

        Ok(serde_json::Value::Object(pytorch_dict))
    }

    /// Validate SafeTensors format compatibility
    ///
    /// # Arguments
    /// * `safetensors` - SafeTensors instance to validate
    ///
    /// # Returns
    /// Validation result with any compatibility issues
    ///
    /// # Errors
    /// Returns `NNError::SerializationError` if validation fails
    pub fn validate_safetensors_format(safetensors: &SafeTensors) -> Result<Vec<String>> {
        let mut issues = Vec::new();

        // Check for supported data types
        for (name, metadata) in &safetensors.header {
            match metadata.dtype {
                SafeDtype::F32 | SafeDtype::F64 => {
                    // Supported
                }
                _ => {
                    issues.push(format!(
                        "Tensor '{}' has unsupported dtype {:?}",
                        name, metadata.dtype
                    ));
                }
            }

            // Check shape consistency
            let expected_size: usize = metadata.shape.iter().product();
            let data_size = (metadata.data_offsets[1] - metadata.data_offsets[0])
                / metadata.dtype.size_in_bytes();

            if expected_size != data_size {
                issues.push(format!(
                    "Tensor '{}' shape/size mismatch: shape implies {} elements, data has {} elements",
                    name, expected_size, data_size
                ));
            }
        }

        Ok(issues)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::linear::Linear;

        #[test]
        fn test_module_conversion() {
            // Create a simple linear layer
            let layer =
                Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 5).unwrap();

            // Convert to SafeTensors
            let safetensors = module_to_safetensors(&layer).unwrap();

            // Convert back to tensors
            let recovered_state_dict = safetensors_to_state_dict::<Float32>(&safetensors).unwrap();

            // Get original parameters
            let original_params = layer.parameters();

            // Verify we have the same number of parameters
            assert_eq!(original_params.len(), recovered_state_dict.len());

            // Verify each parameter matches
            for (i, original_param) in original_params.iter().enumerate() {
                let name = format!("param_{}", i);
                assert!(recovered_state_dict.contains_key(&name));
                let recovered_tensor = &recovered_state_dict[&name];

                // Compare shapes
                assert_eq!(
                    original_param.data().shape().dims(),
                    recovered_tensor.shape().dims()
                );

                // Compare data (within floating point precision)
                let orig_data: Vec<f32> = original_param
                    .data()
                    .as_slice()
                    .iter()
                    .map(|x| x.get())
                    .collect();
                let recovered_data: Vec<f32> = recovered_tensor
                    .as_slice()
                    .iter()
                    .map(|x| x.get())
                    .collect();

                for (a, b) in orig_data.iter().zip(recovered_data.iter()) {
                    assert!((a - b).abs() < 1e-6, "Data mismatch: {} vs {}", a, b);
                }
            }
        }

        #[test]
        fn test_pytorch_json_conversion() {
            // Create sample PyTorch-style JSON
            let pytorch_json = serde_json::json!({
                "weight": {
                    "shape": [5, 10],
                    "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                             11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                             21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                             31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                             41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0],
                    "dtype": "float32"
                },
                "bias": {
                    "shape": [5],
                    "data": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "dtype": "float32"
                }
            });

            // Convert to SafeTensors
            let safetensors = pytorch_json_to_safetensors(&pytorch_json).unwrap();

            // Verify structure
            assert_eq!(safetensors.header.len(), 2);
            assert!(safetensors.header.contains_key("weight"));
            assert!(safetensors.header.contains_key("bias"));

            // Check weight tensor
            let weight_meta = &safetensors.header["weight"];
            assert_eq!(weight_meta.shape, vec![5, 10]);
            assert_eq!(weight_meta.dtype, SafeDtype::F32);

            // Check bias tensor
            let bias_meta = &safetensors.header["bias"];
            assert_eq!(bias_meta.shape, vec![5]);
            assert_eq!(bias_meta.dtype, SafeDtype::F32);
        }

        #[test]
        fn test_format_validation() {
            // Create valid SafeTensors
            let mut state_dict = StateDict::new();
            let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
            state_dict.insert("valid_tensor".to_string(), (data, vec![3]));
            let safetensors = SafeTensors::from_state_dict(&state_dict).unwrap();

            // Validate
            let issues = validate_safetensors_format(&safetensors).unwrap();
            assert!(
                issues.is_empty(),
                "Valid SafeTensors should have no issues: {:?}",
                issues
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_dtype_size() {
        assert_eq!(SafeDtype::F32.size_in_bytes(), 4);
        assert_eq!(SafeDtype::F64.size_in_bytes(), 8);
        assert_eq!(SafeDtype::I32.size_in_bytes(), 4);
    }

    #[test]
    fn test_safetensors_roundtrip() {
        use coeus_dtype::float::Float32;

        // Create test data
        let test_data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
        ];
        let test_shape = vec![2, 3];
        let mut state_dict = HashMap::new();
        state_dict.insert("test_tensor".to_string(), (test_data, test_shape));

        // Serialize to SafeTensors
        let safetensors = SafeTensors::from_state_dict(&state_dict).unwrap();

        // Serialize to bytes
        let bytes = safetensors.to_bytes().unwrap();

        // Deserialize from bytes
        let deserialized = SafeTensors::from_bytes(&bytes).unwrap();

        // Deserialize to state dict
        let recovered_state: HashMap<String, (Vec<Float32>, Vec<usize>)> =
            deserialized.to_state_dict().unwrap();

        // Verify the data matches
        assert_eq!(recovered_state.len(), 1);
        assert!(recovered_state.contains_key("test_tensor"));

        let (recovered_data, recovered_shape) = &recovered_state["test_tensor"];
        assert_eq!(recovered_shape, &[2, 3]);
        assert_eq!(recovered_data.len(), 6);
        assert_eq!(recovered_data[0].get(), 1.0);
        assert_eq!(recovered_data[1].get(), 2.0);
        assert_eq!(recovered_data[2].get(), 3.0);
        assert_eq!(recovered_data[3].get(), 4.0);
        assert_eq!(recovered_data[4].get(), 5.0);
        assert_eq!(recovered_data[5].get(), 6.0);
    }
}

// Future enhancement: Add rkyv zero-copy serialization for state dictionaries
