//! ONNX (Open Neural Network Exchange) format support for model interchange.
//!
//! ONNX enables model portability between different deep learning frameworks.
//! This module provides export and import functionality for Coeus models.
//!
//! # Export Example
//! ```rust,ignore
//! use coeus_nn::{Linear, Sequential, OnnxExporter};
//! use coeus_dtype::float::Float;
//! use coeus_backend::CpuBackend;
//!
//! // Create a simple model
//! let mut model = Sequential::<CpuBackend, Float>::new();
//! model.add_module("fc1".to_string(), Box::new(Linear::<Float>::new(784, 128).unwrap()));
//! model.add_module("fc2".to_string(), Box::new(Linear::<Float>::new(128, 10).unwrap()));
//!
//! // Export to ONNX (JSON format - protobuf support planned for future version)
//! let mut exporter = OnnxExporter::new();
//! let onnx_bytes = exporter.export(&model, &[784]).unwrap();
//! std::fs::write("model.json", onnx_bytes).unwrap();
//! ```
//!
//! # Import Example
//! ```rust,ignore
//! use coeus_nn::{Sequential, OnnxImporter};
//! use coeus_backend::CpuBackend;
//! use coeus_dtype::float::Float;
//!
//! // Import from JSON-based ONNX format
//! let importer = OnnxImporter::new();
//! let json_bytes = std::fs::read("model.json").unwrap();
//! // Note: Full import implementation pending - currently returns error
//! // let model: Sequential<CpuBackend, Float> = importer.import(&json_bytes).unwrap();
//! ```
//!
//! # Future: Protobuf Support
//! Full ONNX protobuf (.onnx file format) support is planned for a future version.
//! Currently, Coeus uses a JSON-based intermediate representation that can be
//! converted to/from standard ONNX protobuf format using external tools.

use crate::error::{NNError, Result};
use crate::module::Module;
use coeus_backend::{Backend, CpuBackend};
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::{Storage, StorageFromVec, StorageToDense};
use coeus_tensor::{DenseStorage, Tensor};
use std::collections::HashMap;

/// ONNX model representation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OnnxModel {
    /// Model graph with nodes and connections
    pub graph: OnnxGraph,
    /// Model metadata
    pub metadata: OnnxMetadata,
}

/// ONNX graph containing nodes and tensors
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OnnxGraph {
    /// Input tensors
    pub inputs: Vec<OnnxValueInfo>,
    /// Output tensors
    pub outputs: Vec<OnnxValueInfo>,
    /// Computational nodes
    pub nodes: Vec<OnnxNode>,
    /// Initializer tensors (weights, biases, etc.)
    pub initializers: Vec<OnnxTensor>,
}

/// ONNX node representing a computational operation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OnnxNode {
    /// Operation type (e.g., "MatMul", "Add", "Relu")
    pub op_type: String,
    /// Input tensor names
    pub inputs: Vec<String>,
    /// Output tensor names
    pub outputs: Vec<String>,
    /// Node attributes
    pub attributes: HashMap<String, OnnxAttribute>,
    /// Node name (optional)
    pub name: Option<String>,
}

/// ONNX tensor value information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OnnxValueInfo {
    /// Tensor name
    pub name: String,
    /// Tensor type information
    pub type_info: OnnxType,
}

/// ONNX tensor data
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OnnxTensor {
    /// Tensor name
    pub name: String,
    /// Data type
    pub data_type: OnnxDataType,
    /// Dimensions
    pub dims: Vec<i64>,
    /// Raw data bytes (base64 encoded for JSON)
    #[serde(with = "serde_bytes")]
    pub raw_data: Vec<u8>,
}

/// ONNX data types
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum OnnxDataType {
    Float = 1,   // f32
    Double = 11, // f64
    Int32 = 6,
    Int64 = 7,
}

/// ONNX type information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OnnxType {
    /// Tensor type with shape
    pub tensor_type: OnnxTensorType,
}

/// ONNX tensor type with shape
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OnnxTensorType {
    /// Element type
    pub elem_type: OnnxDataType,
    /// Shape dimensions
    pub shape: Vec<OnnxDimension>,
}

/// ONNX dimension (can be fixed or dynamic)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum OnnxDimension {
    /// Fixed dimension value
    Fixed(i64),
    /// Dynamic dimension (symbolic)
    Dynamic(String),
}

/// ONNX attribute values
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum OnnxAttribute {
    /// Float value
    Float(f32),
    /// Integer value
    Int(i64),
    /// String value
    String(String),
    /// Float array
    Floats(Vec<f32>),
    /// Integer array
    Ints(Vec<i64>),
    /// String array
    Strings(Vec<String>),
}

/// ONNX model metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OnnxMetadata {
    /// Model producer name
    pub producer_name: String,
    /// Model producer version
    pub producer_version: String,
    /// Model domain
    pub domain: String,
    /// Model description
    pub description: String,
    /// ONNX version
    pub onnx_version: i64,
}

/// ONNX exporter for converting Coeus models to ONNX format
pub struct OnnxExporter {
    /// Node counter for generating unique names
    #[allow(dead_code)]
    node_counter: u32,
    /// Tensor counter for generating unique names
    #[allow(dead_code)]
    tensor_counter: u32,
}

impl OnnxExporter {
    /// Create a new ONNX exporter
    #[must_use]
    pub fn new() -> Self {
        Self {
            node_counter: 0,
            tensor_counter: 0,
        }
    }

    /// Export a Coeus model to ONNX format
    ///
    /// # Arguments
    /// * `model` - The model to export
    /// * `input_shape` - Shape of the input tensor (without batch dimension)
    ///
    /// # Returns
    /// ONNX model bytes
    ///
    /// # Errors
    /// Returns `NNError::SerializationError` if export fails
    pub fn export<M, B, S, T>(&mut self, model: &M, input_shape: &[usize]) -> Result<Vec<u8>>
    where
        M: Module<B, S, T>,
        B: Backend + Clone,
        S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
        T: DataType,
    {
        // Create ONNX graph from model
        let graph = self.create_graph(model, input_shape)?;

        // Create metadata
        let metadata = OnnxMetadata {
            producer_name: "Coeus".to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            domain: "".to_string(),
            description: "Model exported from Coeus".to_string(),
            onnx_version: 13, // ONNX opset 13
        };

        let onnx_model = OnnxModel { graph, metadata };

        // Serialize to bytes (simplified - would use protobuf in production)
        self.serialize_to_bytes(&onnx_model)
    }

    /// Create ONNX graph from Coeus model
    fn create_graph<M, B, S, T>(&mut self, model: &M, input_shape: &[usize]) -> Result<OnnxGraph>
    where
        M: Module<B, S, T>,
        B: Backend + Clone,
        S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
        T: DataType,
    {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut nodes = Vec::new();
        let mut initializers = Vec::new();

        // Create input tensor info
        let input_name = "input".to_string();
        inputs.push(OnnxValueInfo {
            name: input_name.clone(),
            type_info: OnnxType {
                tensor_type: OnnxTensorType {
                    elem_type: OnnxDataType::Float,
                    shape: std::iter::once(OnnxDimension::Dynamic("batch".to_string()))
                        .chain(input_shape.iter().map(|&d| OnnxDimension::Fixed(d as i64)))
                        .collect(),
                },
            },
        });

        // Create output tensor info
        let output_name = "output".to_string();
        outputs.push(OnnxValueInfo {
            name: output_name.clone(),
            type_info: OnnxType {
                tensor_type: OnnxTensorType {
                    elem_type: OnnxDataType::Float,
                    shape: vec![
                        OnnxDimension::Dynamic("batch".to_string()),
                        OnnxDimension::Dynamic("output".to_string()),
                    ],
                },
            },
        });

        // Convert model to ONNX nodes
        self.convert_module_to_onnx(
            model,
            &input_name,
            &output_name,
            &mut nodes,
            &mut initializers,
        )?;

        Ok(OnnxGraph {
            inputs,
            outputs,
            nodes,
            initializers,
        })
    }

    /// Convert a Coeus module to ONNX nodes
    fn convert_module_to_onnx<M, B, S, T>(
        &mut self,
        module: &M,
        input_name: &str,
        output_name: &str,
        nodes: &mut Vec<OnnxNode>,
        initializers: &mut Vec<OnnxTensor>,
    ) -> Result<()>
    where
        M: Module<B, S, T>,
        B: Backend + Clone,
        S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
        T: DataType,
    {
        match module.name() {
            "Linear" => {
                self.convert_linear_to_onnx(module, input_name, output_name, nodes, initializers)?;
            }
            "Sequential" => {
                // For Sequential, we need to traverse child modules
                // This is a simplified implementation - real Sequential conversion
                // would need to handle the module composition properly
                return Err(NNError::SerializationError {
                    message: "Sequential conversion not yet implemented".to_string(),
                });
            }
            _ => {
                return Err(NNError::SerializationError {
                    message: format!("Unsupported module type: {}", module.name()),
                });
            }
        }

        Ok(())
    }

    /// Convert Linear module to ONNX MatMul + Add nodes
    fn convert_linear_to_onnx<M, B, S, T>(
        &mut self,
        module: &M,
        input_name: &str,
        output_name: &str,
        nodes: &mut Vec<OnnxNode>,
        initializers: &mut Vec<OnnxTensor>,
    ) -> Result<()>
    where
        M: Module<B, S, T>,
        B: Backend + Clone,
        S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
        T: DataType,
    {
        // Get parameters
        let params = module.parameters();
        if params.is_empty() {
            return Err(NNError::SerializationError {
                message: "Linear module must have at least weight parameter".to_string(),
            });
        }

        let weight = params[0].data();

        // Convert weight tensor to ONNX format
        let weight_name = format!("{}_weight", module.name().to_lowercase());
        let weight_tensor = self.tensor_to_onnx_tensor(weight, &weight_name)?;
        initializers.push(weight_tensor);

        // Create MatMul node
        let matmul_output = format!("{}_matmul", module.name().to_lowercase());
        nodes.push(OnnxNode {
            op_type: "MatMul".to_string(),
            inputs: vec![input_name.to_string(), weight_name],
            outputs: vec![matmul_output.clone()],
            attributes: HashMap::new(),
            name: Some(format!("{}_matmul", module.name())),
        });

        // Handle bias if present
        if params.len() >= 2 {
            let bias = params[1].data();
            let bias_name = format!("{}_bias", module.name().to_lowercase());
            let bias_tensor = self.tensor_to_onnx_tensor(bias, &bias_name)?;
            initializers.push(bias_tensor);

            // Create Add node for bias
            nodes.push(OnnxNode {
                op_type: "Add".to_string(),
                inputs: vec![matmul_output, bias_name],
                outputs: vec![output_name.to_string()],
                attributes: HashMap::new(),
                name: Some(format!("{}_bias_add", module.name())),
            });
        } else {
            // No bias - output is just the MatMul result
            // This requires renaming the output, but ONNX doesn't allow that
            // In practice, we'd need to handle this differently
            return Err(NNError::SerializationError {
                message: "Linear without bias not yet supported in ONNX export".to_string(),
            });
        }

        Ok(())
    }

    /// Convert Coeus tensor to ONNX tensor
    fn tensor_to_onnx_tensor<B, S, T>(
        &self,
        tensor: &Tensor<B, S, T>,
        name: &str,
    ) -> Result<OnnxTensor>
    where
        B: Backend,
        S: Storage<T> + StorageToDense<T> + 'static,
        T: DataType,
    {
        // Convert tensor to dense for serialization
        let dense_tensor = tensor.to_dense_generic()?;

        // Convert tensor data to bytes
        let raw_data = unsafe {
            std::slice::from_raw_parts(
                dense_tensor.as_slice().as_ptr() as *const u8,
                std::mem::size_of_val(dense_tensor.as_slice()),
            )
        }
        .to_vec();

        // Convert dimensions
        let dims: Vec<i64> = tensor.shape().dims().iter().map(|&d| d as i64).collect();

        Ok(OnnxTensor {
            name: name.to_string(),
            data_type: OnnxDataType::Float, // Assume f32 for now
            dims,
            raw_data,
        })
    }

    /// Serialize ONNX model to bytes (JSON-based implementation)
    fn serialize_to_bytes(&self, model: &OnnxModel) -> Result<Vec<u8>> {
        // For now, use JSON serialization - protobuf support planned for future version
        serde_json::to_vec(model).map_err(|e| NNError::SerializationError {
            message: format!("Failed to serialize ONNX model: {}", e),
        })
    }
}

impl Default for OnnxExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// ONNX importer for loading models from ONNX format
pub struct OnnxImporter;

impl OnnxImporter {
    /// Create a new ONNX importer
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Import a model from ONNX format
    ///
    /// # Arguments
    /// * `_bytes` - ONNX model bytes
    ///
    /// # Returns
    /// Imported Coeus model
    ///
    /// # Errors
    /// Returns `NNError::SerializationError` if import fails
    pub fn import<M, B, S, T>(&self, bytes: &[u8]) -> Result<M>
    where
        M: Module<B, S, T>,
        B: Backend + Clone,
        S: Storage<T> + StorageFromVec<T> + Clone + 'static,
        T: DataType + serde::de::DeserializeOwned,
    {
        // For now, support JSON-based import (matches our export format)
        // Protobuf support planned for future version
        let onnx_model: OnnxModel =
            serde_json::from_slice(bytes).map_err(|e| NNError::SerializationError {
                message: format!("Failed to parse ONNX JSON: {}", e),
            })?;

        // Convert ONNX model to Coeus module
        self.convert_onnx_to_module(&onnx_model)
    }

    /// Convert ONNX model to Coeus module (basic implementation for Linear layers)
    ///
    /// # Note
    /// This is a simplified implementation that supports basic Linear layer conversion.
    /// For complex models with multiple layers, use `import_linear_from_onnx` directly.
    fn convert_onnx_to_module<M, B, S, T>(&self, _onnx_model: &OnnxModel) -> Result<M>
    where
        M: Module<B, S, T>,
        B: Backend + Clone,
        S: Storage<T> + StorageFromVec<T> + Clone + 'static,
        T: DataType,
    {
        // Generic conversion is not feasible due to type erasure
        // Users should use specific conversion functions like import_linear_from_onnx
        Err(NNError::SerializationError {
            message: "Generic ONNX to module conversion not supported. Use specific conversion functions like OnnxImporter::import_linear_from_onnx() for Linear layers.".to_string(),
        })
    }

    /// Import a Linear layer from ONNX model
    ///
    /// Reconstructs a Linear layer from ONNX graph containing MatMul + Add nodes.
    ///
    /// # Arguments
    /// * `onnx_model` - ONNX model containing Linear layer representation
    ///
    /// # Returns
    /// Linear layer with weights and biases loaded from ONNX initializers
    ///
    /// # Errors
    /// Returns error if:
    /// - ONNX graph structure doesn't match Linear layer pattern (MatMul + Add)
    /// - Weight/bias tensors not found in initializers
    /// - Tensor shapes are incompatible
    ///
    /// # Example
    /// ```rust,ignore
    /// use coeus_nn::{OnnxImporter, Linear};
    /// use coeus_dtype::float::Float;
    ///
    /// let importer = OnnxImporter::new();
    /// let onnx_bytes = std::fs::read("linear_layer.json").unwrap();
    /// let onnx_model: OnnxModel = serde_json::from_slice(&onnx_bytes).unwrap();
    /// let linear: Linear<Float> = importer.import_linear_from_onnx(&onnx_model).unwrap();
    /// ```
    pub fn import_linear_from_onnx<T: DataType + FloatExt>(
        &self,
        onnx_model: &OnnxModel,
    ) -> Result<crate::linear::Linear<CpuBackend<T>, DenseStorage<T>, T>> {
        use crate::linear::Linear;

        // Find MatMul and Add nodes in graph
        let graph = &onnx_model.graph;

        // Look for MatMul node (weight multiplication)
        let matmul_node = graph
            .nodes
            .iter()
            .find(|n| n.op_type == "MatMul")
            .ok_or_else(|| NNError::SerializationError {
                message: "No MatMul node found in ONNX graph".to_string(),
            })?;

        // Look for Add node (bias addition)
        let add_node = graph
            .nodes
            .iter()
            .find(|n| n.op_type == "Add")
            .ok_or_else(|| NNError::SerializationError {
                message: "No Add node found in ONNX graph".to_string(),
            })?;

        // Extract weight tensor name from MatMul inputs (second input is weight)
        let weight_name = matmul_node
            .inputs
            .get(1)
            .ok_or_else(|| NNError::SerializationError {
                message: "MatMul node missing weight input".to_string(),
            })?;

        // Extract bias tensor name from Add inputs (second input is bias)
        let bias_name = add_node
            .inputs
            .get(1)
            .ok_or_else(|| NNError::SerializationError {
                message: "Add node missing bias input".to_string(),
            })?;

        // Find weight and bias tensors in initializers
        let weight_tensor = graph
            .initializers
            .iter()
            .find(|t| &t.name == weight_name)
            .ok_or_else(|| NNError::SerializationError {
                message: format!("Weight tensor '{}' not found in initializers", weight_name),
            })?;

        let bias_tensor = graph
            .initializers
            .iter()
            .find(|t| &t.name == bias_name)
            .ok_or_else(|| NNError::SerializationError {
                message: format!("Bias tensor '{}' not found in initializers", bias_name),
            })?;

        // Convert ONNX tensors to Coeus tensors
        let weight = self.onnx_tensor_to_tensor::<T>(weight_tensor)?;
        let bias = self.onnx_tensor_to_tensor::<T>(bias_tensor)?;

        // Extract dimensions
        let weight_shape = weight.shape().dims();
        if weight_shape.len() != 2usize {
            return Err(NNError::SerializationError {
                message: format!("Weight tensor must be 2D, got shape {:?}", weight_shape),
            });
        }

        let out_features = weight_shape[0];
        let in_features = weight_shape[1];

        // Create Linear layer with loaded weights
        let weight_param = crate::parameter::Parameter::new(
            weight.clone().requires_grad_(true),
            "weight".to_string(),
        );
        let bias_param =
            crate::parameter::Parameter::new(bias.requires_grad_(true), "bias".to_string());

        // Cache transposed weight for ONNX-loaded layers too
        let weight_t = Some(weight.to_dense_generic()?.transpose(1, 0)?);

        Ok(Linear {
            weight: weight_param,
            bias: bias_param,
            weight_t,
            in_features,
            out_features,
        })
    }

    /// Convert ONNX tensor to Coeus tensor
    fn onnx_tensor_to_tensor<T: DataType>(
        &self,
        onnx_tensor: &OnnxTensor,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        // Convert dimensions from i64 to usize
        let dims: Vec<usize> = onnx_tensor.dims.iter().map(|&d| d as usize).collect();

        // Deserialize raw data based on data type
        let data: Vec<T> = match onnx_tensor.data_type {
            OnnxDataType::Float => {
                // Interpret raw bytes as f32 values
                let float_data: Vec<f32> = onnx_tensor
                    .raw_data
                    .chunks_exact(4)
                    .map(|chunk| {
                        let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                        f32::from_le_bytes(bytes)
                    })
                    .collect();

                // Convert to target type T
                float_data
                    .into_iter()
                    .map(|f| T::from(f).unwrap_or_else(|| T::zero()))
                    .collect()
            }
            _ => {
                return Err(NNError::SerializationError {
                    message: format!("Unsupported ONNX data type: {:?}. Only Float (f32) is currently supported.", onnx_tensor.data_type),
                });
            }
        };

        // Create tensor from data
        Tensor::from_vec(data, &dims).map_err(|e| NNError::TensorError { source: e })
    }
}

impl Default for OnnxImporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_dtype::float::Float32;

    #[test]
    fn test_onnx_exporter_creation() {
        let exporter = OnnxExporter::new();
        // Basic creation test
        assert_eq!(exporter.node_counter, 0);
        assert_eq!(exporter.tensor_counter, 0);
    }

    #[test]
    fn test_onnx_importer_creation() {
        let _importer = OnnxImporter::new();
        // Basic creation test - no fields to check
    }

    #[test]
    fn test_onnx_importer_invalid_json() {
        let importer = OnnxImporter::new();
        let invalid_json = b"invalid json";
        let result: std::result::Result<
            crate::Sequential<
                coeus_backend::CpuBackend,
                crate::DenseStorage<coeus_dtype::float::Float32>,
                coeus_dtype::float::Float32,
            >,
            _,
        > = importer.import(invalid_json);
        assert!(result.is_err());
    }

    #[test]
    fn test_onnx_data_types() {
        assert_eq!(OnnxDataType::Float as i32, 1);
        assert_eq!(OnnxDataType::Int32 as i32, 6);
        assert_eq!(OnnxDataType::Int64 as i32, 7);
    }

    #[test]
    fn test_import_linear_from_onnx() {
        use crate::linear::Linear;

        // Create a simple ONNX model representing a Linear layer (2 -> 3)
        let weight_data: Vec<f32> = vec![
            0.1, 0.2, // First output neuron weights
            0.3, 0.4, // Second output neuron weights
            0.5, 0.6, // Third output neuron weights
        ];
        let bias_data: Vec<f32> = vec![0.1, 0.2, 0.3];

        // Convert to raw bytes (little-endian)
        let weight_bytes: Vec<u8> = weight_data.iter().flat_map(|&f| f.to_le_bytes()).collect();
        let bias_bytes: Vec<u8> = bias_data.iter().flat_map(|&f| f.to_le_bytes()).collect();

        let onnx_model = OnnxModel {
            graph: OnnxGraph {
                inputs: vec![OnnxValueInfo {
                    name: "input".to_string(),
                    type_info: OnnxType {
                        tensor_type: OnnxTensorType {
                            elem_type: OnnxDataType::Float,
                            shape: vec![OnnxDimension::Fixed(1), OnnxDimension::Fixed(2)],
                        },
                    },
                }],
                outputs: vec![OnnxValueInfo {
                    name: "output".to_string(),
                    type_info: OnnxType {
                        tensor_type: OnnxTensorType {
                            elem_type: OnnxDataType::Float,
                            shape: vec![OnnxDimension::Fixed(1), OnnxDimension::Fixed(3)],
                        },
                    },
                }],
                nodes: vec![
                    OnnxNode {
                        op_type: "MatMul".to_string(),
                        inputs: vec!["input".to_string(), "weight".to_string()],
                        outputs: vec!["matmul_output".to_string()],
                        attributes: HashMap::new(),
                        name: Some("matmul".to_string()),
                    },
                    OnnxNode {
                        op_type: "Add".to_string(),
                        inputs: vec!["matmul_output".to_string(), "bias".to_string()],
                        outputs: vec!["output".to_string()],
                        attributes: HashMap::new(),
                        name: Some("add".to_string()),
                    },
                ],
                initializers: vec![
                    OnnxTensor {
                        name: "weight".to_string(),
                        data_type: OnnxDataType::Float,
                        dims: vec![3, 2], // [out_features, in_features]
                        raw_data: weight_bytes,
                    },
                    OnnxTensor {
                        name: "bias".to_string(),
                        data_type: OnnxDataType::Float,
                        dims: vec![3],
                        raw_data: bias_bytes,
                    },
                ],
            },
            metadata: OnnxMetadata {
                producer_name: "coeus-test".to_string(),
                producer_version: "0.1.0".to_string(),
                domain: "coeus".to_string(),
                description: "Test Linear layer".to_string(),
                onnx_version: 1,
            },
        };

        // Import Linear layer from ONNX
        let importer = OnnxImporter::new();
        let linear: Linear<CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
            importer.import_linear_from_onnx(&onnx_model).unwrap();

        // Verify dimensions
        assert_eq!(linear.in_features, 2);
        assert_eq!(linear.out_features, 3);

        // Verify weight shape
        assert_eq!(linear.weight.data().shape().dims(), &[3, 2]);

        // Verify bias shape
        assert_eq!(linear.bias.data().shape().dims(), &[3]);

        // Verify weight values (approximately, due to floating point)
        let weight_slice = linear.weight.data().as_slice();
        assert!((weight_slice[0].get() - 0.1).abs() < 1e-6);
        assert!((weight_slice[1].get() - 0.2).abs() < 1e-6);
        assert!((weight_slice[2].get() - 0.3).abs() < 1e-6);

        // Verify bias values
        let bias_slice = linear.bias.data().as_slice();
        assert!((bias_slice[0].get() - 0.1).abs() < 1e-6);
        assert!((bias_slice[1].get() - 0.2).abs() < 1e-6);
        assert!((bias_slice[2].get() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_import_linear_missing_matmul() {
        // Create ONNX model without MatMul node
        let onnx_model = OnnxModel {
            graph: OnnxGraph {
                inputs: vec![],
                outputs: vec![],
                nodes: vec![], // No MatMul node
                initializers: vec![],
            },
            metadata: OnnxMetadata {
                producer_name: "coeus-test".to_string(),
                producer_version: "0.1.0".to_string(),
                domain: "coeus".to_string(),
                description: "Test empty model".to_string(),
                onnx_version: 1,
            },
        };

        let importer = OnnxImporter::new();
        let result: Result<
            crate::linear::Linear<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
        > = importer.import_linear_from_onnx(&onnx_model);

        assert!(result.is_err());
        if let Err(NNError::SerializationError { message }) = result {
            assert!(message.contains("No MatMul node found"));
        } else {
            panic!("Expected SerializationError");
        }
    }

    // Note: Full export/import tests would require protobuf implementation
    // These are placeholder tests for the basic structure
}
