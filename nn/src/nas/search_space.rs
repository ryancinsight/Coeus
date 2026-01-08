//! Architecture search space definitions.
//!
//! This module defines the search spaces for neural architecture search,
//! including architecture representations and search space constraints.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::core::error::{NNError, Result};

/// Represents a neural network architecture
#[derive(Debug, Clone, PartialEq)]
pub struct Architecture {
    /// Architecture type (CNN, RNN, Transformer, etc.)
    pub architecture_type: ArchitectureType,
    /// Layers in the architecture
    pub layers: Vec<LayerSpec>,
    /// Connections between layers
    pub connections: Vec<Connection>,
    /// Global architecture parameters
    pub parameters: HashMap<String, f64>,
}

impl Architecture {
    /// Create a new architecture
    pub fn new(architecture_type: ArchitectureType) -> Self {
        Self {
            architecture_type,
            layers: Vec::new(),
            connections: Vec::new(),
            parameters: HashMap::new(),
        }
    }

    /// Add a layer to the architecture
    pub fn add_layer(&mut self, layer: LayerSpec) -> &mut Self {
        self.layers.push(layer);
        self
    }

    /// Add a connection between layers
    pub fn add_connection(&mut self, from: usize, to: usize) -> &mut Self {
        self.connections.push(Connection { from, to });
        self
    }

    /// Set a parameter
    pub fn set_parameter(&mut self, key: String, value: f64) -> &mut Self {
        self.parameters.insert(key, value);
        self
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.layers.iter().map(|layer| layer.num_parameters()).sum()
    }

    /// Validate architecture
    pub fn validate(&self) -> Result<()> {
        // Check that all connections reference valid layers
        for conn in &self.connections {
            if conn.from >= self.layers.len() || conn.to >= self.layers.len() {
                return Err(NNError::InvalidConfiguration {
                    message: format!(
                        "Invalid connection: {} -> {} (only {} layers)",
                        conn.from,
                        conn.to,
                        self.layers.len()
                    ),
                });
            }
        }

        // Check for cycles (simplified check)
        if self.has_cycles() {
            return Err(NNError::InvalidConfiguration {
                message: "Architecture contains cycles".to_string(),
            });
        }

        Ok(())
    }

    /// Check for cycles in the architecture graph
    fn has_cycles(&self) -> bool {
        // Simple cycle detection using DFS
        let mut visited = vec![false; self.layers.len()];
        let mut rec_stack = vec![false; self.layers.len()];

        for i in 0..self.layers.len() {
            if self.has_cycles_util(i, &mut visited, &mut rec_stack) {
                return true;
            }
        }
        false
    }

    fn has_cycles_util(&self, node: usize, visited: &mut [bool], rec_stack: &mut [bool]) -> bool {
        if rec_stack[node] {
            return true;
        }
        if visited[node] {
            return false;
        }

        visited[node] = true;
        rec_stack[node] = true;

        // Check all outgoing connections
        for conn in &self.connections {
            if conn.from == node && self.has_cycles_util(conn.to, visited, rec_stack) {
                return true;
            }
        }

        rec_stack[node] = false;
        false
    }
}

/// Layer specification
#[derive(Debug, Clone, PartialEq)]
pub enum LayerSpec {
    /// Convolutional layer
    Conv2D {
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    },
    /// Fully connected layer
    Linear { out_features: usize },
    /// RNN layer
    RNN {
        hidden_size: usize,
        num_layers: usize,
        bidirectional: bool,
    },
    /// Transformer layer
    Transformer {
        num_heads: usize,
        feedforward_dim: usize,
        dropout: f64,
    },
    /// Attention layer
    Attention {
        num_heads: usize,
        sparse_pattern: Option<String>,
    },
    /// Pooling layer
    Pooling {
        pool_type: PoolType,
        kernel_size: usize,
        stride: usize,
    },
    /// Normalization layer
    Normalization { norm_type: NormType },
    /// Activation layer
    Activation { activation_type: ActivationType },
    /// Dropout layer
    Dropout { dropout_rate: f64 },
}

impl LayerSpec {
    /// Get the number of parameters in this layer
    pub fn num_parameters(&self) -> usize {
        match self {
            LayerSpec::Conv2D { out_channels, .. } => {
                // Rough estimate: in_channels * out_channels * kernel_size * kernel_size + out_channels
                // Using average in_channels = 64 for estimation
                64 * out_channels * 9 + out_channels
            }
            LayerSpec::Linear { out_features } => {
                // Rough estimate: in_features * out_features + out_features
                // Using average in_features = 1024 for estimation
                1024 * out_features + out_features
            }
            LayerSpec::RNN {
                hidden_size,
                num_layers,
                bidirectional,
            } => {
                let directions = if *bidirectional { 2 } else { 1 };
                // Rough estimate for RNN parameters
                directions * hidden_size * (1024 + hidden_size) * num_layers
            }
            LayerSpec::Transformer {
                num_heads: _,
                feedforward_dim,
                ..
            } => {
                // Rough estimate: attention params + feedforward params
                1024 * 1024 + 1024 * feedforward_dim * 2
            }
            LayerSpec::Attention { num_heads, .. } => {
                // Rough estimate for attention parameters
                1024 * 1024 + 1024 * num_heads
            }
            _ => 0, // Other layers have negligible parameters
        }
    }
}

/// Pooling types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolType {
    MaxPool,
    AvgPool,
    AdaptiveAvgPool,
}

/// Normalization types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormType {
    BatchNorm,
    LayerNorm,
    GroupNorm { groups: usize },
}

/// Activation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationType {
    ReLU,
    GELU,
    Tanh,
    Sigmoid,
    Swish,
}

/// Architecture types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchitectureType {
    /// Convolutional Neural Network
    CNN,
    /// Recurrent Neural Network
    RNN,
    /// Transformer architecture
    Transformer,
    /// Hybrid architecture
    Hybrid,
}

/// Layer types available in search space
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LayerType {
    Conv2D,
    Linear,
    Attention,
    Pooling,
    Normalization,
    Activation,
}

/// Connection between layers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Connection {
    /// Source layer index
    pub from: usize,
    /// Target layer index
    pub to: usize,
}

/// Architecture search space definition
#[derive(Debug, Clone)]
pub struct ArchitectureSpace {
    /// Maximum number of layers
    pub max_layers: usize,
    /// Available layer types
    pub layer_types: Vec<LayerType>,
    /// Parameter ranges for each layer type
    pub parameter_ranges: HashMap<LayerType, ParameterRange>,
    /// Architecture type
    pub architecture_type: ArchitectureType,
}

impl ArchitectureSpace {
    /// Create a new architecture search space
    pub fn new(architecture_type: ArchitectureType) -> Self {
        Self {
            max_layers: 20,
            layer_types: Vec::new(),
            parameter_ranges: HashMap::new(),
            architecture_type,
        }
    }

    /// Add a layer type to the search space
    pub fn add_layer_type(&mut self, layer_type: LayerType, params: ParameterRange) -> &mut Self {
        self.layer_types.push(layer_type.clone());
        self.parameter_ranges.insert(layer_type, params);
        self
    }

    /// Sample a random architecture from the search space
    pub fn sample_random(&self, num_layers: usize) -> Result<Architecture> {
        use rand::Rng;

        let mut rng = rand::thread_rng();
        // Ensure at least 3 layers (input + 1 hidden + output)
        let max_layers = num_layers.min(self.max_layers);
        let actual_layers = if max_layers < 3 {
            3
        } else {
            rng.gen_range(3..=max_layers)
        };

        let mut architecture = Architecture::new(self.architecture_type);

        // Add input layer
        architecture.add_layer(LayerSpec::Conv2D {
            out_channels: 32,
            kernel_size: 3,
            stride: 1,
            padding: 1,
        });

        // Add hidden layers
        if !self.layer_types.is_empty() {
            for _ in 1..actual_layers - 1 {
                let layer_type = &self.layer_types[rng.gen_range(0..self.layer_types.len())];
                let layer = self.sample_layer(layer_type)?;
                architecture.add_layer(layer);
            }
        } else {
            // Fallback if no layer types defined (avoid panic)
            for _ in 1..actual_layers - 1 {
                architecture.add_layer(LayerSpec::Conv2D {
                    out_channels: 64,
                    kernel_size: 3,
                    stride: 1,
                    padding: 1,
                });
            }
        }

        // Add output layer
        architecture.add_layer(LayerSpec::Linear { out_features: 10 });

        // Add sequential connections
        for i in 0..actual_layers - 1 {
            architecture.add_connection(i, i + 1);
        }

        architecture.validate()?;
        Ok(architecture)
    }

    /// Sample a layer of the given type
    pub fn sample_layer(&self, layer_type: &LayerType) -> Result<LayerSpec> {
        use rand::Rng;

        let mut rng = rand::thread_rng();

        match layer_type {
            LayerType::Conv2D => {
                let range = self.parameter_ranges.get(layer_type).ok_or_else(|| {
                    NNError::InvalidConfiguration {
                        message: format!("Missing parameter range for layer type {:?}", layer_type),
                    }
                })?;

                let out_channels = if range.out_channels.0 > range.out_channels.1 {
                    range.out_channels.0
                } else {
                    rng.gen_range(range.out_channels.0..=range.out_channels.1)
                };

                let kernel_size = if range.kernel_size.0 > range.kernel_size.1 {
                    range.kernel_size.0
                } else {
                    rng.gen_range(range.kernel_size.0..=range.kernel_size.1)
                };

                let stride = if range.stride.0 > range.stride.1 {
                    range.stride.0
                } else {
                    rng.gen_range(range.stride.0..=range.stride.1)
                };

                let padding = if range.padding.0 > range.padding.1 {
                    range.padding.0
                } else {
                    rng.gen_range(range.padding.0..=range.padding.1)
                };

                Ok(LayerSpec::Conv2D {
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                })
            }
            LayerType::Linear => {
                let range = self.parameter_ranges.get(layer_type).ok_or_else(|| {
                    NNError::InvalidConfiguration {
                        message: format!("Missing parameter range for layer type {:?}", layer_type),
                    }
                })?;

                let out_features = if range.out_features.0 > range.out_features.1 {
                    range.out_features.0
                } else {
                    rng.gen_range(range.out_features.0..=range.out_features.1)
                };

                Ok(LayerSpec::Linear { out_features })
            }
            LayerType::Attention => {
                let range = self.parameter_ranges.get(layer_type).ok_or_else(|| {
                    NNError::InvalidConfiguration {
                        message: format!("Missing parameter range for layer type {:?}", layer_type),
                    }
                })?;

                let num_heads = if range.num_heads.0 > range.num_heads.1 {
                    range.num_heads.0
                } else {
                    rng.gen_range(range.num_heads.0..=range.num_heads.1)
                };

                Ok(LayerSpec::Attention {
                    num_heads,
                    sparse_pattern: None, // Could be randomized
                })
            }
            _ => Err(NNError::InvalidConfiguration {
                message: format!("Layer type {:?} not supported for sampling", layer_type),
            }),
        }
    }
}

/// Parameter ranges for layer sampling
#[derive(Debug, Clone)]
pub struct ParameterRange {
    pub out_channels: (usize, usize),
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub out_features: (usize, usize),
    pub num_heads: (usize, usize),
}

impl Hash for Architecture {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash architecture type
        self.architecture_type.hash(state);

        // Hash layers (simplified - just count and basic info)
        self.layers.len().hash(state);
        for layer in &self.layers {
            match layer {
                LayerSpec::Conv2D {
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                } => {
                    0u8.hash(state); // discriminant
                    out_channels.hash(state);
                    kernel_size.hash(state);
                    stride.hash(state);
                    padding.hash(state);
                }
                LayerSpec::Linear { out_features } => {
                    1u8.hash(state);
                    out_features.hash(state);
                }
                LayerSpec::RNN {
                    hidden_size,
                    num_layers,
                    bidirectional,
                } => {
                    2u8.hash(state);
                    hidden_size.hash(state);
                    num_layers.hash(state);
                    bidirectional.hash(state);
                }
                LayerSpec::Transformer {
                    num_heads,
                    feedforward_dim,
                    dropout,
                } => {
                    3u8.hash(state);
                    num_heads.hash(state);
                    feedforward_dim.hash(state);
                    // Hash dropout as bits
                    ((*dropout * 1000.0) as i32).hash(state);
                }
                LayerSpec::Attention {
                    num_heads,
                    sparse_pattern,
                } => {
                    4u8.hash(state);
                    num_heads.hash(state);
                    sparse_pattern.hash(state);
                }
                LayerSpec::Pooling {
                    pool_type,
                    kernel_size,
                    stride,
                } => {
                    5u8.hash(state);
                    pool_type.hash(state);
                    kernel_size.hash(state);
                    stride.hash(state);
                }
                LayerSpec::Normalization { norm_type } => {
                    6u8.hash(state);
                    norm_type.hash(state);
                }
                LayerSpec::Activation { activation_type } => {
                    7u8.hash(state);
                    activation_type.hash(state);
                }
                LayerSpec::Dropout { dropout_rate } => {
                    8u8.hash(state);
                    ((*dropout_rate * 1000.0) as i32).hash(state);
                }
            }
        }

        // Hash connections
        self.connections.len().hash(state);
        for conn in &self.connections {
            conn.from.hash(state);
            conn.to.hash(state);
        }
    }
}

impl Default for ParameterRange {
    fn default() -> Self {
        Self {
            out_channels: (16, 512),
            kernel_size: (1, 7),
            stride: (1, 4),
            padding: (0, 3),
            out_features: (64, 2048),
            num_heads: (1, 16),
        }
    }
}
