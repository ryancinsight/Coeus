//! Layer fusion operations for quantization optimization

use crate::core::error::Result;
use crate::core::module::Module;
use crate::core::parameter::Parameter;

use quantization::{QuantizationBitwidth, QuantizationScheme};

use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

use serde::{Deserialize, Serialize};

/// Fusion configuration for layer fusion operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Type of fusion pattern to apply
    pub pattern: FusionPattern,
    /// Whether to enable quantization for the fused operation
    pub enable_quantization: bool,
    /// Target bitwidth for quantization (if enabled)
    pub target_bitwidth: QuantizationBitwidth,
    /// Quantization scheme (if enabled)
    pub scheme: QuantizationScheme,
}

/// Supported fusion patterns for neural network layers
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FusionPattern {
    /// Convolution followed by Batch Normalization
    ConvBatchNorm,
    /// Convolution followed by ReLU activation
    ConvReLU,
    /// Linear layer followed by activation function
    LinearActivation(ActivationType),
    /// Multi-head attention with fused projections
    AttentionFusion,
}

/// Activation function types for fusion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
}

/// Fused Convolution + Batch Normalization layer
#[derive(Debug)]
pub struct FusedConvBatchNorm<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + Clone + PartialOrd + Into<f64> + From<f64>,
    f64: From<T>,
{
    /// Fused convolution weights (pre-multiplied with BN parameters)
    pub weight: Parameter<B, S, T>,
    /// Fused bias (incorporating BN bias and running stats)
    pub bias: Option<Parameter<B, S, T>>,
    /// Convolution parameters
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    /// Fusion configuration
    pub config: FusionConfig,
    /// Quantized version (if quantization enabled)
    pub quantized: Option<FusedConvBatchNormQuantized<B, S, T>>,
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

/// Quantized version of fused Conv + BatchNorm
#[derive(Debug)]
pub struct FusedConvBatchNormQuantized<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType,
{
    /// Quantized fused weights
    pub weight: Parameter<B, S, T>,
    /// Quantized bias
    pub bias: Option<Parameter<B, S, T>>,
    /// Quantization parameters
    pub weight_scale: T,
    pub weight_zero_point: T,
    pub input_scale: T,
    pub input_zero_point: T,
    /// Convolution parameters
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

/// Fused Linear + Activation layer
#[derive(Debug)]
pub struct FusedLinearActivation<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + Clone + PartialOrd,
{
    /// Linear layer weights
    pub weight: Parameter<B, S, T>,
    /// Linear layer bias
    pub bias: Option<Parameter<B, S, T>>,
    /// Activation type
    pub activation: ActivationType,
    /// Fusion configuration
    pub config: FusionConfig,
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

/// Fusion policy for automatic layer fusion decisions
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FusionPolicy {
    /// Minimum performance gain threshold for fusion
    pub min_gain_threshold: f64,
    /// Maximum memory overhead allowed for fusion
    pub max_memory_overhead: f64,
    /// Whether to prioritize accuracy over performance
    pub prioritize_accuracy: bool,
    /// Hardware-specific fusion preferences
    pub hardware_hints: Vec<String>,
}

/// Fusion pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionPipelineConfig {
    /// Global fusion policy
    pub policy: FusionPolicy,
    /// Layer-specific fusion configurations
    pub layer_configs: std::collections::HashMap<String, FusionConfig>,
    /// Whether to enable automatic fusion detection
    pub enable_auto_fusion: bool,
    /// Performance profiling data
    pub profiling_data: Option<std::collections::HashMap<String, f64>>,
}
