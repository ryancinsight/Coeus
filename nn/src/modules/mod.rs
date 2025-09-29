//! Neural network modules and layers
//!
//! This module provides PyTorch-like neural network components including:
//! - Linear layers for fully connected networks
//! - Convolutional layers for feature extraction
//! - Pooling layers for dimensionality reduction
//! - Recurrent layers for sequence processing
//! - Normalization layers for training stability
//! - Dropout for regularization
//! - Embedding layers for token representations

pub mod adaptive_avgpool1d;
pub mod adaptive_avgpool2d;
pub mod adaptive_avgpool3d;
pub mod adaptive_maxpool1d;
pub mod adaptive_maxpool2d;
pub mod adaptive_maxpool3d;
pub mod attention;
pub mod attention_config;
pub mod causal_self_attention;
pub mod multihead_attention;
pub mod avgpool1d;
pub mod avgpool2d;
pub mod avgpool3d;
pub mod conv;
pub mod conv1d;
pub mod conv2d;
pub mod conv_transpose2d;
pub mod dropout;
pub mod embedding;
pub mod gpt2;
pub mod batch_norm1d;
pub mod batch_norm2d;
pub mod batch_norm3d;
pub mod group_norm;
pub mod gru;
pub mod instance_norm1d;
pub mod instance_norm2d;
pub mod instance_norm3d;
pub mod layer_norm;
pub mod linear;
pub mod lstm;
pub mod maxpool1d;
pub mod maxpool2d;
pub mod maxpool3d;
pub mod normalization;
pub mod pooling;
pub mod rnn;
pub mod rnn_types;

pub use adaptive_avgpool1d::AdaptiveAvgPool1d;
pub use adaptive_avgpool2d::AdaptiveAvgPool2d;
pub use adaptive_avgpool3d::AdaptiveAvgPool3d;
pub use adaptive_maxpool1d::AdaptiveMaxPool1d;
pub use adaptive_maxpool2d::AdaptiveMaxPool2d;
pub use adaptive_maxpool3d::AdaptiveMaxPool3d;
pub use attention::{
    AttentionConfig as LegacyAttentionConfig, Block, CausalSelfAttention as LegacyCausalSelfAttention,
    MultiHeadAttention as LegacyMultiHeadAttention, Transformer,
    TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer, MLP,
};
pub use attention_config::AttentionConfig;
pub use causal_self_attention::CausalSelfAttention;
pub use multihead_attention::MultiHeadAttention;
pub use avgpool1d::AvgPool1d;
pub use avgpool2d::AvgPool2d;
pub use avgpool3d::AvgPool3d;
pub use conv::{Conv1d, Conv2d, ConvTranspose2d};
pub use conv1d::Conv1d as Conv1dModular;
pub use conv2d::Conv2d as Conv2dModular;
pub use conv_transpose2d::ConvTranspose2d as ConvTranspose2dModular;
pub use dropout::*;
pub use embedding::*;
pub use gpt2::*;
pub use gru::{Gru, GruCell};
pub use linear::*;
pub use lstm::{Lstm, LstmCell, LstmOutput};
pub use maxpool1d::MaxPool1d;
pub use maxpool2d::MaxPool2d;
pub use maxpool3d::MaxPool3d;
pub use batch_norm1d::BatchNorm1d;
pub use batch_norm2d::BatchNorm2d;
pub use batch_norm3d::BatchNorm3d;
pub use group_norm::GroupNorm;
pub use instance_norm1d::InstanceNorm1d;
pub use instance_norm2d::InstanceNorm2d;
pub use instance_norm3d::InstanceNorm3d;
pub use layer_norm::LayerNorm;
// Legacy normalization structs are no longer available - use modular versions
pub use pooling::{
    AdaptiveAvgPool1d as PoolingAdaptiveAvgPool1d, AdaptiveAvgPool2d as PoolingAdaptiveAvgPool2d,
    AdaptiveAvgPool3d as PoolingAdaptiveAvgPool3d, AdaptiveMaxPool1d as PoolingAdaptiveMaxPool1d,
    AdaptiveMaxPool2d as PoolingAdaptiveMaxPool2d, AdaptiveMaxPool3d as PoolingAdaptiveMaxPool3d,
    AvgPool1d as PoolingAvgPool1d, AvgPool2d as PoolingAvgPool2d, AvgPool3d as PoolingAvgPool3d,
    MaxPool1d as PoolingMaxPool1d, MaxPool2d as PoolingMaxPool2d, MaxPool3d as PoolingMaxPool3d,
};
pub use rnn_types::{Rnn, RnnCell};


