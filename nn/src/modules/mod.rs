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

pub mod attention;
pub mod conv;
pub mod dropout;
pub mod embedding;
pub mod gpt2;
pub mod linear;
pub mod normalization;
pub mod pooling;
pub mod rnn;

pub use attention::{
    AttentionConfig, Block, CausalSelfAttention, MultiHeadAttention, Transformer,
    TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer, MLP,
};
pub use conv::{Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d};
pub use dropout::*;
pub use embedding::*;
pub use gpt2::*;
pub use linear::*;
pub use normalization::*;
pub use pooling::{
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d, AdaptiveMaxPool1d, AdaptiveMaxPool2d,
    AdaptiveMaxPool3d, AvgPool2d, MaxPool2d,
};
pub use rnn::{Rnn, Lstm, Gru, RnnCell, LstmCell, GruCell};
