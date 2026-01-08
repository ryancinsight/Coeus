//! # Neural Network Modules and Layers for Coeus
//!
//! Deeply hierarchical neural network implementation for improved maintainability.

// Core infrastructure
pub mod core {
    pub mod autograd_compat;
    pub mod autograd_stub;
    pub mod error;
    pub mod init;
    pub mod module;
    pub mod parameter;
}

// Hierarchical structure
pub mod containers;
pub mod functional;
pub mod io;
pub mod modules;
pub mod research;

pub mod training;

// Primary API re-exports for compatibility
pub use crate::core::error::{NNError, Result};
pub use crate::core::init;
pub use crate::core::module::{Module, ModuleExt};
pub use crate::core::parameter::Parameter;
pub use backend::Backend;

#[cfg(feature = "safetensors")]
pub use crate::core::module::ModuleSerialize;

// Module re-exports
pub use crate::modules::activation::{GeLU, PReLU, ReLU, SiLU, SwiGLU};
pub use crate::modules::attention::{MultiHeadAttention, SparseAttention};
pub use crate::modules::convolution::{Conv1D, Conv2D, Conv3D, ConvTranspose1d};
pub use crate::modules::embedding::Embedding;
pub use crate::modules::linear::{Linear, SparseLinear};
pub use crate::modules::loss::{BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, NLLLoss};
pub use crate::modules::normalization::{
    BatchNorm1d, BatchNorm2d, BatchNorm3d, GroupNorm, LayerNorm, RMSNorm,
};
pub use crate::modules::pooling::{
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveMaxPool2d, AvgPool1d, AvgPool2d, AvgPool3d,
    MaxPool1d, MaxPool2d, MaxPool3d,
};
pub use crate::modules::regularization::dropout::{Dropout, Dropout2d, Dropout3d};
pub use crate::modules::rnn::{GRU, LSTM, RNN};
pub use crate::modules::vision::upsample::Upsample;

// Container re-exports
pub use crate::containers::sequential::Sequential;

// Training re-exports
pub use crate::training::amp;
pub use crate::training::checkpointing;
pub use crate::training::distributed;
pub use crate::training::experiment_tracking;

// Functional API
pub mod functional_api {
    pub use crate::functional::activation::{elu, gelu, leaky_relu, relu, sigmoid, silu, tanh};
    pub use crate::functional::attention::scaled_dot_product_attention;
    pub use crate::functional::convolution::{
        conv1d, conv2d, conv2d_transpose as conv_transpose_2d, conv3d,
    };
    pub use crate::functional::linear::linear;
    pub use crate::functional::loss::{cross_entropy, mse_loss, nll_loss};
    pub use crate::functional::normalization::{batch_norm, layer_norm};
    pub use crate::functional::ops::activations::{softmax, softmax_dim};
    pub use crate::functional::ops::dropout;
    pub use crate::functional::ops::loss::bce_with_logits_loss;
    pub use crate::functional::pooling::{avg_pool2d, max_pool2d};
}

// Specialized modules
#[cfg(feature = "clip")]
pub mod clip;
#[cfg(feature = "clip")]
pub mod datasets;
#[cfg(feature = "clip")]
pub mod evaluation;
#[cfg(feature = "multimodal")]
pub mod multimodal;

pub mod feature;
pub mod hpo;
pub mod meta;
pub mod nas;

// HPO/NAS re-exports
pub use crate::hpo::optimizer::{BenchmarkRunner, HyperparameterOptimizer, OptimizationResult};
pub use crate::hpo::space::{HyperparameterConfig, HyperparameterSpace};
pub use crate::meta::maml::MAML;
pub use crate::meta::prototypical::{FewShotEpisodeGenerator, PrototypicalNetwork};
pub use crate::nas::search_space::{Architecture, ArchitectureSpace};
